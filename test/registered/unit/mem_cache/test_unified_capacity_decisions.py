"""Capacity decisions over a unified pool ask a demand: a scalar read inside one is
a violation, and the prefill budget's token-count questions price demands instead."""

import logging
import random
import unittest
from array import array
from types import MethodType, SimpleNamespace
from unittest.mock import patch

from test_prefill_memory_budget import (
    _cache,
    _multi_window_state_allocator,
    _shared_allocator,
)

from sglang.srt.environ import InvariantCheckLevel, envs
from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.managers.schedule_policy import PrefillAdder
from sglang.srt.mem_cache.base_prefix_cache import (
    EvictParams,
    InsertParams,
    MatchPrefixParams,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.capacity import (
    capacity_decision,
    create_prefill_budget,
    evict_to_free_tokens,
)
from sglang.srt.mem_cache.common import release_plan
from sglang.srt.mem_cache.kv.alloc.demand import KVDemand, WindowRetirement
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixCache, RadixKey
from sglang.srt.mem_cache.static_kv_storage import kv_storage_of
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.runtime_context import reset_context
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.scheduler_test_utils import release_only_retirer
from sglang.test.test_utils import CustomTestCase
from sglang.test.tree_slots import storage_tree_slots
from sglang.test.unified_allocator_fixtures import (
    build_swa_cache,
    build_swa_pool,
    build_tri_cache,
    build_tri_pool,
    largest_take,
    setup_allocator_context,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def setUpModule():
    patcher = patch(
        "sglang.srt.mem_cache.prefill_budget.window_retirement",
        return_value=WindowRetirement(0, 0, 1),
    )
    patcher.start()
    unittest.addModuleCleanup(patcher.stop)


@capacity_decision
def _decide(read):
    return read()


def _rowless(rid):
    """A preemption candidate holding no request row: its release frees no
    window row."""
    return SimpleNamespace(rid=rid, kv=ReqKvInfo())


class TestEstimatesAreObserveOnly(CustomTestCase):
    def setUp(self):
        setup_allocator_context()
        self.addCleanup(reset_context)
        _, self.allocator = build_tri_pool(page_size=4)
        owner, (window,), state = self.allocator.sub_pool_roles()
        self.reads = {
            "available_size": self.allocator.available_size,
            "owner_available_size": self.allocator.owner_available_size,
            "full_available_size": self.allocator.full_available_size,
            "window_available_size": lambda: self.allocator.window_available_size(
                window
            ),
            "window_capacity_and_available": (
                lambda: self.allocator.window_capacity_and_available(window)
            ),
            "usage": self.allocator.usage,
            "state slots": self.allocator.state_slots.schedulable_available_size,
        }

    def test_strict_checks_refuse_an_estimate_inside_a_decision(self):
        with envs.SGLANG_INVARIANT_CHECK.override(InvariantCheckLevel.STRICT):
            for name, read in self.reads.items():
                with self.subTest(read=name):
                    read()  # an observer may read it
                    with self.assertRaisesRegex(AssertionError, "observe-only"):
                        _decide(read)

    def test_warn_logs_and_off_stays_silent(self):
        read = self.reads["full_available_size"]
        with envs.SGLANG_INVARIANT_CHECK.override(InvariantCheckLevel.WARN):
            with self.assertLogs("sglang.srt.mem_cache.capacity", logging.WARNING):
                self.assertEqual(_decide(read), read())
        with envs.SGLANG_INVARIANT_CHECK.override(InvariantCheckLevel.OFF):
            with self.assertNoLogs("sglang.srt.mem_cache.capacity", logging.WARNING):
                self.assertEqual(_decide(read), read())

    def test_demand_questions_pass_inside_a_decision(self):
        allocator = self.allocator
        with envs.SGLANG_INVARIANT_CHECK.override(InvariantCheckLevel.STRICT):
            self.assertTrue(
                _decide(
                    lambda: allocator.ensure_capacity(
                        KVDemand.same_tokens(allocator, 4, state_slots=1)
                    )
                )
            )
            self.assertTrue(_decide(lambda: allocator.state_slots.slots_fit(1)))

    def test_estimate_bounds_every_take_and_keeps_the_census_exact(self):
        allocator = self.allocator
        owner, (window,), state = allocator.sub_pool_roles()
        self.assertIsNotNone(allocator.alloc(12))
        self.assertIsNotNone(allocator.state_slots.alloc(2))
        for name in (owner, window, state):
            with self.subTest(sub_pool=name):
                # Bytes no live page holds bound what any take finds.
                self.assertGreaterEqual(
                    allocator.estimated_units_available(name),
                    largest_take(allocator, name),
                )
                capacity, free = allocator.usage()[name]
                self.assertEqual(capacity - free, allocator.held_units(name))
        self.assertEqual(allocator.held_units(owner), 12)
        self.assertEqual(allocator.held_units(state), 2)

    def test_state_slots_fit_is_the_take(self):
        slots = self.allocator.state_slots
        room = largest_take(self.allocator, self.allocator.sub_pool_roles()[2])
        self.assertTrue(slots.slots_fit(room))
        self.assertFalse(slots.slots_fit(room + 1))
        self.assertIsNotNone(slots.alloc(room))
        self.assertFalse(slots.slots_fit(1))


class TestPrefillBudgetPricesDemands(CustomTestCase):
    def setUp(self):
        self.allocator = _shared_allocator(pool_tokens=256)
        self.addCleanup(reset_context)
        self.budget = create_prefill_budget(self.allocator, _cache())
        self.request = dict(
            extend_input_len=12,
            max_new_tokens=4,
            chunk_limit=16,
            state_slots=0,
            cached_tokens=0,
        )

    def test_the_budget_has_no_remaining_token_count(self):
        self.assertTrue(self.budget.prices_by_demand)
        for name in ("remaining_total", "remaining_current", "remaining_window_tokens"):
            with self.subTest(read=name), self.assertRaises(TypeError):
                getattr(self.budget, name)

    def test_ignore_eos_prices_the_worst_stage(self):
        page = self.allocator.page_size
        # Running requests (tokens left, tokens held), fewest left first.
        states = [(4, 16), (8, 16), (20, 8)]
        # Stages: (4 + 1) * 3 = 15, then (8 + 1) * 2 - 16 = 2, then
        # (20 + 1) - 32 < 0; the first is the worst.
        peak = 15
        request = self.budget.request_demand(
            12, 4, chunk_limit=16, state_slots=0, cached_tokens=0
        )
        for paged_input in (16, 64, 4096):
            with self.subTest(paged_input=paged_input):
                self.assertEqual(
                    self.budget.ignore_eos_decode_fits(
                        states,
                        paged_input=paged_input,
                        reserve_tokens=1,
                        **self.request,
                    ),
                    self.budget._fits(
                        paged_input + peak,
                        request.window_tokens,
                        request.state_slots,
                        now=True,
                    ),
                )
        self.assertTrue(
            self.budget.ignore_eos_decode_fits(
                states, paged_input=2 * page, reserve_tokens=1, **self.request
            )
        )
        self.assertFalse(
            self.budget.ignore_eos_decode_fits(
                states, paged_input=1 << 20, reserve_tokens=1, **self.request
            )
        )

    def test_preemption_releases_the_fewest_candidates_that_admit(self):
        # Running requests' pending decode is what the budget holds back:
        # twice the pool between them.
        owner = self.allocator.sub_pool_roles()[0]
        released = [largest_take(self.allocator, owner) / 2] * 4
        self.budget.total_offset += sum(released)
        candidates = [_rowless(i) for i in range(len(released))]
        demand = self.budget.request_demand(
            12, 4, chunk_limit=16, state_slots=0, extra_tokens=0, cached_tokens=0
        )
        request = dict(total_tokens=16, admits=lambda: self.budget._admits(demand))

        def admitted(count):
            return self.budget._fits(
                demand.full_tokens - sum(released[:count]),
                demand.window_tokens,
                demand.state_slots,
            )

        count = self.budget.preemptions_needed(candidates, released, **request)
        expected = next((c for c in range(1, len(released) + 1) if admitted(c)), None)
        self.assertEqual(count, expected)
        # One release is not enough here, and all of them are.
        self.assertIsNotNone(count)
        self.assertGreater(count, 1)
        # The budget is left as it was.
        self.assertEqual(self.budget.total_offset, sum(released))
        self.assertIsNone(self.budget.preemptions_needed([], [], **request))

    def test_largest_chunk_is_the_largest_admitted(self):
        def fits(tokens, now):
            return self.budget._chunk_fits(
                tokens,
                0,
                chunk_limit=None,
                host_hit_lengths=None,
                state_slots=0,
                now=now,
            )

        self.budget.reserve(12, 4, chunk_limit=16)
        for now in (False, True):
            with self.subTest(now=now):
                largest = self.budget.largest_chunk_tokens(4096, now=now)
                self.assertGreater(largest, 0)
                self.assertTrue(fits(largest, now))
                self.assertFalse(fits(largest + 1, now))


class TestReleasePlanIsWhatRetractionFrees(CustomTestCase):
    """Derived property: the release plan preemption prices a victim at is, by
    sub-pool, exactly what retracting the victim frees: every page its rows touch
    (a partial last page and an over-allocated tail too), none of the tree's
    prefix, no window row retired before, and every state slot it fills (a lazy
    second tracking slot too)."""

    def setUp(self):
        setup_allocator_context()
        self.addCleanup(reset_context)
        # Speculative decoding over-allocates a tail past the committed rows.
        spec = patch(
            "sglang.srt.mem_cache.common.get_spec",
            return_value=SimpleNamespace(speculative_algorithm="EAGLE"),
        )
        spec.start()
        self.addCleanup(spec.stop)

    @staticmethod
    def _caches():
        """One window (the shared free path), one window with state slots, and
        two windows with lazily tracked state slots (the per-window free path)."""
        yield "window", build_swa_cache(occupancy=0, page_size=4, window=8)[1], 24
        yield "window+state", build_tri_cache(page_size=1)[2], 24
        storage = _multi_window_state_allocator(
            extra_buffer=True, lazy_tracking=True, tokens=32
        )
        yield (
            "windows+lazy state",
            UnifiedRadixCache(
                CacheInitParams(
                    disable=False,
                    req_to_token_pool=storage.requests,
                    token_to_kv_pool_allocator=storage.allocator,
                    page_size=4,
                    sliding_window_size=16,
                    tree_slots=storage_tree_slots(storage),
                    enable_mamba_extra_buffer=True,
                )
            ),
            24,
        )

    @staticmethod
    def _victim(cache, rng, rid, max_tokens):
        """A running request whose row holds a random state: a tree prefix it
        locks, a partial last page, an over-allocated tail, window rows retired
        and a second tracking slot taken."""
        allocator = cache.token_to_kv_pool_allocator
        storage = kv_storage_of(cache.req_to_token_pool, allocator)
        page = allocator.page_size
        protected = rng.choice([0, page])
        owned = rng.randint(protected + 1, max_tokens - page)
        allocated = owned + rng.choice([0, rng.randint(1, page + 2)])
        tokens = array("q", range(1000 * rid, 1000 * rid + allocated))
        req = Req(
            rid=str(rid),
            origin_input_text="",
            origin_input_ids=tokens[:owned],
            sampling_params=SamplingParams(temperature=0, max_new_tokens=8),
        )
        (row,) = storage.acquire([req])
        ids = allocator.alloc(-(-allocated // page) * page)
        cache.req_to_token_pool.write((row, slice(0, allocated)), ids[:allocated])
        req.kv.kv_committed_len, req.kv.kv_allocated_len = owned, allocated
        if protected:
            key = RadixKey(tokens[:protected])
            cache.insert(
                InsertParams(
                    key=key, value=ids[:protected].clone(), allow_missing_mamba=True
                )
            )
            node = cache.match_prefix(MatchPrefixParams(key=key)).last_device_node
            req.lock = cache.lock(node)
            req.kv.protect_prefix(protected)
        for name in allocator.window_pool_names():
            cursor = rng.choice(range(protected, owned // page * page + 1, page))
            if cursor > protected:
                allocator.free_window_segment(
                    name, ids[protected:cursor], start_pos=protected
                )
                req.kv.set_evicted_seqlen(name, cursor)
        track = req.kv.mamba_ping_pong_empty
        if track is not None and track[1] and rng.random() < 0.5:
            assert storage.take_tracking_slot(req, 1), "no tracking slot to take"
        return req

    def test_plan_equals_the_retraction_frees(self):
        rng = random.Random(0)
        for kind, cache, max_tokens in self._caches():
            allocator = cache.token_to_kv_pool_allocator
            owner, windows, state = allocator.sub_pool_roles()
            names = [owner, *windows] + ([state] if state is not None else [])
            for case in range(24):
                with self.subTest(cache=kind, case=case):
                    req = self._victim(cache, rng, case + 1, max_tokens)
                    held = {name: allocator.held_units(name) for name in names}
                    plan = release_plan(req, cache)
                    release_only_retirer(tree_cache=cache).retract(
                        req, offload_kv=False
                    )
                    freed = {
                        name: held[name] - allocator.held_units(name) for name in names
                    }
                    self.assertEqual({name: plan.get(name, 0) for name in names}, freed)
                    cache.evict(EvictParams(num_tokens=1 << 20))

    def test_a_row_its_session_keeps_frees_nothing(self):
        _, cache = build_swa_cache(occupancy=0, page_size=4, window=8, sessions=True)
        allocator = cache.token_to_kv_pool_allocator
        names = [allocator.sub_pool_roles()[0], *allocator.window_pool_names()]
        turn = self._victim(cache, random.Random(0), 1, 24)
        turn.session = SimpleNamespace(
            session_id="session",
            streaming=True,
            finish_req=lambda req: None,
            abort_req=lambda: None,
        )
        cache.maybe_hand_to_session(turn)
        held = [allocator.held_units(name) for name in names]
        self.assertFalse(any(release_plan(turn, cache).values()))
        release_only_retirer(tree_cache=cache).retract(turn, offload_kv=False)
        self.assertEqual([allocator.held_units(name) for name in names], held)


class TestDllmBlockWaitsWhenNoneFits(CustomTestCase):
    """Regression: when no dLLM block fits a full FULL + window pool, the block size
    is zero, so a request memory cannot hold waits instead of being staged."""

    def test_a_full_pool_leaves_no_block(self):
        allocator = _shared_allocator()
        budget = create_prefill_budget(allocator, _cache())
        while allocator.alloc(4) is not None:
            pass
        self.assertEqual(budget.largest_chunk_tokens(64), 0)
        adder = SimpleNamespace(
            memory_budget=budget,
            dllm_config=SimpleNamespace(requires_separate_context_encoding=False),
            rem_dllm_tokens=64,
            dllm_block_size=64,
        )
        adder._get_dllm_remain_tokens_by_demand = MethodType(
            PrefillAdder._get_dllm_remain_tokens_by_demand, adder
        )
        self.assertEqual(PrefillAdder._get_dllm_remain_tokens(adder), 0)


class TestPlainTreeReclaimsADemand(CustomTestCase):
    """Regression: a prefix tree keeping no window sub-pools of its own (FlexKV's
    host-loading radix cache) reclaims a token allocation over a shared pool on
    the demand; reading the pool's availability estimate inside that capacity
    decision raised under strict invariant checks."""

    def test_token_rows_are_reclaimed_for_the_allocation(self):
        setup_allocator_context()
        self.addCleanup(reset_context)
        allocator = build_swa_pool(page_size=4)
        cache = RadixCache(
            CacheInitParams(
                disable=False,
                req_to_token_pool=ReqToTokenPool(
                    size=4, max_context_len=64, device="cpu", enable_memory_saver=False
                ),
                token_to_kv_pool_allocator=allocator,
                page_size=4,
            )
        )
        base = 0
        while (ids := allocator.alloc(4)) is not None:
            cache.insert(
                InsertParams(key=RadixKey(array("q", range(base, base + 4))), value=ids)
            )
            base += 4
        with envs.SGLANG_INVARIANT_CHECK.override(InvariantCheckLevel.STRICT):
            self.assertTrue(_decide(lambda: evict_to_free_tokens(cache, allocator, 16)))
        self.assertIsNotNone(allocator.alloc(16))


if __name__ == "__main__":
    unittest.main()
