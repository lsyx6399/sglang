"""Pipeline stages under unified memory: every stage asks each capacity
question of every stage's geometry, so all of them decide alike."""

import random
import unittest
from contextlib import contextmanager
from typing import List

from sglang.srt.mem_cache.kv.alloc.demand import KVDemand
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase
from sglang.test.unified_allocator_fixtures import (
    UnifiedGeometry,
    build_unified_bundle,
    setup_allocator_context,
)

register_cpu_ci(est_time=75, suite="base-a-test-cpu")

# One byte budget, two layer mixes: the first stage holds mostly full
# attention, the second mostly recurrent state.
_STATE_STAGES = (
    UnifiedGeometry(
        chain="mamba",
        full_layer_ids=tuple(range(6)),
        state_layer_ids=(6,),
        full_tokens=256,
        state_slots=16,
        page_size=4,
        byte_budget=1 << 16,
    ),
    UnifiedGeometry(
        chain="mamba",
        full_layer_ids=(7,),
        state_layer_ids=tuple(range(8, 16)),
        full_tokens=256,
        state_slots=16,
        page_size=4,
        byte_budget=1 << 16,
        start_layer=7,
    ),
)
# With a sliding window (a float between the ends): the first stage mostly
# full attention, the second mostly window layers.
_WINDOW_STAGES = (
    UnifiedGeometry(
        chain="tri",
        full_layer_ids=tuple(range(4)),
        swa_layer_ids=(4,),
        state_layer_ids=(5,),
        full_tokens=256,
        swa_tokens=128,
        state_slots=8,
        page_size=4,
        byte_budget=1 << 16,
        window=16,
    ),
    UnifiedGeometry(
        chain="tri",
        full_layer_ids=(6,),
        swa_layer_ids=(7, 8, 9, 10),
        state_layer_ids=(11, 12),
        full_tokens=256,
        swa_tokens=128,
        state_slots=8,
        page_size=4,
        byte_budget=1 << 16,
        window=16,
        start_layer=6,
    ),
)


def _stages(geometries, checks=False):
    allocators = [
        build_unified_bundle(g).token_to_kv_pool_allocator for g in geometries
    ]
    shared = [a.stage_geometry() for a in allocators]
    for stage, allocator in enumerate(allocators):
        allocator.install_stage_capacity(stage, shared, checks=checks)
    return allocators


@contextmanager
def _alone(allocator):
    """The allocator answering for its own stage only."""
    capacity, allocator.stage_capacity = allocator.stage_capacity, None
    try:
        yield allocator
    finally:
        allocator.stage_capacity = capacity


def _fits_alone(allocator, demands) -> bool:
    with _alone(allocator):
        return allocator.units_fit(demands)


def _admits_alone(allocator, demand, evictable) -> bool:
    with _alone(allocator):
        return allocator.admits(demand, evictable=evictable)


def _on_every_stage(allocators, op):
    """Run ``op`` on every stage, as every stage runs the same operations."""
    return [op(allocator) for allocator in allocators]


def _probe_demands(allocator, rng, count=12) -> List[dict]:
    owner, windows, state = allocator.sub_pool_roles()
    probes = []
    for _ in range(count):
        pages = rng.randrange(0, 80)
        demands = {owner: pages}
        for window in windows:
            demands[window] = min(pages, rng.randrange(0, 8))
        if state is not None:
            demands[state] = rng.randrange(0, 24)
        probes.append(demands)
    return probes


class TestStageCapacity(CustomTestCase):
    def setUp(self):
        setup_allocator_context()

    def test_every_stage_decides_alike_where_the_stages_disagree(self):
        """On empty arenas of one size, a FULL-heavy demand fits only the state-heavy
        stage and a state-heavy demand only the other; both stages refuse both."""
        a0, a1 = _stages(_STATE_STAGES)
        owner, _, state = a0.sub_pool_roles()
        only = {0: [], 1: []}
        for full in range(0, 480, 8):
            for slots in range(0, 48, 4):
                demands = {owner: full, state: slots}
                fit0, fit1 = _fits_alone(a0, demands), _fits_alone(a1, demands)
                for allocator in (a0, a1):
                    self.assertEqual(
                        allocator.units_fit(demands), fit0 and fit1, demands
                    )
                if fit0 != fit1:
                    only[0 if fit0 else 1].append(demands)
        self.assertTrue(only[0] and only[1], only)
        for demands in only[0] + only[1]:
            tokens = demands[owner] * a0.page_size
            demand = KVDemand(tokens, {}, demands[state])
            self.assertFalse(a0.admits(demand), demands)
            self.assertFalse(a1.admits(demand), demands)
            self.assertNotEqual(
                _admits_alone(a0, demand, None), _admits_alone(a1, demand, None)
            )

    def test_a_cache_insert_slot_take_is_decided_alike(self):
        """A state slot a prefix-cache insert takes outside scheduling is decided on
        every stage: the stage that alone would still take it refuses it too."""
        a0, a1 = _stages(_STATE_STAGES)
        _, _, state = a0.sub_pool_roles()
        held = []
        while True:
            taken = _on_every_stage((a0, a1), lambda a: a.alloc_state_slots(1))
            self.assertEqual(taken[0] is None, taken[1] is None)
            if taken[0] is None:
                break
            held.append(taken)
        self.assertTrue(_fits_alone(a0, {state: 1}))
        self.assertFalse(_fits_alone(a1, {state: 1}))

    def test_estimates_are_alike_on_every_stage(self):
        """Observer estimates read every stage's simulated snapshot, so every stage
        observes the same values, unlike the stages alone."""
        a0, a1 = _stages(_STATE_STAGES)
        _on_every_stage((a0, a1), lambda a: a.alloc(16 * a.page_size))
        _on_every_stage((a0, a1), lambda a: a.alloc_state_slots(3))
        self.assertEqual(a0.usage(), a1.usage())
        self.assertEqual(a0.available_size(), a1.available_size())
        with _alone(a0), _alone(a1):
            self.assertNotEqual(a0.usage(), a1.usage())

    def _check_lockstep_simulation(self, geometries, seed):
        """As the stages take and free the same units, every take is decided alike and
        every answer is the conjunction of the stages' answers on their real layouts."""
        stages = _stages(geometries)
        a0, a1 = stages
        rng = random.Random(seed)
        owner, windows, state = a0.sub_pool_roles()
        tokens, slots = [], []
        disagreements = refusals = 0
        for _ in range(160):
            roll = rng.random()
            if roll < 0.4:
                n = rng.randrange(1, 24) * a0.page_size
                taken = _on_every_stage(stages, lambda a: a.alloc(n))
                self.assertEqual([t is None for t in taken], [taken[0] is None] * 2)
                if taken[0] is None:
                    refusals += 1
                else:
                    tokens.append(taken)
            elif roll < 0.6 and state is not None:
                n = rng.randrange(1, 6)
                taken = _on_every_stage(stages, lambda a: a.alloc_state_slots(n))
                self.assertEqual([t is None for t in taken], [taken[0] is None] * 2)
                if taken[0] is not None:
                    slots.append(taken)
            elif roll < 0.85 and tokens:
                pair = tokens.pop(rng.randrange(len(tokens)))
                for allocator, ids in zip(stages, pair):
                    allocator.free(ids)
            elif slots:
                pair = slots.pop(rng.randrange(len(slots)))
                for allocator, ids in zip(stages, pair):
                    allocator.mamba_allocator.free(ids)
            self.assertEqual(a0.stage_counts(), a1.stage_counts())
            for demands in _probe_demands(a0, rng, count=4):
                fit0, fit1 = _fits_alone(a0, demands), _fits_alone(a1, demands)
                disagreements += fit0 != fit1
                self.assertEqual(a0.units_fit(demands), fit0 and fit1, demands)
                self.assertEqual(a1.units_fit(demands), fit0 and fit1, demands)
                evictable = {
                    name: rng.randrange(0, 16) * (1 if name == state else 4)
                    for name in (owner, *windows, state)
                    if name is not None
                }
                demand = KVDemand(
                    demands[owner] * a0.page_size,
                    {w: demands[w] * a0.page_size for w in windows},
                    demands.get(state, 0),
                )
                alone = _admits_alone(a0, demand, evictable) and _admits_alone(
                    a1, demand, evictable
                )
                for allocator in stages:
                    self.assertEqual(
                        allocator.admits(demand, evictable=evictable),
                        alone,
                        (demands, evictable),
                    )
        self.assertGreater(disagreements, 0)
        self.assertGreater(refusals, 0)
        for allocator in stages:
            self.assertEqual(allocator.verify_byte_accounting(), [])

    def test_lockstep_simulation_state_chain(self):
        self._check_lockstep_simulation(_STATE_STAGES, seed=0)

    def test_lockstep_simulation_window_chain(self):
        self._check_lockstep_simulation(_WINDOW_STAGES, seed=1)


class TestStageCapacityDigest(CustomTestCase):
    def setUp(self):
        setup_allocator_context()

    def test_digests_agree_until_a_stage_drifts(self):
        """Stages that ask the same questions and hold the same units give the
        same digest; one that holds a unit more does not."""
        a0, a1 = _stages(_STATE_STAGES, checks=True)

        def digests():
            return [a.stage_capacity.take_digest(a.stage_counts()) for a in (a0, a1)]

        _on_every_stage((a0, a1), lambda a: a.alloc(8 * a.page_size))
        first = digests()
        self.assertIsNotNone(first[0])
        self.assertEqual(first[0], first[1])
        _on_every_stage((a0, a1), lambda a: a.alloc_state_slots(2))
        with _alone(a1):
            a1.alloc_state_slots(1)
        second = digests()
        self.assertNotEqual(second[0], second[1])
        a0.install_stage_capacity(0, [a.stage_geometry() for a in (a0, a1)])
        self.assertIsNone(a0.stage_capacity.take_digest(a0.stage_counts()))

    def test_digests_do_not_depend_on_band_order(self):
        """Stages may hold the same bands in a different arena order; the same
        questions and the same units held give the same digest."""
        a0, a1 = _stages(_STATE_STAGES, checks=True)
        _on_every_stage((a0, a1), lambda a: a.alloc(8 * a.page_size))
        _on_every_stage((a0, a1), lambda a: a.alloc_state_slots(2))
        counts = a1.stage_counts()
        self.assertTrue(all(len(c) > 1 for c in counts))
        reordered = tuple(dict(reversed(c.items())) for c in counts)
        self.assertEqual(
            a0.stage_capacity.take_digest(a0.stage_counts()),
            a1.stage_capacity.take_digest(reordered),
        )

    def test_digests_agree_after_a_request_is_clipped_to_memory(self):
        """A stage clips a request's output limit to memory, then relays the
        request; the next stage must search from the limit as received, or the
        stages' question digests differ although their decisions agree."""
        import pickle

        from sglang.srt.managers.mm_utils import strip_mm_features_for_pp_relay

        stages = [_request_stage(a) for a in _stages(_STATE_STAGES, checks=True)]
        requested = 4096
        recv_req = _generate_req(12, requested)
        for s in stages:
            s.handle_generate_request(recv_req)
            # The relay pickles the request as this stage left it.
            (recv_req,) = pickle.loads(
                pickle.dumps(strip_mm_features_for_pp_relay([recv_req]))
            )
        clipped = [s.queued[0].sampling_params.max_new_tokens for s in stages]
        self.assertLess(clipped[0], requested)
        self.assertEqual(clipped[0], clipped[1])
        for s in stages:
            s._pp_mark_capacity_digest()
        self.assertIsNotNone(stages[0]._pp_capacity_digest)
        self.assertEqual(stages[0]._pp_capacity_digest, stages[1]._pp_capacity_digest)


def _request_stage(
    allocator, *, max_req_len=1 << 14, max_req_input_len=1 << 14, chunk=None
):
    """A scheduler that runs production request intake on ``allocator``."""
    from types import SimpleNamespace as NS

    from sglang.srt.disaggregation.mode import DisaggregationMode
    from sglang.srt.managers.scheduler import Scheduler

    s = Scheduler.__new__(Scheduler)
    s.token_to_kv_pool_allocator = allocator
    if allocator.stage_capacity is not None:
        s._pp_checked_allocator = allocator
    s.page_size = allocator.page_size
    s.max_req_len, s.max_req_input_len = max_req_len, max_req_input_len
    s.max_total_num_tokens = 1 << 14
    s.kv_shard_widening = 1
    s.max_new_tokens_limit = None
    s.sliding_window_size = None
    s.chunked_prefill_size, s.truncation_align_size = chunk, None
    s.tree_cache = NS(
        supports_prefix_sharing=lambda: True,
        window_checkpoint_interval=lambda: 0,
        cache_controller=None,
    )
    s.enable_session_radix_cache = False
    s.disaggregation_mode = DisaggregationMode.NULL
    s.model_config = NS(hf_eos_token_id={1}, vocab_size=128)
    s.metrics_reporter = NS(enable_metrics=False)
    s.tokenizer = s.dllm_config = None
    s.spec_algorithm = NS(is_dflash_family=lambda: False, is_uno=lambda: False)
    s._maybe_namespace_elastic_radix_cache = lambda req: None
    s._record_kv_event_lora_name = lambda req, name: None
    s.invariant_checker = NS(expect_output=lambda req: None)
    s.grammar_manager = NS(process_req_with_grammar=lambda req: False)
    s.queued = []
    s._add_request_to_queue = s.queued.append
    return s


def _generate_req(input_len, max_new_tokens):
    from array import array

    from sglang.srt.managers.io_struct import TokenizedGenerateReqInput
    from sglang.srt.sampling.sampling_params import SamplingParams

    return TokenizedGenerateReqInput(
        rid="r",
        input_text="",
        input_ids=array("q", range(input_len)),
        input_embeds=None,
        mm_inputs=None,
        token_type_ids=None,
        sampling_params=SamplingParams(max_new_tokens=max_new_tokens),
        return_logprob=False,
        logprob_start_len=-1,
        top_logprobs_num=0,
        token_ids_logprob=None,
        stream=False,
    )


# FULL + window stages without state, one holding mostly full attention and
# the other mostly window layers: their empty arenas hold very unequal FULL.
_FULL_WINDOW_STAGES = (
    UnifiedGeometry(
        chain="swa",
        full_layer_ids=(0,),
        swa_layer_ids=tuple(range(1, 17)),
        page_size=1,
        byte_budget=1 << 16,
        window=32,
    ),
    UnifiedGeometry(
        chain="swa",
        full_layer_ids=tuple(range(16)),
        swa_layer_ids=(16,),
        page_size=1,
        byte_budget=1 << 16,
        window=32,
    ),
)


def _limits_worker(rank, path):
    """One stage's startup request limits, then intake of a prompt between the
    stages' own limits, with and without auto-truncation."""
    from datetime import timedelta
    from types import SimpleNamespace as NS

    import torch.distributed as dist

    from sglang.srt.managers.req_types import FINISH_ABORT
    from sglang.srt.managers.tp_worker import TpModelWorker
    from sglang.srt.mem_cache import kv_cache_builder
    from sglang.srt.mem_cache.kv_cache_configurator import KVCacheConfigurator
    from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
    from sglang.srt.model_executor.model_runner import ModelRunner
    from sglang.srt.runtime_context import get_memory, get_parallel, get_serving

    setup_allocator_context()
    dist.init_process_group(
        "gloo",
        init_method="file://" + path,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    allocator = build_unified_bundle(
        _FULL_WINDOW_STAGES[rank]
    ).token_to_kv_pool_allocator
    configurator = KVCacheConfigurator.__new__(KVCacheConfigurator)
    configurator.is_hybrid_swa = True
    runner = ModelRunner.__new__(ModelRunner)
    runner.kv_cache_configurator = configurator
    runner.is_hybrid_swa = True
    runner.max_total_num_tokens = runner.full_max_total_num_tokens = 64
    runner.swa_max_total_num_tokens = 64
    runner.max_running_requests = 4
    runner.token_to_kv_pool = NS(size=0)
    runner.token_to_kv_pool_allocator = allocator
    runner.req_to_token_pool = ReqToTokenPool.__new__(ReqToTokenPool)
    runner.req_to_token_pool.size = 4
    runner.req_to_token_pool.max_context_len = 1 << 14
    runner.req_to_token_pool._aux_cache = None
    runner.forward_stream = None
    worker = NS(
        model_runner=runner,
        model_config=NS(context_len=1 << 14),
        random_seed=0,
        device="cpu",
        dllm_algorithm=None,
    )
    own = [None] * 2
    dist.all_gather_object(own, allocator.size_full)
    assert own[0] != own[1], own
    try:
        with (
            get_parallel().override(
                pp_size=2,
                pp_rank=rank,
                pp_group=NS(cpu_group=dist.group.WORLD),
                tp_size=1,
                tp_rank=0,
            ),
            get_memory().override(enable_unified_memory=True),
        ):
            # The scheduler's startup order: the final pool, then its limits.
            kv_cache_builder.agree_on_stage_request_capacity(tp_worker=worker)
            info = TpModelWorker.get_worker_info(worker)
            max_req_len, max_req_input_len = info[4], info[5]
            outcomes = []
            for truncate in (False, True):
                with get_serving().override(allow_auto_truncate=truncate):
                    stage = _request_stage(
                        allocator,
                        max_req_len=max_req_len,
                        max_req_input_len=max_req_input_len,
                        # Short chunks, so the prompt fits both arenas.
                        chunk=1,
                    )
                    stage.handle_generate_request(_generate_req(min(own) - 5, 0))
                    (req,) = stage.queued
                    outcomes.append(
                        (
                            isinstance(req.to_finish, FINISH_ABORT),
                            len(req.origin_input_ids),
                        )
                    )
        every = [None] * 2
        dist.all_gather_object(every, outcomes)
        assert every[0] == every[1], every
    finally:
        dist.destroy_process_group()


class TestStageRequestLimits(CustomTestCase):
    def test_stages_reject_and_truncate_a_prompt_alike(self):
        """FULL + window stages whose empty arenas hold unequal FULL tokens
        reject, or auto-truncate, a prompt between their own limits alike."""
        import tempfile

        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(_limits_worker, args=(tmp + "/rendezvous",), nprocs=2, join=True)


# A FULL-only stage, as an uneven layer split can leave beside a FULL + window one.
_FULL_ONLY_STAGE = UnifiedGeometry(
    chain="full",
    full_layer_ids=(16, 17),
    page_size=1,
    byte_budget=1 << 16,
    start_layer=16,
)


def _ratio_host_sizes(allocator):
    """Each sub-pool's host-view capacity and the independent host pool that ratio
    sizing builds from it, as storage backends without a shared host arena do."""
    from sglang.srt.mem_cache.kv.layout.base import HostPoolOptions

    sizes = {}
    for name, view in sorted(allocator.views.items()):
        host = allocator.storage.host_layout(name).build(
            HostPoolOptions(
                ratio=2.0,
                size=0,
                page_size=allocator.page_size,
                layout="layer_first",
                allocator_type="default",
                pool_label=name,
                pin_memory=False,
            )
        )
        sizes[name] = (view.host_capacity_tokens, host.size)
    return sizes


def _stage_shares_worker(rank, path, geometries, unlike):
    """One stage's startup share agreement, compared with every stage's."""
    from datetime import timedelta
    from types import SimpleNamespace as NS

    import torch.distributed as dist

    from sglang.srt.mem_cache import kv_cache_builder
    from sglang.srt.runtime_context import get_memory, get_parallel

    setup_allocator_context()
    dist.init_process_group(
        "gloo",
        init_method="file://" + path,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=30),
    )
    allocator = build_unified_bundle(geometries[rank]).token_to_kv_pool_allocator
    worker = NS(model_runner=NS(token_to_kv_pool_allocator=allocator))
    try:
        with (
            get_parallel().override(
                pp_size=2, pp_rank=rank, pp_group=NS(cpu_group=dist.group.WORLD)
            ),
            get_memory().override(enable_unified_memory=True),
        ):
            before = None if unlike else _ratio_host_sizes(allocator)
            try:
                kv_cache_builder.agree_on_stage_request_capacity(tp_worker=worker)
            except ValueError as error:
                record = str(error)
            else:
                record = (
                    allocator.size_full,
                    allocator.size_swa,
                    _ratio_host_sizes(allocator),
                )
        every = [None] * 2
        dist.all_gather_object(every, (before, record))
        if unlike:
            assert every[0] == every[1], every
            assert "must plan the same sub-pools" in record, every
        else:
            # Alone, the stages' FULL host pools differ; agreed, every pool matches.
            assert every[0][0]["full"] != every[1][0]["full"], every
            assert every[0][1] == every[1][1], every
    finally:
        dist.destroy_process_group()


class TestStageShares(CustomTestCase):
    def _spawn(self, geometries, unlike):
        import tempfile

        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as tmp:
            mp.spawn(
                _stage_shares_worker,
                args=(tmp + "/rendezvous", geometries, unlike),
                nprocs=2,
                join=True,
            )

    def test_stages_with_unlike_sub_pools_fail_alike_at_startup(self):
        """A FULL + window stage and a FULL-only stage both raise the same error at
        the startup agreement; neither waits on a collective the other skips."""
        self._spawn((_FULL_WINDOW_STAGES[0], _FULL_ONLY_STAGE), unlike=True)

    def test_ratio_sized_host_pools_agree_after_startup(self):
        """FULL + window stages whose arenas hold unequal FULL tokens keep one FULL
        and window share, and ratio sizing builds equal host pools from them."""
        self._spawn(_FULL_WINDOW_STAGES, unlike=False)


class _LoopDone(Exception):
    pass


def _loop_worker(rank, path, level, drift_round):
    """One stage of the real PP event loop over gloo, scheduling and model stood in;
    with ``drift_round`` the middle stage takes a slot of its own at that iteration."""
    import math
    import os
    from datetime import timedelta
    from types import SimpleNamespace as NS

    import torch.distributed as dist

    os.environ["SGLANG_INVARIANT_CHECK"] = str(level)
    from sglang.srt.managers import scheduler_pp_mixin
    from sglang.srt.managers.scheduler import Scheduler
    from sglang.srt.managers.scheduler_pp_mixin import SchedulerPPMixin
    from sglang.srt.runtime_context import get_parallel

    setup_allocator_context()
    dist.init_process_group(
        "gloo",
        init_method="file://" + path,
        rank=rank,
        world_size=3,
        timeout=timedelta(seconds=60 if drift_round is None else 15),
    )
    # One rank per stage: the attention TP broadcast is the identity.
    scheduler_pp_mixin.attn_cp_tp_broadcast_pyobj = lambda data: data
    geometry = _STATE_STAGES[rank % 2]
    allocator = build_unified_bundle(geometry).token_to_kv_pool_allocator
    page = allocator.page_size
    rounds = 24

    class Stage(SchedulerPPMixin):
        ingest_requests = Scheduler.ingest_requests

        def __init__(self):
            self.token_to_kv_pool_allocator = allocator
            self.tree_cache = None
            self.recv_skipper = None
            self.uocr_runtime = None
            self.device = "cpu"
            self.spec_algorithm = NS(is_none=lambda: True)
            self.pp_group = NS(
                is_first_rank=rank == 0,
                is_last_rank=rank == 2,
                cpu_group=dist.group.WORLD,
            )
            self.world_group = NS(cpu_group=dist.group.WORLD)
            self.pause_state = NS(
                holds_inputs=lambda: False,
                take_inputs=lambda reqs, stop_at_pause: list(reqs),
                may_launch=lambda: True,
            )
            self.request_receiver = NS(recv_requests=self._recv_requests)
            self.metrics_reporter = NS(record_scheduler_active=lambda start: None)
            self.invariant_checker = NS(run=lambda *args: None)
            self.new_token_ratio_tracker = NS(current=0.5)
            self.chunked_req = None
            self.forward_ct = 0
            self.waiting_queue = []
            self.held = []
            self.round = 0
            self.log = []

        def increment_scheduler_iterations(self):
            self.round += 1

        def _poll_timeout_aborts(self):
            return []

        def _recv_requests(self, local_reqs):
            if rank:
                return self._pp_recv_pyobj_from_prev_stage()
            # Requests of growing, then shrinking prompts; some fit no
            # FULL-heavy arena even empty.
            r = self.round
            return [(f"r{r}-{i}", (r * 37 + i * 101) % 360 + 8) for i in range(2)]

        def process_input_requests(self, reqs):
            for rid, tokens in reqs:
                fits = allocator.admits(KVDemand(tokens, {}, 1), empty_pool=True)
                self.log.append(("arrive", rid, tokens, fits))
                if fits:
                    self.waiting_queue.append(NS(rid=rid, tokens=tokens))

        def get_next_batch_to_run(self, running_batch, last_batch):
            if rank == 1 and self.round == drift_round:
                with _alone(allocator):
                    allocator.alloc_state_slots(1)
            if self.held and self.round % 3 == 0:
                req, ids, slots = self.held.pop(0)
                allocator.free(ids)
                allocator.mamba_allocator.free(slots)
                self.log.append(("finish", req.rid))
            while self.waiting_queue:
                req = self.waiting_queue[0]
                pages = math.ceil(req.tokens / page)
                if not allocator.admits(KVDemand(pages * page, {}, 1)):
                    self.log.append(("full", req.rid))
                    break
                ids = allocator.alloc(pages * page)
                slots = allocator.alloc_state_slots(1)
                assert ids is not None and slots is not None, req.rid
                self.waiting_queue.pop(0)
                self.held.append((req, ids, slots))
                self.log.append(("admit", req.rid))
            return NS(running_batch=running_batch, batch_to_run=None)

        def _pp_commit_send_output_work_and_preprocess_output_tensors(self, *a):
            return None, None, None

        def _pp_commit_proxy_send_work(self):
            pass

        def on_idle(self):
            if self.round >= rounds:
                raise _LoopDone()

    stage = Stage()
    try:
        with get_parallel().override(
            pp_size=3,
            pp_rank=rank,
            tp_size=1,
            attn_tp_size=1,
            attn_tp_rank=0,
            attn_cp_size=1,
            attn_cp_rank=0,
            attn_dp_rank=0,
            pp_async_batch_depth=0,
            world_group=NS(cpu_group=dist.group.WORLD),
        ):
            try:
                stage.event_loop_pp()
            except _LoopDone:
                pass
            except AssertionError as error:
                with open(f"{path}.failed{rank}", "w") as f:
                    f.write(str(error))
                raise
            stage._pp_commit_comm_work(stage.send_req_work)
            stage._pp_commit_comm_work(stage._pp_capacity_send_work)
        logs = [None] * 3
        dist.all_gather_object(logs, stage.log)
        assert logs[0] == logs[1] == logs[2], logs
        kinds = {entry[0] for entry in logs[0]}
        assert {"admit", "full", "finish"} <= kinds, logs[0]
        arrivals = [e for e in logs[0] if e[0] == "arrive"]
        assert not all(e[3] for e in arrivals), arrivals
        # The state-heavy stage alone would keep requests the FULL-heavy
        # stages cannot; it drops them as they do.
        if rank == 1:
            with _alone(allocator):
                alone = [
                    allocator.admits(KVDemand(e[2], {}, 1), empty_pool=True)
                    for e in arrivals
                ]
            assert alone != [e[3] for e in arrivals], arrivals
    finally:
        dist.destroy_process_group()


class TestStageCapacityInThePipelineLoop(CustomTestCase):
    def test_three_stages_schedule_alike_under_the_pp_loop(self):
        """Three gloo ranks in the real PP loop admit, defer and finish the same
        requests; with invariant checks on, their decision digests match."""
        import tempfile

        import torch.multiprocessing as mp

        for level in (0, 2):
            with self.subTest(level=level), tempfile.TemporaryDirectory() as tmp:
                mp.spawn(
                    _loop_worker,
                    args=(tmp + "/rendezvous", level, None),
                    nprocs=3,
                    join=True,
                )

    def test_a_stage_that_drifts_fails_its_digest_check(self):
        """A stage holding a slot the others do not fails loudly at the next
        iteration's digest check."""
        import os
        import tempfile

        import torch.multiprocessing as mp

        with tempfile.TemporaryDirectory() as tmp:
            path = tmp + "/rendezvous"
            with self.assertRaises(Exception):
                mp.spawn(_loop_worker, args=(path, 2, 5), nprocs=3, join=True)
            self.assertTrue(os.path.exists(path + ".failed1"), os.listdir(tmp))
            with open(path + ".failed1") as f:
                self.assertIn("stage_capacity_agreement", f.read())


if __name__ == "__main__":
    unittest.main()
