import pytest

from gpuscaler.policy import QueueDepthPolicy
from gpuscaler.sim import PoolSim, SimConfig
from gpuscaler.trace import bursty_trace


def test_bursty_trace_has_quiet_and_burst_phases():
    arrivals = bursty_trace(duration_s=600, base_rate=1.0, burst_rate=10.0, burst_start_s=200, burst_len_s=100, seed=1)
    quiet = sum(1 for t in arrivals if t < 200) / 200
    burst = sum(1 for t in arrivals if 200 <= t < 300) / 100
    assert 0.7 <= quiet <= 1.3 and 8 <= burst <= 12
    assert arrivals == sorted(arrivals)


def test_fixed_pool_at_capacity_serves_without_queue():
    cfg = SimConfig(capacity_per_replica=4, service_time_s=2.0, cold_start_s=90)
    sim = PoolSim(cfg, initial_replicas=2)
    arrivals = [i * 1.0 for i in range(100)]            # 1 req/s, 2 s service -> 2 in flight, capacity 8
    res = sim.run(arrivals, policy=None, tick_s=1.0)
    assert res["requests"] == 100 and res["wait"]["p95"] == pytest.approx(0.0, abs=1e-9)
    assert res["gpu_hours"] == pytest.approx(2 * res["duration_s"] / 3600, rel=1e-6)
    assert res["peak_queue"] == 0


def test_undersized_pool_queues_and_wait_grows():
    cfg = SimConfig(capacity_per_replica=2, service_time_s=2.0, cold_start_s=90)
    sim = PoolSim(cfg, initial_replicas=1)
    arrivals = [i * 0.5 for i in range(200)]            # 2 req/s against capacity 1 req/s
    res = sim.run(arrivals, policy=None, tick_s=1.0)
    assert res["peak_queue"] > 50 and res["wait"]["p95"] > 20


def test_reactive_scaling_adds_replicas_after_cold_start():
    cfg = SimConfig(capacity_per_replica=2, service_time_s=2.0, cold_start_s=60)
    sim = PoolSim(cfg, initial_replicas=1)
    policy = QueueDepthPolicy(target_per_replica=2, min_replicas=1, max_replicas=8, scale_up_cooldown_s=30)
    arrivals = [i * 0.5 for i in range(400)]            # 2 req/s for 200 s
    res = sim.run(arrivals, policy=policy, tick_s=1.0)
    tl = res["timeline"]
    assert max(s["replicas_ready"] for s in tl) >= 2
    first_ready = next(s["t"] for s in tl if s["replicas_ready"] >= 2)
    assert 60 <= first_ready <= 75                       # decided at ~t=1-10, ready 60 s later
    assert res["wait"]["p95"] < 60 and res["peak_queue"] > 0


def test_warm_standby_cuts_the_wait_spike():
    cfg = SimConfig(capacity_per_replica=2, service_time_s=2.0, cold_start_s=60)
    policy = lambda: QueueDepthPolicy(target_per_replica=2, min_replicas=1, max_replicas=8, scale_up_cooldown_s=30)
    arrivals = [i * 0.5 for i in range(400)]
    base = PoolSim(cfg, initial_replicas=1).run(arrivals, policy=policy(), tick_s=1.0)
    warm = PoolSim(cfg, initial_replicas=1, warm_standby=1).run(arrivals, policy=policy(), tick_s=1.0)
    assert warm["wait"]["p95"] < base["wait"]["p95"] * 0.5


def test_warm_standby_costs_one_replica_when_nothing_happens():
    cfg = SimConfig(capacity_per_replica=4, service_time_s=2.0, cold_start_s=60)
    arrivals = [i * 2.0 for i in range(100)]            # 0.5 req/s: one replica is plenty
    policy = lambda: QueueDepthPolicy(target_per_replica=4, min_replicas=1, max_replicas=8)
    base = PoolSim(cfg, initial_replicas=1).run(arrivals, policy=policy(), tick_s=1.0)
    warm = PoolSim(cfg, initial_replicas=1, warm_standby=1).run(arrivals, policy=policy(), tick_s=1.0)
    assert warm["gpu_hours"] == pytest.approx(2 * base["gpu_hours"], rel=1e-6)
    assert base["peak_replicas"] == 1 and warm["peak_replicas"] == 2
