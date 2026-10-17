"""Discrete-time simulation of a replica pool with cold starts (CONCEPTS.md sections 3, 5).

Each tick: ready replicas take queued requests up to their capacity; a replica that was
asked for becomes ready only after `cold_start_s`; the policy sees the queue depth and may
change the desired count. GPU-hours count every replica from launch to termination,
including the cold start during which it serves nothing."""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field


@dataclass
class SimConfig:
    capacity_per_replica: int = 8          # concurrent requests a replica serves at its good operating point
    service_time_s: float = 3.0            # end-to-end time of one request when within capacity
    cold_start_s: float = 90.0             # launch -> ready (servellm measured ~90 s for a 1.5B model on a T4)


@dataclass
class _Replica:
    launched_at: float
    ready_at: float
    terminated_at: float | None = None
    in_flight: list[float] = field(default_factory=list)   # completion times (heap)


def _pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


class PoolSim:
    def __init__(self, cfg: SimConfig, initial_replicas: int = 1, warm_standby: int = 0):
        self.cfg = cfg
        self.initial = initial_replicas
        self.warm_standby = warm_standby

    def run(self, arrivals: list[float], policy=None, tick_s: float = 1.0, drain: bool = True) -> dict:
        cfg = self.cfg
        replicas: list[_Replica] = [_Replica(0.0, 0.0) for _ in range(self.initial + self.warm_standby)]
        desired_serving = self.initial                       # replicas the policy wants, standby excluded
        queue: list[float] = []                              # arrival times of waiting requests
        waits: list[float] = []
        timeline: list[dict] = []
        ai, t = 0, 0.0
        end = (arrivals[-1] if arrivals else 0.0) + (cfg.service_time_s * 3 if drain else 0.0)
        gpu_seconds = 0.0
        while t <= end or (drain and queue):
            # arrivals up to t
            while ai < len(arrivals) and arrivals[ai] <= t:
                queue.append(arrivals[ai]); ai += 1
            # completions
            for r in replicas:
                while r.in_flight and r.in_flight[0] <= t:
                    heapq.heappop(r.in_flight)
            # dispatch to ready, live replicas
            live = [r for r in replicas if r.terminated_at is None]
            ready = [r for r in live if r.ready_at <= t]
            for r in ready:
                while queue and len(r.in_flight) < cfg.capacity_per_replica:
                    a = queue.pop(0)
                    waits.append(t - a)
                    heapq.heappush(r.in_flight, t + cfg.service_time_s)
            # policy
            if policy is not None:
                want = policy.decide(queue_depth=len(queue), current=desired_serving, now=t)
                total_wanted = want + self.warm_standby
                if total_wanted > len(live):
                    for _ in range(total_wanted - len(live)):
                        replicas.append(_Replica(t, t + cfg.cold_start_s))
                elif total_wanted < len(live):
                    # retire the newest idle replicas first; a busy one finishes its work before leaving
                    for r in sorted(live, key=lambda r: -r.launched_at):
                        if len(live) <= total_wanted:
                            break
                        if not r.in_flight:
                            r.terminated_at = t; live.remove(r)
                desired_serving = want
            gpu_seconds += len([r for r in replicas if r.terminated_at is None]) * tick_s
            timeline.append({"t": t, "queue": len(queue), "replicas_live": len(live), "replicas_ready": len(ready),
                             "in_flight": sum(len(r.in_flight) for r in ready), "desired": desired_serving + self.warm_standby})
            t += tick_s
            if t > end + 3600:
                break
        return {
            "requests": len(waits) + len(queue), "served": len(waits), "unserved": len(queue),
            "duration_s": t, "gpu_hours": gpu_seconds / 3600, "peak_queue": max(s["queue"] for s in timeline),
            "peak_replicas": max(s["replicas_live"] for s in timeline),
            "wait": {"mean": sum(waits) / len(waits) if waits else None, "p50": _pct(waits, 50), "p95": _pct(waits, 95), "p99": _pct(waits, 99), "max": max(waits) if waits else None},
            "timeline": timeline, "decisions": list(getattr(policy, "decisions", [])),
        }
