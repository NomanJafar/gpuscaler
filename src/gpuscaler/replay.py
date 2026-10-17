"""Replay one bursty trace under four policies and report GPU-hours against waiting time."""

from __future__ import annotations

import json
import math
from pathlib import Path

from gpuscaler.policy import QueueDepthPolicy
from gpuscaler.sim import PoolSim, SimConfig
from gpuscaler.trace import bursty_trace

POLICIES = ["fixed: peak-sized", "fixed: average-sized", "reactive", "reactive + 1 warm standby"]


def _needed(rate: float, capacity: int, service_time_s: float, headroom: float = 1.0) -> int:
    # Little's law gives the mean in-flight count (rate x service time); a pool sized exactly for the
    # mean still queues on Poisson bursts, so the peak-sized baseline carries 50% headroom.
    return max(1, math.ceil(headroom * rate * service_time_s / capacity))


def run_replay(out_dir, duration_s=3600, base_rate=1.0, burst_rate=10.0, burst_start_s=600, burst_len_s=300,
               repeat_every_s=1200, capacity=8, service_time_s=3.0, cold_start_s=90, seed=0, target_per_replica=4,
               scale_up_cooldown_s=60, scale_down_window_s=180, plots=True) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrivals = bursty_trace(duration_s, base_rate, burst_rate, burst_start_s, burst_len_s, seed, repeat_every_s)
    cfg = SimConfig(capacity_per_replica=capacity, service_time_s=service_time_s, cold_start_s=cold_start_s)
    peak_n, avg_n = _needed(burst_rate, capacity, service_time_s, headroom=1.5), _needed(base_rate, capacity, service_time_s)
    max_n = peak_n + 2

    def policy():
        return QueueDepthPolicy(target_per_replica=target_per_replica, min_replicas=avg_n, max_replicas=max_n,
                                scale_up_cooldown_s=scale_up_cooldown_s, scale_down_window_s=scale_down_window_s)

    results = {
        POLICIES[0]: PoolSim(cfg, initial_replicas=peak_n).run(arrivals, None),
        POLICIES[1]: PoolSim(cfg, initial_replicas=avg_n).run(arrivals, None),
        POLICIES[2]: PoolSim(cfg, initial_replicas=avg_n).run(arrivals, policy()),
        POLICIES[3]: PoolSim(cfg, initial_replicas=avg_n, warm_standby=1).run(arrivals, policy()),
    }
    meta = {"arrivals": len(arrivals), "duration_s": duration_s, "base_rate": base_rate, "burst_rate": burst_rate,
            "burst_start_s": burst_start_s, "burst_len_s": burst_len_s, "repeat_every_s": repeat_every_s,
            "capacity": capacity, "service_time_s": service_time_s, "cold_start_s": cold_start_s,
            "peak_sized_replicas": peak_n, "average_sized_replicas": avg_n, "max_replicas": max_n,
            "target_per_replica": target_per_replica, "scale_up_cooldown_s": scale_up_cooldown_s, "scale_down_window_s": scale_down_window_s}
    (out_dir / "replay.json").write_text(json.dumps({**results, "_meta": meta}, indent=1))
    (out_dir / "comparison.md").write_text(render_markdown(results))
    if plots:
        _plots(results, arrivals, out_dir)
    return results


def render_markdown(results: dict) -> str:
    head = ["policy", "GPU-hours", "peak replicas", "wait p50", "p95", "p99", "max", "peak queue", "served", "unserved"]
    f = lambda x, d=1: "–" if x is None else f"{x:.{d}f}"
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for name, r in results.items():
        if name.startswith("_"):
            continue
        w = r["wait"]
        lines.append("| " + " | ".join([name, f(r["gpu_hours"], 3), str(r["peak_replicas"]), f(w["p50"]) + " s", f(w["p95"]) + " s",
                                        f(w["p99"]) + " s", f(w["max"]) + " s", str(r["peak_queue"]), str(r["served"]), str(r["unserved"])]) + " |")
    return "\n".join(lines) + "\n"


def _plots(results: dict, arrivals: list[float], out_dir: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = [n for n in results if not n.startswith("_")]
    fig, axes = plt.subplots(len(names), 1, figsize=(12, 2.8 * len(names)), sharex=True)
    axes = [axes] if len(names) == 1 else list(axes)
    # arrivals per minute as context
    end = arrivals[-1] if arrivals else 0
    mins = int(end // 60) + 1
    per_min = [0] * mins
    for a in arrivals:
        per_min[int(a // 60)] += 1
    for ax, n in zip(axes, names, strict=True):
        tl = results[n]["timeline"]
        t = [s["t"] / 60 for s in tl]
        ax.step(t, [s["replicas_ready"] for s in tl], where="post", color="tab:blue", label="replicas ready")
        ax.step(t, [s["replicas_live"] for s in tl], where="post", color="tab:blue", alpha=0.35, label="replicas live (incl. starting)")
        ax.set_ylabel("replicas")
        ax2 = ax.twinx()
        ax2.plot(t, [s["queue"] for s in tl], color="tab:red", lw=0.9, label="queue depth")
        ax2.set_ylabel("queued")
        ax.set_title(f"{n}: {results[n]['gpu_hours']:.2f} GPU-h, wait p95 {results[n]['wait']['p95']:.1f} s, max {results[n]['wait']['max']:.0f} s")
        ax.grid(alpha=.3); ax.legend(loc="upper left", fontsize=8); ax2.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("minutes")
    fig.tight_layout(); fig.savefig(out_dir / "timelines.png", dpi=120); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for n in names:
        ax.scatter(results[n]["gpu_hours"], results[n]["wait"]["p95"], s=80, label=n)
    ax.set_xlabel("GPU-hours"); ax.set_ylabel("wait p95 (s)"); ax.set_title("Cost vs latency per policy"); ax.grid(alpha=.3); ax.legend()
    fig.tight_layout(); fig.savefig(out_dir / "cost_vs_latency.png", dpi=130); plt.close(fig)
