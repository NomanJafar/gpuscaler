"""`gpuscaler replay` (simulated pool, the experiment) and `gpuscaler demo` (real mock replicas)."""

from __future__ import annotations

import argparse
import json
import sys

from gpuscaler.replay import render_markdown, run_replay


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    p = argparse.ArgumentParser(prog="gpuscaler", description="queue-depth autoscaling for inference replicas")
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("replay", help="replay a bursty trace under fixed, reactive and warm-standby policies (simulated)")
    r.add_argument("--out", default="results/replay")
    r.add_argument("--duration", type=float, default=3600)
    r.add_argument("--base-rate", type=float, default=1.0)
    r.add_argument("--burst-rate", type=float, default=10.0)
    r.add_argument("--burst-start", type=float, default=600)
    r.add_argument("--burst-len", type=float, default=300)
    r.add_argument("--repeat-every", type=float, default=1200)
    r.add_argument("--capacity", type=int, default=16, help="concurrent requests per replica at its good operating point")
    r.add_argument("--service-time", type=float, default=3.0, help="seconds per request within capacity")
    r.add_argument("--cold-start", type=float, default=90.0, help="seconds from launch to ready")
    r.add_argument("--target", type=float, default=4.0, help="target queued requests per replica")
    r.add_argument("--up-cooldown", type=float, default=60.0)
    r.add_argument("--down-window", type=float, default=180.0)
    r.add_argument("--seed", type=int, default=0)
    d = sub.add_parser("demo", help="drive real tokenbench mock replicas with the policy for a short while")
    d.add_argument("--seconds", type=float, default=60)
    d.add_argument("--capacity", type=int, default=4)
    d.add_argument("--rate", type=float, default=6.0, help="requests per second of synthetic load")
    d.add_argument("--out", default="results/demo.json")
    a = p.parse_args(argv)
    if a.cmd == "replay":
        res = run_replay(a.out, a.duration, a.base_rate, a.burst_rate, a.burst_start, a.burst_len, a.repeat_every, a.capacity,
                         a.service_time, a.cold_start, a.seed, a.target, a.up_cooldown, a.down_window, plots=True)
        print(render_markdown(res), end="")
        print(f"wrote {a.out}/")
    else:
        _demo(a)


def _demo(a) -> None:
    import asyncio
    import threading
    import time

    import httpx

    from gpuscaler.controller import MockReplicaPool, run_controller
    from gpuscaler.policy import QueueDepthPolicy

    pool = MockReplicaPool(capacity=a.capacity, ttft_ms=100, itl_ms=10)
    pool.scale_to(1)
    stop = threading.Event()

    async def load():
        async with httpx.AsyncClient(timeout=60) as c:
            tasks = []
            while not stop.is_set():
                urls = pool.urls()
                if urls:
                    u = urls[int(time.monotonic() * 997) % len(urls)]
                    body = {"model": "mock", "messages": [{"role": "user", "content": "x"}], "max_tokens": 60, "stream": False, "ignore_eos": True}
                    tasks.append(asyncio.create_task(c.post(f"{u}/v1/chat/completions", json=body)))
                await asyncio.sleep(1 / a.rate)
            await asyncio.gather(*tasks, return_exceptions=True)

    th = threading.Thread(target=lambda: asyncio.run(load()), daemon=True)
    th.start()
    try:
        tl = run_controller(pool, QueueDepthPolicy(target_per_replica=2, min_replicas=1, max_replicas=6, scale_up_cooldown_s=5, scale_down_window_s=20), a.seconds)
    finally:
        stop.set(); th.join(timeout=30); pool.close()
    from pathlib import Path
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(tl, indent=1))
    print(f"replicas over time: {[s['replicas'] for s in tl[::5]]}")
    print(f"queue over time:    {[s['queue'] for s in tl[::5]]}")
