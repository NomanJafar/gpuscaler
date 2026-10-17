"""The same policy driving real replica processes: tokenbench mock servers started and stopped
on demand, queue depth read from their /metrics. One integration test proves the loop; the
replay simulator is where the policies are compared."""

from __future__ import annotations

import re
import socket
import subprocess
import sys
import time

import httpx

from gpuscaler.policy import QueueDepthPolicy

_WAITING = re.compile(r"^vllm:num_requests_waiting(?:\{[^}]*\})?\s+(\S+)", re.MULTILINE)
_RUNNING = re.compile(r"^vllm:num_requests_running(?:\{[^}]*\})?\s+(\S+)", re.MULTILINE)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class MockReplicaPool:
    """Replica = one `tokenbench mockserver` subprocess with a fixed concurrency capacity."""

    def __init__(self, capacity: int = 8, ttft_ms: float = 80, itl_ms: float = 12, startup_delay_s: float = 0.0):
        self.capacity, self.ttft_ms, self.itl_ms, self.startup_delay_s = capacity, ttft_ms, itl_ms, startup_delay_s
        self.procs: list[tuple[subprocess.Popen, str, float]] = []   # (proc, url, ready_at)
        self.events: list[tuple[float, str, int]] = []

    def urls(self) -> list[str]:
        return [u for _, u, _ in self.procs]

    def live(self) -> int:
        return len(self.procs)

    def _start_one(self) -> None:
        port = _free_port()
        cmd = [sys.executable, "-m", "tokenbench.mockserver", "--port", str(port), "--capacity", str(self.capacity),
               "--ttft-ms", str(self.ttft_ms), "--itl-ms", str(self.itl_ms)]
        p = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        url = f"http://127.0.0.1:{port}"
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            try:
                if httpx.get(f"{url}/health", timeout=0.5).status_code == 200:
                    break
            except httpx.HTTPError:
                time.sleep(0.05)
        self.procs.append((p, url, time.monotonic() + self.startup_delay_s))

    def scale_to(self, n: int) -> None:
        while self.live() < n:
            self._start_one()
        while self.live() > n:
            p, _, _ = self.procs.pop()        # newest first
            p.terminate()
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()
        self.events.append((time.monotonic(), "scale", n))

    def queue_depth(self) -> int:
        total = 0
        for _, u, _ in self.procs:
            try:
                t = httpx.get(f"{u}/metrics", timeout=2).text
                m = _WAITING.search(t)
                total += int(float(m.group(1))) if m else 0
            except httpx.HTTPError:
                continue
        return total

    def tick(self, policy: QueueDepthPolicy, now: float) -> int:
        want = policy.decide(queue_depth=self.queue_depth(), current=self.live(), now=now)
        if want != self.live():
            self.scale_to(want)
        return want

    def close(self) -> None:
        self.scale_to(0)


def run_controller(pool: MockReplicaPool, policy: QueueDepthPolicy, duration_s: float, interval_s: float = 1.0) -> list[dict]:
    """Poll and act every `interval_s`; returns a timeline of (t, queue, replicas)."""
    t0 = time.monotonic()
    timeline = []
    while time.monotonic() - t0 < duration_s:
        now = time.monotonic() - t0
        q = pool.queue_depth()
        want = policy.decide(queue_depth=q, current=pool.live(), now=now)
        if want != pool.live():
            pool.scale_to(want)
        timeline.append({"t": now, "queue": q, "replicas": pool.live()})
        time.sleep(interval_s)
    return timeline
