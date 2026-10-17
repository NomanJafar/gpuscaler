"""The controller against real tokenbench mock replicas: it reads queue depth from their /metrics
and starts/stops replica processes to match the policy's decision."""

import time

import httpx

from gpuscaler.controller import MockReplicaPool
from gpuscaler.policy import QueueDepthPolicy


def test_pool_starts_and_stops_mock_replicas_and_reads_queue_depth():
    pool = MockReplicaPool(capacity=2, ttft_ms=200, itl_ms=5)
    try:
        pool.scale_to(2)
        assert pool.live() == 2 and all(httpx.get(f"{u}/health", timeout=2).status_code == 200 for u in pool.urls())
        assert pool.queue_depth() == 0
        pool.scale_to(1)
        assert pool.live() == 1 and len(pool.urls()) == 1
    finally:
        pool.close()


def test_controller_tick_scales_up_when_replicas_queue(tmp_path):
    import asyncio
    pool = MockReplicaPool(capacity=1, ttft_ms=50, itl_ms=20)
    policy = QueueDepthPolicy(target_per_replica=1, min_replicas=1, max_replicas=3, scale_up_cooldown_s=0)
    try:
        pool.scale_to(1)
        url = pool.urls()[0]

        async def flood():
            async with httpx.AsyncClient(base_url=url, timeout=30) as c:
                body = {"model": "mock", "messages": [{"role": "user", "content": "x"}], "max_tokens": 40, "stream": False, "ignore_eos": True}
                await asyncio.gather(*[c.post("/v1/chat/completions", json=body) for _ in range(4)])

        import threading
        th = threading.Thread(target=lambda: asyncio.run(flood()), daemon=True); th.start()
        time.sleep(0.3)
        depth = pool.queue_depth()
        assert depth >= 2                                   # capacity 1: the other requests wait
        decided = pool.tick(policy, now=0.0)
        assert decided == 3 and pool.live() == 3
        th.join(timeout=10)
    finally:
        pool.close()
