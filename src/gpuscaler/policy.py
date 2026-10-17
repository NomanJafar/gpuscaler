"""Queue-depth scaling policy with the KEDA/HPA shape (CONCEPTS.md sections 2 and 4):
desired = ceil(queue / target), clamped; scale up at most once per cooldown; scale down
one step at a time only after the metric has stayed at or below the lower threshold for
the whole stabilisation window."""

from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class QueueDepthPolicy:
    target_per_replica: float = 4.0
    min_replicas: int = 1
    max_replicas: int = 8
    scale_up_cooldown_s: float = 60.0
    scale_down_window_s: float = 180.0
    scale_down_step: int = 1
    _last_up: float | None = field(default=None, init=False)
    _last_high: float = field(default=0.0, init=False)   # last time the metric was high or the pool changed
    decisions: list[tuple[float, int, int, float]] = field(default_factory=list)  # (t, from, to, queue)

    def desired(self, queue_depth: float, current: int, now: float) -> int:
        want = math.ceil(queue_depth / self.target_per_replica) if queue_depth > 0 else self.min_replicas
        return max(self.min_replicas, min(self.max_replicas, want))

    def decide(self, queue_depth: float, current: int, now: float) -> int:
        want = self.desired(queue_depth, current, now)
        if want > current:
            self._last_high = now
            if self._last_up is None or now - self._last_up >= self.scale_up_cooldown_s:
                self._last_up = now
                self.decisions.append((now, current, want, queue_depth))
                return want
            return current
        # Scale-down candidate: the metric would fit in (current - step) replicas. Like the HPA's
        # stabilisation window, it must have been that way since the last high sample or pool change.
        fits_smaller = queue_depth <= self.target_per_replica * max(self.min_replicas, current - self.scale_down_step)
        if not fits_smaller:
            self._last_high = now
            return current
        if current > self.min_replicas and now - self._last_high >= self.scale_down_window_s:
            new = max(self.min_replicas, current - self.scale_down_step)
            self._last_high = now
            self.decisions.append((now, current, new, queue_depth))
            return new
        return current
