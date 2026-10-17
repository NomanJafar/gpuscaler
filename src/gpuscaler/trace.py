"""Arrival traces: a Poisson baseline with rectangular bursts."""

from __future__ import annotations

import random


def bursty_trace(duration_s: float, base_rate: float, burst_rate: float, burst_start_s: float, burst_len_s: float,
                 seed: int = 0, repeat_every_s: float | None = None) -> list[float]:
    rng = random.Random(seed)
    out, t = [], 0.0
    while t < duration_s:
        in_burst = False
        if repeat_every_s:
            phase = (t - burst_start_s) % repeat_every_s
            in_burst = t >= burst_start_s and phase < burst_len_s
        else:
            in_burst = burst_start_s <= t < burst_start_s + burst_len_s
        rate = burst_rate if in_burst else base_rate
        t += rng.expovariate(rate)
        if t < duration_s:
            out.append(t)
    return out
