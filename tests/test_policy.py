from gpuscaler.policy import QueueDepthPolicy


def test_desired_replicas_is_ceil_of_metric_over_target_within_bounds():
    p = QueueDepthPolicy(target_per_replica=4, min_replicas=1, max_replicas=10)
    assert p.desired(queue_depth=0, current=1, now=0) == 1
    assert p.desired(queue_depth=9, current=1, now=0) == 3          # ceil(9/4)
    assert p.desired(queue_depth=400, current=1, now=0) == 10       # clamped


def test_scale_up_respects_cooldown():
    p = QueueDepthPolicy(target_per_replica=4, min_replicas=1, max_replicas=10, scale_up_cooldown_s=60)
    assert p.decide(queue_depth=20, current=1, now=0) == 5
    assert p.decide(queue_depth=40, current=5, now=30) == 5          # still cooling down
    assert p.decide(queue_depth=40, current=5, now=61) == 10


def test_scale_down_waits_for_stabilisation_window_and_steps_one_at_a_time():
    p = QueueDepthPolicy(target_per_replica=4, min_replicas=1, max_replicas=10, scale_down_window_s=120, scale_down_step=1)
    p.decide(queue_depth=20, current=1, now=0)                       # -> 5
    assert p.decide(queue_depth=0, current=5, now=10) == 5           # low for 10 s only
    assert p.decide(queue_depth=0, current=5, now=100) == 5
    assert p.decide(queue_depth=0, current=5, now=121) == 4          # window elapsed: one step down
    assert p.decide(queue_depth=0, current=4, now=130) == 4          # window restarts after a change
    assert p.decide(queue_depth=3, current=4, now=200) == 4          # metric rose above target x (current-1)? no: 3 <= 4*3 so still removable
    assert p.decide(queue_depth=0, current=4, now=251) == 3


def test_scale_down_window_resets_when_metric_exceeds_target():
    p = QueueDepthPolicy(target_per_replica=4, min_replicas=1, max_replicas=10, scale_down_window_s=60)
    p.decide(queue_depth=20, current=1, now=0)
    p.decide(queue_depth=0, current=5, now=10)
    p.decide(queue_depth=30, current=5, now=40)                      # burst again: window must reset
    assert p.decide(queue_depth=0, current=5, now=70) == 5
    assert p.decide(queue_depth=0, current=5, now=101) == 4


def test_scale_to_zero_only_when_allowed():
    p0 = QueueDepthPolicy(target_per_replica=4, min_replicas=0, max_replicas=4, scale_down_window_s=10)
    p0.decide(queue_depth=5, current=1, now=0)
    assert p0.decide(queue_depth=0, current=2, now=11) == 1 and p0.decide(queue_depth=0, current=1, now=22) == 0
    p1 = QueueDepthPolicy(target_per_replica=4, min_replicas=1, max_replicas=4, scale_down_window_s=10)
    assert p1.decide(queue_depth=0, current=1, now=100) == 1
