import json

from gpuscaler.replay import POLICIES, render_markdown, run_replay


def test_replay_runs_every_policy_and_reports_cost_vs_latency(tmp_path):
    res = run_replay(out_dir=tmp_path, duration_s=1200, base_rate=1.0, burst_rate=8.0, burst_start_s=300, burst_len_s=180,
                     repeat_every_s=600, capacity=4, service_time_s=3.0, cold_start_s=90, seed=1, plots=True)
    assert set(res) == set(POLICIES)
    peak, avg, react, warm = res["fixed: peak-sized"], res["fixed: average-sized"], res["reactive"], res["reactive + 1 warm standby"]
    assert peak["wait"]["p95"] < 1.0 and peak["gpu_hours"] > avg["gpu_hours"]
    assert avg["wait"]["p95"] > react["wait"]["p95"] > warm["wait"]["p95"]
    assert react["gpu_hours"] < peak["gpu_hours"]
    assert (tmp_path / "timelines.png").exists() and (tmp_path / "cost_vs_latency.png").exists()
    data = json.loads((tmp_path / "replay.json").read_text())
    assert "reactive" in data and len(data["reactive"]["timeline"]) > 100


def test_render_markdown_has_one_row_per_policy():
    rows = {"a": {"gpu_hours": 1.0, "peak_replicas": 2, "wait": {"p50": 0.0, "p95": 1.0, "p99": 2.0, "max": 3.0}, "peak_queue": 5, "served": 10, "unserved": 0},
            "b": {"gpu_hours": 2.0, "peak_replicas": 4, "wait": {"p50": 0.0, "p95": 0.5, "p99": 1.0, "max": 1.5}, "peak_queue": 1, "served": 10, "unserved": 0}}
    md = render_markdown(rows)
    assert md.count("\n| ") == 2 and "GPU-hours" in md
