"""The scaling script's fits recover known curves and read the log headers."""

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("matplotlib")
SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


@pytest.fixture(scope="module")
def scaling():
    sys.path.insert(0, str(SCRIPTS))
    spec = importlib.util.spec_from_file_location("scaling_curves", SCRIPTS / "scaling_curves.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_power_fit_recovers_a_known_curve(scaling):
    D = np.geomspace(1.5, 8.0, 30)
    a, B, beta, rms = scaling.power_fit(D, 2.8 + 0.5 * D**-0.5)
    assert abs(a - 2.8) < 1e-3 and abs(B - 0.5) < 1e-3 and abs(beta - 0.5) < 0.006 and rms < 1e-4


def test_separable_solve_is_exact_through_three_anchors(scaling):
    E, A, B, alpha, beta = 2.0, 0.9, 0.8, 0.3, 0.4
    anchors = [(N, D, E + A * N**-alpha + B * D**-beta) for N, D in ((1.9, 4.8), (1.9, 9.6), (4.3, 10.7))]
    fit = scaling.separable_solve(anchors, alpha, beta)
    assert np.allclose(fit[:3], (E, A, B), atol=1e-9) and fit[3] < 1e-9
    assert abs(scaling.separable_predict(fit, 7.5, 75.0, alpha, beta) - (E + A * 7.5**-alpha + B * 75.0**-beta)) < 1e-9
    assert scaling.separable_solve(anchors[:2], alpha, beta) is None


def test_annealing_areas_follow_the_schedule(scaling):
    m = scaling.schedule_multiplier(1000, 20, 200)
    assert m[0] == pytest.approx(1 / 20) and m[19] == 1.0 and m[799] == 1.0 and m[-1] == 0.0
    S1, S2 = scaling.annealing_areas(m, 0.99)
    assert S1[-1] == pytest.approx(m.sum()) and S2[799] == 0.0 and np.all(np.diff(S2[800:]) >= 0)
    # A unit drop earns at most 1/(1-decay) of annealing area, approached as the cooldown lengthens.
    longer = scaling.annealing_areas(scaling.schedule_multiplier(10000, 20, 2000), 0.99)[1][-1]
    assert 0 < S2[-1] < longer < 100


def test_parse_log_reads_the_screen_recipe(scaling, tmp_path):
    header = (
        "run        | tag=t | data_root=/d | source=s | params=1 | device=cpu | ranks=1 | precision=fp8 | condition=f | seed=1 | "
        "data_seed=0 | seq_len=4096 | batch_rows=128 | steps=9142 | vocab_size=50304 | dim=768 | layers=16 | heads=4 | kv_heads=2 | "
        "head_dim=256 | expert_intermediate=832 | num_routed_experts=15 | experts_per_token=3 | pkda_heads=8 | pkda_head_dim=128 | pkda_conv_size=4 | loop_iterations=2\n"
        "schedule   | warmup_steps=183 | preheat_steps=0 | heat_steps=7131 | cooldown_steps=1828 | recurrence_boundary=5485 | start_step=0 | end_step=9142 | total_steps=9142\n"
    )
    lines = [f"step       | step={s}/9142 | phase=heat | loss=1 | pass1={3 + 1 / s:.4f} | gnorm=0.5\n" for s in range(1, 9143)]
    evals = [f"eval       | step={s}/9142 | val={3 + 1 / s:.4f}\n" for s in range(250, 9143, 250)]
    log = tmp_path / "t.log"
    log.write_text(header + "".join(lines + evals) + "done       | step=9142/9142 | val=2.9310\n")
    run = scaling.parse_log(log)
    assert run["scale"] == "screen" and run["active"] == 191_702_304 and run["tokens_per_step"] == 524_288
    assert run["stable_end"] == 7314 and run["boundary"] == 5485 and run["final"] == 2.931 and run["complete"]
    assert scaling.steps_for(run["active"], run["tokens_per_step"], 25) == 9142
    assert scaling.steps_for(run["active"], run["tokens_per_step"], 50) == 18283
