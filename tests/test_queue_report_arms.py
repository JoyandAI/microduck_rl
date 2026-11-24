"""The queue-verdict readers must not silently drop an arm or a phase.

`scripts/queue_round2_report.py` is what turns four training runs into the
acceptance verdict, and both of its readers have already failed once in a way
that produced a *plausible but incomplete* table rather than an error:

  * `_discover_arms` used to find arms by filename glob `walk*_walk02.json`.
    That silently dropped every arm not named "walk*" — including `turn_first`
    and `turnfirst_yawdc`, which are precisely the arms that target the user's
    requirement ("turn when commanded, straight otherwise"). A verdict table
    missing the decisive arm is worse than no table, because it looks complete.
  * `_parse_sitstand` scrapes `eval_sitstand_head.py`'s stdout; a format change
    makes it return `[]`, and the Q2 section then prints "could not parse"
    instead of the numbers the whole fix was judged on.

These tests run on CPU and use fixture files only.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]


def _load():
    spec = importlib.util.spec_from_file_location(
        "queue_round2_report", REPO / "scripts" / "queue_round2_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture()
def mod():
    return _load()


HEADER = ("name\ttask\titers\tenvs\trun_dir\trc\tfp_before\tfp_after\t"
          "elapsed_s\tonnx\teval_walk02\teval_zerocmd\teval_sitstand\n")


def test_discover_arms_finds_arms_whose_names_do_not_start_with_walk(mod, tmp_path,
                                                                    monkeypatch):
    """The regression: a `walk*` glob dropped exactly the decisive arms."""
    out = tmp_path / "queue"
    out.mkdir()
    # one real file per arm, so the existence check passes
    rows = []
    for arm in ("walk_sym", "walk_yawdc", "turn_first", "turnfirst_yawdc"):
        j = out / f"{arm}_walk02.json"
        j.write_text('{"aggregate": {}}')
        z = out / f"{arm}_zerocmd.json"
        z.write_text('{"aggregate": {}}')
        rows.append(f"{arm}\tT\t9000\t2048\tlogs/rsl_rl/x/{arm}\t0\tfp\tfp\t1\t"
                    f"{arm}.onnx\t{j}\t{z}\t-")
    (out / "results.tsv").write_text(HEADER + "\n".join(rows) + "\n")
    monkeypatch.setattr(mod, "OUT", out)
    monkeypatch.setattr(mod, "ROOT", Path("/"))

    arms = mod._discover_arms()
    assert set(arms) == {"walk_sym", "walk_yawdc", "turn_first", "turnfirst_yawdc"}
    for arm in arms:
        assert "walk02" in arms[arm], arm
        assert "zerocmd" in arms[arm], arm
        assert "sitstand" not in arms[arm], arm


def test_discover_arms_ignores_placeholder_and_missing_paths(mod, tmp_path, monkeypatch):
    out = tmp_path / "queue"
    out.mkdir()
    (out / "results.tsv").write_text(
        HEADER
        + "smoke_fail\tT\t1\t64\t-\t1\tfp\tfp\t0\t-\t-\t-\t-\n"
        + "gone\tT\t1\t64\trun\t0\tfp\tfp\t0\tx.onnx\t/nonexistent/a.json\t-\t-\n")
    monkeypatch.setattr(mod, "OUT", out)
    arms = mod._discover_arms()
    assert arms == {}, arms


def test_discover_arms_returns_empty_without_a_results_table(mod, tmp_path, monkeypatch):
    monkeypatch.setattr(mod, "OUT", tmp_path / "nope")
    assert mod._discover_arms() == {}


SITSTAND_STDOUT = """
=== sitstand head behaviour (faithful BAM env, no pushes) ===
  onnx    : x.onnx
  schedule: [(0, 3.0), (1, 4.0), (0, 4.0)]   (0=STAND, 1=SIT)

phase             n   z_mean  tilt_mean  neck_rel  head_rel  track_err
----------------------------------------------------------------------
seg0_STAND      150    0.115      35.35    -1.932    -0.086     0.6993

  --- STEADY STATE ONLY (last 30% of each segment) ---
phase                 z     tilt  neck_rel  head_rel  neck_abs  hcmd_neck      err  headF_N head_z_mm
-----------------------------------------------------------------------------------------------------
seg0_STAND        0.114     35.1    -1.932    -0.082    -1.583     -0.026   0.6647    3.283     88.0
seg1_SIT          0.056      1.1    -1.932    -0.011    -1.583     -0.045   0.4933    2.324     92.9
seg2_STAND        0.115     35.1    -1.933    -0.075    -1.584      0.012   0.6672    3.273     88.3

  --- DID IT ACHIEVE THE POSTURE? (extreme over the whole segment) ---
"""


def test_parse_sitstand_reads_all_three_phases(mod, tmp_path):
    log = tmp_path / "sitstand_headfix_eval.log"
    log.write_text(SITSTAND_STDOUT)
    rows = mod._parse_sitstand(log)
    assert [r[0] for r in rows] == ["seg0_STAND", "seg1_SIT", "seg2_STAND"]
    assert rows[0][1] == pytest.approx(0.114)
    assert rows[0][2] == pytest.approx(35.1)
    assert rows[0][3] == pytest.approx(-1.932)


def test_parse_sitstand_returns_empty_rather_than_garbage_on_a_format_change(mod,
                                                                            tmp_path):
    log = tmp_path / "x_eval.log"
    log.write_text("something else entirely\n")
    assert mod._parse_sitstand(log) == []


def test_parse_sitstand_ignores_the_pre_steady_block(mod, tmp_path):
    """The header/whole-segment block above must not be mistaken for steady rows."""
    log = tmp_path / "x_eval.log"
    log.write_text(SITSTAND_STDOUT)
    rows = mod._parse_sitstand(log)
    # the earlier `seg0_STAND 150 0.115 35.35 ...` line has 6 fields, not 7, and
    # lives before the marker — it must not appear
    assert len(rows) == 3, rows
    # 6 fields since the physical columns were added: (phase, z, tilt, neck,
    # headF_N, head_z_m). The pre-marker block must still be excluded above all.
    assert all(len(r) == 6 for r in rows)
