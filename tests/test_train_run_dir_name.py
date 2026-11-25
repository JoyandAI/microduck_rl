"""Run directories are ``YYYYMMDD-HHMMSS_<run_name>``.

mjlab still calls ``strftime("%Y-%m-%d_%H-%M-%S")``. ``train`` rewrites that
one format so a saved run (and an onnx named after it) reads
``20261008-161025_openmd-walk`` instead of ``2026-10-08_16-10-25_openmd-walk``.
"""

from datetime import datetime

import pytest

from mjlab_microduck import train_cli


@pytest.fixture
def naming(monkeypatch):
    import mjlab.scripts.train as mjlab_train

    original = mjlab_train.datetime
    monkeypatch.delenv(train_cli._ENV_TOGGLE, raising=False)
    train_cli._install_run_dir_naming()
    try:
        yield mjlab_train
    finally:
        mjlab_train.datetime = original


def test_stamp_is_yyyymmdd_hhmmss(naming):
    before = datetime.now().strftime(train_cli._COMPACT_STAMP_FMT)
    stamp = naming.datetime.now().strftime(train_cli._MJLAB_STAMP_FMT)
    after = datetime.now().strftime(train_cli._COMPACT_STAMP_FMT)
    assert stamp in {before, after}
    assert len(stamp) == 15 and stamp[8] == "-" and stamp[:8].isdigit() and stamp[9:].isdigit()


def test_other_strftime_formats_pass_through(naming):
    assert naming.datetime.now().strftime("%Y") == datetime.now().strftime("%Y")


def test_task_named_runs_still_replace_the_stamp(naming, monkeypatch):
    monkeypatch.setenv(train_cli._ENV_TOGGLE, "1")
    monkeypatch.setattr(train_cli, "_task_slug", lambda: "Mjlab-Velocity-Flat-OpenMD")
    monkeypatch.setattr(train_cli, "_unique_run_name", lambda slug: slug)
    assert naming.datetime.now().strftime(train_cli._MJLAB_STAMP_FMT) == "Mjlab-Velocity-Flat-OpenMD"
