"""`train` console script — deliberately identical to mjlab's own.

This project must keep DECLARING a `train` script even though it no longer
needs one: `--hf-jobs` is intercepted from the `mjlab.tasks` plugin entry point
(train_hook.py), on mjlab's own import path, so the wrapper has nothing left to
do.

Why keep it anyway: same-name console scripts are last-writer-wins, and after
an install BOTH dist-info RECORDs claim `.venv/bin/train`. Removing this
declaration therefore does not hand the name back to mjlab — it makes the next
`uv sync` UNINSTALL the file (ours, per our RECORD) while nothing reinstalls
mjlab's. `train` then vanishes from the venv entirely and `uv run train` falls
through to whatever `train` sits on PATH: on one machine liblinear's, which
answered `can't open input file Mjlab-Velocity-Flat-MicroDuck` (2026-08-31).

So the collision stays, and is made harmless instead: whichever shim wins,
`train` behaves the same, and the flag is handled in exactly one place.
"""

from __future__ import annotations

import os
import sys

_ENV_TOGGLE = "MICRODUCK_TASK_NAMED_RUNS"
_ENV_EXPLICIT_NAME = "MICRODUCK_RUN_DIR_NAME"


def _task_slug() -> str | None:
    """The run-directory name for this invocation: the TASK ID, never a clock.

    `MICRODUCK_RUN_DIR_NAME` wins (the training queue uses it to tag a smoke run
    apart from the real one); otherwise the first non-flag argv entry that looks
    like a task id is taken, which is exactly how `train <TASK_ID>` is spelled.
    """
    explicit = os.environ.get(_ENV_EXPLICIT_NAME)
    if explicit:
        return explicit
    for arg in sys.argv[1:]:
        if not arg.startswith("-") and arg.startswith("Mjlab-"):
            return arg
    return None


def _unique_run_name(slug: str) -> str:
    """`slug`, or `slug_r2` / `slug_r3` … if that run directory already exists.

    Deterministic names must not silently overwrite a previous run: the queue's
    whole point is that a task's directory can be named before it starts, and a
    second attempt at the same task is a normal thing to do.
    """
    from pathlib import Path

    root = Path("logs") / "rsl_rl"
    if not (root / slug).exists() and not list(root.glob(f"*/{slug}")):
        return slug
    n = 2
    while list(root.glob(f"*/{slug}_r{n}")):
        n += 1
    return f"{slug}_r{n}"


# mjlab/scripts/train.py, launch_training: the only clock read that names a run.
_MJLAB_STAMP_FMT = "%Y-%m-%d_%H-%M-%S"
_COMPACT_STAMP_FMT = "%Y%m%d-%H%M%S"


def _install_run_dir_naming() -> None:
    """Rewrite the run-directory clock, without forking ``launch_training``.

    mjlab composes the directory as
    ``f"{datetime.now():%Y-%m-%d_%H-%M-%S}_{run_name}"``. That function also
    owns GPU selection and the torchrunx path, so a copy here would drift.
    Only the ``strftime`` call is redirected:

    - the stamp becomes ``YYYYMMDD-HHMMSS`` (``20261008-161025_openmd-walk``);
    - ``MICRODUCK_TASK_NAMED_RUNS=1`` still replaces the stamp with the task
      id, so a queue can name the directory before the run starts.

    If mjlab stops calling ``strftime`` with that format, this degrades to
    mjlab's own naming — the run still starts, only the directory name differs.
    """
    from datetime import datetime as real_datetime

    from mjlab.scripts import train as mjlab_train

    class _Now:
        def strftime(self, fmt: str) -> str:
            if os.environ.get(_ENV_TOGGLE) == "1":
                slug = _task_slug()
                if slug is not None:
                    return _unique_run_name(slug)
            if fmt == _MJLAB_STAMP_FMT:
                fmt = _COMPACT_STAMP_FMT
            return real_datetime.now().strftime(fmt)

    class _Datetime:
        @staticmethod
        def now() -> _Now:
            return _Now()

    mjlab_train.datetime = _Datetime  # type: ignore[assignment]

    if os.environ.get(_ENV_TOGGLE) != "1":
        return
    slug = _task_slug()
    if slug is None:
        return

    # mjlab then appends `_{run_name}` when the task's runner cfg carries one
    # (e.g. `microduck_roulade`), which would give
    # `Mjlab-Roulade-Flat-MicroDuck_microduck_roulade`. Blank it for the length of
    # this invocation so the directory is exactly the task id. `launch_training`
    # is WRAPPED rather than forked — it also owns GPU selection and the torchrunx
    # path, and `main()` resolves it as a module global, so the wrapper is used.
    import dataclasses

    _orig_launch_training = mjlab_train.launch_training

    def launch_training(task_id, args=None):  # noqa: ANN001, ANN202
        if args is None:
            args = mjlab_train.TrainConfig.from_task(task_id)
        args = dataclasses.replace(
            args, agent=dataclasses.replace(args.agent, run_name="")
        )
        return _orig_launch_training(task_id=task_id, args=args)

    mjlab_train.launch_training = launch_training  # type: ignore[assignment]
    print(f"[train] run directory named by task: {slug}", flush=True)


def main() -> int | None:
    # This import runs mjlab's plugin loader, which imports
    # mjlab_microduck.tasks -> train_hook.maybe_submit_to_hf_jobs(). A
    # `--hf-jobs` invocation submits and exits inside the import below; it
    # never comes back here.
    from mjlab.scripts.train import main as mjlab_train_main

    _install_run_dir_naming()
    return mjlab_train_main()


if __name__ == "__main__":
    sys.exit(main())
