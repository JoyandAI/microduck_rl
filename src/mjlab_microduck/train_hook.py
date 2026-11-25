"""Keep `train <task> ... --hf-jobs` working, whoever owns the `train` script.

The flag used to live in a `train` console script of our own, declared in
`[project.scripts]` and documented as "shadowing" mjlab's. It shadows nothing:
mjlab 1.3.0 declares `train` too, two distributions declaring the SAME script
name is last-writer-wins at install time, and mjlab won — `uv sync` left
`mjlab.scripts.train:main` in `.venv/bin/train`, so our wrapper was never
invoked and `uv run train ... --hf-jobs` died on tyro's
`Unrecognized options: --hf-jobs` (2026-08-31). Nothing warns about it: the
install succeeds and the flag silently disappears.

So the flag is not implemented in a console script at all any more. It is
intercepted here, from the `mjlab.tasks` plugin entry point: mjlab's own
`mjlab/__init__.py` calls `_import_registered_packages()` at module scope,
which imports `mjlab_microduck.tasks` — and mjlab's `train` reaches that while
executing `from mjlab.scripts.train import main`, i.e. before its two-stage
tyro parse ever sees argv. That path is mjlab's own, so no install order can
take it away from us.

`uv run scripts/hf/train_hf.py <task> ...` calls `submit()` directly and stays
the escape hatch if this interception ever stops firing.
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from datetime import datetime
from pathlib import Path

_FLAG = "--hf-jobs"

#: Set on the job's environment by ``hf_jobs.submit`` — inside the job,
#: ``uv run train`` must always mean "train locally".
_IN_JOB_ENV = "MICRODUCK_IN_HF_JOB"

#: Actuator parameters the simulated physics depends on. A silent swap *during* a
#: run already invalidated two policies (kt/kd replaced at 2026-09-10 18:51 while a
#: velocity run was live — see docs/policy_failure_analysis.md), and nothing in the
#: run directory recorded which values it had used. Every process that imports the
#: tasks plugin now fingerprints them, so a swap is visible after the fact.
_PARAMS_DIR = Path(__file__).resolve().parents[2] / "vendor" / "bam" / "bam" / "params" / "hd1910"
_FINGERPRINT_LOG = Path(__file__).resolve().parents[2] / "logs" / "actuator_params_seen.tsv"


def actuator_params_fingerprint(params_dir: Path | None = None) -> tuple[str, dict]:
    """``(sha256-16, summary)`` of the HD-1910 parameter set currently deployed.

    Hashes every ``m*.json`` plus ``characterization.json`` (names and bytes), so
    both a value edit and an added/removed model change the digest.
    """
    d = Path(params_dir) if params_dir is not None else _PARAMS_DIR
    h = hashlib.sha256()
    files = sorted(list(d.glob("m*.json")) + list(d.glob("characterization.json")))
    for f in files:
        h.update(f.name.encode())
        h.update(f.read_bytes())
    summary: dict = {"dir": str(d), "n_files": len(files)}
    # Summarize the file training actually loads, not a sibling tier. m6 is the
    # xgoduck's fit (no kd); m5 is the older HD-1910 file.
    from mjlab_microduck.robot.microduck_constants import _BAM_ACTUATOR_KWARGS
    deployed = _BAM_ACTUATOR_KWARGS.get("model", "m6")
    summary["model"] = deployed
    deployed_json = d / f"{deployed}.json"
    if deployed_json.exists():
        p = json.loads(deployed_json.read_text(encoding="utf-8"))
        summary.update({k: p.get(k) for k in (
            "kt", "R", "error_gain", "error_gain_ratio", "kd",
            "friction_base", "command_delay", "actuator", "source")})
    return h.hexdigest()[:16], summary


def ignore_removed_wandb_start_method() -> None:
    """Let rsl_rl 5.0.1 call ``wandb.Settings(start_method=...)`` on wandb >= 0.29.

    wandb 0.29.0 deleted ``start_method`` (it had already done nothing for
    several releases) and ``Settings`` rejects unknown fields. rsl_rl 5.0.1,
    which mjlab 1.3.0 requires, still passes ``start_method="thread"`` inside
    ``WandbSummaryWriter``, so ``wandb.init`` dies before the first iteration.
    Drop the argument. On older wandb the field still exists; dropping it is
    harmless because the setting was already a no-op.
    """
    try:
        import wandb
    except ModuleNotFoundError:
        return
    settings_cls = wandb.Settings
    if getattr(settings_cls, "_microduck_start_method_ignored", False):
        return
    removed = "start_method" not in getattr(settings_cls, "model_fields", {})
    orig_init = settings_cls.__init__

    def _init(self, *args, **kwargs):
        kwargs.pop("start_method", None)
        return orig_init(self, *args, **kwargs)

    settings_cls.__init__ = _init  # type: ignore[method-assign]
    settings_cls._microduck_start_method_ignored = True
    if removed:
        print(
            "[wandb] Settings.start_method was removed in wandb>=0.29; "
            "dropping it so rsl_rl 5.0.1 can start a run",
            flush=True,
        )


def record_actuator_params() -> str | None:
    """Append one fingerprint line per process; never raises into the caller."""
    try:
        fp, s = actuator_params_fingerprint()
        e_gain = s.get("error_gain")
        if e_gain is None:
            e_gain = s.get("error_gain_ratio")
        line = (f"{datetime.now():%Y-%m-%d %H:%M:%S}\tpid={os.getpid()}\t{fp}\t"
                f"kt={s.get('kt')}\tkd={s.get('kd')}\te_gain={e_gain}\t"
                f"fric={s.get('friction_base')}\tdelay={s.get('command_delay')}\t"
                f"argv={' '.join(Path(a).name for a in sys.argv[:3])}\n")
        _FINGERPRINT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with _FINGERPRINT_LOG.open("a", encoding="utf-8") as fh:
            fh.write(line)
        kd = s.get("kd")
        # A missing kd is xgoduck's fit (no D term), not the 192.68 brake fit.
        warn = ""
        if kd is not None and float(kd) >= 5.0:
            warn = "  ⚠️ kd 异常偏大(参见 policy_failure_analysis)"
        print(f"[bam] actuator params fingerprint {fp} "
              f"({s.get('model')}: actuator={s.get('actuator')} kt={s.get('kt')} kd={kd}){warn}",
              flush=True)
        return fp
    except Exception as exc:  # never break training because of bookkeeping
        print(f"[bam] WARN: could not fingerprint actuator params: {exc}", flush=True)
        return None



def _invoked_as_train() -> bool:
    """True when argv[0] is mjlab's trainer (console script or `-m`).

    `play --hf-jobs` must NOT submit a training job; let that command's own
    parser reject the flag instead.
    """
    prog = Path(sys.argv[0]).name
    return prog.removesuffix(".py").removesuffix("-script") == "train"


def maybe_submit_to_hf_jobs() -> None:
    """Consume `--hf-jobs` and exit the process; a no-op without the flag.

    Called at import time of `mjlab_microduck.tasks`, so it runs inside mjlab's
    plugin loader. `SystemExit` is a `BaseException`, so it propagates through
    the loader's `except Exception` and out of `import mjlab` — the local
    trainer never starts.
    """
    if _FLAG not in sys.argv[1:]:
        return
    if os.environ.get(_IN_JOB_ENV):
        return
    if not _invoked_as_train():
        return

    from mjlab_microduck.hf_jobs import submit

    sys.exit(submit([a for a in sys.argv[1:] if a != _FLAG]))
