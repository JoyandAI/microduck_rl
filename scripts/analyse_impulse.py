#!/usr/bin/env python3
"""Per-half-step yaw IMPULSE for every per-frame CSV in a directory.

Reuses agent D's own analysis (`docs/forensics_2026-09-18/agent_D_phase_impulse.py`)
rather than re-deriving it — D validated that probe against `eval_straightness.py`,
and its published numbers are the self-check printed at the bottom.

WHY THIS QUANTITY: agent D showed the official walks straight because its gait does
not PRODUCE a yaw disturbance (0.46 deg per half-step) while ours produces a large
one (6.35 deg) and cancels it to a few percent, leaving a large random residual.
The flight penalty moved its own metric (flight 0.229 -> 0.031) without buying
straightness, so the question this prints is: did ANY arm reduce the impulse itself?

Usage: python scripts/analyse_impulse.py <dir-with-csvs>
"""
from __future__ import annotations

import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent
SRC = REPO / "docs/forensics_2026-09-18/agent_D_phase_impulse.py"


def load_agent_module():
    src = SRC.read_text()
    src = src.split('if __name__ == "__main__":')[0]  # keep defs, drop the demo
    ns: dict = {}
    exec(compile(src, str(SRC), "exec"), ns)
    return ns["analyse"], ns["rep"]


def main() -> int:
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "logs/impulse")
    analyse, rep = load_agent_module()

    rows_out = []
    print()
    print(
        f"{'policy':20s}{'J_L':>9s}{'J_R':>9s}{'J_net':>9s}{'|L|-|R|':>9s}"
        f"{'per-halfstep':>13s}{'frac_D':>8s}{'step Hz':>8s}{'travel m':>9s}"
    )
    for csv in sorted(out.glob("*.csv")):
        try:
            rows = analyse(str(csv))
            mean, _med = rep(csv.stem, rows)
        except Exception as e:  # noqa: BLE001 - report and continue
            print(f"{csv.stem:20s} ERROR {type(e).__name__}: {e}")
            continue
        half_steps = max(mean["n_events"] / 2.0, 1e-9)
        per_half = abs(mean["J_L"]) / half_steps
        rows_out.append((csv.stem, mean, per_half))
        print(
            f"{csv.stem:20s}{mean['J_L']:+9.2f}{mean['J_R']:+9.2f}{mean['J_net']:+9.2f}"
            f"{mean['mismatch']:+9.2f}{per_half:13.2f}{mean['frac_D']:8.3f}"
            f"{mean['step_hz']:8.2f}{mean['speed_along']:9.3f}"
        )

    print(
        "\n[self-check] official per-half-step impulse should be ~0.46 deg and "
        "ck7000 ~6.35 deg (agent D's published numbers)."
    )
    print(
        "[read] per-halfstep is the DISTURBANCE the gait generates per step; "
        "J_net is what survives. Straight-by-not-producing => small per-halfstep."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
