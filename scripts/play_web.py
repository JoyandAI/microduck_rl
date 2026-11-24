#!/usr/bin/env python3
"""`play` with mjlab's **web (viser) viewer** on a free port.

mjlab's browser viewer uses viser. This entry point selects the viser backend
by default and reads its port from PLAY_VISER_PORT (default 8081). It delegates
checkpoint loading and simulation to mjlab's `play` entry point.

Usage (same args as `uv run play`):
    # Use the exact task ID and checkpoint from training.
    uv run --no-sync python scripts/play_web.py Mjlab-Velocity-Flat-Backlash-OpenMD \
        --checkpoint-file logs/rsl_rl/openmd_velocity/<run>/model_9999.pt

    PLAY_VISER_PORT=8082 uv run python scripts/play_web.py ...   # 换端口
    ... --viewer native                                          # 仍想用原生窗口

Open the printed URL in a browser. Extra flags (`--num-envs`, `--device`,
`--camera`, `--video`, `--no-terminations`) pass straight through.
"""

from __future__ import annotations

import argparse
import os
import socket
import sys

import viser

_PORT = int(os.environ.get("PLAY_VISER_PORT", "8081"))

_orig_viser_server = viser.ViserServer


def _viser_server_on_free_port(*args, **kwargs):
    """Inject the port when the caller (mjlab's viewer) didn't pass one."""
    kwargs.setdefault("port", _PORT)
    return _orig_viser_server(*args, **kwargs)


viser.ViserServer = _viser_server_on_free_port  # type: ignore[assignment]


def _host_hint() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "<this-machine-ip>"


def main() -> int:
    argv = sys.argv[1:]
    # Default to the web viewer: with DISPLAY set, play's "auto" picks the native
    # window, which is invisible to a remote/web client.
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    parser.add_argument("--viewer")
    options, _ = parser.parse_known_args(argv)
    if options.viewer is None:
        argv += ["--viewer", "viser"]

    viewer = options.viewer or "viser"
    print(f"[play_web] viewer={viewer}", flush=True)
    if viewer == "viser":
        print(f"[play_web] http://127.0.0.1:{_PORT} or http://{_host_hint()}:{_PORT}", flush=True)

    sys.argv = [sys.argv[0]] + argv
    from mjlab.scripts.play import main as mjlab_play_main

    mjlab_play_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
