#!/usr/bin/env python3
"""`play` with mjlab's **web (viser) viewer** on a free port.

Why this exists: `uv run play <TASK> --viewer viser` is mjlab's official browser
viewer, but `PlayConfig` exposes no port and `viser.ViserServer` defaults to
**8080** — which on this box is already taken by the DSH Web GUI. So this shim
just forces the viser port (default 8081) and defaults the viewer backend to
`viser`, then hands off to mjlab's own `play` entrypoint unchanged.

Usage (same args as `uv run play`):
    uv run python scripts/play_web.py Mjlab-Velocity-Flat-MicroDuck \
        --checkpoint-file logs/rsl_rl/velocity/<run>/model_14999.pt

    PLAY_VISER_PORT=8082 uv run python scripts/play_web.py ...   # 换端口
    ... --viewer native                                          # 仍想用原生窗口

Open the printed URL in a browser. Extra flags (`--num-envs`, `--device`,
`--camera`, `--video`, `--no-terminations`) pass straight through.
"""

from __future__ import annotations

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
    if "--viewer" not in argv:
        argv += ["--viewer", "viser"]

    print("=" * 72, flush=True)
    print(f"[play_web] viewer=viser  port={_PORT}", flush=True)
    print(f"[play_web] open:  http://127.0.0.1:{_PORT}", flush=True)
    print(f"[play_web]   or:  http://{_host_hint()}:{_PORT}", flush=True)
    print("=" * 72, flush=True)

    sys.argv = [sys.argv[0]] + argv
    from mjlab.scripts.play import main as mjlab_play_main

    mjlab_play_main()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
