#!/usr/bin/env python3
"""Thin launcher for the RecaMem desktop app.

``python app.py`` opens the desktop UI (Tauri shell when built, browser fallback
otherwise). ``streamlit run app.py`` still starts the legacy console from
:mod:`recamem.console_legacy`, kept for one release; the supported surface is
`recamem ui` / `recamem serve --web`.

Everything model-related lives in the Python API (:mod:`recamem.api`); this file
holds no model code.
"""

from __future__ import annotations

import runpy
import sys


def running_under_streamlit() -> bool:
    try:
        from streamlit import runtime
    except Exception:  # noqa: BLE001 - streamlit is optional
        return False
    try:
        return bool(runtime.exists())
    except Exception:  # noqa: BLE001 - older streamlit builds
        return False


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if "--legacy-console" in argv or running_under_streamlit():
        args = [arg for arg in argv if arg != "--legacy-console"]
        sys.argv = ["app.py", *args]
        runpy.run_module("recamem.console_legacy", run_name="__main__")
        return 0

    from recamem.cli import main as cli_main

    return cli_main(["ui", *argv])


if __name__ == "__main__":
    raise SystemExit(main())
