"""Headless runner: chat/teach/recall/decide without Streamlit, Node or Tauri.

Everything here depends on core dependencies only. Each invocation prints one
JSON object to stdout, so scripts and CI can consume the output without parsing
human-readable text::

    recamem run --prompt "What is the calibration constant?" --name demo
    printf 'hello\\nstatus?\\n' | recamem run --name demo

The runner never fabricates an answer: when the profile has no generation
backend it reports the recall evidence it found and says so in ``backend``.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, Iterable, Optional, TextIO

DEFAULT_PROFILE = "droid-alpha"


class RunnerError(RuntimeError):
    """A runner invocation that cannot produce an honest answer."""


def _anchor_mode() -> str:
    from recamem.anchor import resolve_anchor_mode

    return resolve_anchor_mode()


def load_engine(name: Optional[str] = None, base_dir: Optional[str] = None):
    """Load (or create) the named profile through the manager."""
    from recamem import identity
    from recamem.manager import DroidManager

    home = base_dir or str(identity.default_base_dir())
    manager = DroidManager(base_dir=home)
    return manager, manager.get_droid(name or DEFAULT_PROFILE), home


def _memory_payload(engine, prompt: str, top_k: int = 3) -> list[Dict[str, Any]]:
    """Recall evidence for the prompt, as plain JSON-serialisable rows."""
    try:
        hits = engine.recall(prompt, top_k=top_k)
    except Exception as exc:  # noqa: BLE001 - a recall failure must not kill chat
        return [{"error": f"{type(exc).__name__}: {exc}"}]
    rows = []
    for hit in hits or []:
        if isinstance(hit, dict):
            rows.append(
                {
                    "text": hit.get("text"),
                    "source": hit.get("source"),
                    "score": hit.get("score"),
                    "channels": hit.get("channels"),
                }
            )
        else:  # pragma: no cover - defensive for older engine versions
            rows.append({"text": str(hit), "source": None, "score": None})
    return rows


def run_once(prompt: str, engine, *, mode: str = "chat", profile: Optional[str] = None,
             top_k: int = 3) -> Dict[str, Any]:
    """Execute one prompt and return the JSON-serialisable payload."""
    if not isinstance(prompt, str) or not prompt.strip():
        raise RunnerError("prompt must be a non-empty string")
    if mode not in {"chat", "recall", "teach"}:
        raise RunnerError(f"unknown runner mode {mode!r}")

    payload: Dict[str, Any] = {
        "prompt": prompt,
        "profile": profile or getattr(engine, "name", None),
        "anchor": _anchor_mode(),
        "backend": "memory" if mode != "teach" else "teach",
    }
    if mode == "teach":
        result = engine.teach(prompt)
        if not isinstance(result, dict):
            result = {"propositions": int(result or 0)}
        stored = int(result.get("propositions", 0) or 0)
        payload["response"] = f"stored {stored} fact(s)"
        payload["memory"] = []
        payload["facts_stored"] = stored
        payload["teach"] = result
        return payload

    if mode == "recall":
        memory = _memory_payload(engine, prompt, top_k=top_k)
        payload["memory"] = memory
        payload["response"] = (
            memory[0].get("text") if memory and memory[0].get("text") else ""
        )
        return payload

    payload["memory"] = _memory_payload(engine, prompt, top_k=top_k)
    payload["response"] = engine.chat(prompt)
    return payload


def run_loop(engine, stdin: Iterable[str], stdout: TextIO, *, mode: str = "chat",
             profile: Optional[str] = None, top_k: int = 3) -> int:
    """Read prompts (one per line) and emit one JSON object per line."""
    count = 0
    for line in stdin:
        prompt = line.rstrip("\n")
        if not prompt.strip():
            continue
        try:
            payload = run_once(prompt, engine, mode=mode, profile=profile, top_k=top_k)
        except RunnerError as exc:
            payload = {"prompt": prompt, "error": str(exc)}
        stdout.write(json.dumps(payload, sort_keys=True) + "\n")
        stdout.flush()
        count += 1
    return count


def main(argv: Optional[list[str]] = None) -> int:
    """``recamem run`` implementation (kept here so tests can drive it)."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="recamem run",
        description="Headless runner: one JSON object per prompt (core deps only)",
    )
    parser.add_argument("--prompt", default=None, help="run this prompt and exit")
    parser.add_argument("--name", default=DEFAULT_PROFILE, help="profile name")
    parser.add_argument("--base-dir", default=None, help="profile home")
    parser.add_argument("--mode", default="chat", choices=["chat", "recall", "teach"])
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args(argv)

    try:
        _manager, engine, home = load_engine(args.name, args.base_dir)
    except Exception as exc:  # noqa: BLE001 - exit 1 with a JSON error, no traceback
        json.dump(
            {"error": f"{type(exc).__name__}: {exc}", "home": args.base_dir or os.environ.get("RECAMEM_HOME")},
            sys.stdout,
        )
        sys.stdout.write("\n")
        return 1

    if args.prompt is not None:
        try:
            payload = run_once(
                args.prompt, engine, mode=args.mode, profile=args.name, top_k=args.top_k
            )
        except RunnerError as exc:
            json.dump({"prompt": args.prompt, "error": str(exc)}, sys.stdout)
            sys.stdout.write("\n")
            return 2
        sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
        return 0

    if sys.stdin.isatty():
        sys.stderr.write(
            "recamem run: nothing to do; pass --prompt or pipe prompts on stdin\n"
        )
        return 1
    count = run_loop(
        engine, sys.stdin, sys.stdout, mode=args.mode, profile=args.name, top_k=args.top_k
    )
    return 0 if count else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
