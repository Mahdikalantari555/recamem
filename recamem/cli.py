"""The ``recamem`` command.

New subcommands live here (``ingest``, ``recall``, ``doctor``, ``migrate``,
``rtu``); the historical commands keep their behaviour and are delegated to the
previous parser so nothing regresses. ``python -m src.cli`` and ``tmt-droid``
still work and print a deprecation notice naming ``recamem``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

LEGACY_COMMANDS = {
    "chat",
    "teach",
    "sleep",
    "decide",
    "package",
    "eval",
    "research",
    "bench",
    "download-model",
    "generate",
}

NEW_COMMANDS = {"ingest", "recall", "doctor", "migrate", "rtu", "run", "ui", "serve"}

USAGE = """recamem <command> [options]

memory
  chat                 interactive terminal chat with a profile
  teach                teach facts from stdin or a file
  ingest <dir>         ingest Markdown, text and (optional) PDF files
  recall <query>       two-channel recall with sources
  decide               System-1 option scoring
  sleep                run one consolidation cycle

models
  rtu                  inspect or run a byte-RTU checkpoint
  download-model       download a GGUF model
  generate             grounded generation through the profile backend

evaluation and operations
  eval                 retrieval evaluation (exit 1 when it cannot run)
  bench                throughput / perplexity / grammar probes
  research             acquire knowledge from an external source
  serve                start the MCP stdio server or the REST daemon
  package              export, import or list .droid archives
  doctor               readiness report (anchor, backend, storage)
  serve --web          versioned loopback API + built frontend
  run                  headless runner (one JSON object per prompt)
  ui                   desktop app (Tauri), or the browser fallback
  migrate <dir>        copy legacy droids/ profiles into the recamem home
"""


# --------------------------------------------------------------------------- #
# New subcommands
# --------------------------------------------------------------------------- #
def cmd_migrate(args) -> int:
    from recamem import identity

    report = identity.migrate_droids(args.source, args.dest)
    for line in identity.describe_migration(report):
        print(line)
    return 0 if report.get("status") == "ok" and report.get("counts", {}).get("failed", 0) == 0 else 1


def cmd_recall(args) -> int:
    from recamem.manager import DroidManager
    from recamem.retrieval import TwoChannelRecall

    manager = _manager(args)
    engine = manager.get_droid(args.name)
    recall = TwoChannelRecall(engine)
    result = recall.recall(args.query, top_k=args.top_k)
    print(f"profile={args.name}")
    print(f"query={args.query}")
    print(f"results={len(result['results'])}")
    print(f"scorer={result['scorer']}")
    if result.get("reason"):
        print(f"reason={result['reason']}")
    for item in result["results"]:
        text = " ".join(str(item["text"]).split())
        print(f"fact={item['id']} score={item['scores']['fused']:.4f} source={item['source']} text={text}")
    return 0


def cmd_ingest(args) -> int:
    from recamem.ingest import describe_report, ingest_directory

    manager = _manager(args)
    engine = manager.get_droid(args.name)
    report = ingest_directory(engine, args.directory)
    for line in describe_report(report):
        print(line)
    engine.save_profile(str(Path(manager.base_dir) / args.name)) if args.save else None
    return 0


def cmd_doctor(args) -> int:
    from recamem import identity
    from recamem.anchor import AnchorUnavailableError, OnnxMiniLM, resolve_anchor_mode
    from recamem.paths import configs_dir

    items: List[Dict[str, Any]] = []
    mode = "mock" if args.mock else resolve_anchor_mode()
    anchor_item: Dict[str, Any] = {"item": "anchor", "mode": mode}
    if mode == "mock":
        anchor_item.update(
            ready=False,
            detail="mock embeddings selected explicitly; no real anchor is loaded",
        )
    else:
        started = time.perf_counter()
        try:
            anchor = OnnxMiniLM()
            load_seconds = time.perf_counter() - started
            anchor_item.update(
                ready=True,
                detail=f"model={anchor.model_path}",
                load_seconds=round(load_seconds, 4),
            )
        except AnchorUnavailableError as exc:
            anchor_item.update(ready=False, detail=str(exc))
    items.append(anchor_item)

    home = identity.agents_dir()
    try:
        home.mkdir(parents=True, exist_ok=True)
        probe = home / ".write-probe"
        probe.write_text("ok")
        probe.unlink()
        storage_ready, storage_detail = True, str(home)
    except OSError as exc:
        storage_ready, storage_detail = False, f"{home} not writable: {exc}"
    items.append({"item": "storage", "ready": storage_ready, "detail": storage_detail})

    manifest = configs_dir() / "recamem_selected_core.json"
    items.append(
        {
            "item": "selected_core",
            "ready": manifest.exists(),
            "detail": str(manifest) if manifest.exists() else f"missing {manifest}",
        }
    )
    items.append(
        {
            "item": "generation_backend",
            "ready": True,
            "detail": "none configured (prompt/recall paths work offline)",
        }
    )
    items.append({"item": "network", "ready": True, "detail": "offline by default; no request made"})

    # Memory stats: profiles under the home and the facts they hold. Read-only,
    # never opens a writer, never fabricates a number.
    profiles = sorted(d for d in home.iterdir() if d.is_dir() and not d.name.startswith(".")) if home.is_dir() else []
    facts = 0
    unreadable = []
    import sqlite3

    for profile in profiles:
        for candidate in ("knowledge.db", "facts.db", "episodic.db"):
            db = profile / candidate
            if not db.is_file():
                continue
            try:
                with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as conn:
                    facts += int(conn.execute("SELECT COUNT(*) FROM facts").fetchone()[0])
            except sqlite3.Error:
                unreadable.append(str(db))
            break
    items.append(
        {
            "item": "memory",
            "ready": True,
            "detail": f"profiles={len(profiles)} facts={facts}"
            + (f" unreadable={len(unreadable)}" if unreadable else ""),
        }
    )

    print(f"anchor={mode if mode == 'mock' else ('real' if anchor_item['ready'] else 'unavailable')}")
    print(f"anchor_detail={anchor_item['detail']}")
    print(f"storage={storage_detail}")
    print(f"profiles={home}")
    print(f"selected_core={manifest}")
    print(f"profiles_count={len(profiles)}")
    print(f"facts={facts}")
    print("generation_backend=none")
    print("offline=yes")
    ready = all(item["ready"] for item in items[1:]) and bool(anchor_item["ready"])
    for item in items:
        print(f"item={item['item']} ready={'yes' if item['ready'] else 'no'} detail={item['detail']}")
    return 0 if ready else 1


def cmd_rtu(args) -> int:
    from recamem.byte_rtu import ByteRtuBackend
    from src.model.rtu import MissingCheckpointError

    try:
        backend = ByteRtuBackend(
            checkpoint=args.checkpoint, dim=args.dim, layers=args.layers
        )
    except MissingCheckpointError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    params = sum(p.numel() for p in backend.model.parameters())
    print(f"checkpoint={args.checkpoint}")
    print(f"params={params}")
    print(f"dim={args.dim}")
    print(f"layers={args.layers}")
    if args.prompt:
        text = backend.generate(args.prompt, max_tokens=args.max_tokens)
        print(f"prompt={args.prompt}")
        print(f"generated={text}")
    return 0


def cmd_run(args) -> int:
    from recamem import runner

    return runner.main(
        [
            *(["--prompt", args.prompt] if args.prompt is not None else []),
            "--name", args.name,
            *(["--base-dir", args.base_dir] if args.base_dir else []),
            "--mode", args.mode,
            "--top-k", str(args.top_k),
        ]
    )


def cmd_serve(args) -> int:
    if args.mcp:
        from src.mcp_server import MCPServer

        MCPServer(base_dir=args.base_dir).run_stdio()
        return 0
    if not (args.rest or args.web):
        print("Must specify --mcp, --rest or --web", file=sys.stderr)
        return 1
    from recamem import api

    try:
        return api.serve(
            host=args.host,
            port=args.port or 8000,
            base_dir=args.base_dir,
            profile=args.name,
            static=args.web,
        )
    except api.ApiError as exc:
        print(f"error: {exc.message}", file=sys.stderr)
        return 1


def cmd_ui(args) -> int:
    """Launch the Tauri desktop shell, or fall back to the browser."""
    from recamem import api
    from recamem.paths import repo_root

    binary = repo_root() / "src-tauri" / "target" / "release" / (
        "recamem.exe" if os.name == "nt" else "recamem"
    )
    dist = api.default_static_dir()
    shell = "tauri" if binary.is_file() else "browser"

    if args.dry_run:
        print(f"shell={shell}")
        print(f"dist={dist} exists={'yes' if dist.is_dir() else 'no'}")
        print(f"binary={binary} exists={'yes' if binary.is_file() else 'no'}")
        print(f"url=http://127.0.0.1:{args.port or 8000}/")
        return 0

    if shell == "tauri":
        import subprocess

        return subprocess.call([str(binary)])

    if not dist.is_dir():
        print(
            f"error: the desktop shell is not built and {dist} is missing;\n"
            "       run `npm --prefix web install && npm --prefix web run build`\n"
            "       (or install the Rust toolchain and build src-tauri) and retry.",
            file=sys.stderr,
        )
        return 1
    if args.no_browser:
        return api.serve(
            host="127.0.0.1", port=args.port or 8000, profile=args.name, static=True
        )
    import threading
    import webbrowser

    server = api.create_server(
        host="127.0.0.1",
        port=args.port or 8000,
        profile=args.name,
        static_dir=str(dist),
    )
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    print(f"recamem ui: browser fallback at {url}")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


# --------------------------------------------------------------------------- #
# Plumbing
# --------------------------------------------------------------------------- #
def _manager(args):
    from recamem import identity
    from recamem.manager import DroidManager

    base = getattr(args, "base_dir", None) or identity.default_base_dir()
    return DroidManager(base_dir=str(base))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recamem", description=USAGE,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="command", required=True)

    p_ingest = subparsers.add_parser("ingest", help="Ingest a folder of Markdown, text and PDF files")
    p_ingest.add_argument("directory")
    p_ingest.add_argument("--name", default="droid-alpha")
    p_ingest.add_argument("--base-dir", default=None)
    p_ingest.add_argument("--save", action="store_true", help="Save the profile after ingest")
    p_ingest.set_defaults(func=cmd_ingest)

    p_recall = subparsers.add_parser("recall", help="Two-channel recall with source attribution")
    p_recall.add_argument("query")
    p_recall.add_argument("--name", default="droid-alpha")
    p_recall.add_argument("--base-dir", default=None)
    p_recall.add_argument("--top-k", type=int, default=3)
    p_recall.set_defaults(func=cmd_recall)

    p_doctor = subparsers.add_parser("doctor", help="Readiness report; exit 0 only when ready")
    p_doctor.add_argument("--mock", action="store_true", help="Report the explicit mock anchor")
    p_doctor.set_defaults(func=cmd_doctor)

    p_migrate = subparsers.add_parser("migrate", help="Copy legacy droids/ profiles into the recamem home")
    p_migrate.add_argument("source", nargs="?", default=None)
    p_migrate.add_argument("--dest", default=None)
    p_migrate.set_defaults(func=cmd_migrate)

    p_rtu = subparsers.add_parser("rtu", help="Inspect or run a byte-RTU checkpoint")
    p_rtu.add_argument("--checkpoint", required=True)
    p_rtu.add_argument("--prompt", default=None)
    p_rtu.add_argument("--max-tokens", type=int, default=32)
    p_rtu.add_argument("--dim", type=int, default=64)
    p_rtu.add_argument("--layers", type=int, default=2)
    p_rtu.set_defaults(func=cmd_rtu)

    p_run = subparsers.add_parser("run", help="Headless runner: one JSON object per prompt")
    p_run.add_argument("--prompt", default=None, help="run a single prompt and exit")
    p_run.add_argument("--name", default="droid-alpha")
    p_run.add_argument("--base-dir", default=None)
    p_run.add_argument("--mode", default="chat", choices=["chat", "recall", "teach"])
    p_run.add_argument("--top-k", type=int, default=3)
    p_run.set_defaults(func=cmd_run)

    p_serve = subparsers.add_parser("serve", help="MCP stdio server, REST API or web app")
    p_serve.add_argument("--mcp", action="store_true", help="serve MCP over stdio")
    p_serve.add_argument("--rest", action="store_true", help="serve the versioned loopback API")
    p_serve.add_argument("--web", action="store_true", help="serve the API and web/dist")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.add_argument("--name", default="droid-alpha")
    p_serve.add_argument("--base-dir", default=None)
    p_serve.set_defaults(func=cmd_serve)

    p_ui = subparsers.add_parser("ui", help="Launch the desktop app (browser fallback)")
    p_ui.add_argument("--port", type=int, default=8000)
    p_ui.add_argument("--name", default="droid-alpha")
    p_ui.add_argument("--no-browser", action="store_true")
    p_ui.add_argument("--dry-run", action="store_true", help="print the resolved shell and exit")
    p_ui.set_defaults(func=cmd_ui)
    return parser


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help"}:
        print(USAGE)
        return 0
    command = argv[0]
    if command in LEGACY_COMMANDS:
        from src import cli as legacy_cli

        return int(legacy_cli.main(argv, _deprecation_notice=False) or 0)
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


def entry() -> None:
    sys.exit(main())


def legacy_entry() -> None:
    print(
        "deprecated: the tmt-droid command is deprecated; use `recamem` instead",
        file=sys.stderr,
    )
    entry()


if __name__ == "__main__":  # pragma: no cover
    entry()
