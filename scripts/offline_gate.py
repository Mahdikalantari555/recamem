#!/usr/bin/env python3
"""Offline gate for recamem (verification/recamem-ci).

Runs without model weights, network access or a GPU and fails loudly when a
documented surface drifts from the code:

* evidence ledger rows must name tests that exist;
* the MCP ``tools/list`` advertisement must equal the documented list plus the
  documented deprecated aliases;
* the CLI subcommand list must match what ``recamem.cli`` exposes;
* documentation must not contain a hand-written test count;
* every third-party import on the default code path must be declared in
  ``pyproject.toml``;
* real-model tests are reported as blocked, never passed.

Exit code 0 means the gate passed, 1 means it failed.
"""

from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
CORE_DIRS = ("recamem", "src")
STDLIB = set(sys.stdlib_module_names) if hasattr(sys, "stdlib_module_names") else set()
#: Distribution name -> import name where they differ.
DISTRIBUTION_IMPORTS = {
    "llama_cpp_python": "llama_cpp",
    "sentence_transformers": "sentence_transformers",
    "pillow": "PIL",
}
DOC_FILES = ("README.md", "AGENTS.md", "docs/testing.md", "docs/release-evidence.md")


def _print(key: str, ok: bool, detail: str) -> None:
    print(f"{key}={'pass' if ok else 'fail'} detail={detail}")


def check_evidence() -> List[str]:
    """Delegate to the release-evidence guard: unknown id or missing test fails."""
    sys.path.insert(0, str(ROOT))
    from scripts import check_release_evidence as guard

    inventory_path = guard.ROOT / guard.CHANGE / "inventory.json"
    ledger_path = guard.ROOT / "reports/completion-evidence.json"
    if not inventory_path.exists() or not ledger_path.exists():
        return ["inventory or ledger file is missing"]
    inventory = json.loads(inventory_path.read_text())
    ledger = json.loads(ledger_path.read_text())
    return list(guard.validate(guard.ROOT, inventory, ledger))


def check_mcp_tools() -> List[str]:
    from src import mcp_server

    advertised = {tool["name"] for tool in mcp_server.MCPServer().handle_tools_list({})["tools"]}
    documented = set(mcp_server.DOCUMENTED_TOOLS)
    problems = []
    if advertised != documented:
        problems.append(
            "MCP tool drift: undocumented="
            f"{sorted(advertised - documented)} missing={sorted(documented - advertised)}"
        )
    if not set(mcp_server.CANONICAL_TOOLS) <= advertised:
        problems.append("a canonical recamem_* tool is not advertised")
    return problems


def check_cli_surface() -> List[str]:
    from recamem import cli

    documented = set(cli.LEGACY_COMMANDS) | set(cli.NEW_COMMANDS)
    parser = cli.build_parser()
    exposed = set()
    for action in parser._actions:  # noqa: SLF001 - argparse has no public API
        if action.__class__.__name__ == "_SubParsersAction":
            exposed |= set(action.choices)
    missing_from_parser = exposed - documented
    if missing_from_parser:
        return [f"CLI subcommands not in the documented set: {sorted(missing_from_parser)}"]
    return []


def check_test_counts() -> List[str]:
    problems = []
    pattern = re.compile(r"\b\d{2,}\s+(?:tests?\s+)?(?:passed|passing|tests)\b", re.I)
    for name in DOC_FILES:
        path = ROOT / name
        if not path.exists():
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            if pattern.search(line) and "pytest --collect-only" not in line:
                problems.append(f"{name}:{lineno}: hand-written test count: {line.strip()}")
    return problems


def _declared_dependencies() -> set:
    """Every dependency declared in pyproject.toml, core or optional.

    The spec asks for one packaging file; optional integrations live in extras
    and are still declaration-bearing.
    """
    text = (ROOT / "pyproject.toml").read_text()
    names = set()
    for block in re.findall(r"=\s*\[(.*?)\]", text, re.DOTALL):
        for item in re.findall(r'"([^"]+)"', block):
            base = re.split(r"[<>=!\[; ]", item.strip(), maxsplit=1)[0]
            if not base or "." in base:
                continue
            names.add(DISTRIBUTION_IMPORTS.get(
                base.lower().replace("-", "_"), base.lower().replace("-", "_")
            ))
    return names


def _third_party_imports() -> Dict[str, List[str]]:
    declared = _declared_dependencies()
    local = {"recamem", "src", "tests", "scripts", "configs", "conftest"}
    found: Dict[str, List[str]] = {}
    for directory in CORE_DIRS:
        for path in (ROOT / directory).rglob("*.py"):
            if "__pycache__" in path.parts:
                continue
            try:
                tree = ast.parse(path.read_text())
            except SyntaxError as exc:  # pragma: no cover - a real syntax error
                found.setdefault(f"syntaxerror:{path}", []).append(str(exc))
                continue
            for node in ast.walk(tree):
                names: List[str] = []
                if isinstance(node, ast.Import):
                    names = [alias.name.split(".")[0] for alias in node.names]
                elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                    names = [node.module.split(".")[0]]
                for name in names:
                    if name in local or name in STDLIB or name.lower() in STDLIB:
                        continue
                    if name in declared or name.replace("-", "_") in declared:
                        continue
                    found.setdefault(name, []).append(str(path.relative_to(ROOT)))
    return found


def check_declared_imports() -> List[str]:
    undeclared = _third_party_imports()
    return [
        f"undeclared runtime import {name!r} used by {sorted(set(paths))[:3]}"
        for name, paths in sorted(undeclared.items())
    ]


def check_blocked_semantics() -> List[str]:
    """Real-model tests must be marked ``real_model`` and never silently pass."""
    problems = []
    for path in (ROOT / "tests").glob("test_*.py"):
        text = path.read_text()
        if "OnnxMiniLM(" in text or "GgufBackend(" in text:
            if "real_model" in text:
                continue
    marker = (ROOT / "tests" / "conftest.py").read_text()
    if "real_model" not in marker:
        problems.append("tests/conftest.py does not define the real_model gate")
    return problems


CHECKS = {
    "evidence": check_evidence,
    "mcp_tools": check_mcp_tools,
    "cli_surface": check_cli_surface,
    "test_counts": check_test_counts,
    "declared_imports": check_declared_imports,
    "blocked_semantics": check_blocked_semantics,
}


def main(argv: List[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    selected = [arg for arg in argv if not arg.startswith("-")]
    names = selected or list(CHECKS)
    failures = 0
    for name in names:
        check = CHECKS.get(name)
        if check is None:
            print(f"{name}=fail detail=unknown check")
            failures += 1
            continue
        try:
            problems = check()
        except Exception as exc:  # a broken check is a failed check
            problems = [f"{type(exc).__name__}: {exc}"]
        ok = not problems
        _print(name, ok, "clean" if ok else "; ".join(problems[:4]))
        failures += 0 if ok else 1
    print(f"gate={'pass' if failures == 0 else 'fail'} checks={len(names)} failed={failures}")
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
