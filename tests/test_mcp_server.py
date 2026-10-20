"""MCP tool/spec parity (product-surface-completion 5.3/5.4).

The documented tool list lives in the main spec and, while the change is still
active, in that change's delta. Reading both locations keeps the test correct
before and after archive: the delta gains the sixth tool now, the main spec
gains it at archive, and the union is unchanged.
"""

import re
from pathlib import Path

from src.mcp_server import MCPServer

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_SPEC = REPO_ROOT / "openspec" / "specs" / "mcp-server" / "spec.md"
CHANGES_DIR = REPO_ROOT / "openspec" / "changes"

# "- `recall_memory(query, top_k)`: ..." -> ("recall_memory", "query, top_k")
TOOL_LINE = re.compile(r"^\s*-\s+`([a-z_][a-z0-9_]*)\(([^)]*)\)`")


def _documented_tools(path: Path) -> dict:
    tools = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = TOOL_LINE.match(line)
        if match:
            tools[match.group(1)] = match.group(2)
    return tools


def _spec_files() -> list:
    files = [MAIN_SPEC]
    if CHANGES_DIR.exists():
        for change_dir in sorted(CHANGES_DIR.iterdir()):
            if change_dir.name == "archive" or not change_dir.is_dir():
                continue
            candidate = change_dir / "specs" / "mcp-server" / "spec.md"
            if candidate.exists():
                files.append(candidate)
    return files


def documented_tool_names() -> dict:
    names = {}
    for path in _spec_files():
        for name, args in _documented_tools(path).items():
            names.setdefault(name, (path, args))
    return names


def exposed_tool_names() -> dict:
    server = MCPServer(base_dir="droids")
    tools = server.handle_tools_list({})["tools"]
    return {tool["name"]: tool for tool in tools}


def test_every_exposed_tool_is_documented():
    documented = documented_tool_names()
    exposed = exposed_tool_names()
    missing = sorted(set(exposed) - set(documented))
    assert not missing, (
        f"tools exposed by src/mcp_server.py but absent from the spec: {missing}. "
        f"Spec files checked: {[str(p) for p in _spec_files()]}"
    )


def test_every_documented_tool_is_exposed():
    documented = documented_tool_names()
    exposed = exposed_tool_names()
    extra = sorted(set(documented) - set(exposed))
    assert not extra, (
        f"tools documented but not exposed by src/mcp_server.py: {extra}. "
        f"Spec files checked: {[str(p) for p in _spec_files()]}"
    )


def test_generate_with_memory_is_documented_and_exposed():
    exposed = exposed_tool_names()
    assert "generate_with_memory" in exposed
    properties = exposed["generate_with_memory"]["inputSchema"]["properties"]
    assert {"query", "strategy", "droid_name"} <= set(properties)

    documented = documented_tool_names()
    assert "generate_with_memory" in documented
    spec_path, args = documented["generate_with_memory"]
    for argument in ("query", "strategy", "droid_name"):
        assert argument in args, f"{argument} missing from {spec_path}"
