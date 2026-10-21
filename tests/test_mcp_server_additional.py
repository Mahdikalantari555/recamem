"""MCP tool/spec parity (product-surface-completion 5.3/5.4).

The documented tool list lives in the main spec and, while the change is still
active, in that change's delta. Reading both locations keeps the test correct
before and after archive: the delta gains the sixth tool now, the main spec
gains it at archive, and the union is unchanged.

Scope rule: a delta joins the *enforced* contract only once every task in its
change is checked, because that is when its code has landed. A change that is
still a plan may document a tool that does not exist yet, and a plan must not
fail the build. The "exposed" direction still reads every delta, and the
"documented" direction is strict again the moment the change's tasks are done.
"""

import json
import re
from pathlib import Path

from src.mcp_server import MCPServer

REPO_ROOT = Path(__file__).resolve().parents[1]
MAIN_SPEC = REPO_ROOT / "openspec" / "specs" / "mcp-server" / "spec.md"
CHANGES_DIR = REPO_ROOT / "openspec" / "changes"

# "- `recall_memory(query, top_k)`: ..." -> ("recall_memory", "query, top_k")
TOOL_LINE = re.compile(r"^\s*-\s+`([a-z_][a-z0-9_]*)\(([^)]*)\)`")
# "- [ ] 1.1 ..." / "- [x] 1.1 ..." -> the checkbox marker
TASK_LINE = re.compile(r"^\s*-\s*\[([ xX])\]")


def _documented_tools(path: Path) -> dict:
    tools = {}
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        match = TOOL_LINE.match(line)
        if match:
            tools[match.group(1)] = match.group(2)
    return tools


def _change_is_implemented(change_dir: Path) -> bool:
    """True when every task in the change is checked, i.e. its code has landed."""
    tasks = change_dir / "tasks.md"
    if not tasks.exists():
        return False
    markers = [m for m in (TASK_LINE.match(line) for line in
                            tasks.read_text(encoding="utf-8").splitlines()) if m]
    if not markers:
        return False
    return all(m.group(1) in ("x", "X") for m in markers)


def _change_spec_files() -> list:
    """(change_dir, delta_spec) for every active change that has an MCP delta."""
    files = []
    if CHANGES_DIR.exists():
        for change_dir in sorted(CHANGES_DIR.iterdir()):
            if change_dir.name == "archive" or not change_dir.is_dir():
                continue
            candidate = change_dir / "specs" / "mcp-server" / "spec.md"
            if candidate.exists():
                files.append((change_dir, candidate))
    return files


def _spec_files() -> list:
    """Every spec that may document a tool: the main spec plus every delta."""
    return [MAIN_SPEC] + [spec for _, spec in _change_spec_files()]


def _live_spec_files() -> list:
    """The enforced contract: the main spec plus the deltas of applied changes."""
    return [MAIN_SPEC] + [
        spec for change_dir, spec in _change_spec_files()
        if _change_is_implemented(change_dir)
    ]


def documented_tool_names() -> dict:
    names = {}
    for path in _spec_files():
        for name, args in _documented_tools(path).items():
            names.setdefault(name, (path, args))
    return names


def live_documented_tool_names() -> dict:
    names = {}
    for path in _live_spec_files():
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
    documented = live_documented_tool_names()
    exposed = exposed_tool_names()
    extra = sorted(set(documented) - set(exposed))
    assert not extra, (
        f"tools documented but not exposed by src/mcp_server.py: {extra}. "
        f"Spec files checked: {[str(p) for p in _live_spec_files()]}"
    )


def test_a_delta_is_enforced_only_once_its_change_is_applied(tmp_path):
    """The scope rule: unchecked tasks mean the delta is still a plan."""
    change = tmp_path / "planned-change"
    change.mkdir()
    (change / "tasks.md").write_text("- [ ] 1.1 not done yet\n", encoding="utf-8")
    assert not _change_is_implemented(change)
    (change / "tasks.md").write_text("- [x] 1.1 done\n", encoding="utf-8")
    assert _change_is_implemented(change)
    (change / "tasks.md").write_text("- [x] 1.1 done\n- [ ] 1.2 not done\n", encoding="utf-8")
    assert not _change_is_implemented(change)


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


# ---------------------------------------------------------------------------
# Behavioral coverage for src/mcp_server.py (surface-test-coverage 3.10).
# Exercised directly: no stdio loop, no subprocess; temp profile dirs only.
# ---------------------------------------------------------------------------


class _FakeDroid:
    def __init__(self):
        self.name = "droid-alpha"
        self.dim = 8
        self.step_count = 3

    class knowledge:
        @staticmethod
        def active_count():
            return 2

    def recall(self, query, top_k=3):
        return [
            {"id": 1, "text": "Paris is the capital of France.", "similarity": 0.91},
            {"id": 2, "text": "Lidar measures distance with laser pulses.", "similarity": 0.4},
        ][:top_k]

    def teach(self, text, source="general"):
        return {"propositions": 1, "memory_norm": 0.5, "total_knowledge": 3}

    def get_memory_health(self):
        return {"interference": 0.1}


class _FakeManager:
    def __init__(self, base_dir="droids"):
        self.base_dir = base_dir

    def get_droid(self, name):
        return _FakeDroid()

    def save_droid(self, name):
        pass


def test_handle_initialize_returns_protocol_and_capabilities():
    server = MCPServer(base_dir="droids")
    result = server.handle_initialize({})
    assert result["protocolVersion"] == "2024-11-05"
    assert "tools" in result["capabilities"]
    assert result["serverInfo"]["name"] == "tmt-droid-mcp"


def test_handle_tools_list_names_all_six_tools():
    server = MCPServer(base_dir="droids")
    tools = server.handle_tools_list({})["tools"]
    names = [tool["name"] for tool in tools]
    assert names == [
        "recall_memory",
        "teach_fact",
        "decide_action",
        "list_droids",
        "inspect_state",
        "generate_with_memory",
    ]
    for tool in tools:
        assert "inputSchema" in tool


def test_handle_tools_call_unknown_tool_returns_json_rpc_style_error_envelope():
    server = MCPServer(base_dir="droids")
    result = server.handle_tools_call({"name": "not_a_tool", "arguments": {}})
    assert result["isError"] is True
    assert "Unknown tool: not_a_tool" in result["content"][0]["text"]


def test_handle_tools_call_list_droids_with_empty_base(tmp_path):
    server = MCPServer(base_dir=str(tmp_path / "droids"))
    result = server._tool_list_droids({})
    assert result.get("isError") is not True
    assert "[]" in result["content"][0]["text"]


def test_handle_tools_call_recall_memory(tmp_path):
    server = MCPServer(base_dir=str(tmp_path / "droids"))
    server._manager = _FakeManager()
    result = server._tool_recall_memory({"query": "paris", "top_k": 1})
    text = result["content"][0]["text"]
    assert "Recall for 'paris'" in text
    hits = json.loads(result["content"][1]["text"])
    assert len(hits) == 1


def test_handle_tools_call_teach_fact(tmp_path):
    server = MCPServer(base_dir=str(tmp_path / "droids"))
    server._manager = _FakeManager()
    result = server._tool_teach_fact({"text": "The sky is blue.", "domain": "general"})
    assert result.get("isError") is not True
    assert "Taught 1 facts" in result["content"][0]["text"]


def test_handle_tools_call_decide_action(tmp_path):
    server = MCPServer(base_dir=str(tmp_path / "droids"))
    server._manager = _FakeManager()
    result = server._tool_decide_action({"query": "x", "options": ["a", "b"]})
    assert result.get("isError") is not True
    text = result["content"][0]["text"]
    payload = json.loads(text)
    assert "action" in payload and "confidence" in payload


def test_handle_tools_call_inspect_state(tmp_path):
    server = MCPServer(base_dir=str(tmp_path / "droids"))
    server._manager = _FakeManager()
    result = server._tool_inspect_state({})
    assert result.get("isError") is not True
    text = result["content"][0]["text"]
    state = json.loads(text)
    assert state["name"] == "droid-alpha"
    assert state["step_count"] == 3
