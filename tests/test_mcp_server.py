"""Contract tests for src/mcp_server.py.

The stdio loop is exercised by injecting ``sys.stdin``/``sys.stdout``; no thread,
socket, or real profile is used. What is pinned:

- ``initialize`` advertises the protocol version, tool capability and server info;
- ``tools/list`` returns the documented tool-name set with object schemas;
- ``tools/call`` works for every tool against a temp profile and reports an
  unknown tool or a raised exception in an ``isError`` envelope;
- the stdio loop emits one JSON-RPC 2.0 line per request, with id echo, -32601 for
  an unknown method, and -32700 for unparsable input (logs never leak to stdout);
- the exposed tool set stays in parity with the tools documented in the OpenSpec
  specs (main spec plus any active change delta).
"""

import io
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.mcp_server import MCPServer

EXPECTED_TOOLS = {
    "recall_memory",
    "teach_fact",
    "decide_action",
    "list_droids",
    "inspect_state",
    "generate_with_memory",
}


class FakeAnchor:
    """Mirrors the production all-MiniLM-L6-v2 embedding width."""

    dim = 384

    def embed(self, text):
        return torch.randn(self.dim, generator=torch.Generator().manual_seed(len(str(text))))


class FakeDroid:
    def __init__(self, name="droid-alpha"):
        self.name = name
        self.dim = 384
        self.step_count = 5
        self.anchor = FakeAnchor()
        self.knowledge = SimpleNamespace(active_count=lambda: 4)
        self.memory = SimpleNamespace(states=torch.tensor([3.0, 4.0]))
        self.logs = [{"event": "teach"}, {"event": "recall"}]
        self.teach_calls = []

    def recall(self, query, top_k=3):
        return [
            {"text": "Radar uses radio waves.", "similarity": 0.9},
            {"text": "Lidar uses lasers.", "similarity": 0.6},
        ][:top_k]

    def teach(self, text, source=None):
        self.teach_calls.append((text, source))
        return {"propositions": 2, "memory_norm": 1.5, "total_knowledge": 4}

    def chat(self, query):
        return f"chat answer for {query}"

    def generate_with_memory(self, query, strategy="auto"):
        return {"text": f"slow answer for {query}", "strategy": strategy}

    def get_memory_health(self):
        return {"norm": 5.0}


class FakeManager:
    def __init__(self, base_dir="droids"):
        self.droid = FakeDroid()
        self.saved = []

    def get_droid(self, name):
        self.droid.name = name
        return self.droid

    def save_droid(self, name):
        self.saved.append(name)
        return True


@pytest.fixture
def server(tmp_path):
    srv = MCPServer(base_dir=str(tmp_path / "droids"))
    srv._manager = FakeManager(base_dir=str(tmp_path / "droids"))
    return srv


def _texts(result):
    return [item["text"] for item in result["content"]]


# --------------------------------------------------------------------------- #
# initialize / tools/list
# --------------------------------------------------------------------------- #
def test_handle_initialize(server):
    result = server.handle_initialize({})
    assert result["protocolVersion"] == "2024-11-05"
    assert result["capabilities"] == {"tools": {}}
    assert result["serverInfo"] == {"name": "tmt-droid-mcp", "version": "1.0.0"}


def test_handle_tools_list_name_set_and_schemas(server):
    tools = server.handle_tools_list({})["tools"]
    names = {tool["name"] for tool in tools}
    assert names == EXPECTED_TOOLS
    assert len(tools) == len(EXPECTED_TOOLS)

    for tool in tools:
        assert tool["description"]
        assert tool["inputSchema"]["type"] == "object"
        assert isinstance(tool["inputSchema"]["properties"], dict)

    required = {tool["name"]: tool["inputSchema"].get("required", []) for tool in tools}
    assert required["recall_memory"] == ["query"]
    assert required["teach_fact"] == ["text"]
    assert required["decide_action"] == ["query"]
    assert required["generate_with_memory"] == ["query"]
    assert required["list_droids"] == []


# --------------------------------------------------------------------------- #
# tools/call
# --------------------------------------------------------------------------- #
def test_call_recall_memory(server):
    result = server.handle_tools_call(
        {"name": "recall_memory", "arguments": {"query": "radar", "top_k": 1}}
    )
    assert result.get("isError") is None
    texts = _texts(result)
    assert "Recall for 'radar' (1 hits):" in texts[0]
    assert "[0.900] Radar uses radio waves." in texts[0]
    hits = json.loads(texts[1])
    assert len(hits) == 1


def test_call_teach_fact_saves_profile(server):
    result = server.handle_tools_call(
        {"name": "teach_fact", "arguments": {"text": "Radar uses radio waves.", "domain": "physics"}}
    )
    assert result.get("isError") is None
    assert "Taught 2 facts. Memory norm: 1.50. Total: 4" in _texts(result)[0]
    assert server._manager.saved == ["droid-alpha"]
    assert server._manager.droid.teach_calls == [("Radar uses radio waves.", "physics")]


def test_call_decide_action_returns_json(server):
    result = server.handle_tools_call(
        {"name": "decide_action", "arguments": {"query": "pick a tool", "options": ["radar", "lidar"]}}
    )
    payload = json.loads(_texts(result)[0])
    assert set(payload) >= {"action", "probabilities", "confidence", "prototype_names"}
    assert payload["prototype_names"] == ["radar", "lidar"]


def test_call_decide_action_with_non_384_anchor_embeddings(server):
    class NarrowAnchor:
        def embed(self, text):
            return torch.randn(16, generator=torch.Generator().manual_seed(len(str(text))))

    server._manager.droid.anchor = NarrowAnchor()
    result = server.handle_tools_call(
        {"name": "decide_action", "arguments": {"query": "pick a tool", "options": ["radar", "lidar"]}}
    )
    assert result.get("isError") is None


def test_call_decide_action_without_anchor_uses_generic_head(server):
    original = server._manager
    try:
        server._manager = SimpleNamespace(get_droid=lambda name: SimpleNamespace())
        result = server.handle_tools_call({"name": "decide_action", "arguments": {"query": "pick"}})
    finally:
        server._manager = original

    payload = json.loads(_texts(result)[0])
    assert payload["action"] in payload["prototype_names"] + ["ambiguous"]


def test_call_list_droids(server):
    base = server._manager  # ensure the fixture manager is used
    assert base is not None
    from pathlib import Path

    root = Path(server.base_dir)
    (root / "alpha").mkdir(parents=True)
    (root / "beta").mkdir()
    (root / "notes.txt").write_text("not a droid")

    result = server.handle_tools_call({"name": "list_droids", "arguments": {}})
    assert sorted(json.loads(_texts(result)[0])) == ["alpha", "beta"]


def test_call_list_droids_without_directory(tmp_path):
    srv = MCPServer(base_dir=str(tmp_path / "missing"))
    srv._manager = FakeManager()
    result = srv.handle_tools_call({"name": "list_droids", "arguments": {}})
    assert "No droids directory found" in _texts(result)[0]


def test_call_inspect_state(server):
    result = server.handle_tools_call(
        {"name": "inspect_state", "arguments": {"droid_name": "droid-beta"}}
    )
    state = json.loads(_texts(result)[0])
    assert state["name"] == "droid-beta"
    assert state["step_count"] == 5
    assert state["total_facts"] == 4
    assert state["memory_norm"] == pytest.approx(5.0)
    assert state["logs_count"] == 2
    assert state["health"] == {"norm": 5.0}


def test_call_generate_with_memory(server):
    result = server.handle_tools_call(
        {"name": "generate_with_memory", "arguments": {"query": "explain radar", "strategy": "prompt"}}
    )
    payload = json.loads(_texts(result)[0])
    assert payload == {"text": "slow answer for explain radar", "strategy": "prompt"}


def test_call_generate_with_memory_falls_back_to_chat(server):
    server._manager = SimpleNamespace(
        get_droid=lambda name: SimpleNamespace(chat=lambda q: f"fallback {q}"),
        save_droid=lambda name: None,
    )
    result = server.handle_tools_call(
        {"name": "generate_with_memory", "arguments": {"query": "radar"}}
    )
    assert _texts(result)[0] == "fallback radar"


def test_call_unknown_tool_is_an_error(server):
    result = server.handle_tools_call({"name": "does_not_exist", "arguments": {}})
    assert result["isError"] is True
    assert "Unknown tool: does_not_exist" in _texts(result)[0]


def test_call_wraps_tool_exceptions(server):
    def boom(name):
        raise RuntimeError("profile corrupt")

    server._manager = SimpleNamespace(get_droid=boom, save_droid=lambda name: None)
    result = server.handle_tools_call({"name": "recall_memory", "arguments": {"query": "radar"}})
    assert result["isError"] is True
    # `_tool_recall_memory` catches its own failures with a tool-specific message.
    assert _texts(result)[0] == "Recall failed: profile corrupt"


def test_tools_report_missing_manager(monkeypatch, tmp_path):
    srv = MCPServer(base_dir=str(tmp_path))
    monkeypatch.setattr(srv, "_get_manager", lambda: None)
    for tool in ("recall_memory", "teach_fact", "list_droids", "inspect_state", "generate_with_memory"):
        result = srv.handle_tools_call({"name": tool, "arguments": {}})
        assert _texts(result)[0] == "DroidManager not available"


# --------------------------------------------------------------------------- #
# JSON-RPC framing
# --------------------------------------------------------------------------- #
def _run_stdio(monkeypatch, server, lines):
    stdin = io.StringIO("".join(line + "\n" for line in lines))
    stdout = io.StringIO()
    monkeypatch.setattr("sys.stdin", stdin)
    monkeypatch.setattr("sys.stdout", stdout)
    server.run_stdio()
    payloads = [json.loads(line) for line in stdout.getvalue().splitlines() if line.strip()]
    return payloads


def test_stdio_loop_emits_json_rpc_envelopes(monkeypatch, server):
    payloads = _run_stdio(
        monkeypatch,
        server,
        [
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}),
            json.dumps({"jsonrpc": "2.0", "id": 3, "method": "tools/call",
                        "params": {"name": "recall_memory", "arguments": {"query": "radar"}}}),
        ],
    )

    assert len(payloads) == 3
    assert [p["id"] for p in payloads] == [1, 2, 3]
    assert all(p["jsonrpc"] == "2.0" for p in payloads)
    assert payloads[0]["result"]["protocolVersion"] == "2024-11-05"
    assert {tool["name"] for tool in payloads[1]["result"]["tools"]} == EXPECTED_TOOLS
    assert "Recall for 'radar'" in payloads[2]["result"]["content"][0]["text"]


def test_stdio_loop_unknown_method_returns_minus_32601(monkeypatch, server):
    payloads = _run_stdio(
        monkeypatch,
        server,
        [json.dumps({"jsonrpc": "2.0", "id": 9, "method": "resources/list", "params": {}})],
    )
    assert payloads == [
        {
            "jsonrpc": "2.0",
            "id": 9,
            "error": {"code": -32601, "message": "Method not found: resources/list"},
        }
    ]


def test_stdio_loop_parse_error_returns_minus_32700(monkeypatch, server):
    payloads = _run_stdio(monkeypatch, server, ["{not json", ""])
    assert len(payloads) == 1
    assert payloads[0]["id"] is None
    assert payloads[0]["error"]["code"] == -32700
    assert payloads[0]["error"]["message"].startswith("Parse error")


def test_stdio_loop_skips_blank_lines(monkeypatch, server):
    payloads = _run_stdio(
        monkeypatch,
        server,
        [
            "   ",
            "",
            json.dumps({"jsonrpc": "2.0", "id": 5, "method": "initialize", "params": {}}),
        ],
    )
    assert [p["id"] for p in payloads] == [5]
# --------------------------------------------------------------------------- #
# Tool/spec parity (product-surface-completion 5.3/5.4)
#
# The documented tool list lives in the main spec and, while a change is still
# active, in that change's delta. Reading both locations keeps these tests
# correct before and after archive.
# --------------------------------------------------------------------------- #

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
