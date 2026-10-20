"""Guard test: every shipped module under `src/` has a test file (surface-test-coverage 3.14).

`MODULE_TESTS` maps each `src/` module to the test file that covers it. A module
absent from the mapping fails this test with the module path and the expected
test filename; a mapped test file that is missing or has no `def test_` also
fails. The unit is "has a contract test", not line percentage.
"""

from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
TESTS_DIR = REPO_ROOT / "tests"

MODULE_TESTS = {
    # Modules covered by this change (surface-test-coverage).
    "src/model/confidence.py": "tests/test_confidence.py",
    "src/model/decision_head.py": "tests/test_decision_head.py",
    "src/model/memory_lifecycle.py": "tests/test_memory_lifecycle.py",
    "src/model/memory_injection.py": "tests/test_memory_injection.py",
    "src/model/propositions.py": "tests/test_propositions.py",
    "src/model/droid_package.py": "tests/test_droid_package.py",
    "src/model/gguf_backend.py": "tests/test_gguf_backend.py",
    "src/model/plastic_adapter.py": "tests/test_plastic_adapter.py",
    "src/cli.py": "tests/test_cli.py",
    "src/mcp_server.py": "tests/test_mcp_server.py",
    # Modules covered by pre-existing test files.
    "src/model/rtu.py": "tests/test_rtu.py",
    "src/model/droid.py": "tests/test_droid_engine.py",
    "src/model/droid_manager.py": "tests/test_droid_profiles.py",
    "src/model/knowledge_store.py": "tests/test_knowledge_store.py",
    "src/model/onnx_anchor.py": "tests/test_onnx_anchor.py",
    "src/model/teacher.py": "tests/test_teacher.py",
    "src/data.py": "tests/test_data.py",
    "src/model/byte_rtu_backend.py": "tests/test_droid_rtu_fusion.py",
}


def _module_paths():
    modules = []
    for path in sorted(SRC_DIR.rglob("*.py")):
        if path.name == "__init__.py":
            continue
        modules.append("src/" + path.relative_to(SRC_DIR).as_posix())
    return modules


def _test_has_test_functions(test_path: Path) -> bool:
    if not test_path.exists():
        return False
    return any(
        line.startswith("def test_") or line.startswith("async def test_")
        for line in test_path.read_text(encoding="utf-8").splitlines()
    )


def test_every_module_is_mapped_to_a_test_file():
    untested = []
    for module in _module_paths():
        if module not in MODULE_TESTS:
            expected = "tests/" + module.removeprefix("src/").replace(".py", "_test.py")
            untested.append(f"  {module} -> expected {expected}")

    assert not untested, (
        "Untested modules (add a test file or an entry to MODULE_TESTS):\n"
        + "\n".join(untested)
    )


def test_every_mapped_test_file_exists_and_has_tests():
    missing_or_empty = []
    for module, test_file in MODULE_TESTS.items():
        test_path = REPO_ROOT / test_file
        if not _test_has_test_functions(test_path):
            missing_or_empty.append(f"  {module} -> {test_file} (missing or no def test_)")

    assert not missing_or_empty, (
        "Mapped test files missing or empty:\n" + "\n".join(missing_or_empty)
    )
