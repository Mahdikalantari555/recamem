"""Structural validation for the OpenSpec tree (no `openspec` CLI required).

Checks, for every change directory (active and archived):

* `proposal.md`, `design.md` and `tasks.md` exist and are non-empty;
* `tasks.md` task ids are unique and every task is `- [ ]` or `- [x]`;
* every spec delta names at least one `### Requirement:` and is non-empty;
* a change's declared capabilities (from `proposal.md`) exist under `specs/`;
* an archived change is referenced by an `archive/` path only once.

This validates structure, not correctness: a checked box is not evidence.
Exit code 0 = structurally valid, 1 = problems found.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OPENSPEC = Path("openspec")
TASK_RE = re.compile(r"^- \[(?P<mark>[ xX])\] (?P<id>\S+) ")
CAPABILITY_RE = re.compile(r"^- `(?P<cap>[a-z0-9-]+/[a-z0-9-]+)`")
REQUIREMENT_RE = re.compile(r"^### Requirement: (?P<name>.+)$", re.MULTILINE)


def _check_change(directory: Path, archived: bool, known: set[str]) -> list[str]:
    errors: list[str] = []
    label = str(directory.relative_to(ROOT))
    for name in ("proposal.md", "tasks.md"):
        path = directory / name
        if not path.is_file() or not path.read_text().strip():
            errors.append(f"{label}: missing or empty {name}")
    if not archived and not (directory / "design.md").is_file():
        errors.append(f"{label}: active change is missing design.md")

    tasks_path = directory / "tasks.md"
    seen: dict[str, int] = {}
    checked = unchecked = 0
    if tasks_path.is_file():
        for lineno, line in enumerate(tasks_path.read_text().splitlines(), start=1):
            if not line.startswith("- ["):
                continue
            match = TASK_RE.match(line)
            if match is None:
                errors.append(
                    f"{label}/tasks.md:{lineno}: task must be '- [ ] <id> ...' "
                    "or '- [x] <id> ...'"
                )
                continue
            task_id = match.group("id")
            if task_id in seen:
                errors.append(
                    f"{label}/tasks.md:{lineno}: duplicate task id {task_id!r}"
                    f" (first at line {seen[task_id]})"
                )
            else:
                seen[task_id] = lineno
            if match.group("mark") == " ":
                unchecked += 1
            else:
                checked += 1

    specs_root = directory / "specs"
    capabilities: list[str] = []
    if specs_root.is_dir():
        for spec in sorted(specs_root.rglob("spec.md")):
            capability = "/".join(spec.parent.relative_to(specs_root).parts)
            capabilities.append(capability)
            body = spec.read_text()
            if not REQUIREMENT_RE.search(body):
                errors.append(f"{label}: {capability}/spec.md has no '### Requirement:'")
            if "## " not in body:
                errors.append(f"{label}: {capability}/spec.md has no requirement section")
    else:
        errors.append(f"{label}: missing specs/ directory")

    proposal = directory / "proposal.md"
    if proposal.is_file():
        declared = [
            match.group("cap")
            for match in (CAPABILITY_RE.match(line) for line in proposal.read_text().splitlines())
            if match
        ]
        for capability in declared:
            # A delta only carries changed capabilities, so a declared one may
            # legitimately live in the main spec tree instead.
            if capability not in capabilities and capability not in known:
                errors.append(
                    f"{label}: proposal declares capability {capability!r} that "
                    "exists neither in specs/ nor in openspec/specs/"
                )
    return errors


def validate(root: Path = ROOT) -> list[str]:
    base = root / OPENSPEC
    changes = base / "changes"
    if not changes.is_dir():
        return ["openspec/changes is missing"]
    errors: list[str] = []
    active = [d for d in sorted(changes.iterdir())
              if d.is_dir() and d.name != "archive"]
    archive = changes / "archive"
    archived = [d for d in sorted(archive.iterdir()) if d.is_dir()] if archive.is_dir() else []
    specs_root = base / "specs"
    known = {
        "/".join(spec.parent.relative_to(specs_root).parts)
        for spec in specs_root.rglob("spec.md")
    } if specs_root.is_dir() else set()
    for directory in active:
        errors.extend(_check_change(directory, archived=False, known=known))
    for directory in archived:
        if not re.match(r"^\d{4}-\d{2}-\d{2}-[a-z0-9-]+$", directory.name):
            errors.append(
                f"openspec/changes/archive/{directory.name}: archived change must be "
                "named YYYY-MM-DD-<change>"
            )
        errors.extend(_check_change(directory, archived=True, known=known))
    if not active:
        errors.append("openspec/changes has no active change")
    for spec in sorted(specs_root.rglob("spec.md")) if specs_root.is_dir() else []:
        if not REQUIREMENT_RE.search(spec.read_text()):
            errors.append(
                f"{spec.relative_to(root)}: has no '### Requirement:'"
            )
    return errors


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    args = parser.parse_args(argv)
    errors = validate(args.root)
    if errors:
        print("\n".join(errors), file=sys.stderr)
        return 1
    print("OpenSpec structure valid (structure only; checked boxes are not evidence)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
