"""Folder ingestion for recamem.

Walks Markdown, plain text and (optionally) PDF files, keeps the file path, the
Markdown heading path and the source content hash as provenance, skips code
fences, and re-ingests idempotently: unchanged files are a no-op and changed
files supersede their previous facts without deleting them.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

TEXT_SUFFIXES = {".md", ".markdown", ".txt", ".text"}
PDF_SUFFIXES = {".pdf"}
SUPPORTED_SUFFIXES = TEXT_SUFFIXES | PDF_SUFFIXES

_FENCE_RE = re.compile(r"^(```|~~~).*?^\1[^\n]*$", re.DOTALL | re.MULTILINE)
_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")


@dataclass
class IngestReport:
    root: str
    files_read: int = 0
    files_unchanged: int = 0
    facts_stored: int = 0
    superseded: int = 0
    files_skipped: List[Dict[str, str]] = field(default_factory=list)
    files_changed: List[str] = field(default_factory=list)

    @property
    def skipped_count(self) -> int:
        return len(self.files_skipped)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "root": self.root,
            "files_read": self.files_read,
            "files_unchanged": self.files_unchanged,
            "files_skipped": self.files_skipped,
            "files_changed": self.files_changed,
            "facts_stored": self.facts_stored,
            "superseded": self.superseded,
            "skipped_count": self.skipped_count,
        }


def describe_report(report: IngestReport) -> List[str]:
    lines = [
        f"root={report.root}",
        f"files_read={report.files_read}",
        f"files_unchanged={report.files_unchanged}",
        f"files_skipped={report.skipped_count}",
        f"facts_stored={report.facts_stored}",
        f"superseded={report.superseded}",
    ]
    for skipped in report.files_skipped:
        lines.append(f"skipped={skipped['path']} reason={skipped['reason']}")
    if report.facts_stored == 0:
        lines.append("note=zero facts stored")
    return lines


def content_hash(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def strip_code_fences(text: str) -> str:
    """Remove fenced code blocks so they are never segmented as prose."""
    return _FENCE_RE.sub("", text)


def markdown_sections(text: str) -> List[Dict[str, str]]:
    """Split Markdown into ``{heading_path, body}`` sections (fences removed)."""
    sections: List[Dict[str, str]] = []
    stack: List[tuple[int, str]] = []
    body: List[str] = []

    def flush() -> None:
        if not body:
            return
        chunk = "\n".join(body).strip()
        if chunk:
            sections.append(
                {
                    "heading_path": " / ".join(title for _, title in stack),
                    "body": chunk,
                }
            )
        body.clear()

    in_fence = False
    fence = ""
    for line in text.splitlines():
        stripped = line.lstrip()
        if not in_fence and (stripped.startswith("```") or stripped.startswith("~~~")):
            in_fence = True
            fence = stripped[:3]
            continue
        if in_fence:
            if stripped.startswith(fence):
                in_fence = False
            continue
        match = _HEADING_RE.match(line)
        if match:
            flush()
            level = len(match.group(1))
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, match.group(2).strip()))
            continue
        body.append(line)
    flush()
    return sections


def read_pdf(path: Path) -> Dict[str, Any]:
    """Extract PDF text with the optional reader, or explain why it was not."""
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError:
        return {"status": "skipped", "reason": "pdf_reader_missing (install recamem[pdf])"}
    try:
        reader = PdfReader(str(path))
        text = "\n".join((page.extract_text() or "") for page in reader.pages)
    except Exception as exc:  # pragma: no cover - depends on optional library
        return {"status": "skipped", "reason": f"pdf_unreadable: {exc}"}
    if not text.strip():
        return {"status": "skipped", "reason": "pdf_no_text_layer"}
    return {"status": "ok", "text": text}


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.is_file():
            yield path


def _store_propositions(engine, text: str, *, source: str, file_path: str,
                        heading_path: str, digest: str) -> int:
    store = engine.knowledge
    before = store.conn.execute("SELECT COALESCE(MAX(id), 0) FROM facts").fetchone()[0]
    engine.teach(text, source=source, origin="ingest")
    rows = store.conn.execute(
        "SELECT id FROM facts WHERE id > ? ORDER BY id", (int(before),)
    ).fetchall()
    ids = [int(r[0]) for r in rows]
    if not ids:
        return 0
    placeholders = ",".join("?" for _ in ids)
    store.conn.execute(
        f"UPDATE facts SET file_path=?, heading_path=?, content_hash=? WHERE id IN ({placeholders})",
        [file_path, heading_path, digest, *ids],
    )
    return len(ids)


def ingest_file(engine, path: Path, *, root: Optional[Path] = None) -> Dict[str, Any]:
    """Ingest one file; returns counts and any skip reason."""
    path = Path(path)
    result: Dict[str, Any] = {"path": str(path), "facts_stored": 0, "superseded": 0}
    suffix = path.suffix.lower()
    if suffix in PDF_SUFFIXES:
        payload = read_pdf(path)
        if payload["status"] != "ok":
            result.update(status="skipped", reason=payload["reason"])
            return result
        text = payload["text"]
        digest = content_hash(path.read_bytes())
    else:
        raw = path.read_text(encoding="utf-8", errors="replace")
        digest = content_hash(path.read_bytes())
        text = raw

    store = engine.knowledge
    previous = store.file_content_hash(str(path))
    if previous == digest:
        result.update(status="unchanged", facts_stored=0)
        return result

    if previous is not None:
        for row in store.facts_by_file(str(path)):
            store.mark_superseded(int(row[0]))
            result["superseded"] += 1

    sections = (
        markdown_sections(text)
        if suffix in {".md", ".markdown"}
        else [{"heading_path": "", "body": text}]
    )
    stored = 0
    for section in sections:
        prose = section["body"].strip()
        if not prose:
            continue
        stored += _store_propositions(
            engine,
            prose,
            source=str(path),
            file_path=str(path),
            heading_path=section["heading_path"],
            digest=digest,
        )
    result.update(status="stored", facts_stored=stored)
    return result


def ingest_directory(engine, root: str | Path) -> IngestReport:
    """Ingest every supported file under ``root``; unsupported files are skipped."""
    root_path = Path(root)
    report = IngestReport(root=str(root_path))
    if not root_path.is_dir():
        report.files_skipped.append({"path": str(root_path), "reason": "not_a_directory"})
        return report
    for path in _iter_files(root_path):
        suffix = path.suffix.lower()
        if suffix not in SUPPORTED_SUFFIXES:
            report.files_skipped.append({"path": str(path), "reason": "unsupported_type"})
            continue
        result = ingest_file(engine, path, root=root_path)
        status = result.get("status")
        if status == "skipped":
            report.files_skipped.append({"path": str(path), "reason": result["reason"]})
            continue
        report.files_read += 1
        if status == "unchanged":
            report.files_unchanged += 1
            continue
        report.files_changed.append(str(path))
        report.facts_stored += int(result.get("facts_stored", 0))
        report.superseded += int(result.get("superseded", 0))
    return report
