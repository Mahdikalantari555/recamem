"""Folder ingestion with provenance, idempotency and honest skips."""

import sqlite3
import sys
import types
from pathlib import Path

import pytest

from recamem.engine import DroidEngine
from recamem.ingest import (
    content_hash,
    describe_report,
    ingest_directory,
    ingest_file,
    markdown_sections,
    read_pdf,
    strip_code_fences,
)
from recamem.segmenter import extract_propositions
pytestmark = pytest.mark.phase0



@pytest.fixture()
def engine():
    return DroidEngine(name="ingest-test")


def test_title_abbreviation_is_not_a_boundary():
    props = extract_propositions(
        "The project lead is Dr. Mina Rostami. The budget is 5 million.",
        min_length=5, filter_conversational=False,
    )
    assert props[0] == "The project lead is Dr. Mina Rostami."
    assert len(props) == 2


def test_initials_and_decimals_are_not_boundaries():
    props = extract_propositions(
        "J. R. R. Tolkien wrote the book and the value is 3.5 metres.",
        min_length=5, filter_conversational=False,
    )
    assert len(props) == 1
    assert "J. R. R. Tolkien" in props[0]
    assert "3.5 metres" in props[0]


def test_real_sentence_ends_still_split():
    props = extract_propositions(
        "The sensor detects moisture. The gateway forwards readings.",
        min_length=5, filter_conversational=False,
    )
    assert props == ["The sensor detects moisture.", "The gateway forwards readings."]


def test_code_fences_are_not_prose():
    text = "# Title\n\nReal prose here.\n\n```python\nx = 1\n```\n\nMore prose.\n"
    assert "x = 1" not in strip_code_fences(text)
    sections = markdown_sections(text)
    assert all("x = 1" not in s["body"] for s in sections)
    assert sections[0]["heading_path"] == "Title"


def test_headings_become_the_heading_path():
    text = "# Project\n\nIntro line here.\n\n## Sensors\n\nSoil probes are deployed.\n"
    sections = markdown_sections(text)
    paths = {s["heading_path"] for s in sections}
    assert "" in paths or "Project" in paths
    assert "Project / Sensors" in paths


def test_markdown_folder_is_ingested_with_provenance(engine, tmp_path):
    (tmp_path / "notes.md").write_text("# Lab\n\nThe calibration constant is 0.98.\n")
    report = ingest_directory(engine, tmp_path)
    assert report.facts_stored >= 1
    rows = engine.knowledge.conn.execute(
        "SELECT text, file_path, heading_path, content_hash FROM facts"
    ).fetchall()
    assert rows
    for text, file_path, heading_path, digest in rows:
        assert file_path.endswith("notes.md")
        assert heading_path == "Lab"
        assert digest == content_hash((tmp_path / "notes.md").read_bytes())


def test_unchanged_file_is_a_no_op(engine, tmp_path):
    (tmp_path / "a.md").write_text("The gateway is on site C.\n")
    ingest_directory(engine, tmp_path)
    before = engine.knowledge.total_count()
    report = ingest_directory(engine, tmp_path)
    assert report.facts_stored == 0
    assert report.files_unchanged == 1
    assert engine.knowledge.total_count() == before


def test_changed_file_supersedes_its_old_facts(engine, tmp_path):
    path = tmp_path / "b.md"
    path.write_text("The gateway is on site C.\n")
    ingest_directory(engine, tmp_path)
    first_ids = [row[0] for row in engine.knowledge.conn.execute("SELECT id FROM facts")]
    path.write_text("The gateway is on site D.\n")
    report = ingest_directory(engine, tmp_path)
    assert report.superseded == len(first_ids)
    rows = engine.knowledge.conn.execute(
        "SELECT id, superseded FROM facts ORDER BY id"
    ).fetchall()
    superseded = [row for row in rows if row[0] in first_ids]
    assert all(row[1] == 1 for row in superseded), "old facts stay queryable for audit"
    assert engine.knowledge.total_count() > len(first_ids)


def test_unsupported_files_are_skipped_with_a_reason(engine, tmp_path):
    (tmp_path / "data.bin").write_bytes(b"\x00\x01")
    report = ingest_directory(engine, tmp_path)
    assert report.facts_stored == 0
    assert any("unsupported" in s["reason"] for s in report.files_skipped)
    assert "note=zero facts stored" in describe_report(report)
    assert not any("recall" in line for line in describe_report(report))


def test_pdf_without_the_reader_is_skipped_with_its_reason(engine, tmp_path, monkeypatch):
    monkeypatch.setitem(sys.modules, "pypdf", None)
    pdf = tmp_path / "doc.pdf"
    pdf.write_bytes(b"%PDF-1.4 not really")
    result = ingest_file(engine, pdf)
    assert result["status"] == "skipped"
    assert "pdf_reader_missing" in result["reason"]
    assert engine.knowledge.total_count() == 0
    assert read_pdf(pdf)["status"] == "skipped"


def test_pdf_with_no_text_layer_is_skipped(tmp_path, monkeypatch):
    class _Page:
        def extract_text(self):
            return "   "

    class _Reader:
        def __init__(self, path):
            self.pages = [_Page()]

    module = types.ModuleType("pypdf")
    module.PdfReader = _Reader
    monkeypatch.setitem(sys.modules, "pypdf", module)
    assert read_pdf(tmp_path / "x.pdf") == {"status": "skipped", "reason": "pdf_no_text_layer"}


def test_ingest_directory_missing_reports_not_a_directory(engine, tmp_path):
    report = ingest_directory(engine, tmp_path / "absent")
    assert report.files_skipped[0]["reason"] == "not_a_directory"
