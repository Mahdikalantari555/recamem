# Ingestion

```bash
recamem ingest ./notes --name agent-notes --save     # Markdown + text (+ PDF)
recamem recall "What is the calibration constant?" --name agent-notes
```

## What is stored

Each fact records:

* `file_path` — the source file,
* `heading_path` — the Markdown heading trail (`Project / Sensors`),
* `content_hash` — SHA-256 of the source file.

Code fences are never segmented as prose. Prose is split by the
abbreviation-safe segmenter (`recamem/segmenter.py`): `Dr.`, initials
(`J. R. R.`) and decimals (`3.5 metres`) are not sentence boundaries.

## Idempotent re-ingest

* Unchanged file (same content hash): nothing is stored; the report says
  `files_unchanged=N`.
* Changed file: the facts from the previous version are marked superseded (they
  stay queryable for audit) and the new text is stored.
* Unsupported file types are listed as skipped with a reason.

## PDF

PDF text needs the optional reader:

```bash
pip install "recamem[pdf]"
```

Without it a PDF is reported as `skipped reason=pdf_reader_missing` and no fact
is stored. A PDF with no text layer is reported as `pdf_no_text_layer`.

## Report

The report contains only measured counts (`files_read`, `files_unchanged`,
`files_skipped`, `facts_stored`, `superseded`) and no recall or quality figure.
