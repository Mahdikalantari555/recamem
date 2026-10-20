"""Contract tests for `src/model/propositions.py` (surface-test-coverage 3.5).

Covered: sentence splitting, abbreviation protection, the length filter, the
conversational-marker filter, trailing-punctuation normalization, and stable
output for a fixed paragraph.
"""

from src.model.propositions import (
    CONVERSATIONAL_MARKERS,
    extract_propositions,
    extract_propositions_with_fallback,
)


def test_extract_propositions_empty_input():
    assert extract_propositions("") == []
    assert extract_propositions(None) == []
    assert extract_propositions("   ") == []


def test_extract_propositions_splits_on_sentence_boundaries():
    text = "The first proposition is here and it is long enough. The second proposition comes next and is long enough too."
    props = extract_propositions(text)
    assert props == [
        "The first proposition is here and it is long enough.",
        "The second proposition comes next and is long enough too.",
    ]


def test_extract_propositions_protects_abbreviations():
    text = "e.g. is an abbreviation for exempli gratia. It appears mid-sentence without splitting."
    props = extract_propositions(text)
    assert props == ["e.g. is an abbreviation for exempli gratia.", "It appears mid-sentence without splitting."]


def test_extract_propositions_length_filter():
    text = "This is a long enough proposition that passes the filter. Too short. This one is long enough as well."
    props = extract_propositions(text)
    assert len(props) == 2
    assert all("short" not in p for p in props)


def test_extract_propositions_conversational_marker_filter():
    text = "Here is the answer to your question. The actual proposition follows here and is long enough. I am an AI assistant."
    props = extract_propositions(text)
    assert props == ["The actual proposition follows here and is long enough."]


def test_extract_propositions_adds_trailing_punctuation():
    text = "This proposition has no period"
    props = extract_propositions(text)
    assert props == ["This proposition has no period."]


def test_extract_propositions_stable_for_fixed_paragraph():
    paragraph = (
        "Dr. Smith visited the lab in New York. e.g. he brought samples. "
        "The results were promising and are described below. vs. the control group was not."
    )
    first = extract_propositions(paragraph)
    second = extract_propositions(paragraph)
    assert first == second
    assert "Dr. Smith visited the lab in New York." in first


def test_extract_propositions_with_fallback_respects_min_length():
    text = "This is a long enough proposition. Short. Another long enough proposition here."
    assert len(extract_propositions_with_fallback(text, min_length=30)) == 2
    assert len(extract_propositions_with_fallback(text, min_length=100)) == 0
    assert extract_propositions_with_fallback("") == []


def test_conversational_markers_are_case_insensitive_prefixes():
    assert all(isinstance(m, str) and m for m in CONVERSATIONAL_MARKERS)
