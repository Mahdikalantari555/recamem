"""Contract tests for src/model/propositions.py.

These pin the implemented contract of the proposer:

- text is split on sentence terminators followed by whitespace;
- abbreviations (``Dr.``, ``e.g.``, ...) do not create sentence boundaries;
- propositions shorter than the minimum length (default 30 characters) are
  dropped;
- sentences whose (quote/bullet/number-stripped) prefix is a conversational
  marker are dropped;
- output is stable for a fixed paragraph.

Note on scope: this module does not rewrite pronouns or resolve anaphora - the
anaphora regex lives in ``src/model/droid.py`` and rewrites at teach time - and
there is no word-count filter here, only the character-length filter. The tests
below assert the split/filter contract that the implementation actually
provides rather than inventing a behaviour it does not have.
"""

from src.model.propositions import (
    ABBREVIATIONS,
    CONVERSATIONAL_MARKERS,
    extract_propositions,
    extract_propositions_with_fallback,
)


def test_empty_input_returns_empty_list():
    assert extract_propositions("") == []
    assert extract_propositions("   \n  ") == []
    assert extract_propositions_with_fallback("") == []


def test_splits_on_sentence_boundaries():
    text = (
        "Remote sensing acquires information without physical contact with the object. "
        "The term is applied to Earth observation in particular. "
        "It is used in geophysics and meteorology today."
    )
    props = extract_propositions(text)
    assert len(props) == 3
    assert props[0].startswith("Remote sensing acquires")
    assert props[1].startswith("The term is applied")
    assert props[2].startswith("It is used")


def test_drops_sentences_shorter_than_thirty_characters():
    text = "Too short. This sentence is comfortably longer than thirty characters."
    props = extract_propositions(text)
    assert props == ["This sentence is comfortably longer than thirty characters."]


def test_min_length_is_configurable_in_fallback():
    text = "Short one here. Another sentence that is a bit longer than the first."
    strict = extract_propositions_with_fallback(text, min_length=60)
    assert strict == []
    loose = extract_propositions_with_fallback(text, min_length=10)
    assert len(loose) == 2


def test_abbreviations_do_not_split_sentences():
    text = (
        "Dr. Smith studies river basins, e.g. the Rhine delta, across several countries. "
        "Prof. Jones compares them with the Amazon basin in Brazil."
    )
    props = extract_propositions(text)
    assert len(props) == 2
    assert props[0].startswith("Dr. Smith studies")
    assert "e.g. the Rhine delta" in props[0]
    assert "e.g." in ABBREVIATIONS


def test_conversational_markers_are_filtered_case_insensitively():
    text = (
        "Based on the retrieved facts, the river is long and navigable. "
        "The river is indeed long and navigable for cargo ships. "
        "HERE IS the summary of the retrieved facts about the river."
    )
    props = extract_propositions(text)
    assert props == ["The river is indeed long and navigable for cargo ships."]
    assert "based on" in CONVERSATIONAL_MARKERS


def test_marker_filter_ignores_leading_quotes_bullets_and_numbers():
    text = (
        '"Based on the context, the answer would be forty two percent. '
        "- Sure, the answer is forty two percent for the tested sample. "
        "1. According to the sources, the answer is forty two percent overall. "
        "The measured value is forty two percent in the benchmark run."
    )
    props = extract_propositions(text)
    assert props == ["The measured value is forty two percent in the benchmark run."]


def test_missing_terminal_punctuation_is_added():
    props = extract_propositions("The sensor array samples at one hertz without gaps")
    assert props == ["The sensor array samples at one hertz without gaps."]


def test_output_is_stable_for_a_fixed_paragraph():
    text = (
        "Radar uses radio waves to detect objects at a distance. "
        "It works through clouds and at night, unlike optical sensors. "
        "Lidar instead uses laser pulses and gives finer 3D geometry data."
    )
    first = extract_propositions(text)
    second = extract_propositions(text)
    assert first == second
    assert len(first) == 3
    assert all(p.endswith(".") for p in first)


def test_punctuation_only_and_short_fragments_are_dropped():
    assert extract_propositions("Hi there. Ok. Fine.") == []
