"""Chat must answer questions, not learn them.

Regression: ``DroidEngine.chat("Define remote sensing")`` was classified as a
declarative statement -- the auto-absorption guard only recognised ``what``,
``who``, ``when``, ``where``, ``why``, ``how``, ``tell`` and ``explain`` -- so the
imperative query was taught as a new fact and the reply was the absorption
message instead of the stored definition. The same turn is exercised after a
profile reload by ``scripts/verify_remote_sensing_e2e.py`` step 7, which is how
the defect surfaced as two failing end-to-end tests.
"""

from __future__ import annotations

import pytest

from recamem.engine import QUERY_OPENERS, DroidEngine

PARAGRAPH = (
    "Remote sensing is the acquisition of information about an object or "
    "phenomenon without making physical contact with the object, in contrast to "
    "in situ or on-site observation. The term is applied especially to acquiring "
    "information about Earth and other planets. Remote sensing is used in "
    "numerous fields, including geophysics, geography and hydrology."
)


@pytest.fixture()
def engine():
    return DroidEngine(name="chat-intent")


def test_imperative_query_recalls_instead_of_teaching(engine):
    engine.teach(PARAGRAPH, source="user_text")
    before = engine.knowledge.active_count()

    reply = engine.chat("Define remote sensing")

    assert "acquisition of information" in reply.lower()
    assert "absorbed this into my memory" not in reply.lower()
    assert engine.knowledge.active_count() == before, (
        "a question must not be stored as a fact"
    )


def test_question_and_imperative_forms_take_the_same_path(engine):
    engine.teach(PARAGRAPH, source="user_text")

    with_question_mark = engine.chat("Can you define remote sensing?")
    without_question_mark = engine.chat("Define remote sensing")

    for reply in (with_question_mark, without_question_mark):
        assert "acquisition of information" in reply.lower()


def test_declarative_statement_is_still_absorbed(engine):
    reply = engine.chat("Paris is the capital of France.")

    assert "absorbed this into my memory" in reply.lower()
    assert engine.knowledge.active_count() >= 1


def test_every_query_opener_counts_as_a_question():
    for opener in sorted(QUERY_OPENERS):
        assert DroidEngine._is_question_clause(f"{opener} something"), opener


def test_statement_is_not_mistaken_for_a_question():
    assert not DroidEngine._is_question_clause(
        "Photogrammetry extracts 3D measurements from overlapping photographs."
    )
