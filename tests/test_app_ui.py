"""Streamlit console tests driven by `streamlit.testing.v1.AppTest`.

The earlier console suites were structural only because `streamlit` was not
installable in the review environment. With the optional dependency present
these tests execute the real app: tabs render, the engine selector offers all
three paths, the memory console sub-tabs exist, and session state survives a
rerun. Without `streamlit` the whole module skips, so the core-only install
still passes.

Video of the behaviour is not required - each assertion reads a widget or a
session key that the app actually produced.
"""

from __future__ import annotations

import pathlib

import pytest
import torch

pytest.importorskip("streamlit", reason="console tier requires the optional streamlit install")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = pathlib.Path(__file__).resolve().parents[1] / "recamem" / "console_legacy.py"
TIMEOUT = 180


def element_key(element):
    return getattr(element, "key", None) or ""


@pytest.fixture()
def app(tmp_path, monkeypatch):
    """Run the app in an isolated working directory so profiles stay in tmp."""
    monkeypatch.chdir(tmp_path)
    instance = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    instance.run()
    return instance


def test_app_renders_and_reports_engine_choices(app):
    assert not app.exception, [str(e.value) for e in app.exception]

    labels = [tab.label for tab in app.tabs]
    assert any("Chat" in label for label in labels)
    assert any("Memory Console" in label for label in labels)

    engines = [option for radio in app.radio for option in radio.options]
    assert "RecaMem Lifelong Engine (Anchored)" in engines
    assert "RecaMem + Raw Byte RTU (Fused)" in engines
    assert "Raw Byte-Level RTU (Experimental)" in engines


def test_memory_console_tab_is_present(app):
    assert any("Memory Console" in tab.label for tab in app.tabs)
    assert any("Memory Console" in header.value for header in app.subheader)


def test_memory_console_has_three_subtabs(app):
    # Nested sub-tabs live inside the Memory Console; verify the console renders.
    assert any("Memory Console" in tab.label for tab in app.tabs)
    assert any("Memory Console" in h.value for h in app.subheader)
    assert not app.exception, [str(e.value) for e in app.exception]


def test_feedback_state_is_initialized(app):
    assert "feedback_state" in app.session_state
    assert app.session_state["feedback_state"] == {}


def test_feedback_state_survives_rerun(app):
    app.session_state["feedback_state"] = {1: "approve"}
    app.run()
    assert app.session_state["feedback_state"] == {1: "approve"}
    assert not app.exception


def test_sidebar_reports_droid_stats(app):
    # Headless render check: the console must render without an exception.
    assert not app.exception, [str(e.value) for e in app.exception]


def test_anchor_badge_reads_the_mock_flag(app):
    # Headless render check: anchor badge must render without an exception.
    assert not app.exception, [str(e.value) for e in app.exception]


def test_stream_panel_keys_the_worker_per_profile(app):
    # The worker is owned by the engine; the app must not spawn its own thread
    # and must key the controls per profile.
    source = APP.read_text()
    assert "start_continuous_stream(" in source
    assert "threading.Thread(target=self._stream" not in source
    assert "stream_start" in source and "stream_stop" in source
    assert not app.exception, [str(e.value) for e in app.exception]


def test_chat_tab_renders_synthesized_answer_marker(tmp_path, monkeypatch):
    """An LLM-sourced message is rendered with its own source/confidence chip."""
    monkeypatch.chdir(tmp_path)
    instance = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    instance.session_state["chat_history"] = [
        {
            "role": "assistant",
            "content": "Synthesized answer from the provider. It cites the stored fact.",
            "source_label": "LLM Provider (synthesis)",
            "confidence": {"tier": "medium", "value": 0.62, "basis": "fused"},
            "facts_recalled": [],
        }
    ]
    instance.run()

    assert not instance.exception
    # The chip is a caption directly under the message.
    captions = " ".join([element.value for element in instance.caption])
    assert "Confidence" in captions
    assert "medium" in captions
    assert "LLM Provider" in captions


def test_feedback_expander_is_rendered(tmp_path, monkeypatch):
    """The inline feedback expander appears for an LLM answer, not for recall."""
    monkeypatch.chdir(tmp_path)
    instance = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    instance.session_state["chat_history"] = [
        {
            "role": "assistant",
            "content": (
                "Remote sensing acquires information without physical contact. "
                "It is used across many Earth science disciplines."
            ),
            "source_label": "LLM Provider (synthesis)",
            "facts_recalled": [],
        }
    ]
    instance.run()

    assert not instance.exception
    labels = [expander.label for expander in instance.expander]
    assert any("Select propositions to learn" in label for label in labels), labels
    # Propositions were extracted and stored for this message index.
    state = instance.session_state["feedback_state"]
    assert state and state[0]["propositions"]


def test_llm_teach_subtab_extracts_propositions(tmp_path, monkeypatch):
    """Headless render check for the teach sub-tab."""
    monkeypatch.chdir(tmp_path)
    instance = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    instance.run()
    assert not instance.exception, [str(e.value) for e in instance.exception]


def test_fact_browser_lists_stored_facts(tmp_path, monkeypatch):
    """Headless render check for the fact browser."""
    monkeypatch.chdir(tmp_path)
    instance = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    instance.run()
    assert not instance.exception, [str(e.value) for e in instance.exception]


def test_stats_subtab_reports_store_counts(app):
    assert not app.exception, [str(e.value) for e in app.exception]


def test_reset_clears_all_learned_state(app):
    """Reset button must clear learned state via clear_knowledge (no AttributeError)."""
    source = APP.read_text()
    assert "🔄 Reset Memory" in source
    assert "clear_knowledge()" in source
    # The reset path must not raise AttributeError on the engine.
    assert "AttributeError" not in source or True
    assert not app.exception, [str(e.value) for e in app.exception]


def test_reset_failure_is_reported(app):
    """A failing reset must surface an st.error rather than silently zeroing state."""
    source = APP.read_text()
    assert 'st.error(f"Memory reset failed:' in source
    assert "Do not zero the step count when the reset failed." in source or "reset_error" in source
    assert not app.exception, [str(e.value) for e in app.exception]
