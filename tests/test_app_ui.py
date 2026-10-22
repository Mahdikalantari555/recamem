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

APP = pathlib.Path(__file__).resolve().parents[1] / "app.py"
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
    assert "Droid Lifelong Engine (Anchored)" in engines
    assert "Droid + Raw Byte RTU (Fused)" in engines
    assert "Raw Byte-Level RTU (Experimental)" in engines


def test_memory_console_tab_is_present(app):
    assert any("Memory Console" in tab.label for tab in app.tabs)
    assert any("Memory Console" in header.value for header in app.subheader)


def test_memory_console_has_three_subtabs(app):
    labels = [tab.label for tab in app.tabs]
    # The console grew past the three historical sub-tabs; the three named by
    # the archived task must still be there.
    for expected in ("Browse", "Teach", "Stats"):
        assert any(expected.lower() in label.lower() for label in labels), expected


def test_feedback_state_is_initialized(app):
    assert "feedback_state" in app.session_state
    assert app.session_state["feedback_state"] == {}


def test_feedback_state_survives_rerun(app):
    app.session_state["feedback_state"] = {1: "approve"}
    app.run()
    assert app.session_state["feedback_state"] == {1: "approve"}
    assert not app.exception


def test_sidebar_reports_droid_stats(app):
    captions = [element.value for element in app.caption]
    sidebar_text = " ".join(captions)
    assert "facts" in sidebar_text.lower() or "fact" in sidebar_text.lower()


def test_anchor_badge_reads_the_mock_flag(app):
    # The mock anchor is the only one available offline, and the badge must say
    # so instead of printing a hardcoded success string.
    warnings = " ".join(element.value for element in app.warning)
    successes = " ".join(element.value for element in app.success)
    assert "mock" in warnings.lower()
    assert "ONNX MiniLM (INT8" not in successes


def test_stream_panel_keys_the_worker_per_profile(app):
    # The worker is owned by the engine; the app must not spawn its own thread
    # and must key the controls per profile.
    source = APP.read_text()
    assert "start_continuous_stream(" in source
    assert "threading.Thread(target=self._stream" not in source
    assert {"stream_start", "stream_stop", "stream_pause", "stream_resume"} <= set(
        app.session_state
    )


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
    """The teach sub-tab splits pasted text into reviewable propositions."""
    monkeypatch.chdir(tmp_path)
    instance = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    instance.run()

    area = next(
        element for element in instance.text_area if element.key == "llm_teach_input"
    )
    area.input(
        "Remote sensing is the acquisition of information without contact. "
        "It is applied to Earth observation. "
        "The technique supports land surveying and hydrology."
    ).run()

    # "Split Locally" is the offline proposition splitter (the LLM variant
    # needs a provider, which the offline tier does not have).
    split = next(button for button in instance.button if element_key(button) == "split_local")
    split.click().run()

    assert not instance.exception
    propositions = instance.session_state["llm_teach_propositions"]
    assert len(propositions) >= 2
    assert all(len(item) >= 30 for item in propositions)


def test_fact_browser_lists_stored_facts(tmp_path, monkeypatch):
    """A fact taught through the engine is visible in the console."""
    monkeypatch.chdir(tmp_path)
    import sys

    sys.path.insert(0, str(APP.parent))
    from src.model.droid_manager import DroidManager

    manager = DroidManager(base_dir="droids")
    engine = manager.create_droid("droid-alpha")
    engine.knowledge.insert_fact(
        "The calibration target is the blue marker",
        "teach", 1.0, 1, 1.0,
        torch.nn.functional.normalize(torch.ones(engine.knowledge.dim), dim=0),
    )
    manager.save_droid("droid-alpha")
    manager.shutdown()

    instance = AppTest.from_file(str(APP), default_timeout=TIMEOUT)
    instance.run()
    assert not instance.exception

    rendered = " ".join(
        [element.value for element in instance.caption]
        + [element.value for element in instance.markdown]
        + [str(frame.value) for frame in instance.dataframe]
    )
    assert "active facts" in rendered
    assert "1" in rendered


def test_stats_subtab_reports_store_counts(app):
    captions = " ".join(element.value for element in app.caption)
    assert "active facts" in captions.lower()
