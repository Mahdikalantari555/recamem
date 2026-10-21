import shutil
import tempfile
import pytest
from src.model.droid_manager import DroidManager

@pytest.fixture
def temp_manager():
    d = tempfile.mkdtemp()
    mgr = DroidManager(base_dir=d)
    yield mgr
    shutil.rmtree(d)

def test_droid_manager_lifecycle(temp_manager):
    assert temp_manager.list_droids() == []

    d1 = temp_manager.create_droid("geospatial")
    d2 = temp_manager.create_droid("medical")

    assert "geospatial" in temp_manager.list_droids()
    assert "medical" in temp_manager.list_droids()

    # Teach distinct domain knowledge
    d1.teach("Synthetic Aperture Radar provides day-and-night all-weather radar imaging.")
    temp_manager.save_droid("geospatial")

    d2.teach("Cardiology is the study and treatment of disorders of the heart and blood vessels.")
    temp_manager.save_droid("medical")

    # Re-instantiate a fresh manager to test disk reload
    mgr2 = DroidManager(base_dir=temp_manager.base_dir)
    loaded_geo = mgr2.get_droid("geospatial")
    loaded_med = mgr2.get_droid("medical")

    # Check isolation
    geo_reply = loaded_geo.chat("What is synthetic aperture radar?")
    assert "day-and-night all-weather" in geo_reply

    med_reply = loaded_med.chat("What is cardiology?")
    assert "disorders of the heart" in med_reply

    # Medical droid should not have radar facts
    med_geo_reply = loaded_med.chat("What is synthetic aperture radar?")
    assert "day-and-night all-weather" not in med_geo_reply

def test_droid_delete(temp_manager):
    temp_manager.create_droid("temp-bot")
    assert "temp-bot" in temp_manager.list_droids()
    deleted = temp_manager.delete_droid("temp-bot")
    assert deleted is True
    assert "temp-bot" not in temp_manager.list_droids()


# --------------------------------------------------------------------------- #
# Research configuration in the manifest (web-grounded-learning 5.4)
# --------------------------------------------------------------------------- #
import json


def test_a_profile_without_a_research_block_loads_disabled(temp_manager):
    droid = temp_manager.create_droid("plain-profile")
    temp_manager.save_droid("plain-profile")
    manifest = json.loads((temp_manager.base_dir and __import__("pathlib").Path(
        temp_manager.base_dir) / "plain-profile" / "config.json").read_text())
    assert "research" not in manifest
    assert droid.research_config["enabled"] is False
    assert droid.search_learner is None

    loaded = DroidManager(base_dir=temp_manager.base_dir).get_droid("plain-profile")
    assert loaded.research_enabled is False
    assert loaded.research("radar")["available"] is False


def test_a_profile_with_a_research_block_loads_its_provider_and_bounds(temp_manager):
    droid = temp_manager.create_droid("research-profile")
    droid.configure_research({
        "enabled": True,
        "provider": "duckduckgo",
        "max_pages_per_search": 2,
        "max_queries_per_session": 4,
        "suppression_window_seconds": 60,
    })
    temp_manager.save_droid("research-profile")

    manifest = json.loads((__import__("pathlib").Path(temp_manager.base_dir)
                           / "research-profile" / "config.json").read_text())
    assert manifest["research"]["provider"] == "duckduckgo"
    assert manifest["research"]["max_pages_per_search"] == 2

    loaded = DroidManager(base_dir=temp_manager.base_dir).get_droid("research-profile")
    assert loaded.research_enabled is True
    assert loaded.research_config["max_queries_per_session"] == 4
    assert loaded.research_config["suppression_window_seconds"] == 60.0
    assert loaded.search_learner is not None
    assert loaded.search_learner.available is True
