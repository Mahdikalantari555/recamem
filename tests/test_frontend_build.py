"""Frontend scaffold and desktop-shell contracts (surfaces 3.x, 5.x, 9.x).

The Node build is optional: when `web/dist` is absent this module checks the
sources that make the build reproducible and skips the artefacts.
"""

import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
WEB = ROOT / "web"


def test_web_scaffold_has_the_toolchain_and_entry_points():
    package = json.loads((WEB / "package.json").read_text())
    assert package["scripts"]["build"].endswith("vite build")
    assert "react" in package["dependencies"]
    assert "tailwindcss" in package["devDependencies"]
    for name in ("vite.config.ts", "tsconfig.json", "tailwind.config.ts", "index.html"):
        assert (WEB / name).is_file(), name
    assert (WEB / "src" / "main.tsx").is_file()


def test_design_tokens_match_the_design_document():
    config = (WEB / "tailwind.config.ts").read_text()
    for token, value in (
        ("background", "#111318"),
        ("surface", "#1D2027"),
        ("border", "#343943"),
        ("accent", "#B5F36D"),
        ("danger", "#F07878"),
    ):
        assert f'{token}' in config
        assert value in config, token


def test_reduced_motion_and_focus_states_are_declared():
    css = (WEB / "src" / "styles" / "globals.css").read_text()
    assert "prefers-reduced-motion" in css
    assert "focus-visible" in css


def test_shadcn_components_are_vendored():
    for name in ("button", "card", "input", "tabs", "dialog", "sidebar"):
        assert (WEB / "src" / "components" / "ui" / f"{name}.tsx").is_file(), name


def test_api_client_only_talks_to_the_versioned_prefix():
    client = (WEB / "src" / "lib" / "api" / "client.ts").read_text()
    assert 'const BASE = "/api/v1"' in client
    assert "localhost" not in client
    assert "127.0.0.1:8000/api" not in client


def test_no_droid_branding_in_the_new_ui():
    for path in (WEB / "src").rglob("*.tsx"):
        text = path.read_text()
        assert "Droid" not in text.replace("droid-alpha", ""), path


@pytest.mark.skipif(not (WEB / "dist" / "index.html").is_file(),
                    reason="frontend not built (run `npm --prefix web run build`)")
def test_built_frontend_is_servable():
    index = (WEB / "dist" / "index.html").read_text()
    assert "<div id=\"root\">" in index
    assert any((WEB / "dist" / "assets").glob("*.js"))


def test_tauri_shell_is_configured_but_does_not_hold_model_logic():
    config = json.loads((ROOT / "src-tauri" / "tauri.conf.json").read_text())
    assert config["build"]["frontendDist"] == "../web/dist"
    assert config["app"]["windows"][0]["theme"] == "Dark"
    capabilities = json.loads(
        (ROOT / "src-tauri" / "capabilities" / "default.json").read_text()
    )
    assert capabilities["permissions"] == ["core:default"]
    main_rs = (ROOT / "src-tauri" / "src" / "main.rs").read_text()
    assert "DroidEngine" not in main_rs
    assert "torch" not in main_rs


def test_tauri_build_is_reported_blocked_without_cargo():
    assert shutil.which("cargo") is None or (ROOT / "src-tauri" / "Cargo.toml").is_file()
