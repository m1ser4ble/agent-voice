import pytest

from agent_voice.voice_config import VoicePresetError, resolve_voice_settings


def test_default_voice_settings_use_bundled_jarvis_style_preset():
    settings = resolve_voice_settings()

    assert settings.preset == "jarvis_style"
    assert settings.voice == "am_michael"
    assert settings.kokoro_voice == "am_michael"
    assert settings.supertonic_voice == "M2"
    assert settings.lang == "ko"
    assert settings.speed == 0.94
    assert "not a celebrity" in settings.description


def test_voice_settings_can_be_loaded_from_config_file(tmp_path):
    config_path = tmp_path / "voice-presets.toml"
    config_path.write_text(
        """
        [defaults]
        preset = "workstation"

        [presets.workstation]
        voice = "bm_george"
        supertonic_voice = "M3"
        lang = "en-gb"
        speed = 0.9
        description = "Local workstation voice."
        """,
        encoding="utf-8",
    )

    settings = resolve_voice_settings(config_path=config_path)

    assert settings.preset == "workstation"
    assert settings.voice == "bm_george"
    assert settings.kokoro_voice == "bm_george"
    assert settings.supertonic_voice == "M3"
    assert settings.lang == "en-gb"
    assert settings.speed == 0.9


def test_voice_settings_cli_overrides_take_precedence():
    settings = resolve_voice_settings(
        preset_name="jarvis_style",
        voice_override="af_bella",
        supertonic_voice_override="F2",
        lang_override="en-us",
        speed_override=1.05,
    )

    assert settings.preset == "jarvis_style"
    assert settings.voice == "af_bella"
    assert settings.kokoro_voice == "af_bella"
    assert settings.supertonic_voice == "F2"
    assert settings.lang == "en-us"
    assert settings.speed == 1.05


def test_kokoro_voice_override_does_not_replace_supertonic_voice():
    settings = resolve_voice_settings(
        preset_name="jarvis_style",
        voice_override="af_bella",
    )

    assert settings.kokoro_voice == "af_bella"
    assert settings.supertonic_voice == "M2"


def test_unknown_voice_preset_is_reported():
    with pytest.raises(VoicePresetError, match="unknown voice preset"):
        resolve_voice_settings(preset_name="missing")


def test_missing_voice_config_file_is_reported(tmp_path):
    with pytest.raises(VoicePresetError, match="voice config file not found"):
        resolve_voice_settings(config_path=tmp_path / "missing.toml")
