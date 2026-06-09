from __future__ import annotations

import tomllib
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any


class VoicePresetError(ValueError):
    pass


@dataclass(frozen=True)
class VoiceSettings:
    preset: str
    voice: str
    lang: str
    speed: float
    description: str = ""


def resolve_voice_settings(
    *,
    config_path: Path | None = None,
    preset_name: str | None = None,
    voice_override: str | None = None,
    lang_override: str | None = None,
    speed_override: float | None = None,
) -> VoiceSettings:
    config = _load_merged_config(config_path)
    defaults = _mapping(config.get("defaults", {}), "defaults")
    presets = _mapping(config.get("presets", {}), "presets")
    selected_preset = preset_name or str(defaults.get("preset", "jarvis_style"))

    preset = _mapping(
        presets.get(selected_preset),
        f"presets.{selected_preset}",
        missing_message=f"unknown voice preset '{selected_preset}'",
    )
    voice = voice_override or _required_str(preset, "voice", selected_preset)
    lang = lang_override or _required_str(preset, "lang", selected_preset)
    speed = speed_override if speed_override is not None else _required_float(
        preset,
        "speed",
        selected_preset,
    )
    _validate_speed(speed, selected_preset)
    description = str(preset.get("description", ""))
    return VoiceSettings(
        preset=selected_preset,
        voice=voice,
        lang=lang,
        speed=speed,
        description=description,
    )


def _load_merged_config(config_path: Path | None) -> dict[str, Any]:
    config = _load_bundled_config()
    if config_path is None:
        return config

    try:
        user_config = tomllib.loads(config_path.read_text(encoding="utf-8"))
    except OSError as error:
        raise VoicePresetError(f"voice config file not found: {config_path}") from error
    except tomllib.TOMLDecodeError as error:
        raise VoicePresetError(f"voice config file is invalid TOML: {error}") from error
    return _merge_config(config, user_config)


def _load_bundled_config() -> dict[str, Any]:
    content = resources.files("agent_voice").joinpath("voice_presets.toml").read_text()
    return tomllib.loads(content)


def _merge_config(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    if "defaults" in override:
        merged["defaults"] = {
            **_mapping(base.get("defaults", {}), "defaults"),
            **_mapping(override["defaults"], "defaults"),
        }
    if "presets" in override:
        merged["presets"] = {
            **_mapping(base.get("presets", {}), "presets"),
            **_mapping(override["presets"], "presets"),
        }
    return merged


def _mapping(
    value: Any,
    name: str,
    *,
    missing_message: str | None = None,
) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if value is None and missing_message:
        raise VoicePresetError(missing_message)
    raise VoicePresetError(f"{name} must be a table")


def _required_str(preset: dict[str, Any], key: str, preset_name: str) -> str:
    value = preset.get(key)
    if isinstance(value, str) and value:
        return value
    raise VoicePresetError(f"voice preset '{preset_name}' requires string field '{key}'")


def _required_float(preset: dict[str, Any], key: str, preset_name: str) -> float:
    value = preset.get(key)
    if not isinstance(value, bool) and isinstance(value, int | float):
        return float(value)
    raise VoicePresetError(f"voice preset '{preset_name}' requires numeric field '{key}'")


def _validate_speed(speed: float, preset_name: str) -> None:
    if 0.5 <= speed <= 2.0:
        return
    raise VoicePresetError(
        f"voice preset '{preset_name}' speed must be between 0.5 and 2.0"
    )
