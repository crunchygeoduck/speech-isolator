from __future__ import annotations

from pathlib import Path
from copy import deepcopy
import sys
from typing import Any

import yaml


RESOURCE_ROOT = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent.parent))
PROJECT_ROOT = Path(sys.executable).resolve().parent if getattr(sys, "frozen", False) else Path(__file__).resolve().parent.parent
AUDIO_EXTENSIONS = {".wav", ".flac", ".mp3", ".m4a", ".aac", ".ogg", ".aif", ".aiff", ".wma"}
SUPPORTED_EXTENSIONS = {".mov", ".mp4", ".mkv"} | AUDIO_EXTENSIONS
PRESET_NAMES = ("reference", "natural", "strong", "maximum")


def load_preset(name: str, presets_dir: Path | None = None) -> dict[str, Any]:
    if name not in PRESET_NAMES:
        raise ValueError(f"Unknown preset {name!r}; choose from {', '.join(PRESET_NAMES)}")
    path = (presets_dir or RESOURCE_ROOT / "presets") / f"{name}.yaml"
    with path.open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise ValueError(f"Preset is not a mapping: {path}")
    return data


def validate_strength(value: int | float) -> int:
    """Return a whole-number processing strength in the public 0-100 range."""
    strength = int(round(float(value)))
    if not 0 <= strength <= 100:
        raise ValueError("Strength must be between 0 and 100")
    return strength


def settings_for_strength(value: int | float, presets_dir: Path | None = None) -> dict[str, Any]:
    """Derive slider settings while preserving Reference exactly at strength 50."""
    strength = validate_strength(value)
    settings = deepcopy(load_preset("reference", presets_dir))
    settings["name"] = f"strength_{strength}"
    settings["strength"] = strength

    texture = settings.setdefault("speech_texture_blend", {})
    suppression = settings.setdefault("noise_suppression", {})
    if strength < 50:
        # Gradually restore the source; strength 0 itself is handled as an exact
        # container copy by process_video rather than passing through this chain.
        texture["amount"] = 0.04 + (1.0 - strength / 50.0) * 0.96
        texture["gate_dry"] = True
        texture["range"] = 0.001 + 0.999 * (1.0 - strength / 50.0)
        suppression.update(enabled=False, reduction_db=0.0)
    elif strength > 50:
        scale = (strength - 50.0) / 50.0
        texture["amount"] = 0.04 * (1.0 - scale)
        texture["gate_dry"] = True
        suppression.update(enabled=True, reduction_db=8.0 * scale)
        suppression["protect_speech"] = True
    return settings
