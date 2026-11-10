from pathlib import Path

import pytest

from src.config import PRESET_NAMES, load_preset, settings_for_strength, validate_strength
from src.main import clean_output_path, discover_inputs
from src.gui import format_time


def test_all_presets_load():
    for name in PRESET_NAMES:
        preset = load_preset(name)
        assert preset["name"] == name
        assert -24 <= preset["normalization"]["target_lufs"] <= -12


def test_strength_50_is_reference_audio_chain():
    reference = load_preset("reference")
    strength = settings_for_strength(50)
    for section in ("speech_isolation", "noise_suppression", "high_pass", "eq", "speech_texture_blend", "normalization"):
        assert strength[section] == reference[section]


def test_strength_endpoints_are_bounded():
    gentle = settings_for_strength(1)
    maximum = settings_for_strength(100)
    assert gentle["speech_texture_blend"]["amount"] > 0.9
    assert gentle["speech_texture_blend"]["range"] > 0.9
    assert maximum["speech_texture_blend"]["amount"] == 0
    assert maximum["noise_suppression"] == {"enabled": True, "reduction_db": 8.0, "protect_speech": True}


def test_strength_rejects_out_of_range_values():
    for value in (-1, 101):
        try:
            validate_strength(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected {value} to be rejected")


def test_output_does_not_overwrite_source(tmp_path: Path):
    video = tmp_path / "clip.MOV"
    video.touch()
    output = clean_output_path(video, tmp_path)
    assert output.name == "clip_CLEAN.mov"
    assert output != video


def test_discover_inputs_ignores_clean_outputs(tmp_path: Path):
    (tmp_path / "a.mov").touch()
    (tmp_path / "b.mp4").touch()
    (tmp_path / "b_CLEAN.mp4").touch()
    (tmp_path / "notes.txt").touch()
    assert [path.name for path in discover_inputs(tmp_path)] == ["a.mov", "b.mp4"]


def test_rejects_unknown_extension(tmp_path: Path):
    path = tmp_path / "audio.txt"
    path.touch()
    with pytest.raises(ValueError):
        discover_inputs(path)


def test_gui_time_format():
    assert format_time(0) == "00:00"
    assert format_time(65_999) == "01:05"
