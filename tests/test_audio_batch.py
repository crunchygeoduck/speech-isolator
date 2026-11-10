import json
import logging
from pathlib import Path

import pytest

from src import main, media


def test_discover_mixed_audio_folder(tmp_path):
    for name in ("one.wav", "two.flac", "three.mp3", "four.mov", "one_CLEAN.wav", "notes.txt"):
        (tmp_path / name).touch()
    assert len(main.discover_inputs(tmp_path)) == 4


def test_audio_probe_ignores_cover_art(monkeypatch, tmp_path):
    import subprocess
    raw = {"streams": [{"codec_type": "video", "disposition": {"attached_pic": 1}},
                       {"codec_type": "audio", "codec_name": "mp3", "channels": 2}], "format": {"duration": "10"}}
    monkeypatch.setattr(media, "require_binary", lambda name: name)
    monkeypatch.setattr(media, "run_command", lambda *args, **kwargs: subprocess.CompletedProcess([], 0, json.dumps(raw)))
    assert media.probe_media(tmp_path / "music.mp3", logging.getLogger())["has_video"] is False


def test_failed_replacement_preserves_previous_output(monkeypatch, tmp_path):
    source = tmp_path / "source.wav"
    source.write_bytes(b"original")
    output = tmp_path / "out"
    output.mkdir()
    previous = output / "source_CLEAN.wav"
    previous.write_bytes(b"previous accepted export")
    def fail(_source, staging, *args):
        (staging / "source_CLEAN.wav").write_bytes(b"partial")
        raise RuntimeError("simulated model failure")
    monkeypatch.setattr(main, "_process_media_staged", fail)
    with pytest.raises(RuntimeError):
        main.process_video(source, output, "reference", "clearvoice", logging.getLogger(), False, True, 50)
    assert previous.read_bytes() == b"previous accepted export"
    assert source.read_bytes() == b"original"
    assert list(output.iterdir()) == [previous]


def test_preflight_rejects_colliding_exports(tmp_path):
    with pytest.raises(ValueError, match="collision"):
        main.batch_preflight([tmp_path / "a" / "clip.wav", tmp_path / "b" / "clip.wav"], tmp_path / "out", 50, False)


def test_mixed_formats_have_different_output_names(tmp_path):
    main.batch_preflight([tmp_path / "clip.wav", tmp_path / "clip.mp3"], tmp_path / "out", 50, False)
