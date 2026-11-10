from pathlib import Path

import pytest

from src import media


def _source_info() -> dict:
    return {
        "duration_seconds": 10.0,
        "video_codec": "h264",
        "resolution": "1920x1080",
        "frame_rate": "60/1",
    }


def _output_info() -> dict:
    return {
        "duration_seconds": 10.0,
        "video_codec": "h264",
        "resolution": "1920x1080",
        "frame_rate": "60/1",
        "video_start_seconds": 0.0,
        "audio_start_seconds": 0.0,
    }


def test_verify_muxed_output_accepts_copied_synchronized_video(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(media, "probe_media", lambda *args: _output_info())
    monkeypatch.setattr(media, "media_stream_hash", lambda *args: "abc123")
    result = media.verify_muxed_output(_source_info(), tmp_path / "in.mov", tmp_path / "out.mov", media.logging.getLogger())
    assert result["video_stream_copied"] is True
    assert result["audio_video_start_offset_seconds"] == 0.0


def test_verify_muxed_output_rejects_sync_drift(monkeypatch, tmp_path: Path):
    output = _output_info()
    output["audio_start_seconds"] = 0.2
    monkeypatch.setattr(media, "probe_media", lambda *args: output)
    monkeypatch.setattr(media, "media_stream_hash", lambda *args: "abc123")
    with pytest.raises(RuntimeError, match="start offset"):
        media.verify_muxed_output(_source_info(), tmp_path / "in.mov", tmp_path / "out.mov", media.logging.getLogger())


def test_verify_muxed_output_rejects_changed_video_packets(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(media, "probe_media", lambda *args: _output_info())
    hashes = iter(("source", "output"))
    monkeypatch.setattr(media, "media_stream_hash", lambda *args: next(hashes))
    with pytest.raises(RuntimeError, match="not preserved"):
        media.verify_muxed_output(_source_info(), tmp_path / "in.mov", tmp_path / "out.mov", media.logging.getLogger())
