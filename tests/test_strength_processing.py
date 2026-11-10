import hashlib
import logging
from pathlib import Path

from src import main


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_strength_zero_is_an_exact_copy(monkeypatch, tmp_path: Path):
    source = tmp_path / "source.mov"
    source.write_bytes(b"synthetic-container\x00audio-and-video")
    monkeypatch.setattr(main, "probe_media", lambda *_args, **_kwargs: {"duration_seconds": 1.0, "audio_streams": 1})
    logger = logging.getLogger("test.strength.zero")

    result = main.process_video(source, tmp_path / "out", "reference", "clearvoice", logger, False, False, 0)
    output = Path(result["output"])

    assert output.read_bytes() == source.read_bytes()
    assert _digest(output) == _digest(source)
    assert result["engine"] == "bypass"
    assert result["verification"]["exact_source_copy"] is True


def test_zero_wav_export_needs_no_ai_runtime(monkeypatch, tmp_path):
    import numpy as np
    import soundfile as sf

    source = tmp_path / "source.mov"
    source.touch()
    monkeypatch.setattr(main, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(main, "probe_media", lambda *_args: {"audio_streams": 1})
    monkeypatch.setattr(main, "select_engine", lambda *_args: (_ for _ in ()).throw(AssertionError("AI must not load")))
    monkeypatch.setattr(main, "analyze_audio", lambda *_args: {})
    def extract(_source, destination, _logger, **_kwargs):
        sf.write(destination, np.zeros((480, 2)), 48000)
    monkeypatch.setattr(main, "extract_audio", extract)
    result = main.process_video(source, tmp_path / "out", "reference", "clearvoice", logging.getLogger(), False, False, 0, audio_only=True)
    assert result["engine"] == "bypass"
    assert sf.info(result["output"]).frames == 480
