import logging
import shutil

import numpy as np
import pytest
import soundfile as sf

from src import preview
from src.media import extract_audio
from src.overlap import enhance_overlap


def test_preview_reuses_models_and_invalidates_source(monkeypatch, tmp_path):
    source = tmp_path / "source.wav"
    sf.write(source, np.zeros((4800, 2)), 48000)
    calls = []
    monkeypatch.setattr(preview, "extract_audio", lambda src, dst, *args, **kwargs: shutil.copy2(src, dst))
    def model(self, src, dst, *args, output_range=None):
        calls.append(type(self).__name__)
        if output_range is None:
            shutil.copy2(src, dst)
        else:
            data, rate = sf.read(src, always_2d=True)
            left, right = output_range
            sf.write(dst, data[left:right], rate, subtype="FLOAT")
    monkeypatch.setattr(preview.ClearVoiceEngine, "process", model)
    monkeypatch.setattr(preview.LegacyClearVoiceEngine, "process", model)
    monkeypatch.setattr(preview.DeepFilterNetEngine, "process", model)
    monkeypatch.setattr(preview, "post_process", lambda wet, out, *args: shutil.copy2(wet, out))
    cache = preview.PreviewCache()
    try:
        def render(value, detail=False, start=0):
            return cache.render(source, start, 10, 0, detail, value, logging.getLogger("test"))
        first = render(50)
        assert "Previous Reference" in first["variants"]
        render(80)
        zero = render(0)
        assert sf.read(zero["variants"]["Live preview"])[0].shape == (4800, 2)
        assert calls == ["ClearVoiceEngine", "LegacyClearVoiceEngine"]
        render(50, True)
        render(80, True)
        assert calls == ["ClearVoiceEngine", "LegacyClearVoiceEngine", "DeepFilterNetEngine"]
        assert render(50)["variants"] == first["variants"]
        render(50, start=0.025)
        assert calls.count("ClearVoiceEngine") == 2
        sf.write(source, np.zeros((9600, 2)), 48000)
        render(50)
        assert calls.count("ClearVoiceEngine") == 3
    finally:
        cache.close()
    assert not cache.root.exists()


def test_failed_render_is_not_cached(monkeypatch, tmp_path):
    raw = tmp_path / "raw.wav"
    final = tmp_path / "selected.wav"
    sf.write(raw, np.zeros(4800), 48000)
    def fail(wet, out, *args):
        out.write_bytes(b"partial")
        raise RuntimeError("render failed")
    monkeypatch.setattr(preview, "post_process", fail)
    import pytest
    with pytest.raises(RuntimeError):
        preview.PreviewCache._finish(raw, raw, final, 50, logging.getLogger("test"))
    assert not final.exists()


@pytest.mark.parametrize("start,duration", [(0, 2), (13.2, 4.1), (24.1, 3)])
def test_smooth_excerpt_matches_full_window_context(monkeypatch, tmp_path, start, duration):
    source, full = tmp_path / "source.wav", tmp_path / "full.wav"
    raw, isolated = tmp_path / "raw.wav", tmp_path / "isolated.wav"
    logger = logging.getLogger("test")
    # Exercise FFmpeg resampling and a context-sensitive stand-in for the model.
    source_rate = 44100
    time = np.arange(25 * source_rate) / source_rate
    sf.write(source, .1 * np.sin(2 * np.pi * 317 * time) + .001 * time, source_rate, subtype="PCM_24")
    extract_audio(source, full, logger)
    full_audio, rate = sf.read(full, always_2d=True, dtype="float32")

    def model(chunk):
        return chunk + np.mean(chunk, axis=0)

    def process(self, src, dst, settings, logger, *, output_range=None):
        audio, rate = sf.read(src, always_2d=True, dtype="float32")
        result = enhance_overlap(audio, model, self.window_samples, self.hop_samples,
                                 output_range=output_range)
        sf.write(dst, result, rate, subtype="FLOAT")

    monkeypatch.setattr(preview.ClearVoiceEngine, "process", process)
    preview.prepare_smooth_excerpt(source, raw, isolated, start, duration, 0, logger)
    full_isolated = enhance_overlap(full_audio, model, 12 * rate, 6 * rate)
    left = round(start * rate)
    right = left + round(duration * rate)
    np.testing.assert_array_equal(sf.read(raw, always_2d=True, dtype="float32")[0], full_audio[left:right])
    np.testing.assert_array_equal(sf.read(isolated, always_2d=True, dtype="float32")[0], full_isolated[left:right])
