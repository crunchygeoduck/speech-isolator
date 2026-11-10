import numpy as np
import pytest
import soundfile as sf

from src.processing import speech_protection_envelope, validate_enhanced_audio


def test_protection_covers_word_boundaries_and_releases_in_gaps():
    rate = 48000
    audio = np.zeros((rate * 2, 2), dtype=np.float32)
    audio[rate // 2:rate] = 0.05
    envelope = speech_protection_envelope(audio, rate)
    assert np.isfinite(envelope).all()
    assert envelope[rate // 4] == 0
    assert envelope[rate * 3 // 4] > 0.99
    assert envelope[rate + 500] > 0.9
    assert envelope[rate * 3 // 2] == 0


def test_model_validation_rejects_truncated_output(tmp_path):
    source, result = tmp_path / "source.wav", tmp_path / "result.wav"
    sf.write(source, np.zeros((48000, 2)), 48000)
    sf.write(result, np.zeros((24000, 2)), 48000)
    with pytest.raises(ValueError, match="duration"):
        validate_enhanced_audio(source, result)
