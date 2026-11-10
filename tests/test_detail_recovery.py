import numpy as np

from src.processing import blend_speech_detail


def test_detail_recovery_preserves_identical_models():
    rng = np.random.default_rng(4)
    audio = rng.normal(0, .02, (48000, 2)).astype(np.float32)
    result, details = blend_speech_detail(audio, audio.copy(), 48000)
    np.testing.assert_allclose(result, audio, atol=1e-7)
    assert details["enabled_channels"] == [True, True]


def test_detail_recovery_rejects_cancelling_model():
    audio = np.full((48000, 2), .05, dtype=np.float32)
    result, details = blend_speech_detail(audio, -audio, 48000)
    np.testing.assert_array_equal(result, audio)
    assert details["enabled_channels"] == [False, False]


def test_detail_recovery_does_not_add_noise_in_silent_primary():
    primary = np.zeros((48000, 2), dtype=np.float32)
    secondary = np.ones_like(primary) * .01
    result, _ = blend_speech_detail(primary, secondary, 48000)
    assert np.isfinite(result).all()
    assert np.count_nonzero(result) == 0
