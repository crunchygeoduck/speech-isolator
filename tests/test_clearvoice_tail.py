import pytest
from src.processing import clearvoice_tail_padding


@pytest.mark.parametrize("seconds", [22, 25, 28, 31, 34])
def test_exact_grid_requires_context(seconds):
    assert clearvoice_tail_padding(seconds * 48000, 48000) == 384


@pytest.mark.parametrize("samples", [1, 48000, 20 * 48000, 21 * 48000, 1472800, 22 * 48000 + 1])
def test_other_lengths_unchanged(samples):
    assert clearvoice_tail_padding(samples, 48000) == 0


def test_other_window_and_invalid_rate():
    assert clearvoice_tail_padding(26 * 48000, 48000, 8, 20) == 384
    with pytest.raises(ValueError):
        clearvoice_tail_padding(48000, 44100)


@pytest.mark.parametrize("seconds", [22, 23])
def test_wrapper_padding_and_crop(monkeypatch, tmp_path, seconds):
    import logging
    import sys
    from types import SimpleNamespace
    import numpy as np
    import soundfile as sf
    from src import processing

    seen = []
    class FakeModel:
        models = [SimpleNamespace(args=SimpleNamespace(decode_window=4, one_time_decode_length=20))]

        def __call__(self, input_path, online_write):
            data, rate = sf.read(input_path, always_2d=True, dtype="float32")
            seen.append(len(data))
            # Simulate the upstream exact-grid tail omission.
            if len(data) > 20 * rate and (len(data) - 4 * rate) % (3 * rate) == 0:
                data[-rate // 2:] = 0
            return data

        def write(self, enhanced, output_path):
            sf.write(output_path, enhanced, 48000, subtype="FLOAT")

    monkeypatch.setitem(sys.modules, "clearvoice", SimpleNamespace(ClearVoice=FakeModel))
    monkeypatch.setattr(processing, "_CLEARVOICE_MODEL", FakeModel())
    monkeypatch.setattr(processing, "_clearvoice_runtime_root", lambda: tmp_path)
    source, output = tmp_path / "input.wav", tmp_path / "output.wav"
    audio = np.full((seconds * 48000, 2), .125, dtype=np.float32)
    sf.write(source, audio, 48000, subtype="FLOAT")
    processing.LegacyClearVoiceEngine().process(source, output, {}, logging.getLogger("test"))
    actual, rate = sf.read(output, always_2d=True)
    np.testing.assert_array_equal(actual, audio)
    assert rate == 48000
    assert seen == [len(audio) + (384 if seconds == 22 else 0)]
    assert not list(tmp_path.glob("clearvoice_tail_*"))
