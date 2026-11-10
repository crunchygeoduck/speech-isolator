"""Session-local, serialized preview cache. Never writes to the export folder."""
from pathlib import Path
import hashlib
import tempfile
import time

import soundfile as sf

from .config import load_preset, settings_for_strength, validate_strength
from .media import extract_audio
from .processing import (ClearVoiceEngine, LegacyClearVoiceEngine, DeepFilterNetEngine, blend_speech_detail,
                         post_process, validate_enhanced_audio)


def prepare_smooth_excerpt(source, raw, isolated, start, duration, track, logger):
    """Use the same context windows for an excerpt as a full-file export."""
    engine = ClearVoiceEngine()
    first = round(start * engine.sample_rate)
    last = first + round(duration * engine.sample_rate)
    hop = engine.hop_samples
    overlap = engine.window_samples - hop
    context_start = max(0, first // hop * hop - overlap)
    context_end = (last + hop - 1) // hop * hop + overlap

    with tempfile.TemporaryDirectory(prefix="preview_context_", dir=raw.parent) as directory:
        context = Path(directory) / "context.wav"
        extract_audio(source, context, logger, audio_track=track, sample_range=(context_start, context_end))
        with sf.SoundFile(context) as decoded:
            left = first - context_start
            right = min(last - context_start, decoded.frames)
            if left >= right:
                raise ValueError("Preview start is past the end of the selected audio track")
            decoded.seek(left)
            excerpt = decoded.read(right - left, always_2d=True, dtype="float32")
            sf.write(raw, excerpt, decoded.samplerate, subtype="PCM_24")
        engine.process(context, isolated, load_preset("reference")["speech_isolation"], logger,
                       output_range=(left, right))
        validate_enhanced_audio(raw, isolated)


class PreviewCache:
    def __init__(self):
        self._temp = tempfile.TemporaryDirectory(prefix="factory_preview_", ignore_cleanup_errors=True)
        self.root = Path(self._temp.name)

    def close(self):
        self._temp.cleanup()

    def render(self, source, start, duration, track, detail, strength, logger):
        started = time.perf_counter()
        source = Path(source).resolve()
        stat = source.stat()
        identity = (str(source), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns,
                    float(start), float(duration), int(track))
        folder = self.root / hashlib.sha256(repr(identity).encode()).hexdigest()
        folder.mkdir(exist_ok=True)
        raw, isolated = folder / "original.wav", folder / "isolated.wav"
        strength = validate_strength(strength)
        if not (raw.exists() and isolated.exists()):
            pending_raw, pending_isolated = folder / "extracting.wav", folder / "isolating.wav"
            prepare_smooth_excerpt(source, pending_raw, pending_isolated, start, duration, track, logger)
            pending_raw.replace(raw)
            pending_isolated.replace(isolated)
        reference = folder / "reference.wav"
        if not reference.exists():
            self._finish(isolated, raw, reference, 50, logger)
        legacy = folder / "legacy_reference.wav"
        if not legacy.exists():
            pending = folder / "legacy_isolated.wav"
            LegacyClearVoiceEngine().process(raw, pending, load_preset("reference")["speech_isolation"], logger)
            validate_enhanced_audio(raw, pending)
            self._finish(pending, raw, legacy, 50, logger)
        wet = isolated
        if detail and strength > 0:
            wet = folder / "detail.wav"
            if not wet.exists():
                secondary = folder / "secondary.wav"
                DeepFilterNetEngine().process(raw, secondary, {"attenuation_limit_db": 35, "post_filter": False}, logger)
                validate_enhanced_audio(raw, secondary)
                a, rate = sf.read(isolated, always_2d=True, dtype="float32")
                b, _ = sf.read(secondary, always_2d=True, dtype="float32")
                mixed, _ = blend_speech_detail(a, b, rate)
                sf.write(wet, mixed, rate, subtype="FLOAT")
        selected = raw if strength == 0 else folder / f"selected_{int(detail)}_{strength}.wav"
        if not selected.exists():
            self._finish(wet, raw, selected, strength, logger)
        variants = {}
        for label, path in {"Original": raw, "Reference 50": reference, "Previous Reference": legacy, "Live preview": selected}.items():
            playback = path.with_name(path.stem + "_playback.wav")
            if not playback.exists():
                data, rate = sf.read(path, always_2d=True)
                sf.write(playback, data, rate, subtype="PCM_16")
            variants[label] = playback
        return {"variants": variants,
                "strength": strength, "detail": detail,
                "seconds": time.perf_counter() - started}

    @staticmethod
    def _finish(wet, raw, destination, strength, logger):
        pending = destination.with_name("rendering.wav")
        post_process(wet, pending, settings_for_strength(strength), logger, raw)
        validate_enhanced_audio(raw, pending)
        pending.replace(destination)
