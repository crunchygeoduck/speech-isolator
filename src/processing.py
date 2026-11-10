from __future__ import annotations

import logging
import importlib.util
import os
import shutil
import sys
import tempfile
from abc import ABC, abstractmethod
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.ndimage import maximum_filter1d, uniform_filter1d

from .audio_analysis import detect_tonal_peaks
from .media import require_binary, run_command


class SpeechIsolationEngine(ABC):
    name = "abstract"

    @abstractmethod
    def available(self) -> bool: ...

    @abstractmethod
    def process(self, input_wav: Path, output_wav: Path, settings: dict[str, Any], logger: logging.Logger) -> None: ...


class DeepFilterNetEngine(SpeechIsolationEngine):
    name = "deepfilternet"

    def available(self) -> bool:
        return importlib.util.find_spec("df") is not None and importlib.util.find_spec("torch") is not None

    def process(self, input_wav: Path, output_wav: Path, settings: dict[str, Any], logger: logging.Logger) -> None:
        attenuation = int(round(min(100.0, max(0.0, float(settings.get("attenuation_limit_db", 20))))))
        post_filter = bool(settings.get("post_filter"))
        logger.info("DeepFilterNet3: attenuation_limit=%s dB post_filter=%s", attenuation, post_filter)
        try:
            import torch
            from df.enhance import enhance, init_df
        except ImportError as exc:
            raise RuntimeError("DeepFilterNet and its PyTorch runtime are not installed") from exc
        model_dir = _bundled_model_dir()
        cache_key = (str(model_dir) if model_dir else "default", post_filter)
        if cache_key not in _DF_MODELS:
            logger.info("Loading DeepFilterNet3 model (%s)", "bundled" if model_dir else "user cache")
            _DF_MODELS[cache_key] = init_df(
                model_base_dir=str(model_dir) if model_dir else None,
                post_filter=post_filter,
                # Windowed PyInstaller apps have no stdout/stderr; DeepFilterNet's
                # logger otherwise attempts to register a None sink.
                log_level="NONE",
                log_file=str(output_wav.parent / "deepfilternet.log"),
                default_model="DeepFilterNet3",
            )[:2]
        model, df_state = _DF_MODELS[cache_key]
        audio, sample_rate = sf.read(input_wav, always_2d=True, dtype="float32")
        if sample_rate != 48000:
            raise ValueError(f"DeepFilterNet requires 48 kHz input; got {sample_rate}")
        tensor = torch.from_numpy(np.ascontiguousarray(audio.T))
        enhanced = enhance(model, df_state, tensor, pad=True, atten_lim_db=attenuation)
        sf.write(output_wav, enhanced.numpy().T, sample_rate, subtype="PCM_24")


_DF_MODELS: dict[tuple[str, bool], tuple[Any, Any]] = {}


class ClearVoiceEngine(SpeechIsolationEngine):
    """Approved 12-second, 50%-overlap full-band speech enhancement."""

    name = "clearvoice"
    sample_rate = 48000
    window_samples = 12 * sample_rate
    hop_samples = 6 * sample_rate

    def available(self) -> bool:
        return importlib.util.find_spec("clearvoice") is not None and _clearvoice_model_dir() is not None

    def _load_model(self, logger: logging.Logger) -> None:
        global _CLEARVOICE_MODEL
        try:
            from clearvoice import ClearVoice
        except ImportError as exc:
            raise RuntimeError("ClearVoice and its PyTorch runtime are not installed") from exc
        runtime_root = _clearvoice_runtime_root()
        if runtime_root is None:
            raise RuntimeError("The MossFormer2 48 kHz checkpoint is missing")
        if _CLEARVOICE_MODEL is None:
            logger.info("Loading ClearVoice MossFormer2 48 kHz model")
            previous = Path.cwd()
            try:
                # ClearVoice 0.1.x resolves its checkpoint path relative to cwd.
                os.chdir(runtime_root)
                with open(os.devnull, "w", encoding="utf-8") as sink:
                    with redirect_stdout(sys.stdout or sink), redirect_stderr(sys.stderr or sink):
                        _CLEARVOICE_MODEL = ClearVoice(
                            task="speech_enhancement",
                            model_names=["MossFormer2_SE_48K"],
                        )
            finally:
                os.chdir(previous)
    def process(self, input_wav: Path, output_wav: Path, settings: dict[str, Any], logger: logging.Logger,
                *, output_range: tuple[int, int] | None = None) -> None:
        from .overlap import enhance_overlap
        from clearvoice.utils.decode import decode_one_audio_mossformer2_se_48k
        import torch
        self._load_model(logger)
        audio, rate = sf.read(input_wav, always_2d=True, dtype="float32")
        if rate != self.sample_rate:
            raise ValueError("Smooth-context isolation requires 48 kHz audio")
        model = _CLEARVOICE_MODEL.models[0]
        def enhance(chunk):
            channels = []
            with torch.no_grad():
                for channel in chunk.T:
                    channels.append(decode_one_audio_mossformer2_se_48k(
                        model.model, model.device, channel[None, :].copy(), model.args))
            logger.info("Smooth-context: completed a 12-second window")
            return np.stack(channels, axis=1)
        result = enhance_overlap(audio, enhance, self.window_samples, self.hop_samples, output_range=output_range)
        sf.write(output_wav, result, rate, subtype="FLOAT")

    def _legacy_process(self, input_wav: Path, output_wav: Path, settings: dict[str, Any], logger: logging.Logger) -> None:
        self._load_model(logger)
        logger.info("ClearVoice: running legacy Reference enhancement")
        info = sf.info(input_wav)
        args = _CLEARVOICE_MODEL.models[0].args
        padding = clearvoice_tail_padding(info.frames, info.samplerate,
                                          args.decode_window, args.one_time_decode_length)
        if padding:
            # The upstream segmented decoder discards both window edges. If
            # input ends exactly on its grid, the last discarded edge is never
            # filled. Extra right context forces another window; crop afterward.
            logger.info("Protecting the final speech tail with %s padding samples", padding)
            with tempfile.TemporaryDirectory(prefix="clearvoice_tail_", dir=output_wav.parent) as folder:
                padded_input, padded_output = Path(folder) / "input.wav", Path(folder) / "output.wav"
                audio, rate = sf.read(input_wav, always_2d=True, dtype="float32")
                sf.write(padded_input, np.pad(audio, ((0, padding), (0, 0))), rate, subtype="FLOAT")
                self._run(padded_input, padded_output)
                enhanced_audio, enhanced_rate = sf.read(padded_output, always_2d=True, dtype="float32")
                if enhanced_rate != rate or len(enhanced_audio) < info.frames or enhanced_audio.shape[1] != info.channels:
                    raise RuntimeError("ClearVoice returned invalid padded audio dimensions")
                if not np.isfinite(enhanced_audio).all():
                    raise RuntimeError("ClearVoice returned non-finite padded audio")
                sf.write(output_wav, enhanced_audio[:info.frames], rate, subtype="FLOAT")
        else:
            self._run(input_wav, output_wav)
        if not output_wav.is_file() or output_wav.stat().st_size == 0:
            raise RuntimeError("ClearVoice did not create an output WAV")

    @staticmethod
    def _run(input_wav: Path, output_wav: Path) -> None:
        # Windowed PyInstaller apps expose stdout/stderr as None; tqdm expects a
        # writable stream even though the GUI reports progress through logging.
        with open(os.devnull, "w", encoding="utf-8") as sink:
            with redirect_stdout(sys.stdout or sink), redirect_stderr(sys.stderr or sink):
                enhanced = _CLEARVOICE_MODEL(input_path=str(input_wav), online_write=False)
        _CLEARVOICE_MODEL.write(enhanced, output_path=str(output_wav))


def clearvoice_tail_padding(samples: int, rate: int, window_seconds: float = 4,
                           threshold_seconds: float = 20) -> int:
    """Protect the upstream exact-window-grid tail case; otherwise no change."""
    window = int(rate * window_seconds)
    stride = int(window * .75)
    if samples < 1 or rate != 48000 or window < 4 or stride < 1:
        raise ValueError("Expected nonempty 48 kHz audio and a valid decoding window")
    if samples > rate * threshold_seconds and samples >= window and (samples - window) % stride == 0:
        return 384  # One STFT hop breaks the exact-grid case without shifting audio.
    return 0


_CLEARVOICE_MODEL: Any | None = None


class LegacyClearVoiceEngine(ClearVoiceEngine):
    """Previous Reference, retained for explicit comparison and rollback."""
    name = "clearvoice-legacy"
    process = ClearVoiceEngine._legacy_process


def blend_speech_detail(primary: np.ndarray, secondary: np.ndarray, rate: int, amount: float = 0.25) -> tuple[np.ndarray, dict[str, Any]]:
    """Recover bounded complementary model detail only around enhanced speech."""
    if primary.shape != secondary.shape or primary.ndim != 2 or not primary.size:
        raise ValueError("Detail models must return matching nonempty audio arrays")
    if not np.isfinite(primary).all() or not np.isfinite(secondary).all():
        raise ValueError("Non-finite samples in detail recovery")
    amount = min(0.4, max(0.0, amount))
    envelope = speech_protection_envelope(primary, rate)
    active = envelope > 0.8
    a, b = primary[active].astype(np.float64), secondary[active].astype(np.float64)
    cross = np.sum(a * b, axis=0)
    aa, bb = np.sum(a * a, axis=0), np.sum(b * b, axis=0)
    agreement = cross / np.sqrt(np.maximum(aa * bb, 1e-24))
    gains = np.clip(cross / np.maximum(bb, 1e-12), .25, 4)
    # A poorly aligned or unrelated secondary channel must not cancel speech.
    enabled = (agreement >= 0.5) & (bb > 1e-10)
    blend = amount * envelope[:, None] * enabled[None, :]
    mixed = primary + blend * (secondary * gains[None, :] - primary)
    return mixed.astype(np.float32), {"amount": amount, "gains": gains.tolist(),
                                      "channel_agreement": agreement.tolist(), "enabled_channels": enabled.tolist()}


class DetailRecoveryEngine(SpeechIsolationEngine):
    """Optional two-model speech enhancement; Reference remains independent."""

    name = "clearvoice-detail"

    def available(self) -> bool:
        return ClearVoiceEngine().available() and DeepFilterNetEngine().available()

    def process(self, input_wav: Path, output_wav: Path, settings: dict[str, Any], logger: logging.Logger) -> None:
        with tempfile.TemporaryDirectory(prefix="detail_models_", dir=output_wav.parent) as folder:
            primary, secondary = Path(folder) / "primary.wav", Path(folder) / "secondary.wav"
            ClearVoiceEngine().process(input_wav, primary, settings, logger)
            DeepFilterNetEngine().process(input_wav, secondary, {"attenuation_limit_db": 35, "post_filter": False}, logger)
            validate_enhanced_audio(input_wav, primary)
            validate_enhanced_audio(input_wav, secondary)
            a, rate = sf.read(primary, always_2d=True, dtype="float32")
            b, _ = sf.read(secondary, always_2d=True, dtype="float32")
            result, details = blend_speech_detail(a, b, rate)
            sf.write(output_wav, result, rate, subtype="FLOAT")
            logger.info("Independent detail recovery: %s", details)


def _clearvoice_runtime_root() -> Path | None:
    bundle = getattr(sys, "_MEIPASS", None)
    candidates = [Path(bundle)] if bundle else []
    candidates += [Path(sys.executable).resolve().parent, Path(__file__).resolve().parent.parent]
    for candidate in candidates:
        checkpoint = candidate / "checkpoints" / "MossFormer2_SE_48K" / "last_best_checkpoint.pt"
        if checkpoint.is_file() and checkpoint.stat().st_size > 1_000_000:
            return candidate
    return None


def _clearvoice_model_dir() -> Path | None:
    root = _clearvoice_runtime_root()
    return root / "checkpoints" / "MossFormer2_SE_48K" if root else None


def _bundled_model_dir() -> Path | None:
    candidates = []
    bundle = getattr(sys, "_MEIPASS", None)
    if bundle:
        candidates.append(Path(bundle) / "models" / "DeepFilterNet3")
    candidates.append(Path(sys.executable).resolve().parent / "models" / "DeepFilterNet3")
    for candidate in candidates:
        if (candidate / "config.ini").exists() and (candidate / "checkpoints").is_dir():
            return candidate
    return None


class FfmpegDenoiseEngine(SpeechIsolationEngine):
    """Portable spectral-denoise baseline; useful for fallback and comparisons, not AI."""

    name = "ffmpeg-afftdn"

    def available(self) -> bool:
        return shutil.which("ffmpeg") is not None

    def process(self, input_wav: Path, output_wav: Path, settings: dict[str, Any], logger: logging.Logger) -> None:
        strength = min(1.0, max(0.0, float(settings.get("strength", 0.7))))
        reduction = 4 + strength * 10
        command = [require_binary("ffmpeg"), "-hide_banner", "-y", "-i", str(input_wav), "-af", f"afftdn=nr={reduction:.1f}:nf=-35:tn=1", "-c:a", "pcm_s24le", str(output_wav)]
        run_command(command, logger)


class PassthroughEngine(SpeechIsolationEngine):
    name = "passthrough"

    def available(self) -> bool:
        return True

    def process(self, input_wav: Path, output_wav: Path, settings: dict[str, Any], logger: logging.Logger) -> None:
        shutil.copy2(input_wav, output_wav)


def select_engine(requested: str, logger: logging.Logger) -> SpeechIsolationEngine:
    choices = {
        "clearvoice": ClearVoiceEngine(),
        "clearvoice-legacy": LegacyClearVoiceEngine(),
        "clearvoice-detail": DetailRecoveryEngine(),
        "deepfilternet": DeepFilterNetEngine(),
        "ffmpeg": FfmpegDenoiseEngine(),
        "passthrough": PassthroughEngine(),
    }
    if requested == "auto":
        for name in ("clearvoice", "deepfilternet", "ffmpeg"):
            if choices[name].available():
                logger.info("Automatically selected speech engine: %s", choices[name].name)
                return choices[name]
        raise RuntimeError("No speech processing engine is available")
    engine = choices[requested]
    if not engine.available():
        raise RuntimeError(f"Requested engine is unavailable: {requested}")
    return engine


def post_process(
    input_wav: Path,
    output_wav: Path,
    preset: dict[str, Any],
    logger: logging.Logger,
    dry_source_wav: Path | None = None,
) -> list[float]:
    secondary = preset.get("noise_suppression", {})
    if secondary.get("enabled") and secondary.get("protect_speech"):
        from copy import deepcopy
        with tempfile.TemporaryDirectory(prefix="speech_protection_", dir=output_wav.parent) as folder:
            protected = Path(folder) / "protected.wav"
            protect_speech_denoise(input_wav, protected, float(secondary.get("reduction_db", 4)), logger)
            finishing = deepcopy(preset)
            finishing["noise_suppression"]["enabled"] = False
            return post_process(protected, output_wav, finishing, logger, dry_source_wav)
    texture = preset.get("speech_texture_blend", {})
    if texture.get("enabled") and dry_source_wav is not None:
        _speech_texture_post_process(input_wav, dry_source_wav, output_wav, preset, logger)
        return []

    filters: list[str] = []
    high_pass = preset.get("high_pass", {})
    if high_pass.get("enabled"):
        frequency = min(150.0, max(40.0, float(high_pass.get("frequency", 80))))
        filters.append(f"highpass=f={frequency:g}:p=2")

    detected: list[float] = []
    notch = preset.get("notch_filter", {})
    if notch.get("auto_detect"):
        detected = detect_tonal_peaks(input_wav, min(3, int(notch.get("max_filters", 3))))
        logger.info("Detected persistent tonal peaks (Hz): %s", detected or "none")
    if notch.get("enabled"):
        reduction = min(6.0, max(0.0, float(notch.get("max_reduction_db", 6))))
        q = min(20.0, max(5.0, float(notch.get("quality_factor", 12))))
        for frequency in detected:
            width = max(4.0, frequency / q)
            filters.append(f"equalizer=f={frequency:.2f}:t=h:w={width:.2f}:g=-{reduction:g}")

    eq = preset.get("eq", {})
    if eq.get("enabled"):
        mud = min(3.0, max(0.0, float(eq.get("mud_reduction_db", 1.5))))
        presence = min(3.0, max(0.0, float(eq.get("presence_gain_db", 1.5))))
        filters += [f"equalizer=f=250:t=q:w=1:g=-{mud:g}", f"equalizer=f=3000:t=q:w=0.8:g={presence:g}"]

    secondary = preset.get("noise_suppression", {})
    if secondary.get("enabled"):
        reduction = min(10.0, max(0.0, float(secondary.get("reduction_db", 4))))
        filters.append(f"afftdn=nr={reduction:g}:nf=-38:tn=1")

    compressor = preset.get("compressor", {})
    if compressor.get("enabled"):
        ratio = min(4.0, max(1.0, float(compressor.get("ratio", 2.5))))
        threshold = min(-6.0, max(-30.0, float(compressor.get("threshold_db", -18))))
        attack = min(100.0, max(1.0, float(compressor.get("attack_ms", 20))))
        release = min(500.0, max(50.0, float(compressor.get("release_ms", 160))))
        filters.append(f"acompressor=threshold={threshold:g}dB:ratio={ratio:g}:attack={attack:g}:release={release:g}:makeup=1dB:knee=2.828")

    normalization = preset.get("normalization", {})
    if normalization.get("enabled"):
        target = min(-12.0, max(-24.0, float(normalization.get("target_lufs", -16))))
        peak = min(-1.0, max(-6.0, float(normalization.get("true_peak_db", -1.5))))
        lra = min(20.0, max(1.0, float(normalization.get("loudness_range", 11))))
        filters.append(f"loudnorm=I={target:g}:TP={peak:g}:LRA={lra:g}")

    if not filters:
        shutil.copy2(input_wav, output_wav)
        return detected
    command = [require_binary("ffmpeg"), "-hide_banner", "-y", "-i", str(input_wav), "-af", ",".join(filters), "-ar", "48000", "-c:a", "pcm_s24le", str(output_wav)]
    run_command(command, logger)
    return detected


def _speech_texture_post_process(
    isolated_wav: Path,
    dry_wav: Path,
    output_wav: Path,
    preset: dict[str, Any],
    logger: logging.Logger,
) -> None:
    """Restore a trace of source texture only while enhanced speech is present."""
    texture = preset.get("speech_texture_blend", {})
    amount = min(1.0, max(0.0, float(texture.get("amount", 0.04))))
    gate_dry = bool(texture.get("gate_dry", True))
    threshold = min(0.12, max(0.005, float(texture.get("threshold", 0.04))))
    ratio = min(100.0, max(2.0, float(texture.get("ratio", 20))))
    gate_range = min(1.0, max(0.0001, float(texture.get("range", 0.001))))
    attack = min(100.0, max(0.1, float(texture.get("attack_ms", 5))))
    release = min(500.0, max(10.0, float(texture.get("release_ms", 80))))

    high_pass = min(150.0, max(30.0, float(preset.get("high_pass", {}).get("frequency", 45))))
    eq = preset.get("eq", {})
    body = min(4.0, max(-4.0, float(eq.get("body_gain_db", 2.5))))
    low_mid = min(4.0, max(-4.0, float(eq.get("low_mid_gain_db", -1.2))))
    presence = min(4.0, max(-4.0, float(eq.get("presence_gain_db", -2.5))))
    air = min(6.0, max(-6.0, float(eq.get("air_gain_db", -3.5))))

    normalization = preset.get("normalization", {})
    target = min(-12.0, max(-24.0, float(normalization.get("target_lufs", -21))))
    peak = min(-1.0, max(-6.0, float(normalization.get("true_peak_db", -4))))
    lra = min(20.0, max(1.0, float(normalization.get("loudness_range", 11))))
    wet_amount = 1.0 - amount
    secondary = preset.get("noise_suppression", {})
    reduction = min(8.0, max(0.0, float(secondary.get("reduction_db", 0)))) if secondary.get("enabled") else 0.0
    wet_filters = (
        f"highpass=f={high_pass:g}:p=2,"
        f"equalizer=f=350:t=q:w=0.9:g={body:g},"
        f"equalizer=f=650:t=q:w=1.2:g={low_mid:g},"
        f"equalizer=f=3000:t=q:w=0.9:g={presence:g},"
        f"highshelf=f=8000:g={air:g}"
    )
    if reduction > 0:
        wet_filters += f",afftdn=nr={reduction:g}:nf=-38:tn=1"
    if gate_dry:
        graph = (
            "[0:a]asplit=2[wet0][key];"
            f"[wet0]{wet_filters},volume={wet_amount:g}[wet];"
            f"[1:a][key]sidechaingate=threshold={threshold:g}:ratio={ratio:g}:range={gate_range:g}:"
            f"attack={attack:g}:release={release:g}:detection=rms,volume={amount:g}[dry];"
            f"[wet][dry]amix=inputs=2:normalize=0,loudnorm=I={target:g}:TP={peak:g}:LRA={lra:g}[out]"
        )
    else:
        graph = (
            f"[0:a]{wet_filters},volume={wet_amount:g}[wet];"
            f"[1:a]volume={amount:g}[dry];"
            f"[wet][dry]amix=inputs=2:normalize=0,loudnorm=I={target:g}:TP={peak:g}:LRA={lra:g}[out]"
        )
    logger.info(
        "Speech texture restoration: %.1f%% %s dry blend; secondary suppression=%.1f dB",
        amount * 100,
        "gated" if gate_dry else "continuous",
        reduction,
    )
    command = [
        require_binary("ffmpeg"), "-hide_banner", "-y",
        "-i", str(isolated_wav), "-i", str(dry_wav),
        "-filter_complex", graph, "-map", "[out]",
        "-ar", "48000", "-c:a", "pcm_s24le", str(output_wav),
    ]
    run_command(command, logger)


def validate_enhanced_audio(source: Path, enhanced: Path) -> None:
    """Reject corrupt or truncated model output before it reaches a user export."""
    original, rate = sf.read(source, always_2d=True, dtype="float32")
    result, result_rate = sf.read(enhanced, always_2d=True, dtype="float32")
    if result_rate != rate or result.shape[1] != original.shape[1]:
        raise ValueError("Speech model changed sample rate or channel count")
    if not result.size or not np.isfinite(result).all():
        raise ValueError("Speech model returned empty or non-finite audio")
    if abs(len(result) - len(original)) > rate * 0.05:
        raise ValueError("Speech model changed audio duration by more than 50 ms")


def speech_protection_envelope(audio: np.ndarray, rate: int) -> np.ndarray:
    """Conservative energy cue, shared across channels; not a trained VAD."""
    power = np.mean(np.square(audio.astype(np.float64)), axis=1)
    rms = np.sqrt(np.maximum(0, uniform_filter1d(power, max(1, round(rate * 0.02)))))
    threshold = max(0.002, float(np.percentile(rms, 85)) * 0.15)
    protection = np.clip((rms - threshold * 0.25) / (threshold * 0.75), 0, 1)
    # Hold protection around word boundaries, then smooth to avoid abrupt changes.
    protection = maximum_filter1d(protection, max(1, round(rate * 0.12)))
    return np.clip(uniform_filter1d(protection, max(1, round(rate * 0.04))), 0, 1).astype(np.float32)


def protect_speech_denoise(source: Path, output: Path, reduction: float, logger: logging.Logger) -> None:
    """Apply extra suppression primarily in gaps, retaining 85% during speech."""
    with tempfile.TemporaryDirectory(prefix="denoise_", dir=output.parent) as folder:
        denoised = Path(folder) / "denoised.wav"
        run_command([require_binary("ffmpeg"), "-hide_banner", "-y", "-i", str(source),
                     "-af", f"afftdn=nr={min(8.0, max(0.01, reduction)):g}:nf=-38:tn=1",
                     "-c:a", "pcm_f32le", str(denoised)], logger)
        validate_enhanced_audio(source, denoised)
        original, rate = sf.read(source, always_2d=True, dtype="float32")
        reduced, _ = sf.read(denoised, always_2d=True, dtype="float32")
        if reduced.shape != original.shape:
            raise ValueError("Secondary denoiser changed sample count")
        protection = speech_protection_envelope(original, rate)
        blend = (1.0 - 0.85 * protection)[:, None]
        result = original + blend * (reduced - original)
        sf.write(output, result, rate, subtype="FLOAT")
        logger.info("Speech-protected suppression: %.2f dB; %.1f%% of frames protected (energy estimate)",
                    reduction, float(np.mean(protection > 0.5)) * 100)
