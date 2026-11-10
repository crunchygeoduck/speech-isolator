from __future__ import annotations

import logging
import math
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from scipy.signal import find_peaks

from .media import require_binary


def analyze_audio(path: Path, logger: logging.Logger) -> dict[str, Any]:
    audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    mono = audio.mean(axis=1)
    abs_peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    rms = float(np.sqrt(np.mean(np.square(audio)))) if audio.size else 0.0
    metrics = {
        "sample_rate": int(sample_rate),
        "channels": int(audio.shape[1]),
        "duration_seconds": float(len(audio) / sample_rate),
        "peak_dbfs": linear_to_db(abs_peak),
        "rms_dbfs": linear_to_db(rms),
        "clipped_samples": int(np.count_nonzero(np.abs(audio) >= 0.9999)),
        "integrated_lufs": measure_lufs(path),
        "speech_activity_estimate": estimate_activity(mono, sample_rate),
    }
    logger.info("Audio metrics for %s: %s", path.name, metrics)
    return metrics


def detect_tonal_peaks(path: Path, max_filters: int = 3) -> list[float]:
    audio, sample_rate = sf.read(path, always_2d=True, dtype="float32")
    mono = audio.mean(axis=1)
    if len(mono) < sample_rate // 2:
        return []
    window_size = min(65536, 2 ** int(math.log2(len(mono))))
    hop = window_size // 2
    spectra = []
    window = np.hanning(window_size)
    for offset in range(0, len(mono) - window_size + 1, hop):
        spectra.append(np.abs(np.fft.rfft(mono[offset:offset + window_size] * window)))
        if len(spectra) >= 120:
            break
    if not spectra:
        return []
    spectrum = np.median(np.stack(spectra), axis=0)
    freqs = np.fft.rfftfreq(window_size, 1 / sample_rate)
    band = (freqs >= 120) & (freqs <= 4000)
    values = 20 * np.log10(spectrum[band] + 1e-12)
    baseline = float(np.median(values))
    indices, props = find_peaks(values, prominence=10, distance=max(1, int(35 / (sample_rate / window_size))))
    candidates = [(float(props["prominences"][i]), float(freqs[band][idx])) for i, idx in enumerate(indices) if values[idx] > baseline + 12]
    candidates.sort(reverse=True)
    return sorted(freq for _, freq in candidates[:max_filters])


def measure_lufs(path: Path) -> float | None:
    command = [require_binary("ffmpeg"), "-hide_banner", "-nostats", "-i", str(path), "-filter_complex", "ebur128=peak=true", "-f", "null", "-"]
    result = subprocess.run(command, capture_output=True, text=True, encoding="utf-8", errors="replace")
    matches = re.findall(r"I:\s*(-?\d+(?:\.\d+)?)\s*LUFS", result.stderr)
    return float(matches[-1]) if matches else None


def estimate_activity(mono: np.ndarray, sample_rate: int) -> float:
    frame = max(1, int(sample_rate * 0.03))
    usable = len(mono) - len(mono) % frame
    if usable == 0:
        return 0.0
    rms = np.sqrt(np.mean(mono[:usable].reshape(-1, frame) ** 2, axis=1) + 1e-12)
    threshold = max(float(np.percentile(rms, 20)) * 2.5, 10 ** (-50 / 20))
    return round(float(np.mean(rms > threshold)), 4)


def linear_to_db(value: float) -> float:
    return round(20 * math.log10(max(value, 1e-12)), 2)

