"""Independent experimental overlap-add scheduling for a short-window enhancer.

No vendor code or weights. The callback owns its model. This scheduler preserves
length and channels, covers both ends, and blends overlapping predictions.
"""
import numpy as np


def enhance_overlap(audio, enhance_chunk, window, hop, *, output_range=None):
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim != 2 or not audio.size or not np.isfinite(audio).all():
        raise ValueError("Expected finite nonempty samples-by-channels audio")
    if not isinstance(window, int) or not isinstance(hop, int) or not 1 <= hop <= window // 2:
        raise ValueError("Require integer window and hop with 1 <= hop <= window/2")
    output_start, output_stop = output_range if output_range is not None else (0, len(audio))
    if not 0 <= output_start < output_stop <= len(audio):
        raise ValueError("Output range must contain samples within the input")
    frames = output_stop - output_start
    output = np.zeros((frames, audio.shape[1]), dtype=np.float64)
    weights = np.zeros(frames, dtype=np.float64)
    taper = np.sin(np.pi * (np.arange(window) + .5) / window) ** 2
    # Every window ends on a hop boundary. Start with the first one whose
    # end is strictly past the requested start, on the original window grid.
    first_window = (output_start // hop) * hop - window + hop
    for start in range(first_window, output_stop, hop):
        left, right = max(0, start), min(len(audio), start + window)
        chunk = np.zeros((window, audio.shape[1]), dtype=np.float32)
        offset = left - start
        chunk[offset:offset + right - left] = audio[left:right]
        prediction = np.asarray(enhance_chunk(chunk), dtype=np.float64)
        if prediction.shape != chunk.shape or not np.isfinite(prediction).all():
            raise ValueError("Enhancer returned invalid samples or changed chunk shape")
        left, right = max(left, output_start), min(right, output_stop)
        window_slice = slice(left - start, right - start)
        output_slice = slice(left - output_start, right - output_start)
        w = taper[window_slice]
        output[output_slice] += prediction[window_slice] * w[:, None]
        weights[output_slice] += w
    if np.any(weights <= 0):
        raise RuntimeError("Uncovered output samples")
    return (output / weights[:, None]).astype(np.float32)
