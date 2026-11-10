from __future__ import annotations

import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence


class CommandError(RuntimeError):
    pass


def require_binary(name: str) -> str:
    executable = f"{name}.exe" if sys.platform == "win32" and not name.lower().endswith(".exe") else name
    bundle = getattr(sys, "_MEIPASS", None)
    candidates = [Path(sys.executable).resolve().parent / executable]
    if bundle:
        candidates.insert(0, Path(bundle) / executable)
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate)
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Required program is not on PATH: {name}")
    return path


def run_command(command: Sequence[str], logger: logging.Logger, *, capture: bool = False) -> subprocess.CompletedProcess[str]:
    printable = subprocess.list2cmdline(list(command))
    logger.debug("Command: %s", printable)
    result = subprocess.run(list(command), text=True, capture_output=True, encoding="utf-8", errors="replace")
    diagnostic = "\n".join(part.strip() for part in (result.stdout, result.stderr) if part and part.strip())
    if diagnostic:
        logger.debug("Command output:\n%s", diagnostic)
    if result.returncode:
        detail = (result.stderr or result.stdout or "").strip()
        raise CommandError(f"Command failed ({result.returncode}): {printable}\n{detail[-4000:]}")
    return result


def probe_media(path: Path, logger: logging.Logger) -> dict[str, Any]:
    command = [require_binary("ffprobe"), "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)]
    result = run_command(command, logger, capture=True)
    raw = json.loads(result.stdout)
    videos = [stream for stream in raw.get("streams", []) if stream.get("codec_type") == "video" and not stream.get("disposition", {}).get("attached_pic")]
    audios = [stream for stream in raw.get("streams", []) if stream.get("codec_type") == "audio"]
    if not audios:
        raise ValueError(f"No audio stream found in {path}")
    video, audio = videos[0] if videos else {}, audios[0]
    info = {
        "path": str(path.resolve()),
        "has_video": bool(videos),
        "format": raw.get("format", {}).get("format_name"),
        "duration_seconds": _number(raw.get("format", {}).get("duration")),
        "size_bytes": _integer(raw.get("format", {}).get("size")),
        "video_codec": video.get("codec_name"),
        "video_start_seconds": _number(video.get("start_time")),
        "video_duration_seconds": _number(video.get("duration")),
        "resolution": f"{video.get('width')}x{video.get('height')}",
        "frame_rate": video.get("avg_frame_rate") or video.get("r_frame_rate"),
        "audio_codec": audio.get("codec_name"),
        "audio_start_seconds": _number(audio.get("start_time")),
        "audio_duration_seconds": _number(audio.get("duration")),
        "sample_rate": _integer(audio.get("sample_rate")),
        "channels": audio.get("channels"),
        "channel_layout": audio.get("channel_layout"),
        "bit_depth": audio.get("bits_per_raw_sample") or audio.get("bits_per_sample"),
        "audio_bitrate": _integer(audio.get("bit_rate")),
        "audio_streams": len(audios),
        "audio_tracks": [
            {
                "track": index + 1,
                "stream_index": stream.get("index"),
                "codec": stream.get("codec_name"),
                "sample_rate": _integer(stream.get("sample_rate")),
                "channels": stream.get("channels"),
                "language": stream.get("tags", {}).get("language"),
                "title": stream.get("tags", {}).get("title"),
            }
            for index, stream in enumerate(audios)
        ],
    }
    logger.info("Media: %s", json.dumps(info, ensure_ascii=False, sort_keys=True))
    return info


def extract_audio(video: Path, wav: Path, logger: logging.Logger, start_seconds: float | None = None, duration_seconds: float | None = None, audio_track: int = 0,
                  *, sample_range: tuple[int, int] | None = None) -> None:
    if audio_track < 0:
        raise ValueError("Audio track index cannot be negative")
    wav.parent.mkdir(parents=True, exist_ok=True)
    seek = ["-ss", f"{start_seconds:.3f}"] if start_seconds is not None and start_seconds > 0 else []
    limit = ["-t", f"{duration_seconds:.3f}"] if duration_seconds is not None and duration_seconds > 0 else []
    if sample_range is not None:
        start, stop = sample_range
        if not 0 <= start < stop or start_seconds is not None or duration_seconds is not None:
            raise ValueError("Sample range must be positive and cannot be combined with timed extraction")
        # Decode before trimming to preserve codec history and the resampler's phase.
        limit = ["-af", f"aresample=48000,atrim=start_sample={start}:end_sample={stop}"]
    command = [require_binary("ffmpeg"), "-hide_banner", "-y", *seek, "-i", str(video), "-map", f"0:a:{audio_track}", "-vn", *limit, "-c:a", "pcm_s24le", "-ar", "48000", str(wav)]
    run_command(command, logger)


def mux_clean_audio(video: Path, audio: Path, output: Path, logger: logging.Logger, duration_seconds: float | None = None) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.resolve() == video.resolve():
        raise ValueError("Output would overwrite the source video")
    codec_args = ["-c:a", "aac", "-b:a", "256k"]
    duration_args = ["-t", f"{duration_seconds:.6f}"] if duration_seconds and duration_seconds > 0 else ["-shortest"]
    command = [
        require_binary("ffmpeg"), "-hide_banner", "-y", "-i", str(video), "-i", str(audio),
        "-map", "0:v:0", "-map", "1:a:0", "-map", "0:s?", "-map", "0:d?", "-map_metadata", "0",
        "-c:v", "copy", "-c:s", "copy", "-c:d", "copy", "-af", "apad", *codec_args, *duration_args, str(output),
    ]
    run_command(command, logger)


def verify_muxed_output(source_info: dict[str, Any], source: Path, output: Path, logger: logging.Logger) -> dict[str, Any]:
    """Verify duration/sync and prove that the compressed video packets were copied."""
    output_info = probe_media(output, logger)
    source_duration = source_info.get("duration_seconds")
    output_duration = output_info.get("duration_seconds")
    duration_delta = None
    if source_duration is not None and output_duration is not None:
        duration_delta = abs(float(output_duration) - float(source_duration))

    video_start = output_info.get("video_start_seconds") or 0.0
    audio_start = output_info.get("audio_start_seconds") or 0.0
    sync_offset = abs(float(audio_start) - float(video_start))
    video_properties_match = all(
        output_info.get(key) == source_info.get(key)
        for key in ("video_codec", "resolution", "frame_rate")
    )
    source_video_hash = media_stream_hash(source, "0:v:0", logger)
    output_video_hash = media_stream_hash(output, "0:v:0", logger)
    video_stream_copied = source_video_hash == output_video_hash
    verification = {
        "duration_delta_seconds": round(duration_delta, 6) if duration_delta is not None else None,
        "audio_video_start_offset_seconds": round(sync_offset, 6),
        "video_properties_match": video_properties_match,
        "video_stream_copied": video_stream_copied,
        "source_video_sha256": source_video_hash,
        "output_video_sha256": output_video_hash,
    }
    logger.info("Output verification: %s", json.dumps(verification, sort_keys=True))
    if duration_delta is not None and duration_delta > 0.1:
        raise RuntimeError(f"Output duration differs from source by {duration_delta:.3f} seconds")
    if sync_offset > 0.05:
        raise RuntimeError(f"Output audio/video start offset is {sync_offset:.3f} seconds")
    if not video_properties_match or not video_stream_copied:
        raise RuntimeError("Output video stream was not preserved exactly")
    return verification


def media_stream_hash(path: Path, stream: str, logger: logging.Logger) -> str:
    command = [
        require_binary("ffmpeg"), "-v", "error", "-i", str(path),
        "-map", stream, "-c", "copy", "-f", "hash", "-hash", "sha256", "-",
    ]
    result = run_command(command, logger, capture=True)
    line = result.stdout.strip()
    if "=" not in line:
        raise RuntimeError(f"Could not hash media stream in {path}")
    return line.rsplit("=", 1)[-1].strip().lower()


def _number(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _integer(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
