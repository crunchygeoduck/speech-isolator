from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from .audio_analysis import analyze_audio
from .config import AUDIO_EXTENSIONS, PRESET_NAMES, PROJECT_ROOT, SUPPORTED_EXTENSIONS, load_preset, settings_for_strength, validate_strength
from .logging_utils import create_session_logger
from .media import extract_audio, mux_clean_audio, probe_media, require_binary, verify_muxed_output
from .preview import prepare_smooth_excerpt
from .processing import ClearVoiceEngine, DeepFilterNetEngine, post_process, select_engine, validate_enhanced_audio


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Isolate speech in audio and video files or folders.")
    parser.add_argument("input", type=Path, help="Audio/video file or directory")
    parser.add_argument("--strength", type=int, default=50, help="Processing strength from 0 (exact original) to 100; Reference is 50")
    parser.add_argument("--preset", choices=PRESET_NAMES, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, default=PROJECT_ROOT / "output")
    parser.add_argument("--engine", choices=("auto", "clearvoice", "clearvoice-legacy", "clearvoice-detail", "deepfilternet", "ffmpeg", "passthrough"), default="auto")
    parser.add_argument("--comparison", action="store_true", help="Create synchronized Original, Reference 50, and selected-strength WAVs")
    parser.add_argument("--comparison-start", type=float, default=0.0, help="Comparison excerpt start in seconds")
    parser.add_argument("--comparison-seconds", type=float, default=30.0, help="Comparison excerpt length (default: 30)")
    parser.add_argument("--keep-temp", action="store_true")
    parser.add_argument("--overwrite", action="store_true", help="Replace an existing CLEAN output, never the source")
    parser.add_argument("--audio-track", type=int, default=1, help="1-based source audio track number (default: 1)")
    parser.add_argument("--audio-only", action="store_true", help="Export the cleaned result as a 48 kHz WAV instead of a video")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def detect_hardware() -> dict[str, Any]:
    data: dict[str, Any] = {"os": platform.platform(), "python": sys.version.split()[0], "ffmpeg": shutil.which("ffmpeg"), "ffprobe": shutil.which("ffprobe"), "cuda_available": False, "gpu": None, "vram_mib": None}
    smi = shutil.which("nvidia-smi")
    if smi:
        result = subprocess.run([smi, "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"], capture_output=True, text=True)
        if result.returncode == 0 and result.stdout.strip():
            name, memory = result.stdout.strip().splitlines()[0].rsplit(",", 1)
            data.update(gpu=name.strip(), vram_mib=int(memory.strip()))
    try:
        import torch
        data["cuda_available"] = bool(torch.cuda.is_available())
        data["torch"] = torch.__version__
    except ImportError:
        data["torch"] = None
    return data


def discover_inputs(source: Path) -> list[Path]:
    source = source.expanduser().resolve()
    if source.is_file():
        if source.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"Unsupported media extension: {source.suffix}")
        return [source]
    if not source.is_dir():
        raise FileNotFoundError(source)
    return sorted(path for path in source.iterdir() if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS and not path.stem.upper().endswith("_CLEAN"))


def clean_output_path(video: Path, output_dir: Path) -> Path:
    return output_dir.resolve() / f"{video.stem}_CLEAN{video.suffix.lower()}"


def clean_audio_output_path(video: Path, output_dir: Path) -> Path:
    # Preserve different source formats with the same basename in mixed folders.
    stem = video.stem if video.suffix.lower() == ".wav" else f"{video.stem}_{video.suffix.lstrip('.').lower()}"
    return output_dir.resolve() / f"{stem}_CLEAN.wav"


def batch_preflight(sources: list[Path], output_dir: Path, strength: int | None, audio_only: bool) -> None:
    targets: dict[Path, Path] = {}
    source_paths = {source.resolve() for source in sources}
    for source in sources:
        export_audio = audio_only or (source.suffix.lower() in AUDIO_EXTENSIONS and strength != 0)
        target = clean_audio_output_path(source, output_dir) if export_audio else clean_output_path(source, output_dir)
        if target in source_paths:
            raise ValueError(f"Output would overwrite a queued source: {target.name}. Choose a separate output folder.")
        if target in targets:
            raise ValueError(f"Output name collision: {source} and {targets[target]}. Process these in separate output folders.")
        targets[target] = source


def save_batch_report(log_path: Path, results: list[dict[str, Any]], cancelled: bool = False) -> None:
    log_path.with_suffix(".json").write_text(json.dumps({"results": results, "cancelled": cancelled}, indent=2), encoding="utf-8")


def process_video(video: Path, output_dir: Path, preset_name: str, engine_name: str, logger: logging.Logger, keep_temp: bool, overwrite: bool, strength_percent: int | None = None, audio_track: int = 0, audio_only: bool = False) -> dict[str, Any]:
    """Process in a private staging folder; publish only validated final outputs."""
    if not video.is_file():
        raise FileNotFoundError(f"Recording is missing or unavailable: {video}")
    strength = validate_strength(strength_percent) if strength_percent is not None else None
    effective_audio_only = audio_only or (video.suffix.lower() in AUDIO_EXTENSIONS and strength != 0)
    target = clean_audio_output_path(video, output_dir) if effective_audio_only else clean_output_path(video, output_dir)
    if target.resolve() == video.resolve():
        raise ValueError("Output would overwrite the source")
    if target.exists() and not overwrite:
        logger.warning("Skipping existing output: %s; enable Replace existing CLEAN files to change its settings", target)
        return {"input": str(video), "output": str(target), "status": "skipped"}
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".factory_job_", dir=target.parent) as folder:
        result = _process_media_staged(video, Path(folder), preset_name, engine_name, logger,
                                       keep_temp, False, strength, audio_track, effective_audio_only)
        staged = Path(result["output"])
        if overwrite:
            staged.replace(target)
        else:
            # On Windows rename refuses an existing destination, including a
            # destination created by another process while enhancement ran.
            os.rename(staged, target)
        result["output"] = str(target)
        result["needs_review"] = bool(result.get("isolation_fallback"))
        report = target.with_suffix(target.suffix + ".json")
        try:
            report_stage = Path(folder) / "report.json"
            report_stage.write_text(json.dumps(result, indent=2), encoding="utf-8")
            report_stage.replace(report)
        except OSError:
            logger.exception("Output completed, but its review report could not be saved")
        logger.info("Published verified output: %s", target)
        return result


def _process_media_staged(video: Path, output_dir: Path, preset_name: str, engine_name: str, logger: logging.Logger, keep_temp: bool, overwrite: bool, strength_percent: int | None = None, audio_track: int = 0, audio_only: bool = False) -> dict[str, Any]:
    started = time.perf_counter()
    output = clean_audio_output_path(video, output_dir) if audio_only else clean_output_path(video, output_dir)
    if output.exists() and not overwrite:
        logger.warning("Skipping existing output: %s", output)
        return {"input": str(video), "output": str(output), "status": "skipped"}
    strength = validate_strength(strength_percent) if strength_percent is not None else None
    output.parent.mkdir(parents=True, exist_ok=True)
    logger.info("Starting file: %s", video)
    if audio_track < 0:
        raise ValueError("Audio track index cannot be negative")
    if strength == 0 and not audio_only:
        info = probe_media(video, logger)
        if audio_track >= int(info.get("audio_streams") or 0):
            raise ValueError(f"Audio track {audio_track + 1} does not exist in {video.name}")
        shutil.copy2(video, output)
        result = {"input": str(video), "output": str(output), "status": "complete", "preset": "strength_0", "strength": 0, "engine": "bypass", "audio_track": audio_track + 1, "isolation_fallback": False, "media": info, "verification": {"exact_source_copy": _sha256(video) == _sha256(output)}, "elapsed_seconds": round(time.perf_counter() - started, 2)}
        if not result["verification"]["exact_source_copy"]:
            raise RuntimeError("Strength 0 copy verification failed")
        logger.info("Strength 0: exact source copy completed -> %s", output)
        return result
    preset = settings_for_strength(strength) if strength is not None else load_preset(preset_name)
    engine = select_engine(engine_name, logger) if strength != 0 else None
    logger.info("Resolved processing settings: %s", json.dumps(preset, sort_keys=True))
    info = probe_media(video, logger)
    if audio_track >= int(info.get("audio_streams") or 0):
        raise ValueError(f"Audio track {audio_track + 1} does not exist in {video.name}")
    temp_root = PROJECT_ROOT / "temp"
    temp_root.mkdir(parents=True, exist_ok=True)
    temp_context = tempfile.TemporaryDirectory(prefix=f"{video.stem}_", dir=temp_root)
    work = Path(temp_context.name)
    try:
        raw, isolated, cleaned = work / "raw_audio.wav", work / "isolated.wav", work / "cleaned.wav"
        extract_audio(video, raw, logger, audio_track=audio_track)
        before = analyze_audio(raw, logger)
        if strength == 0:
            shutil.copy2(raw, cleaned)
            after = before
            shutil.copy2(cleaned, output)
            return {"input": str(video), "output": str(output), "status": "complete", "preset": "strength_0", "strength": 0, "engine": "bypass", "audio_track": audio_track + 1, "audio_only": True, "isolation_fallback": False, "media": info, "before": before, "after": after, "verification": {"wav_export": True}, "elapsed_seconds": round(time.perf_counter() - started, 2)}
        isolation_fallback = False
        used_engine_name = engine.name
        try:
            engine.process(raw, isolated, preset["speech_isolation"], logger)
            validate_enhanced_audio(raw, isolated)
        except Exception:
            logger.exception("Primary speech isolation failed")
            fallback = DeepFilterNetEngine()
            if engine.name != fallback.name and fallback.available():
                logger.warning("Falling back to DeepFilterNet3")
                try:
                    fallback.process(raw, isolated, preset["speech_isolation"], logger)
                    validate_enhanced_audio(raw, isolated)
                    used_engine_name = fallback.name
                except Exception:
                    logger.exception("Fallback isolation failed; preserving extracted original audio")
                    shutil.copy2(raw, isolated)
                    used_engine_name = "passthrough"
                isolation_fallback = True
            else:
                logger.warning("Safely falling back to the original extracted audio")
                shutil.copy2(raw, isolated)
                used_engine_name = "passthrough"
                isolation_fallback = True
        if used_engine_name == "passthrough" and isolation_fallback:
            shutil.copy2(raw, cleaned)
            peaks = []
        else:
            peaks = post_process(isolated, cleaned, preset, logger, raw)
        validate_enhanced_audio(raw, cleaned)
        after = analyze_audio(cleaned, logger)
        if audio_only:
            shutil.copy2(cleaned, output)
            verification = {"wav_export": True}
        else:
            mux_clean_audio(video, cleaned, output, logger, info.get("duration_seconds"))
            verification = verify_muxed_output(info, video, output, logger)
        result = {"input": str(video), "output": str(output), "status": "complete", "preset": preset_name if strength is None else f"strength_{strength}", "strength": strength, "engine": used_engine_name, "audio_track": audio_track + 1, "audio_only": audio_only, "isolation_fallback": isolation_fallback, "media": info, "before": before, "after": after, "tonal_peaks_hz": peaks, "verification": verification, "elapsed_seconds": round(time.perf_counter() - started, 2)}
        logger.info("Completed %s in %.2f seconds -> %s", video.name, result["elapsed_seconds"], output)
        if keep_temp:
            kept = temp_root / f"{video.stem}_kept"
            if kept.exists():
                shutil.rmtree(kept)
            shutil.copytree(work, kept)
            result["temp"] = str(kept)
        return result
    finally:
        temp_context.cleanup()


def create_comparisons(video: Path, engine_name: str, logger: logging.Logger, start_seconds: float = 0.0, duration_seconds: float = 30.0) -> list[Path]:
    destination = PROJECT_ROOT / "comparison"
    destination.mkdir(parents=True, exist_ok=True)
    engine = select_engine(engine_name, logger)
    with tempfile.TemporaryDirectory(prefix="comparison_", dir=PROJECT_ROOT / "temp") as folder:
        work = Path(folder)
        raw = work / "sample_original.wav"
        extract_audio(video, raw, logger, max(0.0, start_seconds), max(1.0, duration_seconds))
        outputs = [destination / "sample_original.wav"]
        shutil.copy2(raw, outputs[0])
        clearvoice = ClearVoiceEngine()
        deepfilter = DeepFilterNetEngine()
        for name in PRESET_NAMES:
            preset = load_preset(name)
            isolated, final = work / f"{name}_isolated.wav", destination / f"sample_{name}.wav"
            variant_engine = engine
            if name == "reference" and clearvoice.available():
                variant_engine = clearvoice
            elif name != "reference" and deepfilter.available():
                variant_engine = deepfilter
            logger.info("Comparison %s uses %s", name, variant_engine.name)
            variant_engine.process(raw, isolated, preset["speech_isolation"], logger)
            post_process(isolated, final, preset, logger, raw)
            outputs.append(final)
    manifest = {
        "source": str(video.resolve()),
        "source_sha256": _sha256(video),
        "created": datetime.now().isoformat(timespec="seconds"),
        "engine": engine.name,
        "start_seconds": max(0.0, start_seconds),
        "duration_seconds": max(1.0, duration_seconds),
        "files": {path.stem.removeprefix("sample_"): str(path.resolve()) for path in outputs},
    }
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return outputs


def create_strength_comparisons(video: Path, strength: int, logger: logging.Logger, start_seconds: float = 0.0, duration_seconds: float = 30.0, audio_track: int = 0, engine_name: str = "clearvoice") -> dict[str, Path]:
    """Build synchronized Original, Reference 50, and selected-strength WAVs."""
    strength = validate_strength(strength)
    destination = PROJECT_ROOT / "comparison"
    destination.mkdir(parents=True, exist_ok=True)
    variants: dict[str, Path] = {}
    (PROJECT_ROOT / "temp").mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="comparison_", dir=PROJECT_ROOT / "temp") as folder:
        work = Path(folder)
        raw = work / "sample_original.wav"
        isolated = work / "reference_isolated.wav"
        prepare_smooth_excerpt(video, raw, isolated, max(0.0, start_seconds), max(1.0, duration_seconds), audio_track, logger)
        original = destination / "sample_original.wav"
        shutil.copy2(raw, original)
        variants["Original"] = original
        needed = [50] if strength == 50 else [50, strength]
        if strength == 0:
            needed = [50]
            variants["Selected 0"] = original
        for value in needed:
            final = destination / f"sample_strength_{value}.wav"
            post_process(isolated, final, settings_for_strength(value), logger, raw)
            variants["Reference 50" if value == 50 else f"Selected {value}"] = final
        legacy_isolated = work / "legacy_isolated.wav"
        select_engine("clearvoice-legacy", logger).process(raw, legacy_isolated, load_preset("reference")["speech_isolation"], logger)
        validate_enhanced_audio(raw, legacy_isolated)
        legacy_final = destination / "sample_previous_reference.wav"
        post_process(legacy_isolated, legacy_final, settings_for_strength(50), logger, raw)
        variants["Previous Reference"] = legacy_final
        if strength == 50:
            variants["Selected 50"] = variants["Reference 50"]
        if engine_name != "clearvoice" and strength > 0:
            detail_isolated = work / "selected_engine.wav"
            selected_engine = select_engine(engine_name, logger)
            selected_engine.process(raw, detail_isolated, load_preset("reference")["speech_isolation"], logger)
            validate_enhanced_audio(raw, detail_isolated)
            selected_final = destination / f"sample_{engine_name}_{strength}.wav"
            post_process(detail_isolated, selected_final, settings_for_strength(strength), logger, raw)
            variants.pop(f"Selected {strength}", None)
            variants[f"Selected {strength} ({engine_name})"] = selected_final
    manifest = {"source": str(video.resolve()), "source_sha256": _sha256(video), "created": datetime.now().isoformat(timespec="seconds"), "strength": strength, "audio_track": audio_track + 1, "start_seconds": max(0.0, start_seconds), "duration_seconds": max(1.0, duration_seconds), "files": {name: str(path.resolve()) for name, path in variants.items()}}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return variants


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logger, log_path = create_session_logger(PROJECT_ROOT / "logs", args.verbose)
    try:
        args.strength = validate_strength(args.strength)
    except ValueError as exc:
        logger.error(str(exc))
        return 2
    logger.info("Session started; strength=%s legacy_preset=%s engine=%s", args.strength, args.preset, args.engine)
    logger.info("Hardware: %s", json.dumps(detect_hardware(), sort_keys=True))
    try:
        require_binary("ffmpeg")
        require_binary("ffprobe")
        videos = discover_inputs(args.input)
        if not videos:
            raise ValueError(f"No supported audio or video files found in {args.input}")
        results = []
        batch_preflight(videos, args.output, None if args.preset else args.strength, args.audio_only)
        effective_engine = args.engine
        if effective_engine == "auto" and args.preset and args.preset != "reference" and DeepFilterNetEngine().available():
            effective_engine = "deepfilternet"
        logger.info("Resolved session engine: %s", effective_engine)
        for video in videos:
            try:
                preset_name = args.preset or "reference"
                strength = None if args.preset else args.strength
                results.append(process_video(video, args.output, preset_name, effective_engine, logger, args.keep_temp, args.overwrite, strength, args.audio_track - 1, args.audio_only))
            except Exception as exc:
                logger.exception("Failed: %s", video)
                results.append({"input": str(video), "status": "failed", "error": str(exc)})
        if args.comparison:
            if len(videos) != 1:
                logger.warning("Comparison mode uses only the first discovered video")
            if args.preset:
                paths = create_comparisons(videos[0], args.engine, logger, args.comparison_start, args.comparison_seconds)
                logger.info("Comparisons: %s", [str(path) for path in paths])
            else:
                comparison_engine = "clearvoice" if args.engine == "auto" else args.engine
                variants = create_strength_comparisons(videos[0], args.strength, logger, args.comparison_start, args.comparison_seconds, args.audio_track - 1, comparison_engine)
                logger.info("Comparisons: %s", {name: str(path) for name, path in variants.items()})
        summary = {
            "complete": sum(item["status"] == "complete" for item in results),
            "skipped": sum(item["status"] == "skipped" for item in results),
            "failed": sum(item["status"] == "failed" for item in results),
        }
        logger.info("Session summary: %s", json.dumps(summary, sort_keys=True))
        save_batch_report(log_path, results)
        print(json.dumps({"log": str(log_path), "results": results}, indent=2))
        return 1 if any(item["status"] == "failed" for item in results) else 0
    except Exception:
        logger.exception("Session failed")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
