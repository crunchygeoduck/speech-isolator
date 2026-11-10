import logging
from pathlib import Path

from src import gui


def _test_logger(tmp_path: Path):
    logger = logging.getLogger(f"test.batch.{id(tmp_path)}")
    logger.handlers.clear()
    logger.addHandler(logging.NullHandler())
    logger.propagate = False
    return logger, tmp_path / "batch.log"


def test_worker_continues_after_one_file_fails(monkeypatch, tmp_path: Path):
    sources = [tmp_path / "one.mov", tmp_path / "bad.mov", tmp_path / "three.mov"]
    processed: list[str] = []

    def fake_process(source, output, preset, engine, logger, keep_temp, overwrite, strength_percent=None, audio_track=0, audio_only=False):
        processed.append(source.name)
        if source.name == "bad.mov":
            raise RuntimeError("broken media")
        return {"input": str(source), "output": str(output / f"{source.stem}_CLEAN.mov"), "status": "complete"}

    monkeypatch.setattr(gui, "create_session_logger", lambda *_args, **_kwargs: _test_logger(tmp_path))
    monkeypatch.setattr(gui, "process_video", fake_process)
    worker = gui.ProcessingWorker(sources, tmp_path, 50, 0, 10, False, False)
    payloads = []
    worker.completed.connect(payloads.append)
    worker.run()

    assert processed == ["one.mov", "bad.mov", "three.mov"]
    assert payloads[0]["counts"] == {"complete": 2, "skipped": 0, "failed": 1}


def test_worker_cancellation_prevents_new_files(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(gui, "create_session_logger", lambda *_args, **_kwargs: _test_logger(tmp_path))
    monkeypatch.setattr(gui, "process_video", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("should not run")))
    worker = gui.ProcessingWorker([tmp_path / "one.mov"], tmp_path, 50, 0, 10, False, False)
    payloads = []
    statuses = []
    worker.completed.connect(payloads.append)
    worker.item_status.connect(lambda row, status, detail: statuses.append((row, status)))
    worker.request_cancel()
    worker.run()

    assert statuses == [(0, "Cancelled")]
    assert payloads[0]["cancelled"] is True
