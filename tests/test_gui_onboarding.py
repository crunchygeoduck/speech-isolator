import pytest
import logging
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication

from src import gui


@pytest.fixture
def window(monkeypatch, tmp_path):
    app = QApplication.instance() or QApplication([])
    monkeypatch.setattr(gui, "detect_hardware", lambda: {})
    settings = gui.QSettings(str(tmp_path / "settings.ini"), gui.QSettings.Format.IniFormat)
    monkeypatch.setattr(gui, "QSettings", lambda *_: settings)
    widget = gui.MainWindow()
    widget.show()
    app.processEvents()
    yield widget
    widget.preview_timer.stop()
    if widget.preview_thread:
        widget.preview_thread.quit()
        widget.preview_thread.wait(2000)
        app.processEvents()
    widget.close()
    widget.deleteLater()
    app.processEvents()


def test_control_room_defaults_and_navigation(window):
    assert [window.tabs.tabText(i) for i in range(window.tabs.count())] == ["Control Room", "How to use"]
    assert window.tabs.currentIndex() == 0
    assert window.sources == []
    assert not window.process_button.isEnabled()
    assert not window.a_button.isEnabled()
    assert not window.b_button.isEnabled()
    assert window.strength_slider.value() == 50
    assert not window.detail_check.isChecked()
    assert not window.overwrite_check.isChecked()
    window.tabs.setCurrentIndex(1)
    assert window.tabs.currentIndex() == 1
    window.tabs.setCurrentIndex(0)
    assert window.advanced_panel.isHidden()
    window.advanced_button.click()
    assert not window.advanced_panel.isHidden()
    window.log_toggle.click()
    assert not window.log_view.isHidden()


def test_selection_and_strength_remain_connected(window, tmp_path):
    window._set_sources([tmp_path / "one.wav"])
    assert window.tabs.currentIndex() == 0
    assert window.process_button.text() == "Clean 1 file"
    assert window.process_button.isEnabled()
    assert window.comparison_check.isChecked()
    window.strength_spin.setValue(0)
    assert window.strength_slider.value() == 0
    assert "exact copy" in window.strength_description.text()
    window._set_sources([tmp_path / "one.wav", tmp_path / "two.wav"])
    assert not window.comparison_check.isChecked()
    assert window.process_button.text() == "Clean 2 files"


def test_pages_fit_small_desktop_width(window):
    window.resize(800, 600)
    window.advanced_button.setChecked(True)
    for index in range(window.tabs.count()):
        window.tabs.setCurrentIndex(index)
        QApplication.processEvents()
        assert window.tabs.widget(index).horizontalScrollBar().maximum() == 0


def test_comparison_ready_state(window, tmp_path):
    window.variants = {"Original": tmp_path / "original.wav", "Reference 50": tmp_path / "ref.wav"}
    window._set_comparison_enabled(True)
    assert window.a_combo.currentText() == "Original"
    assert window.b_combo.currentText() == "Reference 50"
    assert window.play_button.isEnabled()
    assert "ready" in window.compare_hint.text()

def test_legacy_comparisons_choose_different_versions(window, tmp_path):
    window.variants = {"original": tmp_path / "a.wav", "reference": tmp_path / "b.wav"}
    window._set_comparison_enabled(True)
    assert window.a_combo.currentText() == "original"
    assert window.b_combo.currentText() == "reference"
    window._set_comparison_enabled(False)
    assert window.a_combo.count() == window.b_combo.count() == 0


def test_queue_append_preserves_rows_and_removal_keeps_source(window, tmp_path):
    first, second, third = [tmp_path / name for name in ("one.wav", "two.wav", "three.wav")]
    second.touch()
    window._set_sources([first, second])
    window.queue_table.setCurrentCell(1, 0)
    window._update_item_status(0, "Complete", "First export")
    window._set_sources([first, third], append=True)
    assert window.sources == [first, second, third]
    assert window._selected_source() == second
    assert window.queue_table.item(0, 1).text() == "Complete"
    window.remove_button.click()
    assert window.sources == [first, third]
    assert window._selected_source() == third
    assert second.is_file()
    assert window.process_button.text() == "Clean 2 files"


def test_remembers_volume_and_existing_output_folder(window, tmp_path):
    destination = tmp_path / "cleaned"
    destination.mkdir()
    window.output_edit.setText(str(destination))
    window.volume_slider.setValue(28)
    window.close()
    reopened = gui.MainWindow()
    try:
        assert reopened.output_edit.text() == str(destination)
        assert reopened.volume_slider.value() == 28
    finally:
        reopened.close()


def test_outgoing_media_status_does_not_consume_resume_request(window):
    window.changing_source = True
    window.pending_position = 3000
    window.resume_after_switch = True
    window._media_status_changed(gui.QMediaPlayer.MediaStatus.LoadedMedia)
    assert window.pending_position == 3000
    assert window.resume_after_switch
    window.changing_source = False
    window.pending_position = 0
    window.resume_after_switch = False


def test_listening_volume_only_changes_audio_device(window):
    initial_strength = window.strength_slider.value()
    window.preview_active = True
    for value in (0, 25, 100):
        window.volume_slider.setValue(value)
        assert window.audio_output.volume() == pytest.approx(value / 100)
        assert window.strength_slider.value() == initial_strength
        assert not window.preview_timer.isActive()
    window.volume_slider.setValue(0)
    assert window.volume_value.text() == "Muted"
    window._set_processing_controls(False)
    assert window.volume_slider.isEnabled()
    window.preview_active = False


def test_blank_destination_has_helpful_error(window, tmp_path, monkeypatch):
    source = tmp_path / "input.wav"
    source.touch()
    window._set_sources([source])
    window.output_edit.clear()
    messages = []
    monkeypatch.setattr(gui.QMessageBox, "warning", lambda *args: messages.append(args[1]))
    window._start_processing()
    window.open_output.click()
    assert messages == ["Choose a save folder", "Choose a save folder"]
    assert window.thread is None


def test_preview_uses_excerpt_not_export_and_ignores_stale_result(window, tmp_path, monkeypatch):
    calls = []
    def render(*args):
        calls.append(args)
        return {"variants": {"Original": tmp_path / "original.wav", "Reference 50": tmp_path / "ref.wav", "Live preview": tmp_path / "preview.wav"}, "strength": args[5], "detail": args[4], "seconds": .01}
    monkeypatch.setattr(window.preview_cache, "render", render)
    monkeypatch.setattr(gui, "process_video", lambda *args, **kwargs: pytest.fail("Preview must not export"))
    first, selected = tmp_path / "first.wav", tmp_path / "selected.wav"
    window._set_sources([first, selected])
    window.queue_table.setCurrentCell(1, 0)
    assert window.selected_file_label.text() == selected.name
    window.preview_button.click()
    window.strength_slider.setValue(72)
    for _ in range(150):
        QTest.qWait(10)
        if window.preview_thread is None and window.preview_rendered == (72, False):
            break
    assert window.preview_thread is None
    assert calls[-1][5] == 72
    assert all(call[0] == selected for call in calls)
    assert window.strength_slider.value() == 72
    assert "strength 72" in window.preview_status.text()
    window._preview_complete({"variants": {"stale": tmp_path}, "strength": 20, "detail": False, "seconds": .1})
    assert "stale" not in window.variants
    window.start_spin.setValue(5)
    assert not window.preview_active
    assert not window.play_button.isEnabled()


def test_clean_button_runs_worker_with_selected_options(window, tmp_path, monkeypatch):
    source = tmp_path / "input.wav"
    source.touch()
    window._set_sources([source])
    window.output_edit.setText(str(tmp_path / "results"))
    window.detail_check.setChecked(True)
    window.audio_only_check.setChecked(True)
    window.audio_track_spin.setValue(2)
    window.strength_spin.setValue(55)
    window.comparison_check.setChecked(False)
    calls = []

    def process(*args, **kwargs):
        calls.append((args, kwargs))
        return {"input": str(source), "output": str(tmp_path / "results" / "input_CLEAN.wav"), "status": "complete"}

    monkeypatch.setattr(gui, "process_video", process)
    monkeypatch.setattr(gui, "create_session_logger", lambda *_: (logging.getLogger("gui.flow"), tmp_path / "session.log"))
    window.process_button.click()
    assert not window.detail_check.isEnabled()
    for _ in range(100):
        QTest.qWait(10)
        if window.thread is None:
            break
    assert window.thread is None
    assert calls[0][0][3] == "clearvoice-detail"
    assert calls[0][1]["strength_percent"] == 55
    assert calls[0][1]["audio_track"] == 1
    assert calls[0][1]["audio_only"] is True
    assert window.queue_table.item(0, 1).text() == "Complete"
    assert window.process_button.isEnabled()
    assert "1 complete" in window.status_label.text()


def test_missing_recording_does_not_shift_queue_status(window, tmp_path, monkeypatch):
    from src import main

    missing, valid = tmp_path / "missing.wav", tmp_path / "valid.wav"
    valid.write_bytes(b"recording")
    monkeypatch.setattr(main, "probe_media", lambda *_: {"audio_streams": 1})
    errors = []
    monkeypatch.setattr(gui.QMessageBox, "critical", lambda *args: errors.append(args[2]))
    window._set_sources([missing, valid])
    window.output_edit.setText(str(tmp_path / "out"))
    window.strength_spin.setValue(0)
    monkeypatch.setattr(gui, "create_session_logger", lambda *_: (logging.getLogger("gui.missing"), tmp_path / "batch.log"))
    window.process_button.click()
    for _ in range(500):
        QTest.qWait(10)
        if window.thread is None:
            break
    assert window.thread is None
    assert errors == []
    assert window.queue_table.item(0, 1).text() == "Failed"
    assert window.queue_table.item(1, 1).text() == "Complete"
