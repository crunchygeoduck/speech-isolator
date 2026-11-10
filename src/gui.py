from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf
from PySide6.QtCore import QObject, QSettings, QSignalBlocker, Qt, QThread, QTimer, QUrl, Signal, Slot
from PySide6.QtGui import QColor, QCloseEvent, QDesktopServices, QDragEnterEvent, QDropEvent, QFont, QKeySequence, QPainter, QPen, QResizeEvent, QShortcut
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtWidgets import (
    QApplication,
    QAbstractItemView,
    QBoxLayout,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QStyle,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from .config import PROJECT_ROOT, SUPPORTED_EXTENSIONS
from .logging_utils import create_session_logger
from .preview import PreviewCache
from .main import create_strength_comparisons, detect_hardware, discover_inputs, process_video, batch_preflight, save_batch_report


STANDARD_SAMPLE_NAME = "AUDIOPROCESSING SNIP.mov"


class SignalLogHandler(logging.Handler):
    def __init__(self, signal: Signal) -> None:
        super().__init__(logging.INFO)
        self.signal = signal

    def emit(self, record: logging.LogRecord) -> None:
        self.signal.emit(self.format(record))


class ProcessingWorker(QObject):
    message = Signal(str)
    item_status = Signal(int, str, str)
    progress = Signal(int, int)
    eta = Signal(str)
    completed = Signal(dict)
    failed = Signal(str)

    def __init__(self, sources: list[Path], output: Path, strength: int, start: float, duration: float, comparisons: bool, overwrite: bool, audio_track: int = 0, audio_only: bool = False, detail_recovery: bool = False) -> None:
        super().__init__()
        self.sources = sources
        self.output = output
        self.strength = strength
        self.start = start
        self.duration = duration
        self.comparisons = comparisons
        self.overwrite = overwrite
        self.audio_track = audio_track
        self.audio_only = audio_only
        self.detail_recovery = detail_recovery
        self.cancel_requested = False

    def request_cancel(self) -> None:
        self.cancel_requested = True

    @Slot()
    def run(self) -> None:
        logger, log_path = create_session_logger(PROJECT_ROOT / "logs")
        handler = SignalLogHandler(self.message)
        handler.setFormatter(logging.Formatter("%(levelname)s  %(message)s"))
        logger.addHandler(handler)
        try:
            engine = "clearvoice-detail" if self.detail_recovery else "clearvoice"
            batch_preflight(self.sources, self.output, self.strength, self.audio_only)
            logger.info(
                "GUI batch started; files=%d strength=%d engine=%s output=%s overwrite=%s comparisons=%s",
                len(self.sources), self.strength, engine, self.output, self.overwrite, self.comparisons,
            )
            results: list[dict[str, Any]] = []
            batch_started = time.perf_counter()
            for index, source in enumerate(self.sources):
                if self.cancel_requested:
                    for waiting in range(index, len(self.sources)):
                        self.item_status.emit(waiting, "Cancelled", "Not started")
                    break
                self.item_status.emit(index, "Processing", "Inspecting media and cleaning dialogue")
                self.message.emit(f"[{index + 1}/{len(self.sources)}] Processing {source.name}")
                try:
                    result = process_video(
                        source, self.output, "reference", engine, logger,
                        keep_temp=False, overwrite=self.overwrite, strength_percent=self.strength,
                        audio_track=self.audio_track, audio_only=self.audio_only,
                    )
                    results.append(result)
                    status = "Skipped" if result["status"] == "skipped" else "Complete"
                    if result.get("needs_review"):
                        status = "Review: fallback"
                    self.item_status.emit(index, status, result.get("output", ""))
                except Exception as exc:
                    logger.exception("Batch item failed: %s", source)
                    results.append({"input": str(source), "status": "failed", "error": str(exc)})
                    self.item_status.emit(index, "Failed", str(exc))
                self.progress.emit(index + 1, len(self.sources))
                average = (time.perf_counter() - batch_started) / (index + 1)
                remaining = max(0, len(self.sources) - index - 1)
                if remaining:
                    seconds = round(average * remaining)
                    self.eta.emit(f"Estimated time remaining: {seconds // 60}m {seconds % 60}s")
                else:
                    self.eta.emit("File processing complete; finishing comparisons if selected.")

            comparison_paths: dict[str, Path] = {}
            if self.comparisons and self.sources and not self.cancel_requested:
                self.message.emit("Building synchronized A/B versions for the first file…")
                try:
                    comparison_paths = create_strength_comparisons(
                        self.sources[0], self.strength, logger, self.start, self.duration, self.audio_track, engine,
                    )
                except Exception:
                    logger.exception("A/B comparison generation failed")
                    self.message.emit("WARNING  Batch completed, but A/B comparison generation failed.")
            counts = {
                "complete": sum(item["status"] == "complete" for item in results),
                "skipped": sum(item["status"] == "skipped" for item in results),
                "failed": sum(item["status"] == "failed" for item in results),
            }
            logger.info("GUI batch finished; summary=%s cancelled=%s", counts, self.cancel_requested)
            save_batch_report(log_path, results, self.cancel_requested)
            self.completed.emit({
                "results": results,
                "comparisons": {name: str(path) for name, path in comparison_paths.items()},
                "comparison_source": str(self.sources[0]) if comparison_paths else None,
                "log": str(log_path),
                "cancelled": self.cancel_requested,
                "counts": counts,
                "needs_review": sum(bool(item.get("needs_review")) for item in results),
            })
        except Exception as exc:
            logger.exception("GUI processing failed")
            self.failed.emit(str(exc))


class PreviewWorker(QObject):
    completed = Signal(dict)

    def __init__(self, cache, request):
        super().__init__()
        self.cache, self.request = cache, request

    @Slot()
    def run(self):
        try:
            result = self.cache.render(*self.request, logging.getLogger("preview"))
            self.completed.emit(result)
        except Exception as exc:
            self.completed.emit({"error": str(exc)})


class PreviewWaveform(QWidget):
    """A bounded peak envelope of the WAV currently loaded in the player."""

    def __init__(self) -> None:
        super().__init__()
        self.setFixedHeight(72)
        self.setAccessibleName("Listening sample waveform")
        self.path: Path | None = None
        self.peaks: list[float] = []
        self.playhead = 0.0

    def set_audio(self, path: Path | None) -> None:
        if path == self.path:
            return
        self.path, self.peaks = path, []
        self.playhead = 0.0
        if path is not None:
            try:
                with sf.SoundFile(path) as audio:
                    # Read at most 180 blocks and retain a peak per block.
                    block_size = max(1, (audio.frames + 179) // 180)
                    self.peaks = [float(np.max(np.abs(block))) for block in
                                  audio.blocks(blocksize=block_size, dtype="float32")]
            except (OSError, RuntimeError, ValueError):
                pass  # The waveform is optional for files the player can open.
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor("#141c25"))
        if not self.peaks:
            painter.setPen(QColor("#acbdd0"))
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No waveform yet")
            return
        left, width, center = 6, max(1, self.width() - 12), self.height() // 2
        painter.setPen(QPen(QColor("#86baf0"), 1))
        for index, peak in enumerate(self.peaks):
            x = left + round((index + .5) * width / len(self.peaks))
            height = max(1, round(min(1.0, peak) * (center - 6)))
            painter.drawLine(x, center - height, x, center + height)
        painter.setPen(QPen(QColor("#edf2f8"), 2))
        x = left + round(self.playhead * width)
        painter.drawLine(x, 4, x, self.height() - 4)


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("Factory Speech Cleaner")
        self.setMinimumSize(800, 600)
        self.resize(1120, 850)
        self.setAcceptDrops(True)
        self.thread: QThread | None = None
        self.worker: ProcessingWorker | None = None
        self.sources: list[Path] = []
        self.variants: dict[str, Path] = {}
        self.active_side = "A"
        self.pending_position = 0
        self.resume_after_switch = False
        self.changing_source = False
        self.user_seeking = False
        self.preview_cache = PreviewCache()
        self.preview_thread = None
        self.preview_worker = None
        self.preview_active = False
        self.preview_rendered = None
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(400)
        self.preview_timer.timeout.connect(self._render_preview)

        self.settings = QSettings("Local Tools", "Factory Speech Cleaner")
        self.audio_output = QAudioOutput(self)
        volume = self.settings.value("playback/volume", 85, type=int)
        self.audio_output.setVolume(min(100, max(0, volume)) / 100)
        self.player = QMediaPlayer(self)
        self.player.setAudioOutput(self.audio_output)
        self.player.positionChanged.connect(self._position_changed)
        self.player.durationChanged.connect(self._duration_changed)
        self.player.playbackStateChanged.connect(self._playback_state_changed)
        self.player.mediaStatusChanged.connect(self._media_status_changed)
        self.player.errorOccurred.connect(self._player_error)

        self._build_ui()
        self._apply_theme()
        hardware = detect_hardware()
        gpu = hardware.get("gpu") or "No NVIDIA GPU detected"
        mode = "CUDA" if hardware.get("cuda_available") else "CPU"
        self.hardware_label.setText(f"GPU: {gpu}  |  AI runtime: {mode}  |  Reference engine: Smooth-context 48 kHz")
        self.status_label.setText("Ready · Add files or drop a folder into this window.")

    def _build_ui(self) -> None:
        self.tabs = QTabWidget()
        self.tabs.setAccessibleName("Main pages")
        self.setCentralWidget(self.tabs)
        layout = self._page("Control Room")

        title = QLabel("Factory Speech Cleaner")
        title.setObjectName("title")
        title.setFont(QFont("Segoe UI", 24, QFont.Weight.DemiBold))
        layout.addWidget(title)
        layout.addWidget(self._hint("Batch workspace · Queue recordings, check a preview, then export cleaned files."))
        self.hardware_label = QLabel()
        self.hardware_label.setObjectName("hardware")
        self.hardware_label.setWordWrap(True)

        columns = self.control_columns = QHBoxLayout()
        columns.setSpacing(16)
        layout.addLayout(columns, 1)
        queue_column = QVBoxLayout()
        queue_column.setSpacing(12)
        columns.addLayout(queue_column, 5)

        setup = self._section("Processing queue")
        queue_actions = QHBoxLayout()
        self.file_button = QPushButton("Add files…")
        self.file_button.setShortcut(QKeySequence("Ctrl+O"))
        self.file_button.setToolTip("Add recordings to the queue (Ctrl+O)")
        self.file_button.clicked.connect(self._browse_input)
        self.folder_button = QPushButton("Add folder…")
        self.folder_button.clicked.connect(self._browse_input_folder)
        queue_actions.addWidget(self.file_button)
        queue_actions.addWidget(self.folder_button)
        queue_actions.addStretch()
        self.queue_count = self._hint("0 files")
        queue_actions.addWidget(self.queue_count)
        setup.layout().addLayout(queue_actions)

        self.queue_table = QTableWidget(0, 3)
        self.queue_table.setHorizontalHeaderLabels(["Recording", "Status", "Details / output"])
        header = self.queue_table.horizontalHeader()
        header.setDefaultAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        header.setSectionResizeMode(0, header.ResizeMode.Stretch)
        header.setSectionResizeMode(1, header.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, header.ResizeMode.Stretch)
        self.queue_table.verticalHeader().hide()
        self.queue_table.verticalHeader().setDefaultSectionSize(36)
        self.queue_table.setAlternatingRowColors(True)
        self.queue_table.setShowGrid(False)
        self.queue_table.setWordWrap(False)
        self.queue_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.queue_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.queue_table.setAccessibleName("Processing queue")
        self.queue_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.queue_table.setMinimumHeight(180)
        self.queue_table.currentCellChanged.connect(self._queue_cell_changed)
        setup.layout().addWidget(self.queue_table, 1)
        self.remove_button = QPushButton("Remove selected")
        self.remove_button.setToolTip("Remove the selected recording from the queue (Delete while the queue is focused)")
        self.remove_button.setEnabled(False)
        self.remove_button.clicked.connect(self._remove_selected_source)
        remove_shortcut = QShortcut(QKeySequence("Delete"), self.queue_table)
        remove_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        remove_shortcut.activated.connect(self.remove_button.click)
        setup.layout().addWidget(self.remove_button, alignment=Qt.AlignmentFlag.AlignLeft)
        queue_column.addWidget(setup, 1)

        compare = self._section("Selected file")
        columns.addWidget(compare, 4, Qt.AlignmentFlag.AlignTop)
        self.selected_file_label = QLabel("No recording selected")
        self.selected_file_label.setObjectName("selectedFile")
        self.selected_file_label.setWordWrap(True)
        self.selected_file_label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        compare.layout().addWidget(self.selected_file_label)

        strength_label = QLabel("Cleanup strength")
        strength_label.setObjectName("fieldLabel")
        compare.layout().addWidget(strength_label)
        strength_row = QHBoxLayout()
        self.strength_slider = QSlider(Qt.Orientation.Horizontal)
        self.strength_slider.setRange(0, 100)
        self.strength_slider.setValue(50)
        self.strength_slider.setSingleStep(1)
        self.strength_slider.setPageStep(5)
        self.strength_slider.setTickInterval(10)
        self.strength_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.strength_slider.setAccessibleName("Voice isolation strength")
        self.strength_slider.setToolTip("0 bypasses cleanup. 50 is the approved Reference. 100 applies maximum isolation.")
        self.strength_spin = QSpinBox()
        self.strength_spin.setRange(0, 100)
        self.strength_spin.setValue(50)
        self.strength_spin.setSuffix(" %")
        self.strength_spin.setAccessibleName("Voice isolation strength percent")
        self.strength_spin.setMinimumWidth(86)
        self.strength_slider.valueChanged.connect(self.strength_spin.setValue)
        self.strength_spin.valueChanged.connect(self.strength_slider.setValue)
        self.strength_slider.valueChanged.connect(self._strength_changed)
        strength_row.addWidget(self.strength_slider, 1)
        strength_row.addWidget(self.strength_spin)
        compare.layout().addLayout(strength_row)
        self.strength_description = QLabel()
        self.strength_description.setObjectName("hint")
        self.strength_description.setWordWrap(True)
        compare.layout().addWidget(self.strength_description)
        self._strength_changed(50)

        setup = self._section("Export settings")
        destination = QHBoxLayout()
        saved_output = self.settings.value("export/folder", "", type=str)
        output = Path(saved_output) if saved_output and Path(saved_output).is_dir() else PROJECT_ROOT / "output"
        self.output_edit = QLineEdit(str(output))
        self.output_edit.setAccessibleName("Output folder path")
        self.output_button = QPushButton("Browse…")
        self.output_button.clicked.connect(self._browse_output)
        destination.addWidget(self.output_edit, 1)
        destination.addWidget(self.output_button)
        output_label = QLabel("Save cleaned files to")
        output_label.setBuddy(self.output_edit)
        setup.layout().addWidget(output_label)
        setup.layout().addLayout(destination)

        self.comparison_check = QCheckBox("Save A/B samples for the first file")
        self.comparison_check.setChecked(True)
        setup.layout().addWidget(self.comparison_check)

        self.advanced_button = QPushButton("Show advanced options")
        self.advanced_button.setCheckable(True)
        setup.layout().addWidget(self.advanced_button)
        self.advanced_panel = QWidget()
        advanced = QVBoxLayout(self.advanced_panel)
        advanced.setContentsMargins(0, 4, 0, 4)
        advanced.setSpacing(12)
        self.advanced_panel.setVisible(False)
        self.advanced_button.toggled.connect(self.advanced_panel.setVisible)
        self.advanced_button.toggled.connect(lambda checked: self.advanced_button.setText("Hide advanced options" if checked else "Show advanced options"))
        setup.layout().addWidget(self.advanced_panel)

        self.detail_check = QCheckBox("Detail recovery — experimental, slower")
        self.detail_check.setToolTip("Blend complementary speech detail from two local models. Reference 50 stays available in A/B comparisons. Turn this off for the accepted baseline.")
        self.detail_check.toggled.connect(lambda: self._strength_changed(self.strength_slider.value()))
        advanced.addWidget(self.detail_check)
        advanced.addWidget(self._hint("Tries a second local speech model to recover detail. It takes longer and may not sound better on every recording."))

        options = QGridLayout()
        options.addWidget(QLabel("Source audio track"), 0, 0)
        self.audio_track_spin = QSpinBox()
        self.audio_track_spin.setRange(1, 32)
        self.audio_track_spin.setValue(1)
        self.audio_track_spin.setAccessibleName("Source audio track number")
        self.audio_track_spin.setToolTip("Choose which embedded audio track to clean. Most videos use track 1.")
        options.addWidget(self.audio_track_spin, 0, 1)
        self.audio_only_check = QCheckBox("Export cleaned audio only (.wav)")
        self.audio_only_check.setToolTip("Create a 48 kHz WAV instead of recombining the audio with video.")
        options.addWidget(self.audio_only_check, 1, 0, 1, 2)
        advanced.addLayout(options)
        advanced.addWidget(self._hint("Leave track 1 selected unless your video has several audio tracks. Audio-only export saves a WAV, including at strength 0."))

        excerpt = QGridLayout()
        excerpt.addWidget(QLabel("Sample start"), 0, 0)
        self.start_spin = QDoubleSpinBox()
        self.start_spin.setRange(0, 86400)
        self.start_spin.setSuffix(" s")
        self.start_spin.setDecimals(1)
        excerpt.addWidget(self.start_spin, 0, 1)
        excerpt.addWidget(QLabel("Length"), 1, 0)
        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(5, 120)
        self.duration_spin.setValue(10)
        self.duration_spin.setSuffix(" s")
        self.duration_spin.setDecimals(1)
        excerpt.addWidget(self.duration_spin, 1, 1)
        self.overwrite_check = QCheckBox("Replace existing CLEAN files")
        self.overwrite_check.setChecked(False)
        advanced.addLayout(excerpt)
        advanced.addWidget(self.overwrite_check)
        advanced.addWidget(self._hint("Leave replacement off to skip existing results. Turn it on to reprocess with different settings; previous cleaned results will be replaced."))

        actions = QHBoxLayout()
        self.process_button = QPushButton("Clean files")
        self.process_button.setEnabled(False)
        self.process_button.setObjectName("primary")
        self.process_button.setMinimumHeight(44)
        self.process_button.clicked.connect(self._start_processing)
        actions.addWidget(self.process_button)
        self.cancel_button = QPushButton("Stop after this file")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._request_cancel)
        actions.addWidget(self.cancel_button)
        setup.layout().addLayout(actions)
        queue_column.addWidget(setup)

        self.preview_button = QPushButton("Prepare preview")
        self.preview_button.setShortcut(QKeySequence("Ctrl+P"))
        self.preview_button.setToolTip("Prepare the selected recording for listening (Ctrl+P)")
        self.preview_button.setEnabled(False)
        self.preview_button.clicked.connect(self._prepare_preview)
        compare.layout().addWidget(self.preview_button)
        self.preview_status = self._hint("Select a recording in the queue to prepare a preview.")
        compare.layout().addWidget(self.preview_status)
        self.strength_slider.valueChanged.connect(self._schedule_preview)
        self.detail_check.toggled.connect(self._schedule_preview)
        for spin in (self.start_spin, self.duration_spin, self.audio_track_spin):
            spin.valueChanged.connect(self._invalidate_preview)
        self.compare_hint = self._hint("")
        selectors = QGridLayout()
        selectors.setHorizontalSpacing(10)
        selectors.setColumnStretch(1, 1)
        self.a_combo = QComboBox()
        self.b_combo = QComboBox()
        self.a_combo.setAccessibleName("Version for side A")
        self.b_combo.setAccessibleName("Version for side B")
        for combo in (self.a_combo, self.b_combo):
            combo.setEnabled(False)
            combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(8)
        self.a_combo.currentTextChanged.connect(lambda: self._combo_changed("A"))
        self.b_combo.currentTextChanged.connect(lambda: self._combo_changed("B"))
        self.a_button = QPushButton("Listen A")
        self.b_button = QPushButton("Listen B")
        for side, button in (("A", self.a_button), ("B", self.b_button)):
            button.setCheckable(True)
            button.setProperty("ab", True)
            button.clicked.connect(lambda checked=False, selected=side: self._switch_side(selected))
        self.a_button.setChecked(True)
        selectors.addWidget(QLabel("A"), 0, 0)
        selectors.addWidget(self.a_combo, 0, 1)
        selectors.addWidget(self.a_button, 0, 2)
        selectors.addWidget(QLabel("B"), 1, 0)
        selectors.addWidget(self.b_combo, 1, 1)
        selectors.addWidget(self.b_button, 1, 2)
        compare.layout().addLayout(selectors)
        self.waveform = PreviewWaveform()
        compare.layout().addWidget(self.waveform)

        transport = QHBoxLayout()
        self.play_button = QPushButton("Play")
        self.play_button.setShortcut(QKeySequence("Ctrl+Space"))
        self.play_button.setToolTip("Play or pause the listening sample (Ctrl+Space)")
        self.play_button.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay))
        self.play_button.setEnabled(False)
        self.play_button.clicked.connect(self._toggle_playback)
        self.seek = QSlider(Qt.Orientation.Horizontal)
        self.seek.setRange(0, 1000)
        self.seek.setEnabled(False)
        self.seek.sliderPressed.connect(lambda: setattr(self, "user_seeking", True))
        self.seek.sliderReleased.connect(self._seek_released)
        self.time_label = QLabel("00:00 / 00:00")
        transport.addWidget(self.play_button)
        transport.addWidget(self.seek, 1)
        transport.addWidget(self.time_label)
        compare.layout().addLayout(transport)
        volume_row = QHBoxLayout()
        volume_label = QLabel("Playback volume")
        self.volume_slider = QSlider(Qt.Orientation.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(round(self.audio_output.volume() * 100))
        self.volume_slider.setAccessibleName("Listening volume percent; playback only")
        self.volume_slider.setToolTip("Changes only what you hear in this player. 0 is silent. Saved audio is unchanged.")
        volume_label.setBuddy(self.volume_slider)
        self.volume_value = QLabel(f"{self.volume_slider.value()}%")
        self.volume_value.setMinimumWidth(40)
        self.volume_slider.valueChanged.connect(self._listening_volume_changed)
        volume_row.addWidget(volume_label)
        volume_row.addWidget(self.volume_slider, 1)
        volume_row.addWidget(self.volume_value)
        compare.layout().addLayout(volume_row)
        compare.layout().addWidget(self.compare_hint)

        status = self._section("Session")
        status_row = QHBoxLayout()
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        self.open_output = QPushButton("Open cleaned files")
        self.open_output.setIcon(self.style().standardIcon(QStyle.StandardPixmap.SP_DirOpenIcon))
        self.open_output.clicked.connect(self._open_output_folder)
        status_row.addWidget(self.status_label, 1)
        status_row.addWidget(self.open_output)
        status.layout().addLayout(status_row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        status.layout().addWidget(self.progress)
        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMinimumHeight(90)
        self.log_view.setPlaceholderText("Processing details will appear here.")
        self.log_toggle = QPushButton("Show technical details")
        self.log_toggle.setCheckable(True)
        self.log_toggle.toggled.connect(self.log_view.setVisible)
        self.log_toggle.toggled.connect(self.hardware_label.setVisible)
        self.log_toggle.toggled.connect(lambda checked: self.log_toggle.setText("Hide technical details" if checked else "Show technical details"))
        status.layout().addWidget(self.log_toggle)
        status.layout().addWidget(self.hardware_label)
        status.layout().addWidget(self.log_view)
        self.log_view.hide()
        self.hardware_label.hide()
        layout.addWidget(status)
        self._build_guide()
        self._set_comparison_enabled(False)
        for button in self.findChildren(QPushButton):
            button.setCursor(Qt.CursorShape.PointingHandCursor)

    def resizeEvent(self, event: QResizeEvent) -> None:
        super().resizeEvent(event)
        if hasattr(self, "control_columns"):
            direction = (QBoxLayout.Direction.TopToBottom if event.size().width() < 960
                         else QBoxLayout.Direction.LeftToRight)
            self.control_columns.setDirection(direction)

    def _page(self, title: str) -> QVBoxLayout:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setObjectName("workspaceScroll")
        root = QWidget()
        root.setObjectName("root")
        scroll.setWidget(root)
        layout = QVBoxLayout(root)
        layout.setContentsMargins(20, 16, 20, 16)
        layout.setSpacing(12)
        self.tabs.addTab(scroll, title)
        return layout

    def _hint(self, text: str) -> QLabel:
        label = QLabel(text)
        label.setObjectName("hint")
        label.setWordWrap(True)
        return label

    def _build_guide(self) -> None:
        layout = self._page("How to use")
        sections = (
            ("Quick start", "1. Add files or a folder in Control Room.\n2. Select a row, leave strength at 50 and click Prepare preview.\n3. Press Play in the Selected file panel. Use Listen A and Listen B to switch versions at the same position.\n4. Adjust Cleanup strength to update the live preview on side B.\n5. Check the destination and click Clean files to export the entire queue."),
            ("Keyboard shortcuts", "Ctrl+O adds recordings. Ctrl+P prepares the selected recording. Ctrl+Space plays or pauses a sample. Press Delete while the queue is focused to remove the selected row."),
            ("Preview and A/B listening", "Preview processes a 10-second excerpt of the selected recording. Change the excerpt in advanced options. Live preview follows Cleanup strength after a short pause. Original plays the source recording, Reference 50 uses the approved baseline, and Previous Reference uses the earlier engine. Switching recordings or changing the excerpt requires preparing again. Playback volume affects only the player. Exports calculate loudness over the full recording."),
            ("What does strength do?", "0 bypasses cleanup. The app copies the original file or extracts an unprocessed WAV if audio-only export is selected.\n1–49 keeps more of the original sound.\n50 uses the approved Reference baseline.\n51–100 applies stronger cleanup and may affect voice detail.\nTry changes of 5–10 points and listen again."),
            ("How the cleanup works", "A local speech model estimates which parts of the recording belong to speech and reduces other sound. The app then adjusts the result for listening. Detail recovery optionally blends a second model's estimate to try to retain more speech detail. It is experimental and slower. Neither mode can guarantee perfect separation or restore every missing word."),
            ("Full exports versus previews", "Clean files processes each full recording with the current settings. If Save A/B samples is enabled, the app saves comparison clips from the first file. Select that file in the queue to hear them. Saved comparison clips are fixed. Use Prepare preview to return to live tuning."),
            ("Managing the queue", "Add folder includes supported audio and video directly inside the folder. Files and folders add to the queue. Remove selected takes a recording out of the queue and keeps the original on disk. Every file uses the same settings and audio-track number. Stop after this file lets the active file finish and leaves the remaining files queued."),
            ("Where are my results?", "Click Open cleaned files in Session. Results have CLEAN in their filename. Audio inputs export as WAV, and videos normally retain their picture with cleaned audio. Existing results are skipped unless Replace existing CLEAN files is checked. To keep several versions, choose a different destination for each run."),
            ("Which advanced options do I need?", "Usually none. Detail recovery tries to retain more voice detail using two models. Source audio track chooses an embedded track in a video—not the left or right speaker; leave it at 1 unless you know otherwise. Listening sample start and length choose the section used for comparison. Replace existing CLEAN files allows you to redo previously saved results."),
            ("Troubleshooting and sound quality", "Tinny, metallic or missing speech: lower strength and try Detail recovery both off and on.\nToo much background noise: raise strength a little and check that words remain clear.\nSkipped: a cleaned result already exists; use a new folder or enable replacement.\nReview: fallback: the main isolation step did not succeed; listen carefully before using that result.\nFailed: expand technical details, check the file and destination, then retry.\nSlow processing: local models can take time, especially with Detail recovery. The app stays usable while work runs."),
            ("Supported files and portable setup", "Audio: WAV, FLAC, MP3, M4A, AAC, OGG, AIF, AIFF and WMA. Video: MOV, MP4 and MKV. Codec support also depends on the file. Keep the executable together with its _internal folder. Copy the entire FactorySpeechCleaner folder when moving the app; the EXE alone is not a standalone installation."),
        )
        for heading, copy in sections:
            section = self._section(heading)
            section.layout().addWidget(self._hint(copy))
            layout.addWidget(section)
        layout.addStretch()

    def _section(self, heading: str) -> QFrame:
        frame = QFrame()
        frame.setObjectName("section")
        box = QVBoxLayout(frame)
        box.setContentsMargins(14, 12, 14, 12)
        box.setSpacing(9)
        label = QLabel(heading)
        label.setObjectName("sectionTitle")
        box.addWidget(label)
        return frame

    def _apply_theme(self) -> None:
        self.setStyleSheet("""
            QMainWindow, QTabWidget, QWidget#root, QScrollArea#workspaceScroll { background: #11161d; }
            QWidget { font-family: 'Segoe UI'; font-size: 13px; color: #edf2f8; }
            QTabWidget::pane { border: 0; }
            QTabBar::tab { background: #1b232d; padding: 10px 18px; border-bottom: 2px solid #364454; }
            QTabBar::tab:selected { background: #11161d; color: #a4cfff; border-bottom: 2px solid #86baf0; }
            QTabBar::tab:hover { background: #283749; }
            QLabel#title { color: #f3f7fc; font-size: 25px; font-weight: 600; }
            QLabel#hint { color: #acbdd0; }
            QLabel#hardware { color: #b9cadd; background: #242f3c; border-radius: 5px; padding: 6px; }
            QLabel#sectionTitle { color: #edf2f8; font-size: 16px; font-weight: 600; }
            QLabel#fieldLabel, QLabel#selectedFile { color: #edf2f8; font-weight: 600; }
            QFrame#section { background: #1b232d; border: 1px solid #364454; border-radius: 8px; }
            QLineEdit, QComboBox, QDoubleSpinBox, QSpinBox, QTextEdit {
                background: #141c25; border: 1px solid #485b70; border-radius: 5px; padding: 5px 7px;
                selection-background-color: #a4cfff; selection-color: #14263a;
            }
            QComboBox QAbstractItemView { background: #1b232d; color: #edf2f8;
                selection-background-color: #a4cfff; selection-color: #14263a; }
            QTableWidget { background: #141c25; alternate-background-color: #19232e; border: 1px solid #364454;
                selection-background-color: #294663; selection-color: #f3f7fc; }
            QTableWidget::item { padding-left: 7px; }
            QHeaderView::section { background: #242f3c; color: #c4d4e5; border: 0;
                border-bottom: 1px solid #364454; padding: 6px 7px; font-weight: 600; }
            QCheckBox { spacing: 7px; }
            QToolTip { background: #242f3c; color: #edf2f8; border: 1px solid #485b70; }
            QPushButton { background: #242f3c; border: 1px solid #485b70; border-radius: 5px; padding: 6px 10px; }
            QPushButton:hover { background: #304359; border-color: #7292b2; }
            QPushButton:disabled { color: #9bacc0; background: #202934; border-color: #364454; }
            QPushButton#primary:enabled { background: #a4cfff; color: #14263a; border-color: #a4cfff; font-weight: 600; }
            QPushButton#primary:hover:enabled { background: #c2dfff; }
            QPushButton[ab='true']:checked:enabled { background: #294663; color: #f3f7fc; border-color: #86baf0; }
            QPushButton:focus, QLineEdit:focus, QComboBox:focus, QDoubleSpinBox:focus,
            QSpinBox:focus, QTextEdit:focus, QTableWidget:focus { border: 1px solid #a4cfff; }
            QProgressBar { background: #141c25; border: 1px solid #364454; border-radius: 4px;
                min-height: 17px; text-align: center; }
            QProgressBar::chunk { background: #2c5d89; border-radius: 3px; }
            QSlider::groove:horizontal { background: #455b71; height: 5px; border-radius: 2px; }
            QSlider::handle:horizontal { background: #86baf0; width: 14px; margin: -4px 0; border-radius: 7px; }
            QSlider::handle:horizontal:focus { background: #c2dfff; }
        """)

    @Slot()
    def _browse_input(self) -> None:
        patterns = " ".join(f"*{extension}" for extension in sorted(SUPPORTED_EXTENSIONS))
        start = self._selected_source()
        selected, _ = QFileDialog.getOpenFileNames(self, "Add audio or video files", str(start.parent if start else PROJECT_ROOT), f"Audio and video ({patterns})")
        if selected:
            self._set_sources([Path(item) for item in selected], append=True)

    @Slot()
    def _browse_input_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select folder of audio/video files", str(PROJECT_ROOT))
        if selected:
            try:
                sources = discover_inputs(Path(selected))
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "Folder unavailable", str(exc))
                return
            if not sources:
                QMessageBox.information(self, "No media found", "That folder contains no supported audio or video files.")
                return
            self._set_sources(sources, append=True)

    @Slot()
    def _browse_output(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "Select output folder", self.output_edit.text() or str(PROJECT_ROOT))
        if selected:
            self.output_edit.setText(selected)

    @Slot(int)
    def _strength_changed(self, value: int) -> None:
        if value == 0:
            copy = "No cleanup. Saves an exact copy, or an unprocessed WAV in audio-only mode."
        elif value < 50:
            copy = "Gentler cleanup. Keeps more original sound; try this if the voice sounds unnatural."
        elif value == 50:
            copy = "Reference 50 · Recommended starting point for the whole queue."
        else:
            copy = "Stronger cleanup. May remove more noise, but can also affect voice detail. Listen before increasing further."
        if value > 0 and hasattr(self, "detail_check") and self.detail_check.isChecked():
            copy += " Detail recovery is enabled; compare against Reference before batching."
        self.strength_description.setText(copy)

    @Slot()
    def _prepare_preview(self):
        if self._selected_source() is None or self.thread or self.preview_thread:
            return
        self.preview_active = True
        self._render_preview()

    def _schedule_preview(self, *_):
        if self.preview_active:
            self.preview_status.setText("Strength changed — updating shortly. The previous version plays until ready.")
            self.preview_timer.start()

    def _invalidate_preview(self, *_):
        self.preview_active = False
        self.preview_timer.stop()
        self.preview_status.setText("Prepare a preview to hear the selected recording.")
        self.pending_position = 0
        self.resume_after_switch = False
        self.player.stop()
        self.player.setSource(QUrl())
        self.variants = {}
        self._set_comparison_enabled(False)

    def _render_preview(self):
        source = self._selected_source()
        if source is None or not self.preview_active or self.preview_thread or self.thread:
            return
        self.preview_timer.stop()
        self.preview_request = (self.strength_slider.value(), self.detail_check.isChecked())
        request = (source, self.start_spin.value(), self.duration_spin.value(),
                   self.audio_track_spin.value() - 1, self.preview_request[1], self.preview_request[0])
        self.preview_status.setText("Preparing/updating preview… First use of each model takes longer. No full files are being exported.")
        self._set_processing_controls(False)
        self.strength_slider.setEnabled(True)
        self.strength_spin.setEnabled(True)
        self.process_button.setEnabled(False)
        self.preview_button.setEnabled(False)
        self.preview_thread = QThread(self)
        self.preview_worker = PreviewWorker(self.preview_cache, request)
        self.preview_worker.moveToThread(self.preview_thread)
        self.preview_thread.started.connect(self.preview_worker.run)
        self.preview_worker.completed.connect(self._preview_complete)
        self.preview_worker.completed.connect(self.preview_thread.quit)
        self.preview_thread.finished.connect(self.preview_worker.deleteLater)
        self.preview_thread.finished.connect(self._preview_finished)
        self.preview_thread.start()

    @Slot(dict)
    def _preview_complete(self, result):
        if "error" in result:
            self.preview_active = False
            self.preview_status.setText(f"Preview failed: {result['error']}. Choose a valid excerpt and try Prepare preview again.")
            return
        self.preview_rendered = (result["strength"], result["detail"])
        if self.preview_rendered != (self.strength_slider.value(), self.detail_check.isChecked()):
            return  # Never replace playback with an obsolete slider request.
        self.preview_timer.stop()
        first = "Live preview" not in self.variants
        self.variants = result["variants"]
        if first:
            self._set_comparison_enabled(True)
            self.b_combo.setCurrentText("Live preview")
            self._switch_side("B", force=True)
        elif self._variant_for_side(self.active_side) == self.variants["Live preview"]:
            self._switch_side(self.active_side, force=True)
        detail = " · Detail recovery" if result["detail"] else ""
        self.preview_status.setText(f"Ready: strength {result['strength']}{detail} · updated in {result['seconds']:.2f}s")
        self.compare_hint.setText(f"Excerpt starts at {self.start_spin.value():g}s · up to {self.duration_spin.value():g}s. Switch A/B during playback.")

    @Slot()
    def _preview_finished(self):
        self.preview_thread.wait()
        self.preview_thread.deleteLater()
        self.preview_thread = None
        self.preview_worker = None
        self._set_processing_controls(True)
        self.process_button.setEnabled(bool(self.sources))
        self.preview_button.setEnabled(self._selected_source() is not None)
        if self.preview_active and self.preview_rendered != (self.strength_slider.value(), self.detail_check.isChecked()):
            self.preview_timer.start()

    @Slot()
    def _start_processing(self) -> None:
        if self.preview_thread or self.thread:
            return
        sources = list(self.sources)
        if not sources:
            QMessageBox.warning(self, "Queue is empty", "Select audio/video files or a folder to process.")
            return
        output = self._ensure_output_directory()
        if output is None:
            return
        self.player.stop()
        self.preview_active = False
        self.preview_timer.stop()
        self._set_comparison_enabled(False)
        self.process_button.setEnabled(False)
        self._set_processing_controls(False)
        self.cancel_button.setEnabled(True)
        self.progress.setRange(0, len(sources))
        self.progress.setValue(0)
        self.progress.setTextVisible(True)
        self.progress.setFormat("0 of %m files")
        self.status_label.setText(f"Processing {len(sources)} files. The interface will remain responsive.")
        self.log_view.clear()
        for row in range(self.queue_table.rowCount()):
            self._update_item_status(row, "Waiting", "Queued")

        self.thread = QThread(self)
        self.worker = ProcessingWorker(
            sources, output, self.strength_slider.value(), self.start_spin.value(), self.duration_spin.value(),
            self.comparison_check.isChecked(), self.overwrite_check.isChecked(),
            self.audio_track_spin.value() - 1, self.audio_only_check.isChecked(),
            self.detail_check.isChecked(),
        )
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.message.connect(self._append_log)
        self.worker.item_status.connect(self._update_item_status)
        self.worker.progress.connect(self._update_progress)
        self.worker.eta.connect(self.status_label.setText)
        self.worker.completed.connect(self._processing_complete)
        self.worker.failed.connect(self._processing_failed)
        self.worker.completed.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self._thread_finished)
        self.thread.start()

    @Slot(str)
    def _append_log(self, message: str) -> None:
        self.log_view.append(message)
        bar = self.log_view.verticalScrollBar()
        bar.setValue(bar.maximum())

    @Slot(dict)
    def _processing_complete(self, payload: dict[str, Any]) -> None:
        comparison_source = payload.get("comparison_source")
        self.variants = {}
        if comparison_source and Path(comparison_source) == self._selected_source():
            self.variants = {name: Path(path) for name, path in payload["comparisons"].items()}
        counts = payload["counts"]
        self.progress.setValue(sum(counts.values()))
        summary = f"{counts['complete']} complete, {counts['skipped']} skipped, {counts['failed']} failed"
        if payload.get("needs_review"):
            summary += f"; {payload['needs_review']} used fallback audio — review these files"
        if payload.get("cancelled"):
            summary += "; queue cancelled"
        if self.variants:
            self._set_comparison_enabled(True)
            self._switch_side("A", force=True)
        self._append_log(f"Session log: {payload['log']}")
        if self.variants:
            next_step = " Press Play to hear the samples."
        elif comparison_source:
            next_step = f" Select {Path(comparison_source).name} to hear the saved samples."
        else:
            next_step = " Use Open cleaned files to find your results."
        self.status_label.setText(f"Finished: {summary}.{next_step}")

    @Slot(str)
    def _processing_failed(self, error: str) -> None:
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status_label.setText("Processing failed. See the details below.")
        QMessageBox.critical(self, "Processing failed", error)

    @Slot()
    def _thread_finished(self) -> None:
        self.process_button.setEnabled(bool(self.sources))
        self._set_processing_controls(True)
        self.cancel_button.setEnabled(False)
        if self.worker:
            self.worker.deleteLater()
        if self.thread:
            self.thread.deleteLater()
        self.worker = None
        self.thread = None

    @Slot()
    def _request_cancel(self) -> None:
        if self.worker:
            self.worker.request_cancel()
            self.cancel_button.setEnabled(False)
            self.status_label.setText("Cancellation requested. The current file will finish safely; remaining files will not start.")

    @Slot(int, int)
    def _update_progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.progress.setFormat(f"{done} of {total} files")

    @Slot(int, str, str)
    def _update_item_status(self, row: int, status: str, detail: str) -> None:
        if 0 <= row < self.queue_table.rowCount():
            self.queue_table.setItem(row, 1, QTableWidgetItem(status))
            item = QTableWidgetItem(detail)
            item.setToolTip(detail)
            self.queue_table.setItem(row, 2, item)

    def _set_sources(self, sources: list[Path], *, append: bool = False) -> None:
        existing = self.sources if append else []
        unique = existing.copy()
        seen = set(existing)
        for source in sources:
            resolved = source.resolve()
            if resolved not in seen:
                seen.add(resolved)
                unique.append(resolved)
        if append and len(unique) == len(existing):
            self.status_label.setText("Those recordings are already queued.")
            return
        self.sources = unique
        self._update_queue_count()
        select_first = not append or self.queue_table.currentRow() < 0
        with QSignalBlocker(self.queue_table):
            self.queue_table.setRowCount(len(unique))
            for row, source in enumerate(unique[len(existing):], start=len(existing)):
                recording = QTableWidgetItem(source.name)
                recording.setToolTip(str(source))
                self.queue_table.setItem(row, 0, recording)
                self._update_item_status(row, "Waiting", str(source.parent))
            if unique and select_first:
                self.queue_table.setCurrentCell(0, 0)
        if not append or len(existing) < 2:
            self.comparison_check.setChecked(len(unique) == 1)
        if unique:
            self.tabs.setCurrentIndex(0)
        if select_first:
            self._queue_selection_changed()
        if append:
            added = len(unique) - len(existing)
            self.status_label.setText(f"Added {added} recording{'s' if added != 1 else ''} to the queue.")

    def _update_queue_count(self) -> None:
        count = len(self.sources)
        files = f"{count} file{'s' if count != 1 else ''}"
        self.queue_count.setText(files)
        self.process_button.setText(f"Clean {files}" if count else "Clean files")
        self.process_button.setEnabled(bool(count))

    @Slot()
    def _remove_selected_source(self) -> None:
        row = self.queue_table.currentRow()
        if self.thread or self.preview_thread or not 0 <= row < len(self.sources):
            return
        source = self.sources.pop(row)
        with QSignalBlocker(self.queue_table):
            self.queue_table.removeRow(row)
            if self.sources:
                self.queue_table.setCurrentCell(min(row, len(self.sources) - 1), 0)
        self._update_queue_count()
        self._queue_selection_changed()
        self.status_label.setText(f"Removed {source.name} from the queue.")

    def _selected_source(self) -> Path | None:
        row = self.queue_table.currentRow()
        return self.sources[row] if 0 <= row < len(self.sources) else None

    @Slot(int, int, int, int)
    def _queue_cell_changed(self, row: int, column: int, previous_row: int, previous_column: int) -> None:
        if row != previous_row:
            self._queue_selection_changed()

    @Slot()
    def _queue_selection_changed(self) -> None:
        self._invalidate_preview()
        source = self._selected_source()
        self.preview_button.setEnabled(source is not None)
        self.remove_button.setEnabled(source is not None)
        self.selected_file_label.setText(source.name if source else "No recording selected")
        self.selected_file_label.setToolTip(str(source) if source else "")
        if source:
            self._load_manifest_for(source)
        else:
            self.preview_status.setText("Select a recording in the queue to prepare a preview.")

    def _set_processing_controls(self, enabled: bool) -> None:
        for widget in (
            self.file_button, self.folder_button, self.output_button, self.output_edit,
            self.start_spin, self.duration_spin, self.comparison_check, self.overwrite_check,
            self.audio_track_spin, self.audio_only_check,
            self.detail_check,
            self.preview_button, self.queue_table,
        ):
            widget.setEnabled(enabled)
        self.strength_slider.setEnabled(enabled)
        self.strength_spin.setEnabled(enabled)
        self.remove_button.setEnabled(enabled and self._selected_source() is not None)

    def dragEnterEvent(self, event: QDragEnterEvent) -> None:
        paths = [Path(url.toLocalFile()) for url in event.mimeData().urls() if url.isLocalFile()]
        if self.preview_thread or (self.thread and self.thread.isRunning()):
            return
        if any(path.is_dir() or path.suffix.lower() in SUPPORTED_EXTENSIONS for path in paths):
            event.acceptProposedAction()

    def dropEvent(self, event: QDropEvent) -> None:
        if self.preview_thread or (self.thread and self.thread.isRunning()):
            return
        sources: list[Path] = []
        for url in event.mimeData().urls():
            if not url.isLocalFile():
                continue
            path = Path(url.toLocalFile())
            if path.is_dir():
                sources.extend(discover_inputs(path))
            elif path.suffix.lower() in SUPPORTED_EXTENSIONS:
                sources.append(path)
        if sources:
            self._set_sources(sources, append=True)
            event.acceptProposedAction()

    def _load_manifest_for(self, source: Path) -> None:
        manifest_path = PROJECT_ROOT / "comparison" / "manifest.json"
        self.variants = {}
        if manifest_path.is_file():
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                if Path(manifest["source"]).resolve() == source.resolve():
                    variants = {name: Path(path) for name, path in manifest["files"].items()}
                    if all(path.is_file() for path in variants.values()):
                        self.variants = variants
            except (OSError, ValueError, KeyError, TypeError):
                pass
        self._set_comparison_enabled(bool(self.variants))
        if self.variants:
            self.preview_status.setText("Saved samples ready for the selected recording.")
            self._switch_side("A", force=True)

    def _set_comparison_enabled(self, enabled: bool) -> None:
        self.compare_hint.setText("Listening samples are ready. Use A/B to compare the same moment." if enabled else "")
        if not enabled:
            self.waveform.set_audio(None)
        labels = list(self.variants) if enabled else []
        if labels:
            original = next((label for label in labels if label.casefold() == "original"), labels[0])
            alternatives = [label for label in labels if self.variants[label] != self.variants[original]]
            reference = next((label for label in alternatives if label.casefold() in ("reference 50", "reference")),
                             alternatives[0] if alternatives else original)
            preferred = next((label for label in alternatives if label.casefold().startswith("selected")), reference)
        else:
            original = preferred = ""
        for combo, selection in ((self.a_combo, original), (self.b_combo, preferred)):
            with QSignalBlocker(combo):
                combo.clear()
                combo.addItems(labels)
                combo.setCurrentText(selection)
        for widget in (self.a_combo, self.b_combo, self.a_button, self.b_button, self.play_button, self.seek):
            widget.setEnabled(enabled)

    def _variant_for_side(self, side: str) -> Path | None:
        label = self.a_combo.currentText() if side == "A" else self.b_combo.currentText()
        return self.variants.get(label)

    def _switch_side(self, side: str, force: bool = False) -> None:
        if not self.variants or (side == self.active_side and not force):
            return
        path = self._variant_for_side(side)
        if not path or not path.is_file():
            return
        position = self.pending_position or self.player.position()
        resume = self.resume_after_switch or self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState
        self.changing_source = True
        self.player.stop()
        self.pending_position = position
        self.resume_after_switch = resume
        self.active_side = side
        self.a_button.setChecked(side == "A")
        self.b_button.setChecked(side == "B")
        self.waveform.set_audio(path)
        self.player.setSource(QUrl.fromLocalFile(str(path)))
        self.changing_source = False
        if self.player.mediaStatus() == QMediaPlayer.MediaStatus.LoadedMedia:
            self._media_status_changed(self.player.mediaStatus())

    def _combo_changed(self, side: str) -> None:
        if side == self.active_side and self.variants:
            self._switch_side(side, force=True)

    @Slot()
    def _toggle_playback(self) -> None:
        if self.player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    @Slot(int)
    def _listening_volume_changed(self, value: int) -> None:
        # QAudioOutput controls device playback only; never pass this to processing.
        self.audio_output.setVolume(value / 100.0)
        self.volume_value.setText("Muted" if value == 0 else f"{value}%")

    def _media_status_changed(self, status: QMediaPlayer.MediaStatus) -> None:
        if self.changing_source:
            return  # setSource can emit LoadedMedia for the outgoing file.
        if status in (QMediaPlayer.MediaStatus.LoadedMedia, QMediaPlayer.MediaStatus.BufferedMedia):
            position, resume = self.pending_position, self.resume_after_switch
            # Seeking/playing can synchronously emit another media-status event.
            # Consume the request first so it cannot recursively restart playback.
            self.pending_position = 0
            self.resume_after_switch = False
            if position:
                self.player.setPosition(min(position, self.player.duration()))
            if resume:
                self.player.play()

    def _playback_state_changed(self, state: QMediaPlayer.PlaybackState) -> None:
        playing = state == QMediaPlayer.PlaybackState.PlayingState
        self.play_button.setText("Pause" if playing else "Play")
        icon = QStyle.StandardPixmap.SP_MediaPause if playing else QStyle.StandardPixmap.SP_MediaPlay
        self.play_button.setIcon(self.style().standardIcon(icon))

    def _position_changed(self, position: int) -> None:
        duration = self.player.duration()
        if duration > 0 and not self.user_seeking:
            self.seek.setValue(round(position / duration * 1000))
        self.waveform.playhead = min(1.0, position / duration) if duration > 0 else 0.0
        self.waveform.update()
        self.time_label.setText(f"{format_time(position)} / {format_time(duration)}")

    def _duration_changed(self, duration: int) -> None:
        self.time_label.setText(f"{format_time(self.player.position())} / {format_time(duration)}")

    def _seek_released(self) -> None:
        self.user_seeking = False
        duration = self.player.duration()
        if duration > 0:
            self.player.setPosition(round(self.seek.value() / 1000 * duration))

    def _player_error(self, error: QMediaPlayer.Error, message: str) -> None:
        if error != QMediaPlayer.Error.NoError:
            self.status_label.setText(f"Audio playback error: {message}")

    def _ensure_output_directory(self) -> Path | None:
        path = self.output_edit.text().strip()
        if not path:
            QMessageBox.warning(self, "Choose a save folder", "Choose where you want to save the cleaned files.")
            return None
        folder = Path(path)
        try:
            folder.mkdir(parents=True, exist_ok=True)
            return folder.resolve()
        except OSError as exc:
            QMessageBox.warning(self, "Cannot use this save folder", f"Choose a folder you can write to.\n\n{exc}")
            return None

    @Slot()
    def _open_output_folder(self) -> None:
        folder = self._ensure_output_directory()
        if folder is not None and not QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder))):
            QMessageBox.warning(self, "Cannot open folder", f"Open this folder in File Explorer:\n{folder}")

    def closeEvent(self, event: QCloseEvent) -> None:
        if self.preview_thread or (self.thread and self.thread.isRunning()):
            QMessageBox.information(self, "Processing in progress", "Wait for processing to finish before closing the application.")
            event.ignore()
            return
        self.player.stop()
        self.player.setSource(QUrl())
        self.preview_timer.stop()
        self.preview_cache.close()
        self.settings.setValue("playback/volume", self.volume_slider.value())
        destination = self.output_edit.text().strip()
        if destination and Path(destination).is_dir():
            self.settings.setValue("export/folder", str(Path(destination).resolve()))
        event.accept()


def find_standard_sample() -> Path | None:
    candidates = [
        Path.cwd() / STANDARD_SAMPLE_NAME,
        PROJECT_ROOT / STANDARD_SAMPLE_NAME,
        PROJECT_ROOT.parent / STANDARD_SAMPLE_NAME,
        PROJECT_ROOT.parent.parent / STANDARD_SAMPLE_NAME,
        Path.home() / "Downloads" / STANDARD_SAMPLE_NAME,
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return None


def format_time(milliseconds: int) -> str:
    seconds = max(0, milliseconds // 1000)
    return f"{seconds // 60:02d}:{seconds % 60:02d}"


def main() -> int:
    if "--batch" in sys.argv:
        from .main import main as batch_main
        index = sys.argv.index("--batch")
        # A windowed executable has no standard output streams. CLI diagnostics
        # remain available in the persistent processing log and per-file reports.
        import os
        from contextlib import redirect_stdout, redirect_stderr
        with open(os.devnull, "w") as sink, redirect_stdout(sink), redirect_stderr(sink):
            return batch_main(sys.argv[index + 1:])
    if "--pipeline-smoke-test" in sys.argv:
        index = sys.argv.index("--pipeline-smoke-test")
        if index + 1 >= len(sys.argv):
            return 2
        source = Path(sys.argv[index + 1]).resolve()
        try:
            logger, _ = create_session_logger(PROJECT_ROOT / "logs")
            result = process_video(source, PROJECT_ROOT / "output", "reference", "clearvoice", logger, False, True, 50)
            return 0 if result.get("status") == "complete" and not result.get("isolation_fallback") else 3
        except Exception:
            return 4
    app = QApplication(sys.argv)
    app.styleHints().setColorScheme(Qt.ColorScheme.Dark)
    app.setApplicationName("Factory Speech Cleaner")
    app.setOrganizationName("Local Tools")
    window = MainWindow()
    window.show()
    if "--smoke-test" in sys.argv:
        QTimer.singleShot(1200, app.quit)
    return app.exec()
