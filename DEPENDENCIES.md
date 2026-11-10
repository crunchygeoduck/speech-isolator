# Third-party components

This application uses third-party software and model checkpoints under their respective upstream terms.

- **ClearVoice 0.1.2** and **MossFormer2 SE 48 kHz** provide the primary speech model. See the [ClearerVoice-Studio source](https://github.com/modelscope/ClearerVoice-Studio) and [model card and checkpoint](https://huggingface.co/alibabasglab/MossFormer2_SE_48K).
- **DeepFilterNet 0.5.6** and the **DeepFilterNet3** checkpoint provide optional detail recovery and fallback isolation. See the [DeepFilterNet source and model downloads](https://github.com/Rikorose/DeepFilterNet).
- **FFmpeg and FFprobe 8.1.1** in the Windows build are the [Gyan full build](https://github.com/GyanD/codexffmpeg/releases/tag/8.1.1), built with GPL components. See the [FFmpeg source release](https://ffmpeg.org/releases/ffmpeg-8.1.1.tar.xz) and [licensing information](https://ffmpeg.org/legal.html). The app calls the executables as subprocesses.
- **Qt and PySide6 6.9.1** provide the desktop UI. See the [Qt 6.9.1 sources](https://download.qt.io/archive/qt/6.9/6.9.1/single/), [PySide source](https://code.qt.io/cgit/pyside/pyside-setup.git/tag/?h=v6.9.1), and [Qt licensing information](https://doc.qt.io/qt-6/licensing.html). Qt libraries are separate DLLs in the portable folder.
- **Python**, **PyTorch**, and **PyInstaller** provide the runtime, inference, and packaging. See the [Python license](https://docs.python.org/3.11/license.html), [PyTorch source](https://github.com/pytorch/pytorch/tree/v2.2.2), and [PyInstaller license](https://pyinstaller.org/en/stable/license.html).
