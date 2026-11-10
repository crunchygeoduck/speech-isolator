# Speech Isolator

Factory Speech Cleaner is a Windows desktop app for isolating speech in audio and video files. The dark Control Room combines a processing queue, a waveform preview, and synchronized A/B listening. Inference runs locally.

## Windows setup

Clone the repository or download the source ZIP from [Releases](https://github.com/crunchygeoduck/speech-isolator/releases). Follow the setup below to install the dependencies from their upstream providers, run the app, or build a portable Windows folder.

Add files or folders to build a queue, choose a row, and use **Prepare preview** to listen at strength 50. Use **Remove selected** to take out a row. Adjust strength while listening, then use **Clean files** to export the queue. Strength 0 bypasses cleanup. Audio-only mode still extracts a WAV. The app remembers playback volume and the existing output folder across sessions.

Use **Ctrl+O** to add files, **Ctrl+P** to prepare a preview, and **Ctrl+Space** to play or pause. **Delete** removes the selected row when the queue has keyboard focus.

## Run from source

Use 64-bit Python 3.11 on Windows. Put [FFmpeg and FFprobe](https://www.gyan.dev/ffmpeg/builds/) on PATH. Run these commands from the repository root in PowerShell.

```powershell
py -3.11 -m venv .venv
& .\.venv\Scripts\python.exe -m pip install torch==2.2.2 torchaudio==2.2.2 torchvision==0.17.2 --index-url https://download.pytorch.org/whl/cpu
& .\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
& .\.venv\Scripts\hf.exe download alibabasglab/MossFormer2_SE_48K last_best_checkpoint last_best_checkpoint.pt --local-dir checkpoints/MossFormer2_SE_48K
& .\.venv\Scripts\python.exe gui_main.py
```

Optional Detail recovery uses DeepFilterNet3 as a second model. To cache it for local development or packaging, run:

```powershell
& .\.venv\Scripts\python.exe -c "from df.enhance import maybe_download_model; print(maybe_download_model('DeepFilterNet3'))"
```

Build the portable folder with `./build_exe.ps1`. The executable and its runtime are in `dist/FactorySpeechCleaner`. Run the automated checks with `.\.venv\Scripts\python.exe -m pytest -q`.

The CLI can process one recording or a folder:

```powershell
& .\.venv\Scripts\python.exe main.py "D:\Recordings" --output "D:\Cleaned" --strength 50
```

[Third-party component information](DEPENDENCIES.md)
