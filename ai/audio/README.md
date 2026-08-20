# DeepVerify Audio AI

This module is the isolated audio AI development environment for DeepVerify.

Current scope:

- Use Python 3.12.
- Load Microsoft WavLM Base+ from `microsoft/wavlm-base-plus`.
- Verify local PyTorch, Transformers, CPU, CUDA, and MPS environment details.
- Run a local WAV file through WavLM as a pretrained feature extractor.

Out of scope for this stage:

- No ASVspoof download.
- No model training.
- No final deepfake classifier.
- No claims that WavLM alone is a deepfake detector.

WavLM Base+ is only a pretrained speech representation model at this stage. It produces features that may later feed a trained detector.

## Folder Layout

- `data/raw/`: local raw audio files, not committed.
- `data/processed/`: derived local audio data, not committed.
- `data/test/`: small local test WAV files, not committed.
- `models/`: exported model artifacts, not committed.
- `checkpoints/`: downloaded pretrained checkpoints or future fine-tuned checkpoints, not committed.
- `results/`: local experiment outputs, not committed.
- `src/`: module scripts.
- `configs/`: future audio configuration files.

## Environment Setup

Use Python 3.12.

From the repository root on Windows PowerShell:

```powershell
cd ai/audio
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

If `py -3.12` is unavailable, use another Python 3.12 executable explicitly:

```powershell
cd ai/audio
C:\Path\To\Python312\python.exe -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## Checks

Report the environment:

```powershell
.\.venv\Scripts\python.exe src\check_environment.py
```

Download and load WavLM Base+ into `checkpoints/huggingface`:

```powershell
.\.venv\Scripts\python.exe src\download_wavlm.py
```

Run a local PCM WAV file through the cached WavLM model:

```powershell
.\.venv\Scripts\python.exe src\test_wavlm.py data\test\sample.wav
```

`test_wavlm.py` loads a local PCM WAV file, resamples it to 16 kHz, and prints the output tensor shape.
Run `src\download_wavlm.py` first, or pass `--allow-download` if you want the test script to download missing WavLM files.

## Experiment 1: Frozen WavLM Binary Head

Config:

```powershell
configs\exp001_frozen_wavlm.yaml
```

This experiment uses WavLM Base+ as a frozen pretrained encoder and trains only a binary classification head for `bonafide` vs `spoof`. It is not a final detector until it has been trained and evaluated on real held-out data.

Expected manifest:

```csv
path,label,split
..\raw\example_bonafide.wav,bonafide,train
..\raw\example_spoof.wav,spoof,validation
..\raw\example_test.wav,bonafide,test
```

Splits must be explicit: `train`, `validation`, and `test`.

Run the mechanics-only sanity check first:

```powershell
.\.venv\Scripts\python.exe src\train.py --config configs\exp001_frozen_wavlm.yaml --sanity-batch
```

Run training only after a real manifest/protocol exists:

```powershell
.\.venv\Scripts\python.exe src\train.py --config configs\exp001_frozen_wavlm.yaml
```

Evaluate a trained checkpoint:

```powershell
.\.venv\Scripts\python.exe src\evaluate.py --config configs\exp001_frozen_wavlm.yaml --split test
```

Run inference on one local PCM WAV file only after a trained checkpoint exists:

```powershell
.\.venv\Scripts\python.exe src\inference.py data\test\sample.wav
```
