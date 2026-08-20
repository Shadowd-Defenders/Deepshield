"""Run a local WAV file through WavLM Base+ and print the output shape."""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np
import torch
import torchaudio.functional as audio_functional
from transformers import AutoFeatureExtractor, WavLMModel

MODEL_ID = "microsoft/wavlm-base-plus"
TARGET_SAMPLE_RATE = 16_000
MODULE_ROOT = Path(__file__).resolve().parents[1]
HF_CACHE_DIR = MODULE_ROOT / "checkpoints" / "huggingface"


def load_mono_waveform(path: Path) -> tuple[torch.Tensor, int]:
    if not path.exists():
        raise FileNotFoundError(f"WAV file does not exist: {path}")

    with wave.open(str(path), "rb") as wav_file:
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        sample_rate = wav_file.getframerate()
        frame_count = wav_file.getnframes()
        raw_frames = wav_file.readframes(frame_count)

    if channels < 1:
        raise ValueError(f"Expected at least one audio channel, got {channels}")

    if sample_width == 1:
        audio = np.frombuffer(raw_frames, dtype=np.uint8).astype(np.float32)
        audio = (audio - 128.0) / 128.0
    elif sample_width == 2:
        audio = np.frombuffer(raw_frames, dtype="<i2").astype(np.float32)
        audio = audio / 32768.0
    elif sample_width == 4:
        audio = np.frombuffer(raw_frames, dtype="<i4").astype(np.float32)
        audio = audio / 2147483648.0
    else:
        raise ValueError(
            f"Unsupported PCM sample width: {sample_width} bytes. "
            "Use an 8-bit, 16-bit, or 32-bit PCM WAV file."
        )

    if audio.size % channels != 0:
        raise ValueError("WAV data length is not divisible by the channel count.")

    audio = audio.reshape(-1, channels).mean(axis=1)
    return torch.from_numpy(audio), sample_rate


def resample_if_needed(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    if sample_rate == TARGET_SAMPLE_RATE:
        return waveform
    return audio_functional.resample(
        waveform,
        orig_freq=sample_rate,
        new_freq=TARGET_SAMPLE_RATE,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Load a local WAV file, resample it to 16 kHz, run it through "
            "WavLM Base+, and print the output tensor shape."
        )
    )
    parser.add_argument("wav_path", type=Path, help="Path to a local WAV file.")
    parser.add_argument(
        "--allow-download",
        action="store_true",
        help="Allow Hugging Face downloads if WavLM is not already cached.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    print(f"Loading WAV file: {args.wav_path}")
    waveform, original_sample_rate = load_mono_waveform(args.wav_path)
    waveform = resample_if_needed(waveform, original_sample_rate)

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        MODEL_ID,
        cache_dir=HF_CACHE_DIR,
        local_files_only=not args.allow_download,
    )
    model = WavLMModel.from_pretrained(
        MODEL_ID,
        cache_dir=HF_CACHE_DIR,
        local_files_only=not args.allow_download,
    )
    model.eval()

    inputs = feature_extractor(
        waveform.numpy(),
        sampling_rate=TARGET_SAMPLE_RATE,
        return_tensors="pt",
    )

    with torch.no_grad():
        outputs = model(**inputs)

    print("WavLM Base+ feature extraction completed.")
    print("This output is a speech representation tensor, not a deepfake verdict.")
    print(f"Original sample rate: {original_sample_rate}")
    print(f"Resampled sample rate: {TARGET_SAMPLE_RATE}")
    print(f"Last hidden state shape: {tuple(outputs.last_hidden_state.shape)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
