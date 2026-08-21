"""Dataset utilities for WavLM binary classification experiments."""

from __future__ import annotations

import csv
import random
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio
import torchaudio.functional as audio_functional
from torch.utils.data import Dataset


@dataclass(frozen=True)
class AudioExample:
    path: Path
    label: int
    split: str


def load_pcm_wav_mono(path: Path) -> tuple[torch.Tensor, int]:
    """Load an 8/16/32-bit PCM WAV file as mono float32 waveform."""
    if not path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {path}")

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
        raise ValueError("WAV data length is not divisible by channel count.")

    audio = audio.reshape(-1, channels).mean(axis=1)
    return torch.from_numpy(audio), sample_rate


def _load_flac_mono(path: Path) -> tuple[torch.Tensor, int]:
    """Load a FLAC file as mono float32 waveform."""
    if not path.exists():
        raise FileNotFoundError(f"Audio file does not exist: {path}")

    try:
        waveform, sample_rate = torchaudio.load(str(path))
    except Exception as exc:
        raise RuntimeError(f"Failed to load FLAC audio file '{path}': {exc}") from exc

    if waveform.numel() == 0:
        raise ValueError(f"Audio file contains no samples: {path}")

    if waveform.ndim == 1:
        mono = waveform
    elif waveform.ndim == 2:
        if waveform.shape[0] < 1:
            raise ValueError(f"Expected at least one audio channel in {path}")
        mono = waveform.mean(dim=0)
    else:
        raise ValueError(
            f"Expected FLAC waveform with 1 or 2 dimensions, got {waveform.ndim}: {path}"
        )

    return mono.to(dtype=torch.float32), sample_rate


def _load_audio_mono(path: Path) -> tuple[torch.Tensor, int]:
    extension = path.suffix.lower()
    if extension == ".wav":
        waveform, sample_rate = load_pcm_wav_mono(path)
        return waveform.to(dtype=torch.float32), sample_rate
    if extension == ".flac":
        return _load_flac_mono(path)

    raise ValueError(
        f"Unsupported audio extension '{path.suffix}' for file: {path}. "
        "Supported extensions are: .wav, .flac."
    )


def resample_if_needed(
    waveform: torch.Tensor,
    original_sample_rate: int,
    target_sample_rate: int,
) -> torch.Tensor:
    if original_sample_rate == target_sample_rate:
        return waveform
    return audio_functional.resample(
        waveform,
        orig_freq=original_sample_rate,
        new_freq=target_sample_rate,
    )


def crop_or_pad(
    waveform: torch.Tensor,
    target_samples: int,
    random_crop: bool,
) -> torch.Tensor:
    if waveform.numel() == target_samples:
        return waveform

    if waveform.numel() > target_samples:
        max_start = waveform.numel() - target_samples
        start = random.randint(0, max_start) if random_crop else 0
        return waveform[start : start + target_samples]

    padding = target_samples - waveform.numel()
    return torch.nn.functional.pad(waveform, (0, padding))


def read_manifest(
    manifest_path: Path,
    split: str,
    label_map: dict[str, int],
    path_column: str = "path",
    label_column: str = "label",
    split_column: str = "split",
) -> list[AudioExample]:
    if not manifest_path.exists():
        raise FileNotFoundError(
            f"Dataset manifest not found: {manifest_path}. "
            "Create a CSV manifest with path,label,split columns before training."
        )

    examples: list[AudioExample] = []
    base_dir = manifest_path.parent

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {path_column, label_column, split_column}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Manifest is missing required columns: {sorted(missing)}")

        for row in reader:
            row_split = row[split_column].strip().lower()
            if row_split != split:
                continue

            label_name = row[label_column].strip().lower()
            if label_name not in label_map:
                raise ValueError(
                    f"Unknown label '{label_name}' in manifest. "
                    f"Expected one of: {sorted(label_map)}"
                )

            audio_path = Path(row[path_column].strip())
            if not audio_path.is_absolute():
                audio_path = base_dir / audio_path

            examples.append(
                AudioExample(path=audio_path, label=label_map[label_name], split=row_split)
            )

    if not examples:
        raise ValueError(f"No examples found for split '{split}' in {manifest_path}")

    return examples


class WavBinaryDataset(Dataset[dict[str, Any]]):
    """Manifest-backed bonafide/spoof audio dataset."""

    def __init__(
        self,
        manifest_path: Path,
        split: str,
        label_map: dict[str, int],
        sample_rate: int,
        segment_seconds: float,
        path_column: str = "path",
        label_column: str = "label",
        split_column: str = "split",
        random_crop: bool = False,
    ) -> None:
        self.examples = read_manifest(
            manifest_path=manifest_path,
            split=split,
            label_map=label_map,
            path_column=path_column,
            label_column=label_column,
            split_column=split_column,
        )
        self.sample_rate = sample_rate
        self.target_samples = int(round(sample_rate * segment_seconds))
        self.random_crop = random_crop

        if self.target_samples <= 0:
            raise ValueError("segment_seconds must produce at least one sample.")

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        example = self.examples[index]
        waveform, original_sample_rate = _load_audio_mono(example.path)
        waveform = resample_if_needed(waveform, original_sample_rate, self.sample_rate)
        waveform = crop_or_pad(waveform, self.target_samples, self.random_crop)

        return {
            "input_values": waveform.float(),
            "labels": torch.tensor(example.label, dtype=torch.long),
            "path": str(example.path),
            "split": example.split,
        }


def collate_audio_batch(batch: list[dict[str, Any]]) -> dict[str, Any]:
    input_values = torch.stack([item["input_values"] for item in batch])
    labels = torch.stack([item["labels"] for item in batch])
    return {
        "input_values": input_values,
        "labels": labels,
        "paths": [item["path"] for item in batch],
        "splits": [item["split"] for item in batch],
    }
