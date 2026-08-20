"""Inference entrypoint for an exp001 checkpoint on one PCM WAV file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import yaml

from dataset import crop_or_pad, load_pcm_wav_mono, resample_if_needed
from model import HF_CACHE_DIR, WavLMBinaryClassifier


def load_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run binary classifier inference on one PCM WAV file."
    )
    parser.add_argument("wav_path", type=Path)
    parser.add_argument("--config", type=Path, default=Path("configs/exp001_frozen_wavlm.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    outputs = config["outputs"]
    model_config = config["model"]
    audio_config = config["audio"]
    checkpoint_path = args.checkpoint or (
        Path(outputs["checkpoint_dir"]) / outputs["best_checkpoint_name"]
    )

    if not checkpoint_path.exists():
        raise SystemExit(
            f"Checkpoint not found: {checkpoint_path}. "
            "Train and evaluate a real checkpoint before inference."
        )

    waveform, original_sample_rate = load_pcm_wav_mono(args.wav_path)
    waveform = resample_if_needed(
        waveform,
        original_sample_rate,
        int(audio_config["sample_rate"]),
    )
    target_samples = int(float(audio_config["segment_seconds"]) * int(audio_config["sample_rate"]))
    waveform = crop_or_pad(waveform, target_samples, random_crop=False).unsqueeze(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = WavLMBinaryClassifier(
        pretrained_model_id=model_config["pretrained_model_id"],
        freeze_encoder=bool(model_config["freeze_encoder"]),
        classifier_hidden_size=int(model_config["classifier_hidden_size"]),
        dropout=float(model_config["dropout"]),
        cache_dir=HF_CACHE_DIR,
        local_files_only=True,
    )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(waveform.to(device))
        probabilities = torch.softmax(logits, dim=-1).squeeze(0).cpu()

    result = {
        "wav_path": str(args.wav_path),
        "checkpoint": str(checkpoint_path),
        "probabilities": {
            "bonafide": float(probabilities[0]),
            "spoof": float(probabilities[1]),
        },
        "note": (
            "This is classifier inference from a trained checkpoint. "
            "Do not treat it as validated accuracy without held-out evaluation."
        ),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
