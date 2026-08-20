"""Download and load Microsoft WavLM Base+ as a pretrained feature extractor."""

from __future__ import annotations

from pathlib import Path

MODEL_ID = "microsoft/wavlm-base-plus"
MODULE_ROOT = Path(__file__).resolve().parents[1]
HF_CACHE_DIR = MODULE_ROOT / "checkpoints" / "huggingface"


def main() -> int:
    try:
        from transformers import AutoFeatureExtractor, WavLMModel
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency. Install this module with: "
            "python -m pip install -r requirements.txt"
        ) from exc

    print(f"Loading pretrained feature extractor: {MODEL_ID}")
    print("This is not a deepfake detector and no classifier is created.")
    print(f"Cache directory: {HF_CACHE_DIR}")

    feature_extractor = AutoFeatureExtractor.from_pretrained(
        MODEL_ID,
        cache_dir=HF_CACHE_DIR,
    )
    model = WavLMModel.from_pretrained(
        MODEL_ID,
        cache_dir=HF_CACHE_DIR,
    )
    model.eval()

    print("WavLM Base+ loaded successfully.")
    print(f"Feature extractor sampling rate: {feature_extractor.sampling_rate}")
    print(f"Hidden size: {model.config.hidden_size}")
    print(f"Transformer layers: {model.config.num_hidden_layers}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

