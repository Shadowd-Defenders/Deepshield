"""WavLM encoder plus binary classification head."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
from transformers import WavLMModel

MODEL_ID = "microsoft/wavlm-base-plus"
MODULE_ROOT = Path(__file__).resolve().parents[1]
HF_CACHE_DIR = MODULE_ROOT / "checkpoints" / "huggingface"


class WavLMBinaryClassifier(nn.Module):
    """WavLM feature encoder with a small binary classification head."""

    def __init__(
        self,
        pretrained_model_id: str = MODEL_ID,
        freeze_encoder: bool = True,
        classifier_hidden_size: int = 256,
        dropout: float = 0.1,
        cache_dir: Path = HF_CACHE_DIR,
        local_files_only: bool = True,
    ) -> None:
        super().__init__()
        self.pretrained_model_id = pretrained_model_id
        self.freeze_encoder = freeze_encoder
        self.wavlm = WavLMModel.from_pretrained(
            pretrained_model_id,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )

        if freeze_encoder:
            for parameter in self.wavlm.parameters():
                parameter.requires_grad = False
            self.wavlm.eval()

        hidden_size = int(self.wavlm.config.hidden_size)
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, classifier_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(classifier_hidden_size, 2),
        )

    def encode(self, input_values: torch.Tensor) -> torch.Tensor:
        if self.freeze_encoder:
            self.wavlm.eval()
            with torch.no_grad():
                outputs = self.wavlm(input_values=input_values)
        else:
            outputs = self.wavlm(input_values=input_values)
        return outputs.last_hidden_state.mean(dim=1)

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        pooled_features = self.encode(input_values)
        return self.classifier(pooled_features)

    def checkpoint_metadata(self) -> dict[str, Any]:
        return {
            "pretrained_model_id": self.pretrained_model_id,
            "freeze_encoder": self.freeze_encoder,
            "hidden_size": int(self.wavlm.config.hidden_size),
            "num_hidden_layers": int(self.wavlm.config.num_hidden_layers),
            "labels": {"bonafide": 0, "spoof": 1},
            "note": "WavLM is used as a pretrained feature encoder; the head is the binary classifier.",
        }
