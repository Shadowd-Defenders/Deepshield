"""Evaluation utilities for binary bonafide/spoof experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import WavBinaryDataset, collate_audio_batch
from model import HF_CACHE_DIR, WavLMBinaryClassifier


def binary_metrics(labels: list[int], spoof_scores: list[float]) -> dict[str, float | None]:
    y_true = np.asarray(labels, dtype=np.int64)
    y_score = np.asarray(spoof_scores, dtype=np.float64)
    y_pred = (y_score >= 0.5).astype(np.int64)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    total = int(y_true.size)

    accuracy = (tp + tn) / total if total else None
    precision = tp / (tp + fp) if (tp + fp) else None
    recall = tp / (tp + fn) if (tp + fn) else None
    f1 = (
        2 * precision * recall / (precision + recall)
        if precision is not None and recall is not None and (precision + recall)
        else None
    )

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "roc_auc": roc_auc_score(y_true, y_score),
        "eer": equal_error_rate(y_true, y_score),
    }


def roc_auc_score(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    positives = y_true == 1
    negatives = y_true == 0
    pos_count = int(positives.sum())
    neg_count = int(negatives.sum())
    if pos_count == 0 or neg_count == 0:
        return None

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, y_score.size + 1, dtype=np.float64)

    for score in np.unique(y_score):
        tied = y_score == score
        if tied.sum() > 1:
            ranks[tied] = ranks[tied].mean()

    positive_rank_sum = ranks[positives].sum()
    auc = (positive_rank_sum - pos_count * (pos_count + 1) / 2) / (pos_count * neg_count)
    return float(auc)


def equal_error_rate(y_true: np.ndarray, y_score: np.ndarray) -> float | None:
    positives = y_true == 1
    negatives = y_true == 0
    pos_count = int(positives.sum())
    neg_count = int(negatives.sum())
    if pos_count == 0 or neg_count == 0:
        return None

    thresholds = np.unique(np.concatenate(([0.0], y_score, [1.0])))
    best_eer = None
    best_gap = float("inf")

    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(np.int64)
        false_reject_rate = float(((y_pred == 0) & positives).sum() / pos_count)
        false_accept_rate = float(((y_pred == 1) & negatives).sum() / neg_count)
        gap = abs(false_reject_rate - false_accept_rate)
        if gap < best_gap:
            best_gap = gap
            best_eer = (false_reject_rate + false_accept_rate) / 2

    return float(best_eer) if best_eer is not None else None


def evaluate_model(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    loss_fn: torch.nn.Module | None = None,
) -> dict[str, Any]:
    model.eval()
    labels: list[int] = []
    scores: list[float] = []
    losses: list[float] = []

    with torch.no_grad():
        for batch in dataloader:
            input_values = batch["input_values"].to(device)
            batch_labels = batch["labels"].to(device)
            logits = model(input_values)
            probabilities = torch.softmax(logits, dim=-1)[:, 1]

            if loss_fn is not None:
                loss = loss_fn(logits, batch_labels)
                losses.append(float(loss.item()))

            labels.extend(batch_labels.cpu().tolist())
            scores.extend(probabilities.cpu().tolist())

    metrics = binary_metrics(labels, scores)
    if losses:
        metrics["loss"] = float(np.mean(losses))
    return metrics


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_eval_loader(config: dict[str, Any], split: str) -> DataLoader:
    data_config = config["data"]
    audio_config = config["audio"]
    dataset = WavBinaryDataset(
        manifest_path=Path(data_config["manifest_path"]),
        split=split,
        label_map=data_config["labels"],
        sample_rate=int(audio_config["sample_rate"]),
        segment_seconds=float(audio_config["segment_seconds"]),
        path_column=data_config.get("path_column", "path"),
        label_column=data_config.get("label_column", "label"),
        split_column=data_config.get("split_column", "split"),
        random_crop=False,
    )
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["training"]["num_workers"]),
        collate_fn=collate_audio_batch,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate an exp001 checkpoint.")
    parser.add_argument("--config", type=Path, default=Path("configs/exp001_frozen_wavlm.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--split", choices=["validation", "test"], default="test")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    output_config = config["outputs"]
    model_config = config["model"]
    checkpoint_path = args.checkpoint or (
        Path(output_config["checkpoint_dir"]) / output_config["best_checkpoint_name"]
    )

    if not checkpoint_path.exists():
        raise SystemExit(
            f"Checkpoint not found: {checkpoint_path}. Train the model before evaluation."
        )

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

    dataloader = build_eval_loader(config, args.split)
    metrics = evaluate_model(model, dataloader, device, torch.nn.CrossEntropyLoss())

    results_dir = Path(output_config["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    output_path = results_dir / f"{args.split}_metrics.json"
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps({"split": args.split, "metrics": metrics}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
