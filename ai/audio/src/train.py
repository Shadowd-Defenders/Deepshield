"""Training entrypoint for exp001 frozen WavLM binary classification."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from dataset import WavBinaryDataset, collate_audio_batch
from evaluate import evaluate_model
from model import HF_CACHE_DIR, WavLMBinaryClassifier


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def select_device(configured_device: str) -> torch.device:
    if configured_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(configured_device)


def build_loader(config: dict[str, Any], split: str, shuffle: bool) -> DataLoader:
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
        random_crop=shuffle,
    )
    return DataLoader(
        dataset,
        batch_size=int(config["training"]["batch_size"]),
        shuffle=shuffle,
        num_workers=int(config["training"]["num_workers"]),
        collate_fn=collate_audio_batch,
    )


def validate_split_separation(
    train_loader: DataLoader,
    validation_loader: DataLoader,
    test_loader: DataLoader,
) -> None:
    split_paths = {
        "train": {str(example.path.resolve()) for example in train_loader.dataset.examples},
        "validation": {str(example.path.resolve()) for example in validation_loader.dataset.examples},
        "test": {str(example.path.resolve()) for example in test_loader.dataset.examples},
    }
    overlaps = {
        "train_validation": split_paths["train"].intersection(split_paths["validation"]),
        "train_test": split_paths["train"].intersection(split_paths["test"]),
        "validation_test": split_paths["validation"].intersection(split_paths["test"]),
    }
    non_empty_overlaps = {name: paths for name, paths in overlaps.items() if paths}
    if non_empty_overlaps:
        raise ValueError(
            "Dataset split leakage detected. Overlapping paths: "
            f"{ {name: sorted(paths) for name, paths in non_empty_overlaps.items()} }"
        )


def save_checkpoint(
    path: Path,
    model: WavLMBinaryClassifier,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    metrics: dict[str, Any],
    config: dict[str, Any],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "metrics": metrics,
            "config": config,
            "model_metadata": model.checkpoint_metadata(),
        },
        path,
    )


def run_sanity_batch(config: dict[str, Any]) -> None:
    seed = int(config["experiment"]["seed"])
    set_seed(seed)
    device = select_device(config["training"]["device"])
    model_config = config["model"]
    audio_config = config["audio"]

    model = WavLMBinaryClassifier(
        pretrained_model_id=model_config["pretrained_model_id"],
        freeze_encoder=bool(model_config["freeze_encoder"]),
        classifier_hidden_size=int(model_config["classifier_hidden_size"]),
        dropout=float(model_config["dropout"]),
        cache_dir=HF_CACHE_DIR,
        local_files_only=True,
    ).to(device)

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(trainable_parameters, lr=float(config["training"]["learning_rate"]))
    loss_fn = torch.nn.CrossEntropyLoss()

    batch_size = 1
    sample_count = int(float(audio_config["segment_seconds"]) * int(audio_config["sample_rate"]))
    input_values = torch.randn(batch_size, sample_count, device=device)
    labels = torch.tensor([0], dtype=torch.long, device=device)

    model.train()
    optimizer.zero_grad(set_to_none=True)
    logits = model(input_values)
    loss = loss_fn(logits, labels)
    loss.backward()
    optimizer.step()

    print(
        json.dumps(
            {
                "sanity_batch": "passed",
                "input_shape": list(input_values.shape),
                "logits_shape": list(logits.shape),
                "loss": float(loss.item()),
                "note": "Mechanics-only sanity check with random audio tensor; this is not a model metric.",
            },
            indent=2,
        )
    )


def train(config: dict[str, Any]) -> None:
    seed = int(config["experiment"]["seed"])
    set_seed(seed)
    device = select_device(config["training"]["device"])
    outputs = config["outputs"]
    training = config["training"]
    model_config = config["model"]

    train_loader = build_loader(config, "train", shuffle=True)
    validation_loader = build_loader(config, "validation", shuffle=False)
    test_loader = build_loader(config, "test", shuffle=False)
    validate_split_separation(train_loader, validation_loader, test_loader)

    results_dir = Path(outputs["results_dir"])
    checkpoint_dir = Path(outputs["checkpoint_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    model = WavLMBinaryClassifier(
        pretrained_model_id=model_config["pretrained_model_id"],
        freeze_encoder=bool(model_config["freeze_encoder"]),
        classifier_hidden_size=int(model_config["classifier_hidden_size"]),
        dropout=float(model_config["dropout"]),
        cache_dir=HF_CACHE_DIR,
        local_files_only=True,
    ).to(device)

    trainable_parameters = [param for param in model.parameters() if param.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=float(training["learning_rate"]),
        weight_decay=float(training["weight_decay"]),
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    best_validation_f1 = -1.0
    patience = int(training["patience"])
    epochs_without_improvement = 0
    history: list[dict[str, Any]] = []

    for epoch in range(1, int(training["epochs"]) + 1):
        model.train()
        batch_losses: list[float] = []

        for batch in train_loader:
            input_values = batch["input_values"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(input_values)
            loss = loss_fn(logits, labels)
            loss.backward()

            clip_norm = training.get("gradient_clip_norm")
            if clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(trainable_parameters, float(clip_norm))

            optimizer.step()
            batch_losses.append(float(loss.item()))

        validation_metrics = evaluate_model(model, validation_loader, device, loss_fn)
        train_loss = float(np.mean(batch_losses)) if batch_losses else None
        epoch_record = {
            "epoch": epoch,
            "train": {"loss": train_loss},
            "validation": validation_metrics,
        }
        history.append(epoch_record)
        print(json.dumps(epoch_record, indent=2))

        last_path = checkpoint_dir / outputs["last_checkpoint_name"]
        save_checkpoint(last_path, model, optimizer, epoch, validation_metrics, config)

        validation_f1 = validation_metrics.get("f1")
        improved = validation_f1 is not None and validation_f1 > best_validation_f1
        if improved:
            best_validation_f1 = float(validation_f1)
            epochs_without_improvement = 0
            best_path = checkpoint_dir / outputs["best_checkpoint_name"]
            save_checkpoint(best_path, model, optimizer, epoch, validation_metrics, config)
        else:
            epochs_without_improvement += 1

        (results_dir / "training_log.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

        if epochs_without_improvement >= patience:
            print(f"Early stopping after {epoch} epochs without validation F1 improvement.")
            break

    best_checkpoint = checkpoint_dir / outputs["best_checkpoint_name"]
    if not best_checkpoint.exists():
        raise RuntimeError(
            "Training finished without a best checkpoint. Check validation labels and metrics."
        )

    checkpoint = torch.load(best_checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    test_metrics = evaluate_model(model, test_loader, device, loss_fn)
    (results_dir / "test_metrics.json").write_text(
        json.dumps(test_metrics, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({"held_out_test": test_metrics}, indent=2))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train exp001 frozen WavLM binary classifier.")
    parser.add_argument("--config", type=Path, default=Path("configs/exp001_frozen_wavlm.yaml"))
    parser.add_argument(
        "--sanity-batch",
        action="store_true",
        help="Run one forward/loss/backward/optimizer step with random audio.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    if args.sanity_batch:
        run_sanity_batch(config)
        return 0

    try:
        train(config)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(
            f"Cannot run configured training experiment: {exc}\n"
            "Missing requirement: provide a real dataset manifest/protocol with "
            "separate train, validation, and test splits. No training was run."
        ) from exc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
