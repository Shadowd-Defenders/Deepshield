"""Prepare and validate the exp001 dataset manifest.

This script does not download datasets and does not fabricate labels.
It builds the manifest only from official protocol files that already exist
on disk.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path


MODULE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATASET_ROOT = MODULE_ROOT / "data" / "raw" / "ASVspoof2019_LA"
DEFAULT_MANIFEST_PATH = MODULE_ROOT / "data" / "processed" / "exp001_manifest.csv"
LABELS = {"bonafide": 0, "spoof": 1}


@dataclass(frozen=True)
class SplitSpec:
    split: str
    protocol_name: str
    audio_subdir: str


ASVSPOOF2019_LA_SPECS = (
    SplitSpec(
        split="train",
        protocol_name="ASVspoof2019.LA.cm.train.trn.txt",
        audio_subdir="ASVspoof2019_LA_train/flac",
    ),
    SplitSpec(
        split="validation",
        protocol_name="ASVspoof2019.LA.cm.dev.trl.txt",
        audio_subdir="ASVspoof2019_LA_dev/flac",
    ),
    SplitSpec(
        split="test",
        protocol_name="ASVspoof2019.LA.cm.eval.trl.txt",
        audio_subdir="ASVspoof2019_LA_eval/flac",
    ),
)


def resolve_protocol_dir(dataset_root: Path) -> Path:
    candidates = [
        dataset_root / "ASVspoof2019_LA_cm_protocols",
        dataset_root / "LA" / "ASVspoof2019_LA_cm_protocols",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_audio_dir(dataset_root: Path, audio_subdir: str) -> Path:
    candidates = [
        dataset_root / audio_subdir,
        dataset_root / "LA" / audio_subdir,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def required_asvspoof_paths(dataset_root: Path) -> dict[str, Path]:
    protocol_dir = resolve_protocol_dir(dataset_root)
    required: dict[str, Path] = {}
    for spec in ASVSPOOF2019_LA_SPECS:
        required[f"{spec.split}_protocol"] = protocol_dir / spec.protocol_name
        required[f"{spec.split}_audio_dir"] = resolve_audio_dir(dataset_root, spec.audio_subdir)
    return required


def print_missing_dataset_requirements(dataset_root: Path) -> None:
    print("Required dataset/protocol files for exp001 are missing.")
    print("Place the ASVspoof 2019 Logical Access (LA) countermeasure data here:")
    print(f"  {dataset_root}")
    print("")
    print("Expected protocol files:")
    for key, path in required_asvspoof_paths(dataset_root).items():
        if key.endswith("_protocol"):
            print(f"  {path}")
    print("")
    print("Expected audio directories:")
    for key, path in required_asvspoof_paths(dataset_root).items():
        if key.endswith("_audio_dir"):
            print(f"  {path}")
    print("")
    print("Do not create labels manually. The manifest must be derived from the official protocol files.")


def print_missing_dataset_report(missing_paths: list[Path]) -> None:
    print("total examples: 0")
    print("bonafide count: 0")
    print("spoof count: 0")
    print("train count: 0")
    print("validation count: 0")
    print("test count: 0")
    print(f"missing files: {len(missing_paths)}")
    for path in missing_paths:
        print(f"  {path}")
    print("duplicate files: 0")


def parse_protocol_line(line: str, protocol_path: Path, line_number: int) -> tuple[str, str]:
    parts = line.strip().split()
    if len(parts) < 2:
        raise ValueError(f"{protocol_path}:{line_number} is malformed: {line!r}")

    utterance_id = parts[1]
    label = parts[-1].lower()
    if label not in LABELS:
        raise ValueError(
            f"{protocol_path}:{line_number} has unknown label '{parts[-1]}'. "
            "Expected bonafide or spoof."
        )
    return utterance_id, label


def build_manifest_rows(dataset_root: Path) -> list[dict[str, str]]:
    required = required_asvspoof_paths(dataset_root)
    missing = [path for path in required.values() if not path.exists()]
    if missing:
        print_missing_dataset_requirements(dataset_root)
        print("")
        print("Missing paths:")
        for path in missing:
            print(f"  {path}")
        print("")
        print_missing_dataset_report(missing)
        raise SystemExit(2)

    rows: list[dict[str, str]] = []
    protocol_dir = resolve_protocol_dir(dataset_root)

    for spec in ASVSPOOF2019_LA_SPECS:
        protocol_path = protocol_dir / spec.protocol_name
        audio_dir = resolve_audio_dir(dataset_root, spec.audio_subdir)

        with protocol_path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                utterance_id, label = parse_protocol_line(line, protocol_path, line_number)
                audio_path = audio_dir / f"{utterance_id}.flac"
                relative_path = audio_path.relative_to(MODULE_ROOT)
                rows.append(
                    {
                        "path": relative_path.as_posix(),
                        "label": label,
                        "split": spec.split,
                    }
                )

    return rows


def write_manifest(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(manifest_path: Path) -> list[dict[str, str]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")

    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        expected_fields = ["path", "label", "split"]
        if reader.fieldnames != expected_fields:
            raise ValueError(
                f"Manifest must have exactly these columns in this order: {expected_fields}. "
                f"Found: {reader.fieldnames}"
            )
        return list(reader)


def validate_manifest(rows: list[dict[str, str]]) -> dict[str, object]:
    label_counts: Counter[str] = Counter()
    split_counts: Counter[str] = Counter()
    missing_files: list[str] = []
    paths_by_split: dict[str, set[str]] = defaultdict(set)
    duplicate_rows: list[str] = []
    seen_rows: set[tuple[str, str]] = set()

    for index, row in enumerate(rows, start=2):
        path_value = row["path"].strip()
        label = row["label"].strip().lower()
        split = row["split"].strip().lower()

        if label not in LABELS:
            raise ValueError(f"Invalid label on CSV line {index}: {label!r}")
        if split not in {"train", "validation", "test"}:
            raise ValueError(f"Invalid split on CSV line {index}: {split!r}")

        manifest_path = Path(path_value)
        absolute_path = manifest_path if manifest_path.is_absolute() else MODULE_ROOT / manifest_path
        normalized_path = str(absolute_path.resolve())

        row_key = (normalized_path, split)
        if row_key in seen_rows:
            duplicate_rows.append(path_value)
        seen_rows.add(row_key)

        if not absolute_path.exists():
            missing_files.append(path_value)

        paths_by_split[split].add(normalized_path)
        label_counts[label] += 1
        split_counts[split] += 1

    duplicate_across_splits = sorted(
        (
            paths_by_split["train"].intersection(paths_by_split["validation"])
            | paths_by_split["train"].intersection(paths_by_split["test"])
            | paths_by_split["validation"].intersection(paths_by_split["test"])
        )
    )

    return {
        "total_examples": len(rows),
        "bonafide_count": label_counts["bonafide"],
        "spoof_count": label_counts["spoof"],
        "train_count": split_counts["train"],
        "validation_count": split_counts["validation"],
        "test_count": split_counts["test"],
        "missing_files": missing_files,
        "duplicate_files": duplicate_across_splits,
        "duplicate_rows": duplicate_rows,
    }


def print_report(report: dict[str, object]) -> None:
    print(f"total examples: {report['total_examples']}")
    print(f"bonafide count: {report['bonafide_count']}")
    print(f"spoof count: {report['spoof_count']}")
    print(f"train count: {report['train_count']}")
    print(f"validation count: {report['validation_count']}")
    print(f"test count: {report['test_count']}")

    missing_files = report["missing_files"]
    duplicate_files = report["duplicate_files"]
    duplicate_rows = report["duplicate_rows"]

    print(f"missing files: {len(missing_files)}")
    for path in missing_files:
        print(f"  {path}")

    print(f"duplicate files: {len(duplicate_files)}")
    for path in duplicate_files:
        print(f"  {path}")

    if duplicate_rows:
        print(f"duplicate rows within a split: {len(duplicate_rows)}")
        for path in duplicate_rows:
            print(f"  {path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare or validate data/processed/exp001_manifest.csv."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=DEFAULT_DATASET_ROOT,
        help="Root containing ASVspoof2019_LA_* audio folders and CM protocol folder.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help="Manifest path to write or validate.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate an existing manifest without creating a new one.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.output if args.output.is_absolute() else MODULE_ROOT / args.output
    dataset_root = args.dataset_root if args.dataset_root.is_absolute() else MODULE_ROOT / args.dataset_root

    if args.validate_only:
        try:
            rows = read_manifest(manifest_path)
        except FileNotFoundError as exc:
            print(str(exc))
            print("")
            print_missing_dataset_requirements(dataset_root)
            print("")
            print_missing_dataset_report([manifest_path])
            return 2
    else:
        rows = build_manifest_rows(dataset_root)
        write_manifest(rows, manifest_path)
        print(f"created manifest: {manifest_path}")

    report = validate_manifest(rows)
    print_report(report)

    failed = bool(report["missing_files"]) or bool(report["duplicate_files"]) or bool(report["duplicate_rows"])
    missing_split = (
        report["train_count"] == 0
        or report["validation_count"] == 0
        or report["test_count"] == 0
    )
    if missing_split:
        print("error: manifest must contain train, validation, and test examples.")
        failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
