"""Build a manifest for ASVspoof 2019 LA audio classification."""

from __future__ import annotations

import csv
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_ROOT = PROJECT_ROOT / "data" / "raw" / "LA" / "LA"
PROTOCOL_ROOT = DATASET_ROOT / "ASVspoof2019_LA_cm_protocols"

OUTPUT_PATH = PROJECT_ROOT / "data" / "processed" / "exp001_manifest.csv"

LABEL_MAP = {
    "bonafide": 0,
    "spoof": 1,
}

SPLITS = {
    "train": {
        "protocol": PROTOCOL_ROOT / "ASVspoof2019.LA.cm.train.trn.txt",
        "audio_dir": DATASET_ROOT / "ASVspoof2019_LA_train" / "flac",
    },
    "validation": {
        "protocol": PROTOCOL_ROOT / "ASVspoof2019.LA.cm.dev.trl.txt",
        "audio_dir": DATASET_ROOT / "ASVspoof2019_LA_dev" / "flac",
    },
    "test": {
        "protocol": PROTOCOL_ROOT / "ASVspoof2019.LA.cm.eval.trl.txt",
        "audio_dir": DATASET_ROOT / "ASVspoof2019_LA_eval" / "flac",
    },
}

def parse_protocol_line(
    line: str,
    protocol_path: Path,
) -> tuple[str, str, str, str]:
    fields = line.strip().split()

    if len(fields) != 5:
        raise ValueError(
            f"Unexpected protocol format in {protocol_path}: {line!r}"
        )

    speaker_id, utterance_id, _, attack_id, label = fields

    if label not in LABEL_MAP:
        raise ValueError(
            f"Unknown label {label!r} in {protocol_path}: {line!r}"
        )

    return speaker_id, utterance_id, attack_id, label


def build_split(
    split_name: str,
    protocol_path: Path,
    audio_dir: Path,
) -> list[dict[str, object]]:
    if not protocol_path.exists():
        raise FileNotFoundError(
            f"Protocol file not found: {protocol_path}"
        )

    if not audio_dir.exists():
        raise FileNotFoundError(
            f"Audio directory not found: {audio_dir}"
        )

    rows: list[dict[str, object]] = []
    missing_files: list[str] = []

    with protocol_path.open("r", encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            if not raw_line.strip():
                continue

            speaker_id, utterance_id, attack_id, label = parse_protocol_line(
                raw_line,
                protocol_path,
            )

            audio_path = audio_dir / f"{utterance_id}.flac"

            if not audio_path.exists():
                missing_files.append(str(audio_path))
                continue

            rows.append(
                {
                    "path": str(audio_path.relative_to(PROJECT_ROOT)),
                    "label": label,
                    "split": split_name,
                    "speaker_id": speaker_id,
                    "attack_id": attack_id,
                    "utterance_id": utterance_id,
                }
            )

    if missing_files:
        print(
            f"WARNING: {len(missing_files)} audio files referenced by "
            f"{protocol_path.name} were not found."
        )

        for path in missing_files[:10]:
            print(f"  Missing: {path}")

        if len(missing_files) > 10:
            print(f"  ... and {len(missing_files) - 10} more.")

    print(
        f"{split_name}: protocol entries={len(rows) + len(missing_files)}, "
        f"audio files found={len(rows)}, "
        f"missing={len(missing_files)}"
    )

    return rows


def validate_manifest(rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError("Manifest contains no usable audio examples.")

    paths = [str(row["path"]) for row in rows]

    if len(paths) != len(set(paths)):
        duplicates = len(paths) - len(set(paths))
        raise RuntimeError(
            f"Manifest contains {duplicates} duplicate audio paths."
        )

    split_paths: dict[str, set[str]] = {}

    for row in rows:
        split = str(row["split"])
        split_paths.setdefault(split, set()).add(str(row["path"]))

    splits = list(split_paths)

    for index, first_split in enumerate(splits):
        for second_split in splits[index + 1 :]:
            overlap = split_paths[first_split].intersection(
                split_paths[second_split]
            )

            if overlap:
                raise RuntimeError(
                    f"Dataset leakage detected between "
                    f"{first_split} and {second_split}: "
                    f"{len(overlap)} overlapping paths."
                )


def print_statistics(rows: list[dict[str, object]]) -> None:
    print("\nDataset statistics")
    print("==================")

    for split in ("train", "validation", "test"):
        split_rows = [row for row in rows if row["split"] == split]

        bonafide = sum(
            1 for row in split_rows if row["label"] == "bonafide"
        )
        spoof = sum(
            1 for row in split_rows if row["label"] == "spoof"
        )

        print(
            f"{split:10s}: "
            f"total={len(split_rows):6d}, "
            f"bonafide={bonafide:6d}, "
            f"spoof={spoof:6d}"
        )

    print(f"{'total':10s}: {len(rows):6d}")


def main() -> int:
    print("Building ASVspoof 2019 LA manifest...")
    print(f"Dataset root: {DATASET_ROOT}")
    print(f"Output:       {OUTPUT_PATH}")
    print()

    all_rows: list[dict[str, object]] = []

    for split_name, config in SPLITS.items():
        rows = build_split(
            split_name=split_name,
            protocol_path=config["protocol"],
            audio_dir=config["audio_dir"],
        )
        all_rows.extend(rows)

    validate_manifest(all_rows)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "path",
        "label",
        "split",
        "speaker_id",
        "attack_id",
        "utterance_id",
    ]

    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(all_rows)

    print_statistics(all_rows)

    print()
    print(f"Manifest successfully created:")
    print(OUTPUT_PATH)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
