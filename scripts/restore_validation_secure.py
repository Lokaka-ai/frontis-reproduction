#!/usr/bin/env python3
"""Restore only the deterministic hidden validation answers after a pod restart."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from prepare_spooky_assets import (
    CLASSES,
    SOURCE_MD5,
    TASK,
    file_digest,
    validate_asset_frames,
    validate_manifest,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=Path("/workspace/frontis-research/pilot/assets"),
        help="Root containing submit/, validation/, and manifest/ task assets.",
    )
    parser.add_argument(
        "--secure-root",
        type=Path,
        default=Path("/var/lib/frontis-pilot/secure/validation"),
        help="Root for scorer-only validation answers.",
    )
    args = parser.parse_args()
    assets_root = args.assets_root.resolve()

    full_train_path = assets_root / f"submit/{TASK}/train.csv"
    visible_path = assets_root / f"validation/{TASK}/train.csv"
    hidden_public_path = assets_root / f"validation/{TASK}/test.csv"
    sample_submission_path = assets_root / f"validation/{TASK}/sample_submission.csv"
    manifest_path = assets_root / f"manifest/{TASK}.parquet"
    required_paths = (
        full_train_path,
        visible_path,
        hidden_public_path,
        sample_submission_path,
        manifest_path,
    )
    for path in required_paths:
        if not path.is_file():
            raise SystemExit(f"required persistent asset is missing: {path}")

    full_train_md5 = file_digest(full_train_path, "md5")
    if full_train_md5 != SOURCE_MD5:
        raise SystemExit(
            f"persistent full train checksum mismatch: expected {SOURCE_MD5}, got {full_train_md5}"
        )
    full_train = pd.read_csv(full_train_path)
    persisted_visible = pd.read_csv(visible_path)
    persisted_hidden = pd.read_csv(hidden_public_path)
    persisted_sample = pd.read_csv(sample_submission_path)
    try:
        hidden = validate_asset_frames(
            full_train,
            persisted_visible,
            persisted_hidden,
            persisted_sample,
        )
        validate_manifest(pd.read_parquet(manifest_path))
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"persistent asset validation failed: {exc}") from exc

    answers = pd.DataFrame({"id": hidden["id"]})
    for label in CLASSES:
        answers[label] = (hidden["author"] == label).astype(int)
    if list(answers["id"]) != list(persisted_hidden["id"]):
        raise SystemExit("answer order does not match hidden validation input order")

    secure_dir = args.secure_root.resolve() / TASK
    secure_dir.mkdir(parents=True, exist_ok=True)
    secure_dir.chmod(0o700)
    answers_path = secure_dir / "answers.csv"
    answers.to_csv(answers_path, index=False)
    answers_path.chmod(0o600)
    print(
        f"restored {len(answers)} hidden validation answers after exact split checks: "
        f"{answers_path}"
    )


if __name__ == "__main__":
    main()
