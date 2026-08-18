#!/usr/bin/env python3
"""Restore only the deterministic hidden validation answers after a pod restart."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


TASK = "spooky-author-identification"
CLASSES = ["EAP", "HPL", "MWS"]
SEED = 42


def assert_frame_equal(left: pd.DataFrame, right: pd.DataFrame, label: str) -> None:
    try:
        pd.testing.assert_frame_equal(
            left.reset_index(drop=True),
            right.reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError as exc:
        raise SystemExit(f"persistent {label} does not match reconstructed split: {exc}") from exc


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
    manifest_path = assets_root / f"manifest/{TASK}.parquet"
    for path in (full_train_path, visible_path, hidden_public_path, manifest_path):
        if not path.is_file():
            raise SystemExit(f"required persistent asset is missing: {path}")

    full_train = pd.read_csv(full_train_path)
    visible, hidden = train_test_split(
        full_train,
        test_size=0.20,
        random_state=SEED,
        stratify=full_train["author"],
    )
    persisted_visible = pd.read_csv(visible_path)
    persisted_hidden = pd.read_csv(hidden_public_path)
    assert_frame_equal(visible, persisted_visible, "visible training data")
    assert_frame_equal(hidden[["id", "text"]], persisted_hidden, "hidden validation inputs")

    manifest = pd.read_parquet(manifest_path)
    if len(manifest) != 1 or manifest.iloc[0]["metadata"]["task_name"] != TASK:
        raise SystemExit("manifest identity check failed")

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
