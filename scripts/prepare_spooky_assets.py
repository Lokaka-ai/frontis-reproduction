#!/usr/bin/env python3
"""Create the exact Spooky Author Identification assets used by this study."""

from __future__ import annotations

import argparse
import hashlib
import shutil
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


TASK = "spooky-author-identification"
TASK_UUID = "ab7e9a08-7768-5a92-bc2a-62e1bc8a6cba"
CLASSES = ["EAP", "HPL", "MWS"]
SEED = 42
FULL_ROWS = 17_621
VISIBLE_ROWS = 14_096
HIDDEN_ROWS = 3_525
SOURCE_COLUMNS = ["id", "text", "author"]
SOURCE_MD5 = "1124166bf7b8b1e337a32fe568732447"
LEADERBOARD_SHA256 = "1087afc671c8eddcca7f8dc792a4d55877213ce86ac9d1f8c7b280edd0eb2514"
DESCRIPTION_SHA256 = "35d7de46d74377f997bdc7f5859e36ca3311fb4ae2e37ab63491a9d7c72661bd"
SANDBOX_DATA_DIR = f"/datasets/validation/{TASK}"

PUBLIC_SYSTEM_PROMPT = "Use only the visible files and installed offline libraries."
PUBLIC_USER_PROMPT = (
    "Output Instructions:\n"
    "Provide one self-contained Python solution in a fenced python code block. The script "
    "must read DATA_DIR, use a deterministic local validation split, print exactly "
    "`Final Validation Score: <number>` for its own validation, fit on all visible "
    "training rows, and write ./submission.csv with columns id,EAP,HPL,MWS in the test "
    "row order. Probability rows must be finite, in [0,1], and sum to one."
)
DATA_DESCRIPTION = (
    "Read the directory from the DATA_DIR environment variable. It contains train.csv "
    "(id,text,author), test.csv (id,text), and sample_submission.csv (id,EAP,HPL,MWS). "
    "The hidden labels are not present. Available libraries include Python 3.12, numpy, "
    "pandas, scipy, and scikit-learn."
)
TASK_SUFFIX = (
    "This pilot evaluates multiclass log loss on a fixed hidden holdout. Lower scores "
    "are better. Do not attempt network access."
)


def file_digest(path: Path, algorithm: str) -> str:
    digest = hashlib.new(algorithm)
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def assert_frame_equal(expected: pd.DataFrame, actual: pd.DataFrame, label: str) -> None:
    try:
        pd.testing.assert_frame_equal(
            expected.reset_index(drop=True),
            actual.reset_index(drop=True),
            check_dtype=False,
        )
    except AssertionError as exc:
        raise ValueError(f"{label} does not match the expected data: {exc}") from exc


def validate_source_frame(
    frame: pd.DataFrame,
    *,
    expected_rows: int = FULL_ROWS,
) -> None:
    if list(frame.columns) != SOURCE_COLUMNS:
        raise ValueError(
            f"source columns must be {SOURCE_COLUMNS}, got {list(frame.columns)}"
        )
    if len(frame) != expected_rows:
        raise ValueError(f"source must have {expected_rows} rows, got {len(frame)}")
    if frame[SOURCE_COLUMNS].isnull().any().any():
        raise ValueError("source contains missing id, text, or author values")
    if frame["id"].duplicated().any():
        raise ValueError("source contains duplicate ids")
    labels = sorted(frame["author"].unique().tolist())
    if labels != sorted(CLASSES):
        raise ValueError(f"source labels must be {CLASSES}, got {labels}")


def split_full_train(full_train: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    visible, hidden = train_test_split(
        full_train,
        test_size=0.20,
        random_state=SEED,
        stratify=full_train["author"],
    )
    return visible, hidden


def make_sample_submission(hidden: pd.DataFrame) -> pd.DataFrame:
    sample = pd.DataFrame({"id": hidden["id"]})
    for label in CLASSES:
        sample[label] = 1.0 / len(CLASSES)
    return sample


def make_manifest(description: str) -> pd.DataFrame:
    if not description.endswith("\n"):
        raise ValueError("MLE-Bench task description must end with a newline")
    task_description = f"{description}\n\n{TASK_SUFFIX}"
    prompt = [
        {"role": "system", "content": PUBLIC_SYSTEM_PROMPT},
        {"role": "user", "content": PUBLIC_USER_PROMPT},
    ]
    metadata = {
        "uuid": TASK_UUID,
        "task_name": TASK,
        "task": TASK,
        "source": "MLE-Bench",
        "modality": "text",
        "data_dir": SANDBOX_DATA_DIR,
        "higher_is_better": False,
        "theoretical_max": None,
        "theoretical_min": 0.0,
        "leaderboard_max": 35.23192,
        "leaderboard_min": 0.02467,
        "cpu_gpu": "cpu",
        "task_description": task_description,
        "data_description": DATA_DESCRIPTION,
    }
    return pd.DataFrame([{"prompt": prompt, "metadata": metadata}])


def _plain(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if hasattr(value, "tolist"):
        return _plain(value.tolist())
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return value


def validate_manifest(manifest: pd.DataFrame) -> None:
    if list(manifest.columns) != ["prompt", "metadata"] or len(manifest) != 1:
        raise ValueError("manifest must contain one row with prompt and metadata columns")
    prompt = _plain(manifest.iloc[0]["prompt"])
    metadata = _plain(manifest.iloc[0]["metadata"])
    expected_prompt = [
        {"role": "system", "content": PUBLIC_SYSTEM_PROMPT},
        {"role": "user", "content": PUBLIC_USER_PROMPT},
    ]
    if prompt != expected_prompt:
        raise ValueError("manifest prompt does not match the pilot prompt")
    expected_metadata = {
        "uuid": TASK_UUID,
        "task_name": TASK,
        "task": TASK,
        "source": "MLE-Bench",
        "modality": "text",
        "data_dir": SANDBOX_DATA_DIR,
        "higher_is_better": False,
        "theoretical_max": None,
        "theoretical_min": 0.0,
        "leaderboard_max": 35.23192,
        "leaderboard_min": 0.02467,
        "cpu_gpu": "cpu",
        "data_description": DATA_DESCRIPTION,
    }
    expected_keys = set(expected_metadata) | {"task_description"}
    if set(metadata) != expected_keys:
        raise ValueError(
            f"manifest metadata keys must be {sorted(expected_keys)}, got {sorted(metadata)}"
        )
    for key, expected in expected_metadata.items():
        if metadata.get(key) != expected:
            raise ValueError(
                f"manifest metadata {key!r} must be {expected!r}, got {metadata.get(key)!r}"
            )
    task_description = metadata.get("task_description")
    fixed_ending = f"\n\n{TASK_SUFFIX}"
    if not isinstance(task_description, str) or not task_description.endswith(fixed_ending):
        raise ValueError("manifest task description is missing the fixed pilot instructions")
    description = task_description[: -len(fixed_ending)]
    description_sha256 = hashlib.sha256(description.encode("utf-8")).hexdigest()
    if description_sha256 != DESCRIPTION_SHA256:
        raise ValueError("manifest contains the wrong MLE-Bench task description")


def validate_asset_frames(
    full_train: pd.DataFrame,
    visible: pd.DataFrame,
    hidden_public: pd.DataFrame,
    sample_submission: pd.DataFrame | None = None,
) -> pd.DataFrame:
    validate_source_frame(full_train)
    expected_visible, hidden = split_full_train(full_train)
    if len(expected_visible) != VISIBLE_ROWS or len(hidden) != HIDDEN_ROWS:
        raise ValueError(
            "split row counts changed: "
            f"visible={len(expected_visible)}, hidden={len(hidden)}"
        )
    assert_frame_equal(expected_visible, visible, "visible training data")
    expected_hidden_public = hidden[["id", "text"]]
    assert_frame_equal(expected_hidden_public, hidden_public, "hidden validation inputs")
    if "author" in hidden_public.columns:
        raise ValueError("hidden validation inputs must not contain author labels")
    if sample_submission is not None:
        expected_sample = make_sample_submission(hidden)
        assert_frame_equal(expected_sample, sample_submission, "sample submission")
    return hidden


def _atomic_write(path: Path, writer: Callable[[Path], None]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as tmp:
        temporary_path = Path(tmp.name)
    try:
        writer(temporary_path)
        temporary_path.replace(path)
    finally:
        temporary_path.unlink(missing_ok=True)


def _write_or_check_csv(path: Path, expected: pd.DataFrame) -> None:
    if path.exists():
        assert_frame_equal(expected, pd.read_csv(path), str(path))
        return
    _atomic_write(path, lambda temporary: expected.to_csv(temporary, index=False))


def _copy_or_check(path: Path, source: Path, expected: pd.DataFrame) -> None:
    if path.exists():
        assert_frame_equal(expected, pd.read_csv(path), str(path))
        return
    _atomic_write(path, lambda temporary: shutil.copyfile(source, temporary))


def _write_or_check_manifest(path: Path, expected: pd.DataFrame) -> None:
    if path.exists():
        existing = pd.read_parquet(path)
        validate_manifest(existing)
        if _plain(existing.iloc[0].to_dict()) != _plain(expected.iloc[0].to_dict()):
            raise ValueError(f"{path} does not match the expected manifest")
        return
    _atomic_write(path, lambda temporary: expected.to_parquet(temporary, index=False))


def load_mlebench_task(data_dir: Path) -> Any:
    try:
        from mlebench.registry import registry
    except ImportError as exc:
        raise ValueError(
            f"cannot import mlebench or one of its dependencies ({exc}); "
            "install the pinned MLE-Bench checkout first"
        ) from exc
    return registry.set_data_dir(data_dir).get_competition(TASK)


def prepare_assets(mlebench_data_dir: Path, assets_root: Path) -> None:
    competition = load_mlebench_task(mlebench_data_dir.resolve())
    source_path = competition.public_dir / "train.csv"
    leaderboard_path = competition.leaderboard
    if not source_path.is_file():
        raise ValueError(
            f"prepared MLE-Bench source is missing: {source_path}; run mlebench prepare first"
        )
    source_md5 = file_digest(source_path, "md5")
    if source_md5 != SOURCE_MD5:
        raise ValueError(
            f"wrong MLE-Bench train.csv: expected MD5 {SOURCE_MD5}, got {source_md5}"
        )
    if not leaderboard_path.is_file():
        raise ValueError(f"MLE-Bench leaderboard is missing: {leaderboard_path}")
    leaderboard_sha256 = file_digest(leaderboard_path, "sha256")
    if leaderboard_sha256 != LEADERBOARD_SHA256:
        raise ValueError(
            "wrong MLE-Bench leaderboard; run `git lfs pull` in the pinned MLE-Bench checkout "
            f"(expected SHA256 {LEADERBOARD_SHA256}, got {leaderboard_sha256})"
        )
    description_sha256 = hashlib.sha256(competition.description.encode("utf-8")).hexdigest()
    if description_sha256 != DESCRIPTION_SHA256:
        raise ValueError(
            "wrong MLE-Bench task description; install commit "
            "507f92e1138bb6e40dac5c6ee7a6758e6424bf97 "
            f"(expected SHA256 {DESCRIPTION_SHA256}, got {description_sha256})"
        )

    full_train = pd.read_csv(source_path)
    validate_source_frame(full_train)
    visible, hidden = split_full_train(full_train)
    if len(visible) != VISIBLE_ROWS or len(hidden) != HIDDEN_ROWS:
        raise ValueError(f"unexpected split sizes: visible={len(visible)}, hidden={len(hidden)}")
    hidden_public = hidden[["id", "text"]]
    sample_submission = make_sample_submission(hidden)
    manifest = make_manifest(competition.description)

    submit_train = assets_root / "submit" / TASK / "train.csv"
    validation_dir = assets_root / "validation" / TASK
    manifest_path = assets_root / "manifest" / f"{TASK}.parquet"
    leaderboard_target = assets_root / "leaderboards" / f"{TASK}.csv"

    _copy_or_check(submit_train, source_path, full_train)
    _write_or_check_csv(validation_dir / "train.csv", visible)
    _write_or_check_csv(validation_dir / "test.csv", hidden_public)
    _write_or_check_csv(validation_dir / "sample_submission.csv", sample_submission)
    _write_or_check_manifest(manifest_path, manifest)
    _copy_or_check(leaderboard_target, leaderboard_path, pd.read_csv(leaderboard_path))

    persisted_manifest = pd.read_parquet(manifest_path)
    validate_asset_frames(
        pd.read_csv(submit_train),
        pd.read_csv(validation_dir / "train.csv"),
        pd.read_csv(validation_dir / "test.csv"),
        pd.read_csv(validation_dir / "sample_submission.csv"),
    )
    validate_manifest(persisted_manifest)

    written_paths = (
        submit_train,
        validation_dir / "train.csv",
        validation_dir / "test.csv",
        validation_dir / "sample_submission.csv",
        manifest_path,
        leaderboard_target,
    )
    print(f"prepared and verified assets under {assets_root.resolve()}")
    print(
        f"source rows: {len(full_train)}; visible rows: {len(visible)}; "
        f"hidden rows: {len(hidden)}"
    )
    for path in written_paths:
        print(f"SHA256 {file_digest(path, 'sha256')}  {path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the deterministic Spooky Author Identification pilot assets."
    )
    parser.add_argument(
        "--mlebench-data-dir",
        type=Path,
        required=True,
        help="Data directory passed to `mlebench prepare --data-dir`.",
    )
    parser.add_argument(
        "--assets-root",
        type=Path,
        default=Path("/workspace/frontis-research/pilot/assets"),
        help="Destination root for submit, validation, manifest, and leaderboard assets.",
    )
    args = parser.parse_args()
    try:
        prepare_assets(args.mlebench_data_dir, args.assets_root.resolve())
    except (OSError, ValueError, KeyError) as exc:
        raise SystemExit(f"asset preparation failed: {exc}") from exc


if __name__ == "__main__":
    main()
