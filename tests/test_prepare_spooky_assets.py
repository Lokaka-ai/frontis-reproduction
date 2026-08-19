from __future__ import annotations

import sys
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pandas as pd


SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from prepare_spooky_assets import (  # noqa: E402
    CLASSES,
    FULL_ROWS,
    HIDDEN_ROWS,
    VISIBLE_ROWS,
    file_digest,
    make_manifest,
    make_sample_submission,
    prepare_assets,
    split_full_train,
    validate_asset_frames,
    validate_manifest,
    validate_source_frame,
)


def make_full_train() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [f"id{index:05d}" for index in range(FULL_ROWS)],
            "text": [f"text {index}" for index in range(FULL_ROWS)],
            "author": [CLASSES[index % len(CLASSES)] for index in range(FULL_ROWS)],
        }
    )


class PrepareSpookyAssetsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.full_train = make_full_train()
        cls.visible, cls.hidden = split_full_train(cls.full_train)

    def test_split_is_deterministic_and_has_expected_sizes(self) -> None:
        second_visible, second_hidden = split_full_train(self.full_train)
        self.assertEqual(len(self.visible), VISIBLE_ROWS)
        self.assertEqual(len(self.hidden), HIDDEN_ROWS)
        pd.testing.assert_frame_equal(self.visible, second_visible)
        pd.testing.assert_frame_equal(self.hidden, second_hidden)

    def test_public_hidden_files_have_no_labels_and_keep_order(self) -> None:
        hidden_public = self.hidden[["id", "text"]]
        sample = make_sample_submission(self.hidden)
        reconstructed = validate_asset_frames(
            self.full_train,
            self.visible,
            hidden_public,
            sample,
        )
        self.assertNotIn("author", hidden_public.columns)
        self.assertEqual(sample["id"].tolist(), hidden_public["id"].tolist())
        self.assertEqual(reconstructed["id"].tolist(), self.hidden["id"].tolist())

    def test_wrong_source_shape_fails(self) -> None:
        with self.assertRaisesRegex(ValueError, "source must have"):
            validate_source_frame(self.full_train.iloc[:-1])

    def test_changed_row_order_fails(self) -> None:
        wrong_visible = self.visible.iloc[::-1]
        with self.assertRaisesRegex(ValueError, "visible training data"):
            validate_asset_frames(
                self.full_train,
                wrong_visible,
                self.hidden[["id", "text"]],
            )

    def test_manifest_survives_parquet_round_trip(self) -> None:
        description = "Competition description\n"
        manifest = make_manifest(description)
        with tempfile.TemporaryDirectory() as temporary_dir:
            path = Path(temporary_dir) / "manifest.parquet"
            manifest.to_parquet(path, index=False)
            restored = pd.read_parquet(path)
        description_hash = sha256(description.encode("utf-8")).hexdigest()
        with patch("prepare_spooky_assets.DESCRIPTION_SHA256", description_hash):
            validate_manifest(restored)

    def test_changed_manifest_fails(self) -> None:
        manifest = make_manifest("Competition description\n")
        manifest.iloc[0]["metadata"]["data_dir"] = "/wrong/path"
        with self.assertRaisesRegex(ValueError, "data_dir"):
            validate_manifest(manifest)

    def test_preparation_is_repeatable_and_rejects_conflicts(self) -> None:
        description = "Competition description\n"
        with tempfile.TemporaryDirectory() as temporary_dir:
            root = Path(temporary_dir)
            public_dir = root / "mle-data" / "prepared" / "public"
            public_dir.mkdir(parents=True)
            source_path = public_dir / "train.csv"
            self.full_train.to_csv(source_path, index=False)
            leaderboard_path = root / "leaderboard.csv"
            leaderboard_path.write_text("score\n0.1\n0.2\n", encoding="utf-8")
            competition = SimpleNamespace(
                public_dir=public_dir,
                leaderboard=leaderboard_path,
                description=description,
            )
            assets_root = root / "assets"
            with (
                patch("prepare_spooky_assets.load_mlebench_task", return_value=competition),
                patch("prepare_spooky_assets.SOURCE_MD5", file_digest(source_path, "md5")),
                patch(
                    "prepare_spooky_assets.LEADERBOARD_SHA256",
                    file_digest(leaderboard_path, "sha256"),
                ),
                patch(
                    "prepare_spooky_assets.DESCRIPTION_SHA256",
                    sha256(description.encode("utf-8")).hexdigest(),
                ),
                patch("builtins.print"),
            ):
                prepare_assets(root / "mle-data", assets_root)
                prepare_assets(root / "mle-data", assets_root)

                visible_path = assets_root / "validation" / "spooky-author-identification/train.csv"
                wrong_visible = pd.read_csv(visible_path).iloc[::-1]
                wrong_visible.to_csv(visible_path, index=False)
                with self.assertRaisesRegex(ValueError, "does not match the expected data"):
                    prepare_assets(root / "mle-data", assets_root)


if __name__ == "__main__":
    unittest.main()
