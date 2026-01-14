"""Data validation node for the Kaggle Agents workflow."""

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ...core.state import KaggleState
from ...utils.precomputed_features import detect_precomputed_features
from ...utils.submission_format import detect_audio_submission_format


def data_validation_node(state: KaggleState) -> dict[str, Any]:
    """
    Validate and normalize data paths before planning.

    Ensures image competitions use image directories (not label CSVs).
    """
    print("\n" + "=" * 60)
    print("= DATA VALIDATION")
    print("=" * 60)

    working_dir = Path(state["working_directory"])
    data_files = dict(state.get("data_files") or {})
    data_type = str(data_files.get("data_type", "")).lower()
    forced_type = (
        os.getenv("KAGGLE_AGENTS_FORCE_DATA_TYPE")
        or os.getenv("KAGGLE_AGENTS_DATA_TYPE")
        or ""
    ).strip().lower()
    if forced_type in {"tabular", "image", "audio", "text"}:
        print(f"   Data type forced via env: {forced_type}")
        data_type = forced_type
        data_files["data_type"] = forced_type

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tif", ".tiff"}

    def _dir_has_ext(dir_path: Path, exts: set[str], limit: int = 200) -> bool:
        if not dir_path.exists() or not dir_path.is_dir():
            return False
        checked = 0
        for p in dir_path.rglob("*"):
            if not p.is_file():
                continue
            checked += 1
            if p.suffix.lower() in exts:
                return True
            if checked >= limit:
                break
        return False

    def _first_dir_with_images(candidates: list[Path | str | None]) -> Path | None:
        for candidate in candidates:
            if not candidate:
                continue
            path = Path(candidate)
            if _dir_has_ext(path, image_exts):
                return path
        return None

    train_candidates = [
        data_files.get("train"),
        working_dir / "train",
        working_dir / "train_images",
        working_dir / "train_imgs",
        working_dir / "images" / "train",
        working_dir / "images",
    ]
    test_candidates = [
        data_files.get("test"),
        working_dir / "test",
        working_dir / "test_images",
        working_dir / "test_imgs",
        working_dir / "images" / "test",
        working_dir / "images",
    ]

    train_dir = _first_dir_with_images(train_candidates)
    test_dir = _first_dir_with_images(test_candidates)

    if forced_type == "tabular":
        train_dir = None
        test_dir = None

    if test_dir and not train_dir:
        base = test_dir.parent
        train_dir = _first_dir_with_images(
            [
                base / "train",
                base / "train_images",
                base / "train_imgs",
                base / "images" / "train",
                base / "images",
            ]
        )

    if train_dir and not test_dir:
        base = train_dir.parent
        test_dir = _first_dir_with_images(
            [
                base / "test",
                base / "test_images",
                base / "test_imgs",
                base / "images" / "test",
                base / "images",
            ]
        )

    train_csv = data_files.get("train_csv") or str(working_dir / "train.csv")
    train_csv_path = Path(train_csv) if train_csv else None
    has_train_csv = bool(train_csv_path and train_csv_path.exists())

    detected_type = data_type or ""
    if not forced_type:
        if train_dir or test_dir:
            detected_type = "image"
        elif has_train_csv and data_type in {"", "tabular"}:
            detected_type = "tabular"

    updates: dict[str, Any] = {"last_updated": datetime.now()}

    if detected_type and detected_type != data_type:
        print(f"   Data type override: {data_type or 'unknown'} -> {detected_type}")
        data_files["data_type"] = detected_type

    if train_dir:
        print(f"   Train images: {train_dir}")
        data_files["train"] = str(train_dir)
        updates["current_train_path"] = str(train_dir)
        updates["train_data_path"] = str(train_dir)
    elif has_train_csv and detected_type == "tabular":
        print(f"   Train CSV: {train_csv_path}")
        data_files["train_csv"] = str(train_csv_path)
        updates["train_data_path"] = str(train_csv_path)

    if test_dir:
        print(f"   Test images: {test_dir}")
        data_files["test"] = str(test_dir)
        updates["current_test_path"] = str(test_dir)
        updates["test_data_path"] = str(test_dir)

    # Audio-specific detection: submission format and precomputed features
    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".wma", ".aiff", ".aif"}
    is_audio = detected_type == "audio" or forced_type == "audio"

    # Check for audio files if not already detected
    if not is_audio and not train_dir:
        if _dir_has_ext(working_dir, audio_exts, limit=50):
            is_audio = True
            detected_type = "audio"
            data_files["data_type"] = "audio"
            print("   Data type detected: audio (based on file extensions)")

    if is_audio:
        print("\n   --- Audio Competition Detection ---")

        # Detect submission format (wide vs long)
        sample_sub_path = data_files.get("sample_submission") or working_dir / "sample_submission.csv"
        sample_sub_path = Path(sample_sub_path)
        if sample_sub_path.exists():
            submission_format_info = detect_audio_submission_format(sample_sub_path)
            updates["submission_format_info"] = submission_format_info.to_dict()
            print(f"   Submission format: {submission_format_info.format_type}")
            if submission_format_info.id_pattern:
                print(f"   ID pattern: {submission_format_info.id_pattern}")
            if submission_format_info.num_classes:
                print(f"   Num classes: {submission_format_info.num_classes}")

        # Detect precomputed features
        data_dir = working_dir
        # Check common data subdirectories
        for subdir in ["essential_data", "data", "features", "prepared"]:
            candidate = working_dir / subdir
            if candidate.exists():
                data_dir = candidate
                break

        precomputed_features_info = detect_precomputed_features(data_dir)
        if precomputed_features_info.has_features():
            updates["precomputed_features_info"] = precomputed_features_info.to_dict()
            print(f"   Precomputed features found: {len(precomputed_features_info.features_found)}")
            for ft, path in precomputed_features_info.features_found.items():
                shape_str = ""
                if ft in precomputed_features_info.feature_shapes:
                    shape_str = f" {precomputed_features_info.feature_shapes[ft]}"
                print(f"     - {ft}: {path.name}{shape_str}")

            # CVfolds: Extract train/test split if CVfolds file found
            if "cv_folds" in precomputed_features_info.features_found:
                cv_folds_path = precomputed_features_info.features_found["cv_folds"]
                try:
                    cv_df = pd.read_csv(cv_folds_path)
                    if len(cv_df.columns) >= 2:
                        id_col = cv_df.columns[0]
                        fold_col = cv_df.columns[1]

                        # Auto-detect fold semantics based on unique values
                        unique_folds = set(cv_df[fold_col].unique())
                        if unique_folds == {0, 1}:
                            # MLSP-style: fold=0 is train, fold=1 is test
                            train_rec_ids = cv_df[cv_df[fold_col] == 0][id_col].tolist()
                            test_rec_ids = cv_df[cv_df[fold_col] == 1][id_col].tolist()
                            print(f"   CVfolds semantics: 0=train, 1=test")
                        else:
                            # Standard semantics: fold=1 is train, fold=2 is test
                            train_rec_ids = cv_df[cv_df[fold_col] == 1][id_col].tolist()
                            test_rec_ids = cv_df[cv_df[fold_col] == 2][id_col].tolist()
                            print(f"   CVfolds semantics: 1=train, 2=test")

                        if train_rec_ids or test_rec_ids:
                            updates["train_rec_ids"] = train_rec_ids
                            updates["test_rec_ids"] = test_rec_ids
                            updates["cv_folds_used"] = True
                            print(f"   CVfolds: {len(train_rec_ids)} train, {len(test_rec_ids)} test")
                except Exception as e:
                    print(f"   Warning: Failed to parse CVfolds file: {e}")

        print("   ---------------------------------")

    updates["data_files"] = data_files
    return updates
