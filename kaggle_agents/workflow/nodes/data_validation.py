"""Data validation node for the Kaggle Agents workflow."""

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ...core.state import KaggleState
from ...utils.csv_utils import read_csv_auto
from ...utils.data_audit import check_id_integrity
from ...utils.precomputed_features import PrecomputedFeaturesInfo, detect_precomputed_features
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
            # Don't override audio to image - audio competitions may have spectrogram PNGs
            if data_type not in ("audio", "audio_classification"):
                detected_type = "image"
            else:
                print(f"   Keeping audio type despite image directory (spectrograms)")
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

        label_files = [Path(lf) for lf in data_files.get("label_files", []) if lf]

        # Extract train/test split from labels (shared helper)
        label_split_set = False
        from ...utils.label_parser import extract_train_test_split_from_labels

        label_split = extract_train_test_split_from_labels(label_files, verbose=True)
        if label_split:
            updates.update(label_split)
            label_split_set = True

        def _merge_feature_infos(infos: list[PrecomputedFeaturesInfo]) -> PrecomputedFeaturesInfo:
            merged = PrecomputedFeaturesInfo()
            for info in infos:
                for key, path in info.features_found.items():
                    if key not in merged.features_found:
                        merged.features_found[key] = path
                        if key in info.feature_shapes:
                            merged.feature_shapes[key] = info.feature_shapes[key]
                        if key in info.feature_columns:
                            merged.feature_columns[key] = info.feature_columns[key]
                merged.warnings.extend(info.warnings)
            merged.total_features = 0
            for key, shape in merged.feature_shapes.items():
                if len(shape) >= 2 and key not in ("cv_folds", "id_mapping"):
                    merged.total_features += shape[1]
            return merged

        def _looks_numeric_ids(ids: list[Any]) -> bool:
            if not ids:
                return False
            for rid in ids[:20]:
                rid_str = str(rid)
                if not rid_str.isdigit():
                    return False
            return True

        def _parse_cvfolds(cv_path: Path) -> tuple[list[Any], list[Any], str] | None:
            cv_df = read_csv_auto(cv_path)
            if len(cv_df.columns) < 2:
                return None
            id_col = cv_df.columns[0]
            fold_col = cv_df.columns[1]
            id_series = pd.to_numeric(cv_df[id_col], errors="coerce")
            use_numeric_ids = id_series.notna().all()
            id_values = id_series.astype(int) if use_numeric_ids else cv_df[id_col].astype(str)
            fold_series = pd.to_numeric(cv_df[fold_col], errors="coerce")
            unique_folds = set(fold_series.dropna().unique())
            train_mask = None
            test_mask = None
            semantics = ""
            if {0, 1}.issubset(unique_folds):
                train_mask = fold_series == 0
                test_mask = fold_series == 1
                semantics = "0=train, 1=test"
            elif {1, 2}.issubset(unique_folds):
                train_mask = fold_series == 1
                test_mask = fold_series == 2
                semantics = "1=train, 2=test"
            elif unique_folds:
                sorted_folds = sorted(unique_folds)
                if len(sorted_folds) >= 2:
                    train_fold = sorted_folds[0]
                    test_fold = sorted_folds[-1]
                    train_mask = fold_series == train_fold
                    test_mask = fold_series == test_fold
                    semantics = f"{train_fold}=train, {test_fold}=test (inferred)"
            if train_mask is None or test_mask is None:
                return None
            train_ids = id_values[train_mask].tolist()
            test_ids = id_values[test_mask].tolist()
            return train_ids, test_ids, semantics

        # Detect precomputed features across multiple roots
        candidate_dirs: list[Path] = []
        for subdir in ["essential_data", "supplemental_data", "data", "features", "prepared"]:
            candidate = working_dir / subdir
            if candidate.exists():
                candidate_dirs.append(candidate)
        for lf in label_files:
            if lf.exists():
                candidate_dirs.append(lf.parent)
        audio_source = data_files.get("audio_source") or ""
        if audio_source:
            candidate_dirs.append(Path(audio_source).parent)
        if not candidate_dirs:
            candidate_dirs = [working_dir]

        seen_dirs: set[Path] = set()
        search_dirs: list[Path] = []
        for candidate in candidate_dirs:
            resolved = candidate.resolve() if candidate.exists() else candidate
            if resolved in seen_dirs:
                continue
            seen_dirs.add(resolved)
            search_dirs.append(candidate)

        feature_infos: list[PrecomputedFeaturesInfo] = []
        for data_dir in search_dirs:
            info = detect_precomputed_features(data_dir)
            if info.has_features():
                feature_infos.append(info)

        precomputed_features_info = (
            _merge_feature_infos(feature_infos) if feature_infos else PrecomputedFeaturesInfo()
        )
        if precomputed_features_info.has_features():
            updates["precomputed_features_info"] = precomputed_features_info.to_dict()
            print(f"   Precomputed features found: {len(precomputed_features_info.features_found)}")
            for ft, path in precomputed_features_info.features_found.items():
                shape_str = ""
                if ft in precomputed_features_info.feature_shapes:
                    shape_str = f" {precomputed_features_info.feature_shapes[ft]}"
                print(f"     - {ft}: {path.name}{shape_str}")

        # CVfolds: Extract train/test split
        cv_folds_path = precomputed_features_info.features_found.get("cv_folds")
        if not cv_folds_path:
            for lf in label_files:
                if "cvfolds" in lf.name.lower():
                    cv_folds_path = lf
                    break
        if cv_folds_path and Path(cv_folds_path).exists():
            try:
                updates["cv_folds_path"] = str(cv_folds_path)
                if not label_split_set:
                    parsed = _parse_cvfolds(Path(cv_folds_path))
                    if parsed:
                        train_rec_ids, test_rec_ids, semantics = parsed
                        updates["train_rec_ids"] = train_rec_ids
                        updates["test_rec_ids"] = test_rec_ids
                        updates["cv_folds_used"] = True
                        updates["train_test_ids_source"] = "cvfolds"
                        print(f"   CVfolds semantics: {semantics}")
                        print(f"   CVfolds: {len(train_rec_ids)} train, {len(test_rec_ids)} test")
                else:
                    print("   CVfolds found; keeping label-based train/test split")
            except Exception as e:
                print(f"   Warning: Failed to parse CVfolds file: {e}")

        # ID mapping hint (keep numeric rec_ids for label alignment)
        id_mapping_path = precomputed_features_info.features_found.get("id_mapping")
        if not id_mapping_path:
            for lf in label_files:
                name = lf.name.lower()
                if "id2filename" in name or "2filename" in name:
                    id_mapping_path = lf
                    break

        train_rec_ids = updates.get("train_rec_ids", [])
        test_rec_ids = updates.get("test_rec_ids", [])
        if id_mapping_path and (train_rec_ids or test_rec_ids):
            if _looks_numeric_ids(train_rec_ids or test_rec_ids):
                updates["id_mapping_path"] = str(id_mapping_path)
                print("   ID mapping available; keeping numeric rec_ids for label alignment")
            else:
                from ...utils.label_parser import read_id_mapping

                audio_dir = Path(audio_source) if audio_source else None
                try:
                    mapping_df = read_id_mapping(
                        id_mapping_path,
                        id_col="rec_id",
                        filename_col="filename",
                        audio_dir=audio_dir,
                        resolve_extensions=True,
                    )
                    id_to_filename = dict(zip(
                        mapping_df["rec_id"].astype(str),
                        mapping_df["filename"]
                    ))
                    train_filenames = [id_to_filename.get(str(rid)) for rid in train_rec_ids]
                    test_filenames = [id_to_filename.get(str(rid)) for rid in test_rec_ids]
                    train_filenames = [f for f in train_filenames if f]
                    test_filenames = [f for f in test_filenames if f]
                    if train_filenames or test_filenames:
                        updates["train_rec_ids"] = train_filenames
                        updates["test_rec_ids"] = test_filenames
                        print(
                            "   ID mapping applied: "
                            f"{len(train_filenames)} train, {len(test_filenames)} test filenames"
                        )
                except Exception as e:
                    print(f"   Warning: Failed to apply ID mapping: {e}")

        # ID Integrity Check: Validate that IDs match actual files
        # This catches early errors like MLSP-2013 where IDs lack .wav extension
        audio_source = data_files.get("audio_source") or ""
        train_rec_ids = updates.get("train_rec_ids", [])

        skip_id_integrity = bool(id_mapping_path and _looks_numeric_ids(train_rec_ids))
        if audio_source and train_rec_ids and not skip_id_integrity:
            audio_dir = Path(audio_source)
            if audio_dir.exists():
                sample_ids = [str(x) for x in train_rec_ids[:20]]
                is_valid, msg, details = check_id_integrity(sample_ids, audio_dir)

                if not is_valid:
                    print(f"\n   ⚠️ ID INTEGRITY WARNING:")
                    print(f"   {msg}")
                    if details.get("suggested_extension"):
                        updates["id_extension_hint"] = details["suggested_extension"]
                        print(f"   HINT: Add '{details['suggested_extension']}' extension to IDs")
                else:
                    print(f"   ✓ ID integrity validated: IDs match files in {audio_dir.name}/")
        elif skip_id_integrity:
            print("   ID integrity check skipped (id_mapping available for numeric rec_ids)")

        print("   ---------------------------------")

    updates["data_files"] = data_files
    return updates
