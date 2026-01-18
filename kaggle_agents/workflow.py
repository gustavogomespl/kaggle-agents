"""
LangGraph Workflow for Autonomous Kaggle Competition Solving.

This module defines the complete agent workflow using LangGraph's StateGraph,
implementing the full pipeline from SOTA search to submission.
"""

import os
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from .agents import (
    developer_agent_node,
    ensemble_agent_node,  # Ensemble Strategy
    planner_agent_node,
    robustness_agent_node,
    search_agent_node,
)
from .agents.meta_evaluator_agent import meta_evaluator_node  # Meta-Evaluator with RL
from .agents.reporting_agent import reporting_agent_node
from .agents.submission_agent import submission_agent_node
from .core.config import get_config
from .core.state import KaggleState, create_initial_state
from .domain import detect_competition_domain
from .nodes.curriculum_learning import (
    curriculum_learning_node,
    inject_subtask_guidance,
)
from .nodes.prompt_refinement import prompt_refinement_node
from .tools.data_format_discovery import (
    DataFormatDiscoverer,
    detect_traditional_format,
)
from .tools.kaggle_api import KaggleAPIClient
from .utils.csv_utils import read_csv_auto
from .utils.data_audit import (
    AuditFailedError,
    audit_audio_competition,
    print_audit_report,
)
from .utils.data_contract import prepare_canonical_data
from .utils.precomputed_features import (
    PrecomputedFeaturesInfo,
    detect_precomputed_features,
)
from .utils.submission_format import (
    detect_audio_submission_format,
)


# ==================== Agent Nodes ====================


def data_download_node(state: KaggleState) -> dict[str, Any]:
    """
    Download competition data from Kaggle.

    Args:
        state: Current state

    Returns:
        State updates with data file paths
    """
    print("\n" + "=" * 60)
    print("= DATA DOWNLOAD")
    print("=" * 60)

    competition_info = state["competition_info"]
    working_dir = Path(state["working_directory"])

    print(f"\n📥 Downloading data for: {competition_info.name}")
    print(f"   Destination: {working_dir}")

    try:
        # Initialize Kaggle API client
        kaggle_client = KaggleAPIClient()

        # Download competition data
        data_files = kaggle_client.download_competition_data(
            competition=competition_info.name, path=str(working_dir), quiet=False
        )

        print("\n✓ Download complete!")
        print(f"   Train: {data_files.get('train', 'N/A')}")
        print(f"   Test: {data_files.get('test', 'N/A')}")
        target_col = "target"  # Default
        if data_files.get("sample_submission"):
            print(f"   Sample Submission: {data_files['sample_submission']}")
            try:
                # Infer target column from sample submission (usually 2nd column)
                sample_sub = pd.read_csv(data_files["sample_submission"])
                if len(sample_sub.columns) >= 2:
                    target_col = sample_sub.columns[1]
                    print(f"   🎯 Target Column Detected: {target_col}")
            except Exception as e:
                print(f"   ⚠️ Could not read sample submission to infer target: {e}")

        # GENERATE FIXED FOLDS (Consistent CV)
        if data_files.get("train"):
            try:
                from .utils.cross_validation import generate_folds

                folds_path = str(working_dir / "folds.csv")
                # Use train_csv if available (for image competitions where 'train' is a dir/zip)
                train_path_for_folds = data_files.get("train_csv", data_files["train"])

                generate_folds(
                    train_path=train_path_for_folds,
                    target_col=target_col,
                    output_path=folds_path,
                    n_folds=5,
                    seed=42,
                )
                data_files["folds"] = folds_path
            except Exception as e:
                print(f"   ⚠️ Failed to generate fixed folds: {e}")

        return {
            "data_files": data_files,
            "train_data_path": data_files.get("train", ""),
            "test_data_path": data_files.get("test", ""),
            "sample_submission_path": data_files.get("sample_submission", ""),
            "target_col": target_col,
            "last_updated": datetime.now(),
        }

    except RuntimeError as e:
        # Authentication error
        error_msg = str(e)
        print("\n❌ Kaggle API Authentication Failed")
        print(f"   {error_msg}")
        print("\n💡 To fix:")
        print("   1. Set KAGGLE_USERNAME and KAGGLE_KEY environment variables")
        print("   2. Or create ~/.kaggle/kaggle.json with your credentials")
        print("   3. Get credentials from: https://www.kaggle.com/settings/account")

        return {
            "errors": [f"Kaggle authentication failed: {error_msg}"],
            "last_updated": datetime.now(),
        }

    except Exception as e:
        # Download error
        error_msg = str(e)
        print("\n❌ Data Download Failed")
        print(f"   {error_msg}")
        print("\n💡 Possible causes:")
        print(f"   - Competition '{competition_info.name}' doesn't exist")
        print("   - You haven't accepted the competition rules")
        print("   - Network connectivity issues")

        return {
            "errors": [f"Data download failed: {error_msg}"],
            "last_updated": datetime.now(),
        }


def data_format_discovery_node(state: KaggleState) -> dict[str, Any]:
    """
    Intelligent data format discovery with fallback mechanism.

    This node acts as a fallback when traditional CSV detection fails.
    It fetches information from the competition's Kaggle page and uses
    an LLM to generate adaptive parsing instructions.

    Args:
        state: Current state

    Returns:
        State updates with data format information and parsing instructions
    """
    print("\n" + "=" * 60)
    print("= DATA FORMAT DISCOVERY")
    print("=" * 60)

    working_dir = Path(state["working_directory"])
    competition_info = state["competition_info"]
    competition = competition_info.name

    # Step 1: Try traditional detection first
    print("\n   Checking for standard CSV format...")
    traditional_files = detect_traditional_format(working_dir)

    if traditional_files:
        print("   ✓ Traditional CSV format detected")
        print(f"     Train: {traditional_files.get('train', 'N/A')}")
        print(f"     Test: {traditional_files.get('test', 'N/A')}")
        return {
            "data_format_type": "traditional",
            "last_updated": datetime.now(),
        }

    # Step 2: Fallback - discover format from multiple sources
    print("\n   ⚠️  Non-standard format detected, initiating discovery...")

    discoverer = DataFormatDiscoverer()

    # Gather information from multiple sources
    print("   📄 Fetching competition data page...")
    data_page_content = discoverer.fetch_data_page(competition)

    print("   📁 Listing data files...")
    file_listing = discoverer.list_data_files(working_dir)

    print("   🔍 Analyzing SOTA notebooks for data loading patterns...")
    sota_loading_code = discoverer.analyze_sota_data_loading(competition, max_notebooks=3)

    context = {
        "competition": competition,
        "data_page_content": data_page_content,
        "file_listing": file_listing,
        "description": competition_info.description or "",
        "sota_loading_code": sota_loading_code,
    }

    # Step 3: Use LLM to generate parsing instructions
    print("   🤖 Generating parsing instructions with LLM...")

    from .core.config import get_llm_for_role

    try:
        llm = get_llm_for_role(role="planner", temperature=0.0)
        parsing_info = discoverer.generate_parsing_instructions(llm, context)

        print(f"   ✓ Format type: {parsing_info.get('format_type', 'unknown')}")
        print(f"   ✓ ID column: {parsing_info.get('id_column', 'unknown')}")
        print(f"   ✓ Target column: {parsing_info.get('target_column', 'unknown')}")

        if parsing_info.get("notes"):
            print(f"   📝 Notes: {parsing_info.get('notes')}")

    except Exception as e:
        print(f"   ⚠️  LLM parsing failed: {e}")
        parsing_info = {
            "format_type": "unknown",
            "id_column": "unknown",
            "target_column": "unknown",
            "loading_code": "",
            "can_generate_csv": False,
            "error": str(e),
        }

    # Step 4: Pass parsing instructions to developer agent
    # NOTE: We intentionally do NOT execute the LLM-generated loading code here.
    # The code will be incorporated into component code by the developer agent
    # and executed through the sandboxed CodeExecutor for security.
    updates: dict[str, Any] = {
        "data_format_type": parsing_info.get("format_type", "custom"),
        "parsing_info": parsing_info,
        "last_updated": datetime.now(),
    }

    if parsing_info.get("loading_code"):
        print("\n   📝 Passing loading code to developer agent (will run in sandbox)")
        updates["data_loading_code"] = parsing_info.get("loading_code", "")
    else:
        print("\n   ⚠️  No loading code generated - developer will need to infer format")

    return updates


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
                parsed = _parse_cvfolds(Path(cv_folds_path))
                if parsed:
                    train_rec_ids, test_rec_ids, semantics = parsed
                    updates["train_rec_ids"] = train_rec_ids
                    updates["test_rec_ids"] = test_rec_ids
                    updates["cv_folds_used"] = True
                    updates["cv_folds_path"] = str(cv_folds_path)
                    print(f"   CVfolds semantics: {semantics}")
                    print(f"   CVfolds: {len(train_rec_ids)} train, {len(test_rec_ids)} test")
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
                from .utils.label_parser import read_id_mapping

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

        print("   ---------------------------------")

    updates["data_files"] = data_files
    return updates


def domain_detection_node(state: KaggleState) -> dict[str, Any]:
    """
    Detect competition domain using LLM-First approach.

    Args:
        state: Current state

    Returns:
        State updates with domain detection
    """
    print("\n" + "=" * 60)
    print("= DOMAIN DETECTION")
    print("=" * 60)

    competition_info = state["competition_info"]
    working_dir = state["working_directory"]

    submission_format_type = None
    submission_format_metadata: dict[str, Any] = {}

    # ========================================================================
    # FORCE DOMAIN OVERRIDE via environment variable
    # This takes ABSOLUTE precedence over all detection methods
    # ========================================================================
    env_force_data_type = os.getenv("KAGGLE_AGENTS_FORCE_DATA_TYPE", "")
    env_data_type = os.getenv("KAGGLE_AGENTS_DATA_TYPE", "")
    env_force_domain = os.getenv("KAGGLE_AGENTS_FORCE_DOMAIN", "")

    forced_type = (env_force_data_type or env_data_type or env_force_domain).strip().lower()

    # Debug: show which env vars are set
    if env_force_data_type or env_data_type or env_force_domain:
        print(f"   [ENV] KAGGLE_AGENTS_FORCE_DATA_TYPE={env_force_data_type!r}")
        print(f"   [ENV] KAGGLE_AGENTS_DATA_TYPE={env_data_type!r}")
        print(f"   [ENV] KAGGLE_AGENTS_FORCE_DOMAIN={env_force_domain!r}")
        print(f"   [ENV] Resolved forced_type={forced_type!r}")

    FORCE_TYPE_TO_DOMAIN = {
        # Seq2seq / text normalization
        "seq2seq": "seq_to_seq",
        "seq_to_seq": "seq_to_seq",
        "text_normalization": "seq_to_seq",
        "text-normalization": "seq_to_seq",
        "textnormalization": "seq_to_seq",
        # Image domains
        "image": "image_classification",
        "image_classification": "image_classification",
        "image-classification": "image_classification",
        "image_to_image": "image_to_image",
        "image-to-image": "image_to_image",
        # Audio domains
        "audio": "audio_classification",
        "audio_classification": "audio_classification",
        "audio-classification": "audio_classification",
        "audio_tagging": "audio_tagging",
        "audio-tagging": "audio_tagging",
        # Text domains
        "text": "text_classification",
        "text_classification": "text_classification",
        "text-classification": "text_classification",
        "nlp": "text_classification",
        # Tabular domains
        "tabular": "tabular_classification",
        "tabular_classification": "tabular_classification",
        "tabular-classification": "tabular_classification",
        "tabular_regression": "tabular_regression",
        "tabular-regression": "tabular_regression",
        "regression": "tabular_regression",
    }

    if forced_type and forced_type in FORCE_TYPE_TO_DOMAIN:
        forced_domain = FORCE_TYPE_TO_DOMAIN[forced_type]
        print(f"   ⚠️ Domain FORCED via env var: {forced_domain}")
        print(f"      (KAGGLE_AGENTS_FORCE_DATA_TYPE={forced_type})")
        competition_info.domain = forced_domain
        return {
            "domain_detected": forced_domain,
            "domain_confidence": 1.0,  # User explicitly requested
            "submission_format_type": None,
            "submission_format_metadata": {},
            "submission_format": {},
            "last_updated": datetime.now(),
        }
    # ========================================================================

    # Get LLM for domain detection (use planner's LLM with low temperature)
    from .core.config import get_llm_for_role

    try:
        llm = get_llm_for_role(role="planner", temperature=0.0)
    except Exception as e:
        print(f"   Warning: Could not get LLM for domain detection: {e}")
        llm = None

    # Use multi-signal domain detection (delegated to detector.py)
    # Pass state for SOTA tags extraction if available
    domain, confidence = detect_competition_domain(
        competition_info, working_dir, llm=llm, state=state
    )

    # Check for image-to-image overrides (pixel-level submission format)
    data_files = state.get("data_files") or {}
    data_type = data_files.get("data_type")

    if data_type == "image":
        # Check for clean/target directory (e.g., train_cleaned for denoising)
        clean_train_path = data_files.get("clean_train", "")
        if clean_train_path and Path(clean_train_path).exists():
            domain = "image_to_image"
            confidence = 0.95
            print("   Override: Detected clean/target directory -> image_to_image")

        # Check for pixel-level submission format (many rows per image)
        sample_sub_path = data_files.get("sample_submission", "") or state.get(
            "sample_submission_path", ""
        )
        test_path = data_files.get("test", "") or state.get("test_data_path", "")
        if sample_sub_path and Path(sample_sub_path).exists():
            from .domain import detect_submission_format

            sub_format, sub_meta = detect_submission_format(
                sample_sub_path, test_path if test_path else None, competition_info
            )
            submission_format_type = sub_format
            submission_format_metadata = sub_meta
            if sub_format == "pixel_level":
                domain = "image_to_image"
                confidence = 0.95
                print("   Override: Detected pixel-level submission format -> image_to_image")
                print(f"      Expected rows: {sub_meta.get('expected_rows', 'unknown')}")
                print(f"      ID pattern: {sub_meta.get('id_pattern', 'unknown')}")

    if not domain or confidence < 0.5:
        data_type = str(data_files.get("data_type", "")).lower()
        fallback_domain = None
        if data_type == "image":
            fallback_domain = "image_classification"
        elif data_type == "audio":
            fallback_domain = "audio_classification"
        elif data_type == "text":
            fallback_domain = "text_classification"
        elif data_type == "tabular":
            fallback_domain = "tabular"
        if fallback_domain:
            print(f"   Fallback domain from data_type='{data_type}': {fallback_domain}")
            domain = fallback_domain
            confidence = max(confidence, 0.6)

    print(f"\n Domain Detected: {domain}")
    print(f"   Confidence: {confidence:.1%}")

    competition_info.domain = domain

    return {
        "domain_detected": domain,
        "domain_confidence": confidence,
        "submission_format_type": submission_format_type,
        "submission_format_metadata": submission_format_metadata,
        "submission_format": {"type": submission_format_type, **submission_format_metadata}
        if submission_format_type
        else {},
        "last_updated": datetime.now(),
    }


def data_audit_node(state: KaggleState) -> dict[str, Any]:
    """
    Audit competition data before expensive processing begins.

    For audio competitions, validates that audio files exist and labels are parseable.
    FAIL-FAST: Raises AuditFailedError if critical data is missing.

    Args:
        state: Current state

    Returns:
        State updates with audit results
    """
    print("\n" + "=" * 60)
    print("= DATA AUDIT")
    print("=" * 60)

    domain = state.get("domain_detected", "")
    data_files = state.get("data_files", {})
    working_dir = Path(state.get("working_directory", "."))

    # Only run domain-specific audits for supported domains
    if domain and "audio" in domain.lower():
        print("   Running audio competition audit...")

        audio_source = data_files.get("audio_source")
        audio_source_dir = Path(audio_source) if audio_source else None

        train_path = Path(data_files.get("train", "")) if data_files.get("train") else None
        test_path = Path(data_files.get("test", "")) if data_files.get("test") else None

        label_files = data_files.get("label_files", [])
        label_paths = [Path(lf) for lf in label_files] if label_files else []

        try:
            result = audit_audio_competition(
                working_dir=working_dir,
                audio_source_dir=audio_source_dir,
                label_files=label_paths,
                train_path=train_path,
                test_path=test_path,
                min_audio_files=10,
                strict=True,  # Fail-fast by default
            )
            print_audit_report(result)

            return {
                "data_audit_result": {
                    "is_valid": result.is_valid,
                    "audio_files_found": result.audio_files_found,
                    "audio_source_dir": str(result.audio_source_dir) if result.audio_source_dir else None,
                    "label_files_found": [str(lf) for lf in result.label_files_found],
                    "warnings": result.warnings,
                },
                "last_updated": datetime.now(),
            }

        except AuditFailedError as e:
            print(f"\n   AUDIT FAILED: {e}")
            print("   Stopping execution to prevent wasted compute.")
            # Re-raise to halt the workflow
            raise

    else:
        print(f"   Skipping domain-specific audit for domain: {domain}")
        print("   (Audio audit only runs for audio_* domains)")

    return {
        "data_audit_result": {"is_valid": True, "skipped": True},
        "last_updated": datetime.now(),
    }


def canonical_data_preparation_node(state: KaggleState) -> dict[str, Any]:
    """
    Prepare canonical data contract for consistent data handling.

    This node creates the canonical data artifacts that ALL model components
    must consume. This ensures consistent:
    - Row sampling (same train_ids across all components)
    - Fold assignments (same folds.npy for all CV)
    - Feature columns (intersection of train/test to prevent schema mismatch)
    - Target alignment (y.npy aligned with train_ids)

    Args:
        state: Current state

    Returns:
        State updates with canonical data paths
    """
    print("\n" + "=" * 60)
    print("= CANONICAL DATA PREPARATION")
    print("=" * 60)

    working_dir = Path(state["working_directory"])
    data_files = state.get("data_files", {})
    target_col = state.get("target_col", "target")

    # Get train and test paths
    train_path = data_files.get("train_csv") or data_files.get("train")
    test_path = data_files.get("test_csv") or data_files.get("test")

    # Handle non-tabular data (images, audio)
    data_type = str(data_files.get("data_type", "")).lower()
    if data_type == "image":
        # For IMAGE competitions: Try to create canonical data from train.csv
        # Image competitions typically have train.csv with columns [image_id, label1, label2, ...]
        train_csv_path = data_files.get("train_csv")
        if not train_csv_path:
            # Also check workspace for train.csv
            train_csv_path = working_dir / "train.csv"
            if not train_csv_path.exists():
                train_csv_path = None

        if train_csv_path and Path(train_csv_path).exists():
            print(f"   Image competition with train.csv detected: {train_csv_path}")
            print("   Creating canonical data from train.csv labels...")
            # Use train.csv for canonical data (contains image IDs and labels)
            train_path = str(train_csv_path)
            # test_path: use test.csv if exists, otherwise use sample_submission for schema
            test_csv_path = data_files.get("test_csv")
            if not test_csv_path or not Path(test_csv_path).exists():
                test_csv_path = data_files.get("sample_submission")
            test_path = str(test_csv_path) if test_csv_path else str(train_csv_path)
            # Continue with normal canonical data preparation below
        else:
            print(f"   Skipping canonical data prep for {data_type} data type")
            print("   (No train.csv found - image competitions without labels CSV)")
            return {
                "canonical_data_prepared": False,
                "canonical_data_skipped_reason": f"{data_type} data type - no train.csv",
                "last_updated": datetime.now(),
            }

    # For audio: try filename-based label extraction if no train.csv
    if data_type == "audio":
        train_csv_path = working_dir / "train.csv"
        if not train_csv_path.exists():
            print("   Audio competition without train.csv detected")
            # Try MLSP-style labels + CVfolds before filename heuristics
            label_files = [Path(lf) for lf in data_files.get("label_files", []) if lf]
            label_path = None
            cvfolds_path = None
            for lf in label_files:
                name = lf.name.lower()
                if "rec_labels" in name and label_path is None:
                    label_path = lf
                if "cvfolds" in name and cvfolds_path is None:
                    cvfolds_path = lf

            if label_path and cvfolds_path and label_path.exists() and cvfolds_path.exists():
                print("   Detected MLSP-style label + CVfolds files, building canonical data...")
                try:
                    from .utils.label_parser import parse_mlsp_multilabel

                    rec_ids, labels = parse_mlsp_multilabel(label_path)
                    if len(rec_ids) == 0:
                        raise ValueError("Parsed 0 rec_ids from MLSP label file")

                    cv_df = read_csv_auto(cvfolds_path)
                    if len(cv_df.columns) < 2:
                        raise ValueError("CVfolds file has fewer than 2 columns")

                    id_col = cv_df.columns[0]
                    fold_col = cv_df.columns[1]
                    id_series = pd.to_numeric(cv_df[id_col], errors="coerce")
                    fold_series = pd.to_numeric(cv_df[fold_col], errors="coerce")
                    fold_lookup = {
                        int(rid): int(fold)
                        for rid, fold in zip(id_series, fold_series)
                        if pd.notna(rid) and pd.notna(fold)
                    }

                    fold_values = np.array([fold_lookup.get(int(rid), -1) for rid in rec_ids])
                    unique_folds = set(fold_values.tolist())
                    if {0, 1}.issubset(unique_folds):
                        train_fold = 0
                        test_fold = 1
                        print("   CVfolds semantics: 0=train, 1=test")
                    elif {1, 2}.issubset(unique_folds):
                        train_fold = 1
                        test_fold = 2
                        print("   CVfolds semantics: 1=train, 2=test")
                    else:
                        sorted_folds = sorted(f for f in unique_folds if f >= 0)
                        if len(sorted_folds) < 2:
                            raise ValueError("CVfolds values do not define a train/test split")
                        train_fold = sorted_folds[0]
                        test_fold = sorted_folds[-1]
                        print(f"   CVfolds semantics inferred: {train_fold}=train, {test_fold}=test")

                    has_labels = labels.sum(axis=1) > 0
                    train_mask = (fold_values == train_fold) & has_labels
                    test_mask = fold_values == test_fold

                    train_ids = rec_ids[train_mask]
                    test_ids = rec_ids[test_mask]
                    y_train = labels[train_mask]

                    if len(train_ids) < 2:
                        raise ValueError("Not enough training samples after filtering")

                    n_folds = min(5, max(2, len(train_ids)))
                    stratify_vals = y_train.sum(axis=1)
                    use_stratified = len(np.unique(stratify_vals)) > 1

                    fold_assignments = np.zeros(len(train_ids), dtype=int)
                    if use_stratified:
                        from sklearn.model_selection import StratifiedKFold

                        kf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
                        for fold, (_, val_idx) in enumerate(kf.split(train_ids, stratify_vals)):
                            fold_assignments[val_idx] = fold
                    else:
                        from sklearn.model_selection import KFold

                        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
                        for fold, (_, val_idx) in enumerate(kf.split(train_ids)):
                            fold_assignments[val_idx] = fold

                    canonical_dir = working_dir / "canonical"
                    canonical_dir.mkdir(parents=True, exist_ok=True)
                    np.save(canonical_dir / "train_ids.npy", train_ids)
                    np.save(canonical_dir / "y.npy", y_train)
                    np.save(canonical_dir / "folds.npy", fold_assignments)
                    with open(canonical_dir / "feature_cols.json", "w") as f:
                        json.dump([], f)

                    metadata = {
                        "original_rows": int(len(rec_ids)),
                        "canonical_rows": int(len(train_ids)),
                        "n_folds": int(n_folds),
                        "cv_strategy": "stratified_kfold" if use_stratified else "kfold",
                        "id_col": "rec_id",
                        "id_is_synthetic": False,
                        "target_col": "label",
                        "n_features": 0,
                        "group_col": None,
                        "is_classification": True,
                        "n_classes": int(labels.shape[1]),
                        "canonical_version": "1.3",
                        "task_type": "multilabel_classification",
                        "is_seq2seq": False,
                        "source_col": None,
                        "class_col": None,
                        "seq2seq_group_col": None,
                        "target_dtype": str(y_train.dtype),
                        "sampled": False,
                        "n_test": int(len(test_ids)),
                    }

                    with open(canonical_dir / "metadata.json", "w") as f:
                        json.dump(metadata, f, indent=2)

                    print(f"   Created MLSP canonical data: {len(train_ids)} train, {len(test_ids)} test")
                    return {
                        "canonical_data_prepared": True,
                        "canonical_dir": str(canonical_dir),
                        "canonical_train_ids_path": str(canonical_dir / "train_ids.npy"),
                        "canonical_y_path": str(canonical_dir / "y.npy"),
                        "canonical_folds_path": str(canonical_dir / "folds.npy"),
                        "canonical_metadata": metadata,
                        "canonical_data_skipped_reason": None,
                        "train_rec_ids": train_ids.tolist(),
                        "test_rec_ids": test_ids.tolist(),
                        "cv_folds_used": True,
                        "last_updated": datetime.now(),
                    }
                except Exception as e:
                    print(f"   MLSP canonical data prep failed: {e}")

            print("   Attempting to create canonical data from audio filenames...")

            # Import detection mixin for filename-based label extraction
            from .mlebench.data_adapter.detection import DetectionMixin

            detector = DetectionMixin()

            # Use actual train path from data_files if available (e.g., train2/ for whale competition)
            train_path_from_state = data_files.get("train")
            if train_path_from_state and Path(train_path_from_state).exists():
                train_dir = Path(train_path_from_state)
                if train_dir.name != "train":
                    print(f"   Using non-standard train directory: {train_dir.name}/")
            else:
                train_dir = working_dir / "train"

            if train_dir.exists():
                result = detector.create_canonical_from_audio_filenames(
                    audio_dir=train_dir,
                    canonical_dir=working_dir / "canonical",
                    n_folds=5,
                )

                if result.get("success"):
                    print(f"   Created canonical data from {result['metadata']['canonical_rows']} audio files")
                    return {
                        "canonical_data_prepared": True,
                        "canonical_dir": result["canonical_dir"],
                        "canonical_train_ids_path": result["train_ids_path"],
                        "canonical_y_path": result["y_path"],
                        "canonical_folds_path": result["folds_path"],
                        "canonical_metadata": result["metadata"],
                        "canonical_data_skipped_reason": None,
                        "last_updated": datetime.now(),
                    }
                else:
                    print(f"   Failed to extract labels from filenames: {result.get('error')}")

            # Fallback: skip canonical data for audio without labels
            print("   Skipping canonical data prep for audio (no train.csv or filename labels)")
            return {
                "canonical_data_prepared": False,
                "canonical_data_skipped_reason": "audio without train.csv or filename labels",
                "last_updated": datetime.now(),
            }

    # Validate paths exist
    if not train_path or not Path(train_path).exists():
        print(f"   Warning: Train path not found: {train_path}")
        return {
            "canonical_data_prepared": False,
            "canonical_data_skipped_reason": "train path not found",
            "last_updated": datetime.now(),
        }

    if not test_path or not Path(test_path).exists():
        print(f"   Warning: Test path not found: {test_path}")
        # Continue anyway - test path is optional for canonical prep

    # Determine max_rows for sampling based on config
    fast_mode = state.get("fast_mode", False)
    timeout_s = state.get("timeout_s")

    # Budget-aware sampling thresholds
    max_rows = None
    if fast_mode:
        max_rows = 50_000
        print(f"   Fast mode: sampling to {max_rows:,} rows")
    elif timeout_s and timeout_s < 1800:  # Less than 30 min
        max_rows = 200_000
        print(f"   Short timeout ({timeout_s}s): sampling to {max_rows:,} rows")

    # Detect task type from domain for seq2seq handling
    domain = state.get("domain_detected", "tabular")
    competition_name = state.get("competition_name", "").lower()
    seq2seq_domains = {"seq_to_seq", "text_normalization", "translation", "summarization"}

    # Determine task_type with priority:
    # 1. Specific text_normalization detection from competition name
    # 2. Domain detected as seq_to_seq variant
    # 3. Default to tabular
    text_norm_keywords = ["normalization", "normalize", "text-norm", "tts"]
    is_text_norm = any(kw in competition_name for kw in text_norm_keywords)

    if is_text_norm:
        task_type = "text_normalization"
        print("   Task type: text_normalization (detected from competition name)")
    elif domain in seq2seq_domains:
        # Map generic seq_to_seq to specific type if possible
        task_type = domain if domain != "seq_to_seq" else "seq2seq"
        print(f"   Task type from domain: {task_type}")
    else:
        task_type = "tabular"

    try:
        canonical_result = prepare_canonical_data(
            train_path=train_path,
            test_path=test_path if test_path and Path(test_path).exists() else train_path,
            target_col=target_col,
            output_dir=working_dir,
            max_rows=max_rows,
            fast_mode=fast_mode,
            timeout_s=timeout_s,
            task_type=task_type,
        )

        print("\n   Canonical data artifacts created:")
        print(f"      train_ids: {canonical_result['metadata']['canonical_rows']:,} rows")
        print(f"      n_folds: {canonical_result['metadata']['n_folds']}")
        print(f"      n_features: {canonical_result['metadata']['n_features']}")

        if canonical_result["metadata"].get("sampled"):
            print(f"      Sampled from {canonical_result['metadata']['original_rows']:,} original rows")

        if canonical_result["metadata"].get("group_col"):
            print(f"      Group column: {canonical_result['metadata']['group_col']} (GroupKFold)")

        return {
            "canonical_data_prepared": True,
            "canonical_dir": canonical_result["canonical_dir"],
            "canonical_train_ids_path": canonical_result["train_ids_path"],
            "canonical_y_path": canonical_result["y_path"],
            "canonical_folds_path": canonical_result["folds_path"],
            "canonical_feature_cols_path": canonical_result["feature_cols_path"],
            "canonical_metadata": canonical_result["metadata"],
            "last_updated": datetime.now(),
        }

    except Exception as e:
        print(f"\n   Error preparing canonical data: {e}")
        print("   Continuing without canonical data contract...")
        return {
            "canonical_data_prepared": False,
            "canonical_data_error": str(e),
            "last_updated": datetime.now(),
        }


def iteration_control_node(state: KaggleState) -> dict[str, Any]:
    """
    Control iteration and check termination conditions.

    Args:
        state: Current state

    Returns:
        State updates with iteration control
    """
    print("\n" + "=" * 60)
    print("= ITERATION CONTROL")
    print("=" * 60)

    current_iteration = state.get("current_iteration", 0)
    max_iterations = state.get("max_iterations", 10)
    best_score = state.get("best_score", 0.0)
    target_percentile = state.get("target_percentile", 20.0)

    # Increment iteration
    new_iteration = current_iteration + 1

    print(f"\nIteration: {new_iteration}/{max_iterations}")
    print(f"   Best Score: {best_score:.4f}")
    print(f"   Target: Top {target_percentile}%")

    # Check if we should continue
    should_continue = new_iteration < max_iterations

    # Check if goal achieved
    # Note: In real scenario, would check actual percentile
    # For now, continue until max iterations

    termination_reason = None
    if not should_continue:
        termination_reason = "max_iterations_reached"

    # Reset component index for refinement iterations (iteration > 1)
    # This ensures new components from refined plan are implemented
    updates = {
        "current_iteration": new_iteration,
        "should_continue": should_continue,
        "termination_reason": termination_reason,
        "last_updated": datetime.now(),
    }

    # If this is a refinement iteration (> 1), reset component index and skip flag
    if new_iteration > 1:
        print("   🔄 Starting refinement iteration - resetting component index")
        updates["current_component_index"] = 0
        # Reset skip_remaining_components so new iteration can run all components
        updates["skip_remaining_components"] = False

    return updates


def performance_evaluation_node(state: KaggleState) -> dict[str, Any]:
    """
    Evaluate performance and decide if refinement is needed.

    Args:
        state: Current state

    Returns:
        State updates with refinement decision
    """
    print("\n" + "=" * 60)
    print("= PERFORMANCE EVALUATION")
    print("=" * 60)

    current_score = state.get("best_score", 0.0)
    # Dynamic target_score from state (set by MLE-bench or config), fallback to top 20% threshold
    target_score = state.get("target_score")
    if target_score is None:
        target_score = 1.0
    elif isinstance(target_score, str):
        try:
            target_score = float(target_score)
        except ValueError:
            target_score = 1.0
    current_iteration = state.get("current_iteration", 0)
    max_iterations = state.get("max_iterations", 10)

    # Get submission results if available
    submissions = state.get("submissions", [])
    public_score = None
    if submissions:
        latest_sub = submissions[-1]
        public_score = latest_sub.public_score
        if public_score is not None:
            print(f"\n📊 Public Score: {public_score:.4f}")
            # Use metric direction for score selection
            from .core.config import compare_scores

            if current_score == 0.0:
                current_score = public_score
            else:
                try:
                    metric_name = state["competition_info"].evaluation_metric
                except Exception:
                    metric_name = ""
                current_score = compare_scores(current_score, public_score, metric_name)

    from .core.config import is_metric_minimization

    metric_name = ""
    try:
        metric_name = state["competition_info"].evaluation_metric
    except Exception:
        metric_name = ""

    minimize = is_metric_minimization(metric_name)
    gap = (current_score - target_score) if minimize else (target_score - current_score)

    print(f"\nCurrent Score: {current_score:.4f}")
    print(f"Target Score:  {target_score:.4f}")
    print(f"Gap:           {gap:.4f} ({'minimize' if minimize else 'maximize'})")

    # Analyze component performance
    dev_results = state.get("development_results", [])
    successful_components = [r for r in dev_results if r.success]

    print(
        f"\n📈 Component Success Rate: {len(successful_components)}/{len(dev_results)} ({len(successful_components) / len(dev_results) * 100:.0f}%)"
        if dev_results
        else "\n📈 No components tested yet"
    )

    # Decision: should we refine?
    needs_refinement = False
    refinement_reason = None

    if minimize:
        target_achieved = current_score <= target_score
    else:
        target_achieved = current_score >= target_score

    if target_achieved:
        comparator = "<=" if minimize else ">="
        print(f"\n🎉 Target achieved! ({current_score:.4f} {comparator} {target_score:.4f})")
        needs_refinement = False
    elif current_iteration >= max_iterations:
        print(f"\n⏱️  Max iterations reached ({current_iteration}/{max_iterations})")
        needs_refinement = False
    else:
        # Check if we have room for improvement
        improvement_potential = gap

        if improvement_potential > 0.001:  # 0.1% gap
            print(f"\n🔄 Refinement needed (gap: {improvement_potential:.4f})")
            needs_refinement = True
            refinement_reason = "score_below_target"
        else:
            print("\n✅ Close enough to target")
            needs_refinement = False

    return {
        "needs_refinement": needs_refinement,
        "refinement_reason": refinement_reason,
        "current_performance_score": current_score,
        "last_updated": datetime.now(),
    }


# ==================== Conditional Functions ====================


def should_continue_workflow(state: KaggleState) -> Literal["continue", "end"]:
    """
    Decide whether to continue or end the workflow.

    Args:
        state: Current state

    Returns:
        "continue" or "end"
    """
    should_continue = state.get("should_continue", True)
    current_iteration = state.get("current_iteration", 0)
    max_iterations = state.get("max_iterations", 10)

    # End conditions
    if not should_continue:
        return "end"

    if current_iteration >= max_iterations:
        return "end"

    # Check if we have components to implement
    ablation_plan = state.get("ablation_plan", [])
    current_component_index = state.get("current_component_index", 0)

    if current_component_index >= len(ablation_plan):
        # All components implemented, could iterate or end
        return "end"

    return "continue"


def should_retry_component(state: KaggleState) -> Literal["retry", "next"]:
    """
    Decide whether to retry current component or move to next.

    Args:
        state: Current state

    Returns:
        "retry" or "next"
    """
    development_results = state.get("development_results", [])

    if not development_results:
        return "next"

    # Check last result
    last_result = development_results[-1]

    if last_result.success:
        return "next"

    # Check retry count
    code_retry_count = state.get("code_retry_count", 0)
    max_retries = 3  # Max retries at workflow level

    if code_retry_count < max_retries:
        return "retry"

    # Max retries reached, move to next component
    return "next"


def route_after_developer(state: KaggleState) -> Literal["iterate", "end"]:
    """
    Route after developer agent completes.

    Simplified routing logic - only stops for:
    1. Explicit skip_remaining_components flag
    2. Medal achievement (MLE-bench success)
    3. Critical errors (data download failed, auth issues)
    4. All components implemented

    Target score checking is delegated to iteration_control to allow
    multiple refinement iterations with meta-evaluator insights.

    Args:
        state: Current state

    Returns:
        "iterate" to continue implementing components, or "end" if done
    """
    # Explicit early-stop flag (e.g., set by DeveloperAgent)
    if state.get("skip_remaining_components"):
        print("\n⏩ skip_remaining_components=True - Moving to validation")
        return "end"

    # Check for medal achievement in MLE-bench mode (immediate success)
    mlebench_grade = state.get("mlebench_grade")
    run_mode = str(state.get("run_mode", "")).lower()

    if run_mode == "mlebench" and isinstance(mlebench_grade, dict):
        if mlebench_grade.get("valid_submission"):
            if any(mlebench_grade.get(m) for m in ["gold_medal", "silver_medal", "bronze_medal"]):
                print("\n🏅 MEDAL ACHIEVED - Moving to validation")
                return "end"

    # Check for critical errors (data download failed, auth issues)
    errors = state.get("errors", [])
    if errors:
        for error in errors:
            if "Data download failed" in error or "authentication failed" in error.lower():
                print("\n⚠️ Critical error detected, stopping workflow")
                return "end"

    ablation_plan = state.get("ablation_plan", [])
    current_component_index = state.get("current_component_index", 0)

    # Check if more components to implement
    if current_component_index < len(ablation_plan):
        # Check if we're stuck on the same component (prevent infinite loop)
        dev_results = state.get("development_results", [])
        if len(dev_results) >= 3:
            # Check if last 3 results all failed on same component
            recent_failures = [r for r in dev_results[-3:] if not r.success]
            if len(recent_failures) == 3:
                # Check if all have same error about data files
                data_errors = [
                    r for r in recent_failures if "Data files not found" in (r.stderr or "")
                ]
                if len(data_errors) == 3:
                    print("\n⚠️ Repeated data file errors, stopping workflow")
                    return "end"

        remaining = len(ablation_plan) - current_component_index
        print(f"\n🔄 {remaining} component(s) remaining - continuing iteration")
        return "iterate"

    # All components done - move to validation
    print(f"\n✅ All {len(ablation_plan)} components implemented - moving to validation")
    return "end"


def route_after_submission(state: KaggleState) -> Literal["retry_developer", "continue"]:
    """
    Route after submission agent - retry if submission is invalid.

    Checks if the submission passed validation. If not, routes back to
    the developer to regenerate with the error context.

    Args:
        state: Current state

    Returns:
        "retry_developer" if submission invalid and retries remaining,
        "continue" otherwise
    """
    submissions = state.get("submissions", [])

    if not submissions:
        # No submission generated at all - retry
        retry_count = state.get("retry_submission_count", 0)
        if retry_count < 3:
            state["retry_submission_count"] = retry_count + 1
            state["submission_validation_error"] = "No submission file generated"
            print(f"⚠️ No submission generated, retrying... ({retry_count + 1}/3)")
            return "retry_developer"
        return "continue"

    last_submission = submissions[-1]

    # Check if submission is valid (handle both dict and object)
    is_valid = True
    error_msg = None

    if isinstance(last_submission, dict):
        is_valid = last_submission.get("valid", True)
        error_msg = last_submission.get("error")
    else:
        # Object with attributes
        is_valid = getattr(last_submission, "valid", True)
        error_msg = getattr(last_submission, "error", None)

    if not is_valid and error_msg:
        retry_count = state.get("retry_submission_count", 0)

        if retry_count < 3:
            state["retry_submission_count"] = retry_count + 1
            state["submission_validation_error"] = error_msg
            print(f"⚠️ Invalid submission: {error_msg[:100]}...")
            print(f"   Retrying with error context... ({retry_count + 1}/3)")
            return "retry_developer"
        print("⚠️ Max submission retries reached, continuing...")

    return "continue"


def route_after_iteration_control(state: KaggleState) -> Literal["refine", "end"]:
    """
    Route after iteration control - decide if we refine or end.

    Uses adaptive iteration logic:
    1. If score gap > threshold, extend iterations
    2. In MLE-bench mode, aggressively refines until medal/max
    3. Respects minimum iterations before early stopping

    Args:
        state: Current state

    Returns:
        "refine" to start refinement iteration, or "end" if done
    """
    from .core.config import get_config

    config = get_config()
    iter_config = config.iteration

    needs_refinement = state.get("needs_refinement", False)
    current_iteration = state.get("current_iteration", 0)
    base_max_iterations = state.get("max_iterations", iter_config.max_iterations)
    run_mode = str(state.get("run_mode", "")).lower()
    mlebench_grade = state.get("mlebench_grade")

    # Calculate effective max_iterations based on score gap (adaptive)
    max_iterations = base_max_iterations
    if iter_config.adaptive_iterations:
        current_score = state.get("current_performance_score", 0.0)
        target_score = state.get("target_score")
        if target_score and isinstance(target_score, (int, float)) and target_score > 0:
            # Calculate gap percentage
            score_gap = abs(float(target_score) - float(current_score)) / float(target_score)
            if score_gap > iter_config.score_gap_threshold:
                # Extend iterations when gap is large
                max_iterations = min(iter_config.extended_max_iterations, base_max_iterations * 2)
                print(f"   📈 Score gap {score_gap:.1%} > {iter_config.score_gap_threshold:.0%} threshold")
                print(f"      Extended max_iterations: {base_max_iterations} → {max_iterations}")

    print("\n🔀 Routing decision:")
    print(f"   Current iteration: {current_iteration}")
    print(f"   Max iterations: {max_iterations}")
    print(f"   Needs refinement: {needs_refinement}")
    print(f"   Run mode: {run_mode}")

    # Check medal status
    has_gold = False
    has_any_medal = False
    if isinstance(mlebench_grade, dict) and mlebench_grade.get("valid_submission"):
        has_gold = mlebench_grade.get("gold_medal", False)
        has_any_medal = any(mlebench_grade.get(m) for m in ["gold_medal", "silver_medal", "bronze_medal"])

    # Handle skip flag - in MLE-bench mode, only end if gold or max iterations reached
    if state.get("skip_remaining_components"):
        if run_mode == "mlebench":
            if has_gold:
                print("   🥇 GOLD MEDAL ACHIEVED - Ending")
                return "end"
            if current_iteration >= max_iterations:
                print(f"   ⏱️  Max iterations reached with medal ({current_iteration}/{max_iterations})")
                return "end"
            # Reset skip flag and continue refining for better medal
            print(f"   🔄 Medal achieved but continuing for gold (iteration {current_iteration + 1}/{max_iterations})")
                # Note: State update happens in iteration_control_node, not here
        else:
            print("   ⏩ skip_remaining_components=True - Ending")
            return "end"

    # Check for gold medal achievement (always stop on gold)
    if has_gold:
        print("   🥇 GOLD MEDAL ACHIEVED - Success!")
        return "end"

    # Max iterations reached
    if current_iteration >= max_iterations:
        print(f"   ⏱️  Max iterations reached ({current_iteration}/{max_iterations})")
        return "end"

    # MLE-bench mode: aggressively refine until medal or max_iterations
    if run_mode == "mlebench":
        # Log refinement guidance if available
        refinement_guidance = state.get("refinement_guidance", {})
        if refinement_guidance:
            print("   📋 Refinement guidance available from meta-evaluator")
            if refinement_guidance.get("planner_guidance"):
                print(f"      Planner: {refinement_guidance['planner_guidance'][:80]}...")
            if refinement_guidance.get("developer_guidance"):
                print(f"      Developer: {refinement_guidance['developer_guidance'][:80]}...")

        print(f"   🔄 MLE-bench mode: Starting refinement iteration {current_iteration + 1}")
        return "refine"

    # Standard Kaggle mode: check target_score
    current_score = state.get("current_performance_score", 0.0)
    target_score = state.get("target_score")
    if target_score is None:
        target_score = 1.0
    elif isinstance(target_score, str):
        try:
            target_score = float(target_score)
        except ValueError:
            target_score = 1.0

    # Respect metric direction when available
    from .core.config import is_metric_minimization

    metric_name = ""
    try:
        metric_name = state["competition_info"].evaluation_metric
    except Exception:
        metric_name = ""

    if isinstance(current_score, str):
        try:
            current_score = float(current_score)
        except ValueError:
            current_score = 0.0

    if isinstance(target_score, str):
        try:
            target_score = float(target_score)
        except ValueError:
            target_score = 1.0

    if isinstance(current_score, (int, float)) and isinstance(target_score, (int, float)):
        goal_achieved = False
        if is_metric_minimization(metric_name):
            goal_achieved = float(current_score) <= float(target_score)
        else:
            goal_achieved = float(current_score) >= float(target_score)

        if goal_achieved:
            # Respect min_iterations before early stopping
            if iter_config.adaptive_iterations and current_iteration < iter_config.min_iterations:
                print(f"   🎯 Goal achieved but below min_iterations ({current_iteration}/{iter_config.min_iterations})")
                print("      Continuing to consolidate improvements...")
                return "refine"
            print(f"   ✅ Goal achieved: {current_score:.4f} vs target {target_score:.4f}")
            return "end"

    # Decide based on refinement flag
    if needs_refinement:
        print(f"   🔄 Starting refinement iteration {current_iteration + 1}")
        return "refine"

    # If below min_iterations, continue even without explicit refinement need
    if iter_config.adaptive_iterations and current_iteration < iter_config.min_iterations:
        print(f"   📊 Below min_iterations ({current_iteration}/{iter_config.min_iterations}) - continuing")
        return "refine"

    print("   ✅ No refinement needed")
    return "end"


# ==================== SOTA Search Node ====================


def auto_sota_search_node(state: KaggleState) -> dict[str, Any]:
    """
    Automatic SOTA search triggered by stagnation or score gap detection.

    Searches for winning solutions and techniques when progress stalls.

    Args:
        state: Current workflow state

    Returns:
        State updates with SOTA search results and guidance
    """
    from .agents.search_agent import SearchAgent

    print("\n" + "=" * 60)
    print("= AUTO SOTA SEARCH: Finding solutions to break stagnation")
    print("=" * 60)

    stagnation = state.get("stagnation_detection", {})
    if not stagnation.get("trigger_sota_search"):
        print("   Skipping - no SOTA search trigger")
        return {}

    competition_name = state.get("competition_name", "")
    domain = state.get("domain_detected", "tabular")
    current_score = state.get("current_performance_score", 0.0)
    competition_info = state.get("competition_info")

    print(f"\n   🔍 Searching SOTA solutions for: {competition_name}")
    print(f"   📊 Current score: {current_score}")
    print(f"   🎯 Trigger reason: {stagnation.get('reason', 'unknown')}")

    try:
        search_agent = SearchAgent()

        # Focus search on areas that could improve the score
        focus_areas = ["feature_engineering", "model_architecture", "ensemble_strategy"]

        # If stagnation is the issue, focus on novel approaches
        if stagnation.get("stagnated"):
            focus_areas.insert(0, "novel_approaches")
            focus_areas.insert(1, "hyperparameter_optimization")

        # Search for solutions
        search_results = search_agent.search_with_focus(
            competition=competition_name,
            domain=domain,
            focus_areas=focus_areas,
            max_results=5,
        ) if hasattr(search_agent, 'search_with_focus') else {}

        # Generate guidance from search results
        sota_guidance = _generate_sota_guidance_from_results(search_results, stagnation)

        print(f"\n   ✅ SOTA search complete - found {len(search_results.get('solutions', []))} relevant solutions")

        return {
            "sota_search_results": search_results,
            "sota_search_triggered": True,
            "refinement_guidance": {
                **state.get("refinement_guidance", {}),
                "sota_guidance": sota_guidance,
                "sota_triggered_by": stagnation.get("reason"),
            },
            "last_updated": datetime.now(),
        }

    except Exception as e:
        print(f"\n   ⚠️ SOTA search failed: {e}")
        # Return minimal guidance even if search fails
        return {
            "sota_search_triggered": True,
            "refinement_guidance": {
                **state.get("refinement_guidance", {}),
                "sota_guidance": _generate_fallback_sota_guidance(domain, stagnation),
            },
        }


def _generate_sota_guidance_from_results(search_results: dict, stagnation: dict) -> str:
    """Generate guidance string from SOTA search results."""
    solutions = search_results.get("solutions", [])

    guidance_parts = [
        "## SOTA Search Results (triggered by stagnation detection)",
        "",
        f"Trigger reason: {stagnation.get('reason', 'unknown')}",
        "",
    ]

    if solutions:
        guidance_parts.append("### Top Solutions Found:")
        for i, sol in enumerate(solutions[:3], 1):
            title = sol.get("title", "Unknown")
            approach = sol.get("approach", "Not specified")
            guidance_parts.append(f"{i}. **{title}**")
            guidance_parts.append(f"   - Approach: {approach}")

        guidance_parts.append("")
        guidance_parts.append("### Recommended Actions:")
        guidance_parts.append("1. Try feature engineering techniques from top solutions")
        guidance_parts.append("2. Consider model architectures used by winners")
        guidance_parts.append("3. Explore ensemble strategies mentioned")
    else:
        guidance_parts.append("### No specific solutions found - general recommendations:")
        guidance_parts.extend(_get_general_improvement_suggestions())

    return "\n".join(guidance_parts)


def _generate_fallback_sota_guidance(domain: str, stagnation: dict) -> str:
    """Generate fallback guidance when SOTA search fails."""
    guidance = [
        "## Stagnation Detected - General Improvement Suggestions",
        "",
        f"Domain: {domain}",
        f"Trigger: {stagnation.get('reason', 'unknown')}",
        "",
    ]
    guidance.extend(_get_general_improvement_suggestions())
    return "\n".join(guidance)


def _get_general_improvement_suggestions() -> list[str]:
    """Get general suggestions for breaking stagnation."""
    return [
        "### General Strategies to Break Stagnation:",
        "1. **Feature Engineering**: Create interaction features, aggregations, or target encoding",
        "2. **Model Diversity**: Try different model families (Neural, Gradient Boosting, Linear)",
        "3. **Hyperparameter Exploration**: Significantly change learning rate, depth, regularization",
        "4. **Ensemble Methods**: Use stacking with diverse base models",
        "5. **Data Augmentation**: For image/audio, add more augmentation strategies",
        "6. **Cross-Validation**: Ensure CV strategy matches competition requirements",
    ]


# ==================== Workflow Construction ====================


def route_after_meta_evaluator(state: KaggleState) -> Literal["sota_search", "curriculum", "continue"]:
    """
    Route after meta-evaluator - check for SOTA search or curriculum learning.

    Priority:
    1. SOTA search if stagnation/score gap detected
    2. Curriculum learning if critical failures
    3. Continue otherwise

    Args:
        state: Current state

    Returns:
        "sota_search", "curriculum", or "continue"
    """
    # Check for SOTA search trigger (stagnation or score gap)
    stagnation = state.get("stagnation_detection", {})
    if stagnation.get("trigger_sota_search"):
        print(f"\n   🔍 SOTA Search triggered: {stagnation.get('reason', 'stagnation detected')}")
        return "sota_search"

    failure_analysis = state.get("failure_analysis", {})
    error_patterns = failure_analysis.get("error_patterns", [])
    failed_components = failure_analysis.get("failed_components", [])

    # Check for critical errors that need curriculum learning
    critical_errors = ["memory_error", "timeout_error", "import_error", "syntax_error", "data_alignment"]
    has_critical = any(e in critical_errors for e in error_patterns)

    # Only trigger curriculum if we have failures and this is a refinement iteration
    current_iteration = state.get("current_iteration", 0)

    if has_critical and current_iteration > 0 and len(failed_components) > 0:
        print("\n   WEBRL: Critical failures detected - triggering curriculum learning")
        return "curriculum"

    return "continue"


def create_workflow() -> StateGraph:
    """
    Create the complete LangGraph workflow.

    Returns:
        Compiled StateGraph
    """
    # Initialize graph
    workflow = StateGraph(KaggleState)

    # Add nodes
    workflow.add_node("data_download", data_download_node)
    workflow.add_node("data_format_discovery", data_format_discovery_node)  # Fallback for non-standard formats
    workflow.add_node("data_validation", data_validation_node)
    workflow.add_node("domain_detection", domain_detection_node)
    workflow.add_node("data_audit", data_audit_node)  # Fail-fast audit for audio competitions
    workflow.add_node("canonical_data_preparation", canonical_data_preparation_node)  # Canonical data contract
    workflow.add_node("search", search_agent_node)
    workflow.add_node("planner", planner_agent_node)
    workflow.add_node("developer", developer_agent_node)
    workflow.add_node("robustness", robustness_agent_node)
    workflow.add_node("submission", submission_agent_node)
    workflow.add_node("iteration_control", iteration_control_node)
    workflow.add_node("performance_evaluation", performance_evaluation_node)
    workflow.add_node("meta_evaluator", meta_evaluator_node)  # RL-based meta-evaluation
    workflow.add_node("auto_sota_search", auto_sota_search_node)  # SOTA search on stagnation
    workflow.add_node("curriculum_learning", curriculum_learning_node)  # WEBRL: sub-tasks from failures
    workflow.add_node("inject_curriculum", inject_subtask_guidance)  # WEBRL: inject guidance
    workflow.add_node("prompt_refinement", prompt_refinement_node)  # RLPrompt/DSPy optimization
    workflow.add_node("ensemble", ensemble_agent_node)
    workflow.add_node("reporting", reporting_agent_node)

    # Define edges
    # Start → Data Download
    workflow.set_entry_point("data_download")

    # Data Download → Data Format Discovery → Data Validation → Domain Detection → Data Audit
    workflow.add_edge("data_download", "data_format_discovery")
    workflow.add_edge("data_format_discovery", "data_validation")
    workflow.add_edge("data_validation", "domain_detection")
    workflow.add_edge("domain_detection", "data_audit")

    # Data Audit → Canonical Data Preparation → Search
    workflow.add_edge("data_audit", "canonical_data_preparation")
    workflow.add_edge("canonical_data_preparation", "search")

    # Search → Planner
    workflow.add_edge("search", "planner")

    # Planner → Developer
    workflow.add_edge("planner", "developer")

    # Developer → Conditional (more components or done?)
    workflow.add_conditional_edges(
        "developer",
        route_after_developer,
        {
            "iterate": "developer",  # More components to implement
            "end": "robustness",  # All components done → validate
        },
    )

    # Robustness → Ensemble
    workflow.add_edge("robustness", "ensemble")

    # Ensemble → Submission
    workflow.add_edge("ensemble", "submission")

    # Submission → Conditional (valid or retry?)
    workflow.add_conditional_edges(
        "submission",
        route_after_submission,
        {
            "retry_developer": "developer",  # Invalid submission → regenerate
            "continue": "performance_evaluation",  # Valid → continue
        },
    )

    # Performance Evaluation → Meta-Evaluator (RL analysis)
    workflow.add_edge("performance_evaluation", "meta_evaluator")

    # Meta-Evaluator → Conditional (SOTA search, curriculum, or continue?)
    workflow.add_conditional_edges(
        "meta_evaluator",
        route_after_meta_evaluator,
        {
            "sota_search": "auto_sota_search",  # Stagnation detected → SOTA search
            "curriculum": "curriculum_learning",  # WEBRL: Generate sub-tasks
            "continue": "prompt_refinement",  # Standard path
        },
    )

    # SOTA Search → Curriculum Learning (to also process any failures)
    workflow.add_edge("auto_sota_search", "curriculum_learning")

    # Curriculum Learning → Inject Guidance → Prompt Refinement
    workflow.add_edge("curriculum_learning", "inject_curriculum")
    workflow.add_edge("inject_curriculum", "prompt_refinement")

    # Prompt Refinement → Iteration Control
    workflow.add_edge("prompt_refinement", "iteration_control")

    # Iteration Control → Conditional (refine or done?)
    workflow.add_conditional_edges(
        "iteration_control",
        route_after_iteration_control,
        {
            "refine": "planner",  # Start refinement cycle
            "end": "reporting",  # Goal achieved or max iterations -> Explain
        },
    )

    # Reporting → END
    workflow.add_edge("reporting", END)

    return workflow


def compile_workflow(checkpointer=None):
    """
    Compile the workflow with optional checkpointing.

    Args:
        checkpointer: Optional checkpointer (e.g., MemorySaver())

    Returns:
        Compiled workflow
    """
    workflow = create_workflow()

    return workflow.compile(checkpointer=checkpointer) if checkpointer else workflow.compile()


# ==================== Workflow Execution ====================


def run_workflow(
    competition_name: str,
    working_dir: str,
    competition_info: dict[str, Any],
    max_iterations: int = 5,
    use_checkpointing: bool = False,
) -> KaggleState:
    """
    Run the complete workflow for a competition.

    Args:
        competition_name: Name of the Kaggle competition
        working_dir: Working directory for artifacts
        competition_info: Competition metadata
        max_iterations: Maximum workflow iterations
        use_checkpointing: Whether to use checkpointing

    Returns:
        Final state
    """
    print("=" * 70)
    print(f"KAGGLE AGENTS WORKFLOW: {competition_name}")
    print("=" * 70)

    # Create initial state
    state = create_initial_state(competition_name, working_dir)

    # Set competition info
    from .core.state import CompetitionInfo

    state["competition_info"] = CompetitionInfo(**competition_info)

    # Set iteration config
    state["max_iterations"] = max_iterations

    # Create workflow
    # Get centralized recursion_limit from config (default 300)
    agent_config = get_config()
    recursion_limit = agent_config.iteration.langgraph_recursion_limit

    if use_checkpointing:
        checkpointer = MemorySaver()
        workflow = compile_workflow(checkpointer=checkpointer)

        # Run with config for checkpointing
        config = {
            "configurable": {"thread_id": competition_name},
            "recursion_limit": recursion_limit,
            "metadata": {
                "competition": competition_name,
                "project": "default",
                "type": "autonomous-run",
            },
        }
        final_state = workflow.invoke(state, config)
    else:
        workflow = compile_workflow()
        config = {
            "recursion_limit": recursion_limit,
            "metadata": {
                "competition": competition_name,
                "project": "default",
                "type": "autonomous-run",
            },
        }
        final_state = workflow.invoke(state, config)

    print("\n" + "=" * 70)
    print("WORKFLOW COMPLETE")
    print("=" * 70)

    # Print summary
    print("\n📊 Summary:")
    print(f"   Iterations: {final_state.get('current_iteration', 0)}")
    print(f"   SOTA Solutions: {len(final_state.get('sota_solutions', []))}")
    print(f"   Components Planned: {len(final_state.get('ablation_plan', []))}")
    print(f"   Components Implemented: {len(final_state.get('development_results', []))}")

    # Success count
    dev_results = final_state.get("development_results", [])
    successful = sum(1 for r in dev_results if r.success)
    if dev_results:
        print(
            f"   Success Rate: {successful}/{len(dev_results)} ({successful / len(dev_results) * 100:.0f}%)"
        )

    # Validation summary
    validation_score = final_state.get("overall_validation_score")
    if validation_score is not None:
        print(f"   Validation Score: {validation_score:.1%}")

    # Submission summary
    submissions = final_state.get("submissions", [])
    if submissions:
        latest_sub = submissions[-1]
        if latest_sub.public_score is not None:
            print(f"   Public Score: {latest_sub.public_score:.4f}")
            if latest_sub.percentile is not None:
                print(f"   Percentile: {latest_sub.percentile:.1f}%")

    print(f"\n   Termination: {final_state.get('termination_reason', 'unknown')}")

    return final_state


# ==================== MLE-bench Workflow ====================


def create_mlebench_workflow() -> StateGraph:
    """
    Create a workflow for MLE-bench evaluation.

    This workflow skips data_download_node since MLE-bench data
    is already prepared and loaded into the state.

    The flow is:
        domain_detection → search → planner → developer (loop) →
        robustness → ensemble → submission → performance_evaluation →
        meta_evaluator → [curriculum_learning] → prompt_refinement →
        iteration_control → [refine → planner | end → reporting]

    Features:
        - WEBRL: Curriculum learning from failures (auto sub-tasks)
        - Iteration loop for refinement

    Returns:
        Compiled StateGraph
    """
    workflow = StateGraph(KaggleState)

    # Add nodes (skip data_download)
    workflow.add_node("data_format_discovery", data_format_discovery_node)  # Fallback for non-standard formats
    workflow.add_node("data_validation", data_validation_node)
    workflow.add_node("domain_detection", domain_detection_node)
    workflow.add_node("data_audit", data_audit_node)  # Fail-fast audit for audio competitions
    workflow.add_node("canonical_data_preparation", canonical_data_preparation_node)  # Canonical data contract
    workflow.add_node("search", search_agent_node)
    workflow.add_node("planner", planner_agent_node)
    workflow.add_node("developer", developer_agent_node)
    workflow.add_node("robustness", robustness_agent_node)
    workflow.add_node("ensemble", ensemble_agent_node)
    workflow.add_node("submission", submission_agent_node)
    workflow.add_node("performance_evaluation", performance_evaluation_node)
    workflow.add_node("meta_evaluator", meta_evaluator_node)
    workflow.add_node("auto_sota_search", auto_sota_search_node)  # SOTA search on stagnation
    workflow.add_node("curriculum_learning", curriculum_learning_node)  # WEBRL
    workflow.add_node("inject_curriculum", inject_subtask_guidance)  # WEBRL guidance injection
    workflow.add_node("prompt_refinement", prompt_refinement_node)
    workflow.add_node("iteration_control", iteration_control_node)
    workflow.add_node("reporting", reporting_agent_node)

    # Entry point: data_format_discovery (data already loaded but may need format discovery)
    workflow.set_entry_point("data_format_discovery")

    # Data Format Discovery → Data Validation → Domain Detection → Data Audit → Canonical → Search
    workflow.add_edge("data_format_discovery", "data_validation")
    workflow.add_edge("data_validation", "domain_detection")
    workflow.add_edge("domain_detection", "data_audit")
    workflow.add_edge("data_audit", "canonical_data_preparation")
    workflow.add_edge("canonical_data_preparation", "search")

    # Search → Planner
    workflow.add_edge("search", "planner")

    # Planner → Developer
    workflow.add_edge("planner", "developer")

    # Developer → Conditional (more components or done?)
    workflow.add_conditional_edges(
        "developer",
        route_after_developer,
        {
            "iterate": "developer",  # More components to implement
            "end": "robustness",  # All components done → validate
        },
    )

    # Robustness → Ensemble
    workflow.add_edge("robustness", "ensemble")

    # Ensemble → Submission
    workflow.add_edge("ensemble", "submission")

    # Submission → Performance Evaluation → Meta-Evaluator
    workflow.add_edge("submission", "performance_evaluation")
    workflow.add_edge("performance_evaluation", "meta_evaluator")

    # Meta-Evaluator → Conditional (WEBRL: curriculum, SOTA search, or continue?)
    workflow.add_conditional_edges(
        "meta_evaluator",
        route_after_meta_evaluator,
        {
            "sota_search": "auto_sota_search",  # Stagnation detected → search SOTA
            "curriculum": "curriculum_learning",  # WEBRL: Generate sub-tasks
            "continue": "prompt_refinement",  # Standard path
        },
    )

    # Auto SOTA Search → Curriculum Learning (with SOTA guidance)
    workflow.add_edge("auto_sota_search", "curriculum_learning")

    # Curriculum Learning → Inject Guidance → Prompt Refinement
    workflow.add_edge("curriculum_learning", "inject_curriculum")
    workflow.add_edge("inject_curriculum", "prompt_refinement")

    # Prompt Refinement → Iteration Control
    workflow.add_edge("prompt_refinement", "iteration_control")

    # Iteration Control → Conditional (refine or end?)
    workflow.add_conditional_edges(
        "iteration_control",
        route_after_iteration_control,
        {
            "refine": "planner",  # Start refinement cycle (back to planner with guidance)
            "end": "reporting",  # Goal achieved or max iterations → Report
        },
    )

    # Reporting → END
    workflow.add_edge("reporting", END)

    return workflow.compile()


# ==================== Simplified Workflow (for testing) ====================


def create_simple_workflow() -> StateGraph:
    """
    Create a simplified workflow for testing (no iterations).

    Returns:
        Compiled StateGraph
    """
    workflow = StateGraph(KaggleState)

    # Add nodes
    workflow.add_node("data_download", data_download_node)
    workflow.add_node("data_format_discovery", data_format_discovery_node)  # Fallback for non-standard formats
    workflow.add_node("data_validation", data_validation_node)
    workflow.add_node("domain_detection", domain_detection_node)
    workflow.add_node("data_audit", data_audit_node)  # Fail-fast audit for audio competitions
    workflow.add_node("canonical_data_preparation", canonical_data_preparation_node)  # Canonical data contract
    workflow.add_node("search", search_agent_node)
    workflow.add_node("planner", planner_agent_node)
    workflow.add_node("developer", developer_agent_node)

    # Linear flow
    workflow.set_entry_point("data_download")
    workflow.add_edge("data_download", "data_format_discovery")
    workflow.add_edge("data_format_discovery", "data_validation")
    workflow.add_edge("data_validation", "domain_detection")
    workflow.add_edge("domain_detection", "data_audit")
    workflow.add_edge("data_audit", "canonical_data_preparation")
    workflow.add_edge("canonical_data_preparation", "search")
    workflow.add_edge("search", "planner")
    workflow.add_edge("planner", "developer")
    workflow.add_edge("developer", END)

    return workflow.compile()


def run_simple_workflow(
    competition_name: str,
    working_dir: str,
    competition_info: dict[str, Any],
) -> KaggleState:
    """
    Run simplified workflow (one pass, no iterations).

    Args:
        competition_name: Competition name
        working_dir: Working directory
        competition_info: Competition metadata

    Returns:
        Final state
    """
    print("=" * 70)
    print(f"SIMPLE WORKFLOW: {competition_name}")
    print("=" * 70)

    # Create initial state
    state = create_initial_state(competition_name, working_dir)

    from .core.state import CompetitionInfo

    state["competition_info"] = CompetitionInfo(**competition_info)

    # Run workflow
    workflow = create_simple_workflow()
    final_state = workflow.invoke(state)

    print("\n" + "=" * 70)
    print(" WORKFLOW COMPLETE")
    print("=" * 70)

    return final_state
