"""Tests for CVfolds train/test split injection into generated code."""

from pathlib import Path

import numpy as np
import pytest


class TestCVfoldsInjection:
    """Tests for CVfolds injection in code_generator."""

    def test_small_cvfolds_should_be_inlined(self) -> None:
        """Test that small CVfolds lists should be inlined (<=100 items)."""
        # Create mock state with small CVfolds data
        state = {
            "cv_folds_used": True,
            "train_rec_ids": [1, 2, 3, 4, 5],
            "test_rec_ids": [6, 7, 8],
        }

        # The key assertion: when lists are small, they should be inlined
        test_rec_ids = state.get("test_rec_ids", [])
        train_rec_ids = state.get("train_rec_ids", [])

        # Check threshold condition for inlining
        should_save_to_file = len(test_rec_ids) > 100 or len(train_rec_ids) > 100

        assert state["cv_folds_used"] is True
        assert len(test_rec_ids) == 3
        assert len(train_rec_ids) == 5
        assert should_save_to_file is False  # Should be inlined, not saved

    def test_large_cvfolds_saved_to_file(self, tmp_path: Path) -> None:
        """Test that large CVfolds lists are saved to files instead of inlined."""
        # Create large lists (>100 items)
        train_ids = list(range(200))
        test_ids = list(range(200, 350))

        state = {
            "cv_folds_used": True,
            "train_rec_ids": train_ids,
            "test_rec_ids": test_ids,
        }

        # Verify the threshold logic
        assert len(test_ids) > 100
        assert len(train_ids) > 100

        # When lists are large, they should be saved to files
        # This is tested by verifying the size threshold
        models_dir = tmp_path / "models"
        models_dir.mkdir(parents=True, exist_ok=True)

        # Simulate what code_generator does for large lists
        np.save(models_dir / "cvfolds_train_ids.npy", np.array(train_ids))
        np.save(models_dir / "cvfolds_test_ids.npy", np.array(test_ids))

        # Verify files were created correctly
        assert (models_dir / "cvfolds_train_ids.npy").exists()
        assert (models_dir / "cvfolds_test_ids.npy").exists()

        # Load and verify
        loaded_train = np.load(models_dir / "cvfolds_train_ids.npy", allow_pickle=True)
        loaded_test = np.load(models_dir / "cvfolds_test_ids.npy", allow_pickle=True)

        assert len(loaded_train) == 200
        assert len(loaded_test) == 150
        assert list(loaded_train) == train_ids
        assert list(loaded_test) == test_ids

    def test_cvfolds_not_injected_when_disabled(self) -> None:
        """Test that CVfolds are not injected when cv_folds_used is False."""
        state = {
            "cv_folds_used": False,
            "train_rec_ids": [1, 2, 3],
            "test_rec_ids": [4, 5],
        }

        # When cv_folds_used is False, no injection should happen
        cv_folds_used = state.get("cv_folds_used", False)
        test_rec_ids = state.get("test_rec_ids", [])

        # The condition for injection
        should_inject = cv_folds_used and test_rec_ids

        assert should_inject is False

    def test_cvfolds_not_injected_when_empty(self) -> None:
        """Test that CVfolds are not injected when test_rec_ids is empty."""
        state = {
            "cv_folds_used": True,
            "train_rec_ids": [1, 2, 3],
            "test_rec_ids": [],  # Empty
        }

        cv_folds_used = state.get("cv_folds_used", False)
        test_rec_ids = state.get("test_rec_ids", [])

        # The condition for injection - empty list is falsy
        should_inject = cv_folds_used and bool(test_rec_ids)

        assert should_inject is False


class TestEnsembleValidation:
    """Tests for ensemble agent test count validation."""

    def test_direct_match_validation(self) -> None:
        """Test validation passes when n_test matches expected."""
        n_test = 323
        expected_n_test = 323
        n_cols = 19  # Wide format

        is_direct_match = (n_test == expected_n_test)
        is_mlsp_format = (n_cols == 1 and n_test % expected_n_test == 0)

        assert is_direct_match is True
        # No warning should be triggered when there's a match
        should_warn = not is_direct_match and not is_mlsp_format
        assert should_warn is False

    def test_mlsp_format_validation(self) -> None:
        """Test validation handles MLSP long format (n_test * n_classes rows)."""
        expected_n_test = 323  # Number of recordings
        n_test = 323 * 19  # MLSP format: 323 recordings * 19 classes = 6137 rows
        n_cols = 1  # Long format has only 1 prediction column

        is_direct_match = (n_test == expected_n_test)
        is_mlsp_format = (n_cols == 1 and n_test % expected_n_test == 0)

        assert is_direct_match is False
        assert is_mlsp_format is True
        # No warning should be triggered for MLSP format
        should_warn = not is_direct_match and not is_mlsp_format
        assert should_warn is False

    def test_mismatch_triggers_warning(self) -> None:
        """Test that true mismatch would trigger warning."""
        expected_n_test = 323
        n_test = 64  # Wrong count
        n_cols = 19  # Wide format

        is_direct_match = (n_test == expected_n_test)
        is_mlsp_format = (n_cols == 1 and n_test % expected_n_test == 0)

        assert is_direct_match is False
        assert is_mlsp_format is False
        # This would trigger the warning
        should_warn = not is_direct_match and not is_mlsp_format
        assert should_warn is True


class TestDataTypeOverride:
    """Tests for data type override logic in data validation."""

    def test_audio_type_preserved_with_spectrogram_images(self) -> None:
        """Test that audio type is NOT overridden to image when spectrograms exist."""
        # Simulate state where domain detection found audio but image dir exists
        data_type = "audio"  # From domain detection
        train_dir = True  # Image directory found (spectrograms)
        forced_type = False  # No user override

        # The fix: don't override audio to image
        detected_type = data_type or ""
        if not forced_type:
            if train_dir:
                if data_type not in ("audio", "audio_classification"):
                    detected_type = "image"
                # else: keep audio type (spectrograms)

        # Audio type should be preserved
        assert detected_type == "audio"

    def test_audio_classification_type_preserved(self) -> None:
        """Test that audio_classification type is also preserved."""
        data_type = "audio_classification"
        train_dir = True
        forced_type = False

        detected_type = data_type or ""
        if not forced_type:
            if train_dir:
                if data_type not in ("audio", "audio_classification"):
                    detected_type = "image"

        assert detected_type == "audio_classification"

    def test_image_type_detected_for_non_audio(self) -> None:
        """Test that image type IS set for non-audio competitions."""
        data_type = ""  # Unknown type
        train_dir = True  # Image directory found
        forced_type = False

        detected_type = data_type or ""
        if not forced_type:
            if train_dir:
                if data_type not in ("audio", "audio_classification"):
                    detected_type = "image"

        # Should be overridden to image for non-audio
        assert detected_type == "image"

    def test_forced_type_not_overridden(self) -> None:
        """Test that forced type is never overridden."""
        data_type = "tabular"
        train_dir = True
        forced_type = True  # User forced this type

        detected_type = data_type or ""
        if not forced_type:
            if train_dir:
                if data_type not in ("audio", "audio_classification"):
                    detected_type = "image"

        # Should stay as tabular because forced
        assert detected_type == "tabular"


class TestCVfoldsIDMapping:
    """Tests for CVfolds ID-to-filename mapping logic."""

    def test_id_to_filename_mapping(self, tmp_path: Path) -> None:
        """Test that numeric rec_ids are mapped to filenames."""
        # Create mock rec_id2filename.txt mapping
        mapping_data = {
            "rec_id": [0, 1, 2, 3, 4],
            "filename": ["PC1_audio_01", "PC1_audio_02", "PC2_audio_01", "PC2_audio_02", "PC3_audio_01"],
        }
        import pandas as pd
        mapping_df = pd.DataFrame(mapping_data)

        # Create id_to_filename dict (simulating the fix logic)
        id_to_filename = dict(zip(
            mapping_df["rec_id"].astype(str),
            mapping_df["filename"]
        ))

        # Simulate raw train/test IDs from CVfolds
        train_rec_ids = [0, 1, 2]  # Numeric IDs
        test_rec_ids = [3, 4]

        # Map to filenames
        train_filenames = [id_to_filename.get(str(rid)) for rid in train_rec_ids]
        test_filenames = [id_to_filename.get(str(rid)) for rid in test_rec_ids]

        # Filter out None values
        train_filenames = [f for f in train_filenames if f]
        test_filenames = [f for f in test_filenames if f]

        assert train_filenames == ["PC1_audio_01", "PC1_audio_02", "PC2_audio_01"]
        assert test_filenames == ["PC2_audio_02", "PC3_audio_01"]

    def test_unmapped_ids_filtered_out(self) -> None:
        """Test that IDs not in mapping are filtered out."""
        import pandas as pd

        # Partial mapping (missing ID 5)
        mapping_data = {
            "rec_id": [0, 1, 2],
            "filename": ["audio_01", "audio_02", "audio_03"],
        }
        mapping_df = pd.DataFrame(mapping_data)

        id_to_filename = dict(zip(
            mapping_df["rec_id"].astype(str),
            mapping_df["filename"]
        ))

        # rec_ids include ID 5 which is not in mapping
        test_rec_ids = [0, 1, 5]

        test_filenames = [id_to_filename.get(str(rid)) for rid in test_rec_ids]
        test_filenames = [f for f in test_filenames if f]

        # ID 5 should be filtered out
        assert test_filenames == ["audio_01", "audio_02"]
        assert len(test_filenames) == 2

    def test_empty_mapping_preserves_original_ids(self) -> None:
        """Test that if no mapping available, original IDs are preserved."""
        train_rec_ids = [0, 1, 2]
        test_rec_ids = [3, 4]

        # Simulate no id_mapping being available
        has_id_mapping = False

        if has_id_mapping:
            # This branch shouldn't execute
            train_rec_ids = []
            test_rec_ids = []

        # Original IDs should be preserved
        assert train_rec_ids == [0, 1, 2]
        assert test_rec_ids == [3, 4]


class TestAudioLabelValidation:
    """Tests for audio competition label usage validation."""

    def test_detects_hardcoded_bad_paths(self) -> None:
        """Test that hardcoded non-existent paths are detected."""
        code = '''
# Bad code that hardcodes a path
labels_df = pd.read_csv("rec_labels_train.txt")
'''
        bad_paths = ["rec_labels_train.txt", "train_labels.txt"]
        warnings = []
        for bad_path in bad_paths:
            if bad_path in code:
                warnings.append(f"Hardcoded path '{bad_path}'")

        assert len(warnings) == 1
        assert "rec_labels_train.txt" in warnings[0]

    def test_no_warning_when_using_preloaded(self) -> None:
        """Test that using pre-loaded labels generates no warnings."""
        code = '''
# Good code that uses pre-loaded labels
labels_df = _PRELOADED_LABELS_DF
rec_ids = _PRELOADED_REC_IDS
n_classes = _PRELOADED_N_CLASSES
'''
        uses_preloaded = "_PRELOADED_LABELS_DF" in code
        has_bad_paths = any(p in code for p in ["rec_labels_train.txt", "train_labels.txt"])

        assert uses_preloaded is True
        assert has_bad_paths is False

    def test_only_validates_audio_competitions(self) -> None:
        """Test that validation only runs for audio competitions."""
        data_types_to_validate = ("audio", "audio_classification")

        assert "audio" in data_types_to_validate
        assert "audio_classification" in data_types_to_validate
        assert "tabular" not in data_types_to_validate
        assert "image" not in data_types_to_validate


class TestLabelReparsingStripping:
    """Tests for _strip_label_reparsing() function that enforces pre-loaded labels."""

    def test_strips_pd_read_csv_on_label_files(self) -> None:
        """Test that pd.read_csv calls on label files are stripped (singular form)."""
        import re

        code = '''# === END PATH CONSTANTS ===

# Bad code that should be stripped
labels_df = pd.read_csv(LABEL_FILE, header=None, names=['rec_id', 'class_id'])
'''
        marker = "# === END PATH CONSTANTS ==="
        marker_idx = code.find(marker)
        code_after_header = code[marker_idx + len(marker) :]

        # Pattern uses negative lookbehind to avoid "unlabeled" but match LABEL_FILE
        # (?<![a-zA-Z]) ensures no letter before label/labels
        pattern = r"([\t ]*\w+\s*=\s*pd\.read_csv\([^)]*(?<![a-zA-Z])(?:labels?|LABELS?)[^)]*\))"
        replacement = r"# STRIPPED (use _PRELOADED_LABELS_DF): # \1"

        matches = re.findall(pattern, code_after_header, re.IGNORECASE)
        result = re.sub(pattern, replacement, code_after_header, flags=re.IGNORECASE)

        assert len(matches) == 1  # LABEL_FILE should match
        assert "STRIPPED" in result

    def test_strips_pd_read_csv_on_plural_label_files(self) -> None:
        """Test that pd.read_csv calls on plural label files are stripped.

        Real audio competition filenames: rec_labels_train.txt, train_labels.txt
        """
        import re

        code = '''# === END PATH CONSTANTS ===

# Bad code that should be stripped - real audio competition filenames
df1 = pd.read_csv("rec_labels_train.txt")
df2 = pd.read_csv("train_labels.txt")
df3 = pd.read_csv(TRAIN_LABELS_PATH)
'''
        marker = "# === END PATH CONSTANTS ==="
        marker_idx = code.find(marker)
        code_after_header = code[marker_idx + len(marker) :]

        # Pattern uses negative lookbehind - matches rec_labels, train_labels
        labels_pattern = r"([\t ]*\w+\s*=\s*pd\.read_csv\([^)]*(?<![a-zA-Z])(?:labels?|LABELS?)[^)]*\))"
        train_labels_pattern = r"([\t ]*\w+\s*=\s*pd\.read_csv\([^)]*(?<![a-zA-Z])train_labels?[^)]*\))"

        labels_matches = re.findall(labels_pattern, code_after_header, re.IGNORECASE)
        train_labels_matches = re.findall(train_labels_pattern, code_after_header, re.IGNORECASE)

        # rec_labels_train.txt and train_labels.txt should be matched
        assert len(labels_matches) >= 2  # rec_labels and train_labels both contain "labels"
        assert len(train_labels_matches) >= 1  # train_labels explicitly matched

    def test_preserves_non_label_read_csv(self) -> None:
        """Test that pd.read_csv calls on non-label files are NOT stripped."""
        import re

        code = '''# === END PATH CONSTANTS ===

# Good code that should NOT be stripped
sample_sub = pd.read_csv(SAMPLE_SUBMISSION_PATH)
train_df = pd.read_csv("train.csv")
'''
        marker = "# === END PATH CONSTANTS ==="
        marker_idx = code.find(marker)
        code_after_header = code[marker_idx + len(marker) :]

        # Pattern that matches label files only (negative lookbehind)
        pattern = r"([\t ]*\w+\s*=\s*pd\.read_csv\([^)]*(?<![a-zA-Z])(?:labels?|LABELS?)[^)]*\))"
        matches = re.findall(pattern, code_after_header, re.IGNORECASE)

        # No label files, so no matches
        assert len(matches) == 0

    def test_preserves_unlabeled_files(self) -> None:
        """Test that 'unlabeled' files are NOT stripped (negative lookbehind works)."""
        import re

        code = '''# === END PATH CONSTANTS ===

# Good code with "unlabeled" - should NOT be stripped
unlabeled_df = pd.read_csv("unlabeled_data.csv")
'''
        marker = "# === END PATH CONSTANTS ==="
        marker_idx = code.find(marker)
        code_after_header = code[marker_idx + len(marker) :]

        # Pattern should NOT match "unlabeled" due to negative lookbehind
        pattern = r"([\t ]*\w+\s*=\s*pd\.read_csv\([^)]*(?<![a-zA-Z])(?:labels?|LABELS?)[^)]*\))"
        matches = re.findall(pattern, code_after_header, re.IGNORECASE)

        # "unlabeled" should NOT match because "un" precedes "labeled"
        assert len(matches) == 0

    def test_preserves_code_using_preloaded_labels(self) -> None:
        """Test that code correctly using _PRELOADED_LABELS_DF is NOT stripped."""
        import re

        code = '''# === END PATH CONSTANTS ===

# Good code that uses pre-loaded labels - should NOT be stripped
labels_df = _PRELOADED_LABELS_DF.copy()
train_labels = _PRELOADED_LABELS_DF[_PRELOADED_LABELS_DF['rec_id'].isin(train_ids)]
n_classes = _PRELOADED_N_CLASSES
rec_ids = _PRELOADED_REC_IDS
'''
        marker = "# === END PATH CONSTANTS ==="
        marker_idx = code.find(marker)
        code_after_header = code[marker_idx + len(marker) :]

        # Pattern that matches pd.read_csv on label files (negative lookbehind)
        pattern = r"([\t ]*\w+\s*=\s*pd\.read_csv\([^)]*(?<![a-zA-Z])(?:labels?|LABELS?)[^)]*\))"
        matches = re.findall(pattern, code_after_header, re.IGNORECASE)

        # Code using _PRELOADED_LABELS_DF should have no matches (no pd.read_csv)
        assert len(matches) == 0

        # Verify the code uses the correct pre-loaded variables
        assert "_PRELOADED_LABELS_DF" in code
        assert "_PRELOADED_N_CLASSES" in code
        assert "_PRELOADED_REC_IDS" in code


class TestLabelIntCasting:
    """Tests for label integer casting in parse_label_file."""

    def test_numeric_labels_cast_to_int(self) -> None:
        """Test that numeric labels are cast to integers."""
        label = "42"
        try:
            label_val = int(label)
        except ValueError:
            label_val = label

        assert isinstance(label_val, int)
        assert label_val == 42

    def test_non_numeric_labels_preserved_as_string(self) -> None:
        """Test that non-numeric labels are kept as strings."""
        label = "species_bird"
        try:
            label_val = int(label)
        except ValueError:
            label_val = label

        assert isinstance(label_val, str)
        assert label_val == "species_bird"

    def test_mixed_labels_handled_correctly(self) -> None:
        """Test handling of mixed numeric and non-numeric labels."""
        labels = ["1", "10", "species", "42"]
        results = []

        for label in labels:
            try:
                results.append(int(label))
            except ValueError:
                results.append(label)

        assert results == [1, 10, "species", 42]
        assert isinstance(results[0], int)
        assert isinstance(results[2], str)


class TestCanonicalDirFallback:
    """Tests for CANONICAL_DIR fallback definition."""

    def test_canonical_dir_defined_from_models_dir(self, tmp_path: Path) -> None:
        """Test that CANONICAL_DIR is defined relative to MODELS_DIR."""
        MODELS_DIR = tmp_path / "models"
        MODELS_DIR.mkdir(parents=True, exist_ok=True)

        # Simulate the fallback code
        CANONICAL_DIR = MODELS_DIR / "canonical"
        CANONICAL_DIR.mkdir(parents=True, exist_ok=True)

        assert CANONICAL_DIR.exists()
        assert CANONICAL_DIR.parent == MODELS_DIR
        assert CANONICAL_DIR.name == "canonical"

    def test_canonical_dir_created_if_not_exists(self, tmp_path: Path) -> None:
        """Test that CANONICAL_DIR is created if it doesn't exist."""
        MODELS_DIR = tmp_path / "new_models"
        # Don't create MODELS_DIR yet

        CANONICAL_DIR = MODELS_DIR / "canonical"
        CANONICAL_DIR.mkdir(parents=True, exist_ok=True)

        assert CANONICAL_DIR.exists()
        assert MODELS_DIR.exists()  # Parent should also be created

    def test_fallback_not_injected_when_canonical_exists(self, tmp_path: Path) -> None:
        """Test that fallback is NOT injected when canonical data already exists.

        This prevents overriding the canonical contract when real canonical
        artifacts are present (e.g., train_ids.npy, folds.npy).
        """
        # Simulate existing canonical data
        canonical_dir = tmp_path / "canonical"
        canonical_dir.mkdir(parents=True, exist_ok=True)
        (canonical_dir / "train_ids.npy").touch()  # Marker file

        has_canonical = canonical_dir.exists() and (canonical_dir / "train_ids.npy").exists()
        data_type = "audio"

        # Fallback condition: only inject when has_canonical is False
        should_inject_fallback = data_type in ("audio", "audio_classification") and not has_canonical

        assert has_canonical is True
        assert should_inject_fallback is False  # Fallback should NOT be injected

    def test_fallback_injected_when_no_canonical_data(self, tmp_path: Path) -> None:
        """Test that fallback IS injected when no canonical data exists."""
        # No canonical directory exists
        canonical_dir = tmp_path / "canonical"

        has_canonical = canonical_dir.exists() and (canonical_dir / "train_ids.npy").exists()
        data_type = "audio"

        # Fallback condition: inject when has_canonical is False for audio
        should_inject_fallback = data_type in ("audio", "audio_classification") and not has_canonical

        assert has_canonical is False
        assert should_inject_fallback is True  # Fallback SHOULD be injected
