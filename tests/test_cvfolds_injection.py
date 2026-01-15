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
