"""Tests for precomputed features detection."""

from pathlib import Path

import pandas as pd
import pytest

from kaggle_agents.utils.precomputed_features import (
    PrecomputedFeaturesInfo,
    detect_precomputed_features,
    generate_feature_loading_code,
    load_precomputed_features,
)


class TestDetectPrecomputedFeatures:
    """Tests for detect_precomputed_features function."""

    def test_detect_histogram_features(self, tmp_path: Path) -> None:
        """Test detection of histogram features file."""
        # Create a histogram features file
        df = pd.DataFrame({
            "rec_id": [1, 2, 3],
            "feat_0": [0.1, 0.2, 0.3],
            "feat_1": [0.4, 0.5, 0.6],
        })
        (tmp_path / "histogram_features.txt").write_text(
            df.to_csv(index=False)
        )

        result = detect_precomputed_features(tmp_path)

        assert result.has_features()
        assert "histogram" in result.features_found
        assert result.feature_shapes.get("histogram") == (3, 3)

    def test_detect_multiple_feature_files(self, tmp_path: Path) -> None:
        """Test detection of multiple feature files."""
        # Create multiple feature files
        df1 = pd.DataFrame({
            "rec_id": range(10),
            "h_0": [0.1] * 10,
            "h_1": [0.2] * 10,
        })
        (tmp_path / "histogram_features.csv").write_text(
            df1.to_csv(index=False)
        )

        df2 = pd.DataFrame({
            "rec_id": range(10),
            "lat": [1.0] * 10,
            "lon": [2.0] * 10,
        })
        (tmp_path / "location_features.txt").write_text(
            df2.to_csv(index=False)
        )

        result = detect_precomputed_features(tmp_path)

        assert result.has_features()
        assert len(result.features_found) == 2
        assert "histogram" in result.features_found
        assert "location" in result.features_found

    def test_detect_cv_folds(self, tmp_path: Path) -> None:
        """Test detection of CV folds file."""
        df = pd.DataFrame({
            "rec_id": [1, 2, 3, 4, 5],
            "fold": [0, 1, 0, 1, 0],
        })
        (tmp_path / "CVfolds_2.txt").write_text(df.to_csv(index=False))

        result = detect_precomputed_features(tmp_path)

        assert result.has_features()
        assert "cv_folds" in result.features_found

    def test_detect_id_mapping(self, tmp_path: Path) -> None:
        """Test detection of ID-to-filename mapping file."""
        df = pd.DataFrame({
            "rec_id": [1, 2, 3],
            "filename": ["file1.wav", "file2.wav", "file3.wav"],
        })
        (tmp_path / "rec_id2filename.txt").write_text(df.to_csv(index=False))

        result = detect_precomputed_features(tmp_path)

        assert result.has_features()
        assert "id_mapping" in result.features_found

    def test_no_features_found(self, tmp_path: Path) -> None:
        """Test when no feature files are found."""
        # Create unrelated files
        (tmp_path / "train.csv").write_text("id,target\n1,0\n2,1")
        (tmp_path / "test.csv").write_text("id\n1\n2")

        result = detect_precomputed_features(tmp_path)

        assert not result.has_features()
        assert len(result.features_found) == 0

    def test_nonexistent_directory(self) -> None:
        """Test handling of nonexistent directory."""
        result = detect_precomputed_features(Path("/nonexistent/path"))

        assert not result.has_features()
        assert len(result.warnings) > 0

    def test_recursive_search(self, tmp_path: Path) -> None:
        """Test recursive search in subdirectories."""
        # Create nested structure
        subdir = tmp_path / "essential_data"
        subdir.mkdir()

        df = pd.DataFrame({
            "rec_id": range(5),
            "feat": [0.1] * 5,
        })
        (subdir / "histogram_features.txt").write_text(df.to_csv(index=False))

        result = detect_precomputed_features(tmp_path, recursive=True)

        assert result.has_features()
        assert "histogram" in result.features_found

    def test_non_recursive_search(self, tmp_path: Path) -> None:
        """Test non-recursive search."""
        subdir = tmp_path / "subdir"
        subdir.mkdir()

        df = pd.DataFrame({"rec_id": [1], "feat": [0.1]})
        (subdir / "histogram_features.txt").write_text(df.to_csv(index=False))

        result = detect_precomputed_features(tmp_path, recursive=False)

        # Should not find the file in subdirectory
        assert "histogram" not in result.features_found


class TestLoadPrecomputedFeatures:
    """Tests for load_precomputed_features function."""

    def test_load_single_feature_file(self, tmp_path: Path) -> None:
        """Test loading a single feature file."""
        df = pd.DataFrame({
            "rec_id": [1, 2, 3],
            "feat_0": [0.1, 0.2, 0.3],
            "feat_1": [0.4, 0.5, 0.6],
        })
        (tmp_path / "histogram_features.csv").write_text(df.to_csv(index=False))

        features_info = detect_precomputed_features(tmp_path)
        loaded_df = load_precomputed_features(features_info, feature_types=["histogram"])

        assert loaded_df is not None
        assert len(loaded_df) == 3
        # Columns should be prefixed with feature type
        assert any("histogram" in col for col in loaded_df.columns)

    def test_load_no_features(self) -> None:
        """Test loading when no features available."""
        features_info = PrecomputedFeaturesInfo()
        loaded_df = load_precomputed_features(features_info)

        assert loaded_df is None


class TestGenerateFeatureLoadingCode:
    """Tests for generate_feature_loading_code function."""

    def test_generate_code_with_csv_features(self, tmp_path: Path) -> None:
        """Test code generation for CSV features."""
        df = pd.DataFrame({"rec_id": [1], "feat": [0.1]})
        (tmp_path / "histogram_features.csv").write_text(df.to_csv(index=False))

        features_info = detect_precomputed_features(tmp_path)
        code = generate_feature_loading_code(features_info)

        assert "histogram" in code.lower()
        assert "pd.read_csv" in code
        assert "import pandas as pd" in code

    def test_generate_code_with_txt_features(self, tmp_path: Path) -> None:
        """Test code generation for .txt features (should use read_csv)."""
        df = pd.DataFrame({"rec_id": [1], "feat": [0.1]})
        (tmp_path / "histogram_features.txt").write_text(df.to_csv(index=False))

        features_info = detect_precomputed_features(tmp_path)
        code = generate_feature_loading_code(features_info)

        assert "pd.read_csv" in code

    def test_generate_code_with_npy_features(self, tmp_path: Path) -> None:
        """Test code generation for .npy features."""
        import numpy as np
        arr = np.array([[0.1, 0.2], [0.3, 0.4]])
        np.save(tmp_path / "histogram_features.npy", arr)

        features_info = detect_precomputed_features(tmp_path)
        code = generate_feature_loading_code(features_info)

        assert "np.load" in code
        assert "import numpy as np" in code
        # Should NOT have pd.read_csv for .npy
        assert "pd.read_csv" not in code or "histogram" not in code.split("pd.read_csv")[0].split("\n")[-1]

    def test_generate_code_with_parquet_features(self, tmp_path: Path) -> None:
        """Test code generation for .parquet features."""
        pytest.importorskip("pyarrow", reason="pyarrow required for parquet tests")

        df = pd.DataFrame({"rec_id": [1, 2], "feat": [0.1, 0.2]})
        df.to_parquet(tmp_path / "histogram_features.parquet")

        features_info = detect_precomputed_features(tmp_path)
        code = generate_feature_loading_code(features_info)

        assert "pd.read_parquet" in code
        assert "import pandas as pd" in code

    def test_generate_code_with_mixed_formats(self, tmp_path: Path) -> None:
        """Test code generation with mixed file formats."""
        import numpy as np

        # CSV feature
        df = pd.DataFrame({"rec_id": [1], "feat": [0.1]})
        (tmp_path / "histogram_features.csv").write_text(df.to_csv(index=False))

        # NPY feature
        arr = np.array([[0.1, 0.2]])
        np.save(tmp_path / "spectrogram_features.npy", arr)

        features_info = detect_precomputed_features(tmp_path)
        code = generate_feature_loading_code(features_info)

        # Should have both imports
        assert "import pandas as pd" in code
        assert "import numpy as np" in code
        # Should have appropriate loaders for each
        assert "pd.read_csv" in code
        assert "np.load" in code

    def test_generate_code_no_features(self) -> None:
        """Test code generation when no features found."""
        features_info = PrecomputedFeaturesInfo()
        code = generate_feature_loading_code(features_info)

        assert "No precomputed features" in code


class TestPrecomputedFeaturesInfo:
    """Tests for PrecomputedFeaturesInfo dataclass."""

    def test_has_features_true(self) -> None:
        """Test has_features returns True when features exist."""
        info = PrecomputedFeaturesInfo(
            features_found={"histogram": Path("/path/to/file.txt")}
        )
        assert info.has_features()

    def test_has_features_false(self) -> None:
        """Test has_features returns False when no features."""
        info = PrecomputedFeaturesInfo()
        assert not info.has_features()

    def test_to_dict(self) -> None:
        """Test conversion to dictionary."""
        info = PrecomputedFeaturesInfo(
            features_found={"histogram": Path("/path/to/file.txt")},
            feature_shapes={"histogram": (100, 50)},
            total_features=50,
        )
        d = info.to_dict()

        assert d["features_found"]["histogram"] == "/path/to/file.txt"
        assert d["feature_shapes"]["histogram"] == (100, 50)
        assert d["total_features"] == 50
