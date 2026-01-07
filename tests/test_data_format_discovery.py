"""
Tests for data format discovery tool.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from kaggle_agents.tools.data_format_discovery import (
    DataFormatDiscoverer,
    detect_traditional_format,
    get_loading_code_for_developer,
)


class TestDetectTraditionalFormat:
    """Tests for detect_traditional_format function."""

    def test_detects_standard_csv_format(self, tmp_path: Path) -> None:
        """Should detect train.csv and test.csv in root directory."""
        # Create standard CSV files
        (tmp_path / "train.csv").write_text("id,target\n1,0\n2,1")
        (tmp_path / "test.csv").write_text("id\n3\n4")

        result = detect_traditional_format(tmp_path)

        assert result is not None
        assert "train" in result
        assert "test" in result
        assert result["train"].endswith("train.csv")
        assert result["test"].endswith("test.csv")

    def test_detects_train_labels_csv(self, tmp_path: Path) -> None:
        """Should detect train_labels.csv variant."""
        (tmp_path / "train_labels.csv").write_text("id,target\n1,0\n2,1")
        (tmp_path / "test.csv").write_text("id\n3\n4")

        result = detect_traditional_format(tmp_path)

        assert result is not None
        assert "train" in result

    def test_detects_image_directories_with_label_csv(self, tmp_path: Path) -> None:
        """Should detect train/test directories with images when label CSV exists."""
        train_dir = tmp_path / "train"
        test_dir = tmp_path / "test"
        train_dir.mkdir()
        test_dir.mkdir()

        # Create sample image files
        (train_dir / "img1.jpg").write_bytes(b"fake image data")
        (train_dir / "img2.jpg").write_bytes(b"fake image data")
        (test_dir / "img3.jpg").write_bytes(b"fake image data")

        # Create label CSV - required for image competitions
        (tmp_path / "train.csv").write_text("id,label\nimg1,0\nimg2,1")

        result = detect_traditional_format(tmp_path)

        assert result is not None
        assert "train" in result
        assert "test" in result

    def test_returns_none_for_image_dirs_without_label_csv(self, tmp_path: Path) -> None:
        """Should return None when image dirs exist but no label CSV.

        This is the MLSP-2013-Birds scenario: train/test dirs have audio/images
        but labels are in non-standard .txt files.
        """
        train_dir = tmp_path / "train"
        test_dir = tmp_path / "test"
        train_dir.mkdir()
        test_dir.mkdir()

        # Create sample audio files (like MLSP-2013-Birds)
        (train_dir / "audio1.wav").write_bytes(b"fake audio data")
        (train_dir / "audio2.wav").write_bytes(b"fake audio data")
        (test_dir / "audio3.wav").write_bytes(b"fake audio data")

        # No label CSV - labels are in non-standard format
        # (e.g., essential_data/rec_labels_test_hidden.txt)

        result = detect_traditional_format(tmp_path)

        # Should return None to trigger LLM-based discovery
        assert result is None

    def test_returns_none_for_nonstandard_format(self, tmp_path: Path) -> None:
        """Should return None when no standard format is found."""
        # Create non-standard files
        (tmp_path / "rec_labels.txt").write_text("0,1,2\n1,3,4")
        (tmp_path / "CVfolds.txt").write_text("0,0\n1,1")

        result = detect_traditional_format(tmp_path)

        assert result is None

    def test_handles_empty_directory(self, tmp_path: Path) -> None:
        """Should return None for empty directory."""
        result = detect_traditional_format(tmp_path)
        assert result is None


class TestDataFormatDiscoverer:
    """Tests for DataFormatDiscoverer class."""

    def test_list_data_files(self, tmp_path: Path) -> None:
        """Should list files with metadata."""
        # Create test files
        (tmp_path / "labels.txt").write_text("id,label\n1,0\n2,1")
        (tmp_path / "mapping.csv").write_text("id,filename\n1,a.wav\n2,b.wav")

        subdir = tmp_path / "audio"
        subdir.mkdir()
        (subdir / "a.wav").write_bytes(b"audio data")

        discoverer = DataFormatDiscoverer()
        files = discoverer.list_data_files(tmp_path)

        assert len(files) >= 2
        assert any(f["name"] == "labels.txt" for f in files)
        assert any(f["name"] == "mapping.csv" for f in files)

    def test_list_data_files_includes_sample_content(self, tmp_path: Path) -> None:
        """Should include sample content for text files."""
        content = "line1\nline2\nline3"
        (tmp_path / "data.txt").write_text(content)

        discoverer = DataFormatDiscoverer()
        files = discoverer.list_data_files(tmp_path)

        txt_file = next(f for f in files if f["name"] == "data.txt")
        assert "sample_content" in txt_file
        assert "line1" in txt_file["sample_content"]

    def test_list_data_files_handles_empty_dir(self, tmp_path: Path) -> None:
        """Should return empty list for empty directory."""
        discoverer = DataFormatDiscoverer()
        files = discoverer.list_data_files(tmp_path)
        assert files == []

    def test_extract_json_from_response(self) -> None:
        """Should extract JSON from LLM response."""
        discoverer = DataFormatDiscoverer()

        # Test with markdown code block
        response = """Here's the analysis:
```json
{
    "format_type": "txt",
    "id_column": "rec_id",
    "target_column": "species"
}
```
"""
        result = discoverer._extract_json_from_response(response)
        assert result["format_type"] == "txt"
        assert result["id_column"] == "rec_id"

    def test_extract_json_handles_plain_json(self) -> None:
        """Should extract plain JSON without code block."""
        discoverer = DataFormatDiscoverer()

        response = '{"format_type": "csv", "id_column": "id"}'
        result = discoverer._extract_json_from_response(response)
        assert result["format_type"] == "csv"

    def test_extract_json_returns_empty_on_invalid(self) -> None:
        """Should return empty dict for invalid JSON."""
        discoverer = DataFormatDiscoverer()

        response = "This is not JSON at all"
        result = discoverer._extract_json_from_response(response)
        assert result == {}

    @patch("kaggle_agents.tools.data_format_discovery.requests.get")
    def test_fetch_data_page_handles_errors(self, mock_get: MagicMock) -> None:
        """Should handle network errors gracefully."""
        mock_get.side_effect = Exception("Network error")

        discoverer = DataFormatDiscoverer()
        result = discoverer.fetch_data_page("test-competition")

        assert result == ""

    def test_generate_parsing_instructions_validates_response(self) -> None:
        """Should ensure required fields in parsing info."""
        discoverer = DataFormatDiscoverer()

        # Mock LLM that returns incomplete response
        mock_llm = MagicMock()
        mock_llm.invoke.return_value = MagicMock(content='{"notes": "test"}')

        context = {
            "competition": "test",
            "data_page_content": "",
            "file_listing": [],
            "description": "",
            "sota_loading_code": [],
        }

        result = discoverer.generate_parsing_instructions(mock_llm, context)

        # Should have required fields with defaults
        assert "format_type" in result
        assert "id_column" in result
        assert "target_column" in result


class TestGetLoadingCodeForDeveloper:
    """Tests for get_loading_code_for_developer function.

    NOTE: This function intentionally does NOT execute code.
    The loading code is passed to the developer agent where it
    will be executed through the sandboxed CodeExecutor.
    """

    def test_returns_loading_code_from_parsing_info(self) -> None:
        """Should return the loading code without modification."""
        loading_code = """
import pandas as pd
train_df = pd.read_csv('train.csv')
"""
        parsing_info = {
            "loading_code": loading_code,
            "format_type": "csv",
        }

        result = get_loading_code_for_developer(parsing_info)

        assert result == loading_code

    def test_returns_empty_string_when_no_code(self) -> None:
        """Should return empty string if no loading code provided."""
        parsing_info = {"format_type": "custom"}

        result = get_loading_code_for_developer(parsing_info)

        assert result == ""

    def test_does_not_execute_code(self) -> None:
        """Should NOT execute any code - just return it as string.

        This is a security test to ensure we don't accidentally
        execute untrusted LLM-generated code.
        """
        # This code would raise if executed
        dangerous_code = "raise Exception('This should not be executed!')"
        parsing_info = {"loading_code": dangerous_code}

        # Should NOT raise - code is returned, not executed
        result = get_loading_code_for_developer(parsing_info)

        assert result == dangerous_code


class TestDataFormatDiscoveryIntegration:
    """Integration tests for the full discovery flow."""

    def test_mlsp_like_format_detection(self, tmp_path: Path) -> None:
        """Should handle MLSP-2013-Birds-like format."""
        # Create MLSP-like structure
        essential = tmp_path / "essential_data"
        essential.mkdir()

        # rec_id2filename.txt
        (essential / "rec_id2filename.txt").write_text("0 audio_0.wav\n1 audio_1.wav")

        # rec_labels_test_hidden.txt
        (essential / "rec_labels_test_hidden.txt").write_text("0,1,2\n1,?\n")

        # CVfolds_2.txt
        (essential / "CVfolds_2.txt").write_text("0,0\n1,1")

        # Audio files
        src_wavs = essential / "src_wavs"
        src_wavs.mkdir()
        (src_wavs / "audio_0.wav").write_bytes(b"audio")
        (src_wavs / "audio_1.wav").write_bytes(b"audio")

        # Traditional detection should fail
        assert detect_traditional_format(tmp_path) is None

        # File listing should find the txt files
        discoverer = DataFormatDiscoverer()
        files = discoverer.list_data_files(tmp_path)

        file_names = [f["name"] for f in files]
        assert any("rec_id2filename" in n for n in file_names)
        assert any("rec_labels" in n for n in file_names)


class TestAudioDomainDetection:
    """Tests for audio domain detection in DomainDetector.

    These tests verify that audio competitions are correctly detected even when
    images (spectrograms) exist in train/test directories.
    """

    def test_detects_audio_when_spectrograms_in_train_dir(self, tmp_path: Path) -> None:
        """Should detect audio_classification when audio files exist in nested dirs.

        This is the MLSP-2013-Birds scenario:
        - train/test dirs have spectrogram images (.bmp)
        - Audio files (.wav) are in essential_data/src_wavs/
        """
        from kaggle_agents.domain.detector import DomainDetector
        from kaggle_agents.core.state import CompetitionInfo

        # Create train/test with spectrograms (images)
        train_dir = tmp_path / "train"
        test_dir = tmp_path / "test"
        train_dir.mkdir()
        test_dir.mkdir()

        for i in range(20):
            (train_dir / f"spectrogram_{i}.bmp").write_bytes(b"fake image")
            (test_dir / f"spectrogram_{i}.bmp").write_bytes(b"fake image")

        # Create audio files in nested directory (like essential_data/src_wavs)
        essential = tmp_path / "essential_data"
        src_wavs = essential / "src_wavs"
        src_wavs.mkdir(parents=True)

        for i in range(50):
            (src_wavs / f"audio_{i}.wav").write_bytes(b"fake audio data")

        # Create label files (non-standard format)
        (essential / "rec_labels.txt").write_text("0,1,2\n1,3,4")
        (essential / "CVfolds.txt").write_text("0,0\n1,1")

        # Detect domain
        detector = DomainDetector(llm=None)
        competition_info = CompetitionInfo(
            name="mlsp-2013-birds",
            description="Bird species identification from audio recordings",
            evaluation_metric="auc",
            problem_type="classification",
        )

        domain, confidence = detector.detect(competition_info, tmp_path)

        # Should detect audio, NOT image
        assert domain == "audio_classification", f"Expected audio_classification, got {domain}"
        assert confidence >= 0.85

    def test_detects_audio_when_only_audio_files_exist(self, tmp_path: Path) -> None:
        """Should detect audio_classification when only audio files exist."""
        from kaggle_agents.domain.detector import DomainDetector
        from kaggle_agents.core.state import CompetitionInfo

        # Create audio files directory
        audio_dir = tmp_path / "audio"
        audio_dir.mkdir()

        for i in range(30):
            (audio_dir / f"sample_{i}.wav").write_bytes(b"fake audio data")

        detector = DomainDetector(llm=None)
        competition_info = CompetitionInfo(
            name="audio-competition",
            description="Audio classification task",
            evaluation_metric="accuracy",
            problem_type="classification",
        )

        domain, confidence = detector.detect(competition_info, tmp_path)

        assert domain == "audio_classification"
        assert confidence >= 0.85

    def test_audio_in_nested_directory_detected(self, tmp_path: Path) -> None:
        """Should detect audio files in deeply nested directories."""
        from kaggle_agents.domain.detector import DomainDetector
        from kaggle_agents.core.state import CompetitionInfo

        # Create deeply nested audio directory (like essential_data/src_wavs/)
        nested = tmp_path / "data" / "raw" / "audio"
        nested.mkdir(parents=True)

        for i in range(15):
            (nested / f"recording_{i}.mp3").write_bytes(b"fake audio data")

        detector = DomainDetector(llm=None)
        competition_info = CompetitionInfo(
            name="test-comp",
            description="Audio task",
            evaluation_metric="",
            problem_type="",
        )

        domain, confidence = detector.detect(competition_info, tmp_path)

        assert domain == "audio_classification"

    def test_pure_image_competition_still_detected(self, tmp_path: Path) -> None:
        """Should still correctly detect image_classification when no audio exists."""
        from kaggle_agents.domain.detector import DomainDetector
        from kaggle_agents.core.state import CompetitionInfo

        # Create image directories (no audio)
        train_dir = tmp_path / "train"
        test_dir = tmp_path / "test"
        train_dir.mkdir()
        test_dir.mkdir()

        for i in range(30):
            (train_dir / f"image_{i}.jpg").write_bytes(b"fake image")
            (test_dir / f"image_{i}.jpg").write_bytes(b"fake image")

        detector = DomainDetector(llm=None)
        competition_info = CompetitionInfo(
            name="image-comp",
            description="Image classification task",
            evaluation_metric="accuracy",
            problem_type="classification",
        )

        domain, confidence = detector.detect(competition_info, tmp_path)

        # Should still detect image when no audio exists
        assert domain == "image_classification"
        assert confidence >= 0.85
