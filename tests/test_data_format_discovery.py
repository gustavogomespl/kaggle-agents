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

    def test_detects_image_directories(self, tmp_path: Path) -> None:
        """Should detect train/test directories with images."""
        train_dir = tmp_path / "train"
        test_dir = tmp_path / "test"
        train_dir.mkdir()
        test_dir.mkdir()

        # Create sample image files
        (train_dir / "img1.jpg").write_bytes(b"fake image data")
        (train_dir / "img2.jpg").write_bytes(b"fake image data")
        (test_dir / "img3.jpg").write_bytes(b"fake image data")

        result = detect_traditional_format(tmp_path)

        assert result is not None
        assert "train" in result
        assert "test" in result

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
