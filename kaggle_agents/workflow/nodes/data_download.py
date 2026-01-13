"""Data download node for the Kaggle Agents workflow."""

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from ...core.state import KaggleState
from ...tools.kaggle_api import KaggleAPIClient


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
                from ...utils.cross_validation import generate_folds

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
