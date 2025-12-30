import sys
from unittest.mock import MagicMock


# Mock code_executor to avoid circular import issues in this test
sys.modules["kaggle_agents.tools.code_executor"] = MagicMock()

import os


os.environ["KAGGLE_USERNAME"] = "dummy_user"
os.environ["KAGGLE_KEY"] = "dummy_key"

from kaggle_agents.tools.kaggle_api import KaggleAPIClient


def test_identify_assets():
    # Setup dummy directory
    test_dir = Path("temp_test_assets")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir()

    # Create dummy files simulating aerial-cactus-identification
    (test_dir / "train.csv").touch()
    (test_dir / "train.zip").touch()
    (test_dir / "test.zip").touch()
    (test_dir / "sample_submission.csv").touch()

    print(f"Created dummy files in {test_dir}")
    for f in test_dir.glob("*"):
        print(f" - {f.name}")

    # Initialize client (mocking auth to avoid error if not set)
    # We only need _identify_data_assets which doesn't use auth
    client = KaggleAPIClient.__new__(KaggleAPIClient)

    print("\nRunning _identify_data_assets...")
    assets = client._identify_data_assets(test_dir)

    print("\nIdentified Assets:")
    for k, v in assets.items():
        print(f" {k}: {v}")

    # Verification logic
    success = True

    # 1. Check if train_csv is present
    if "train_csv" in assets:
        print("\n✅ 'train_csv' correctly identified.")
    else:
        print("\n❌ 'train_csv' MISSING from assets!")
        success = False

    # 2. Check if train points to zip (default behavior for this case)
    if "train" in assets and assets["train"].endswith(".zip"):
        print("✅ 'train' correctly points to zip file (as expected for fallback).")
    else:
        print(f"⚠️ 'train' points to: {assets.get('train')} (Expected zip)")

    # 3. Simulate workflow logic
    train_path_for_folds = assets.get("train_csv", assets.get("train"))
    print(f"\nWorkflow will use for folds: {train_path_for_folds}")

    if train_path_for_folds and train_path_for_folds.endswith(".csv"):
        print("✅ Workflow logic will correctly select the CSV file.")
    else:
        print("❌ Workflow logic FAILED to select the CSV file.")
        success = False

    # Cleanup
    shutil.rmtree(test_dir)

    if success:
        print("\n🎉 Verification PASSED")
    else:
        print("\n💥 Verification FAILED")


if __name__ == "__main__":
    try:
        test_identify_assets()
    except Exception as e:
        print(f"An error occurred: {e}")
