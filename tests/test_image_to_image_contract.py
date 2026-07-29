"""Image-to-image canonical and packed artifact contracts."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from kaggle_agents.agents.developer.validation import ValidationMixin
from kaggle_agents.core.state import AblationComponent, CompetitionInfo
from kaggle_agents.core.state.contracts import CanonicalDataContract
from kaggle_agents.utils.image_to_image_contract import (
    load_packed_images,
    packed_image_rmse,
    prepare_image_to_image_canonical_data,
    save_packed_images,
    validate_image_fold_assignments,
    validate_packed_images,
    write_packed_image_submission,
)
from kaggle_agents.workflow.nodes.canonical_data import (
    canonical_data_preparation_node,
)


def _write_rgb(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB").save(path)


def test_packed_images_round_trip_variable_shapes_without_pickle(
    tmp_path: Path,
) -> None:
    images = [
        np.arange(12, dtype=np.float32).reshape(2, 2, 3),
        np.arange(18, dtype=np.float32).reshape(3, 2, 3),
    ]
    path = save_packed_images(
        tmp_path / "images.npz",
        images,
        image_ids=["café/imagem.png", "nested/犬.png"],
    )

    with np.load(path, allow_pickle=False) as raw:
        assert raw["values"].dtype == np.float32
        assert raw["offsets"].dtype == np.int64
        assert raw["shapes"].dtype == np.int32
        assert raw["image_ids"].dtype.kind == "U"

    packed = load_packed_images(path)

    assert packed.image_ids.tolist() == ["café/imagem.png", "nested/犬.png"]
    assert packed.offsets.tolist() == [0, 12, 30]
    assert packed.shapes.tolist() == [[2, 2, 3], [3, 2, 3]]
    assert np.array_equal(packed.image(0), images[0])
    assert np.array_equal(packed.image(1), images[1])


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"offsets": np.array([0, 1, 5], dtype=np.int64)}, "final offset"),
        (
            {"shapes": np.array([[2, 1]], dtype=np.int32)},
            "shape row count",
        ),
        (
            {"image_ids": np.array(["one", "one"], dtype=str)},
            "duplicate image IDs",
        ),
        (
            {"values": np.array([0.0, np.nan], dtype=np.float32)},
            "NaN or Inf",
        ),
    ],
)
def test_packed_images_reject_malformed_contract(
    tmp_path: Path,
    updates: dict[str, np.ndarray],
    message: str,
) -> None:
    payload = {
        "values": np.array([0.0, 1.0], dtype=np.float32),
        "offsets": np.array([0, 1, 2], dtype=np.int64),
        "shapes": np.array([[1, 1], [1, 1]], dtype=np.int32),
        "image_ids": np.array(["one", "two"], dtype=str),
    }
    payload.update(updates)
    path = tmp_path / "bad.npz"
    np.savez(path, **payload)

    with pytest.raises(ValueError, match=message):
        load_packed_images(path)


def test_packed_images_reject_rank_zero_and_int64_product_overflow() -> None:
    with pytest.raises(ValueError, match="rank 2 or 3"):
        validate_packed_images(
            np.asarray([0.0], dtype=np.float32),
            np.asarray([0, 1], dtype=np.int64),
            np.empty((1, 0), dtype=np.int32),
            np.asarray(["page.png"], dtype=str),
        )

    with pytest.raises(ValueError, match="offsets do not match"):
        validate_packed_images(
            np.asarray([], dtype=np.float32),
            np.asarray([0, 0], dtype=np.int64),
            np.asarray([[2**30, 2**30, 16]], dtype=np.int32),
            np.asarray(["page.png"], dtype=str),
        )


def test_packed_submission_maps_one_based_pixels_from_artifact(
    tmp_path: Path,
) -> None:
    packed_path = save_packed_images(
        tmp_path / "test_candidate.npz",
        [
            np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float32),
            np.asarray([[0.7, 0.8]], dtype=np.float32),
        ],
        image_ids=["page.png", "nested/other.png"],
    )
    sample = tmp_path / "sample_submission.csv"
    ids = [
        "page_2_3",
        "page_1_1",
        "page_2_1",
        "page_1_3",
        "page_1_2",
        "page_2_2",
        "other_1_2",
        "other_1_1",
    ]
    pd.DataFrame(
        {"value": [0.0] * len(ids), "id": ids, "echo": ["NA"] * len(ids)}
    ).to_csv(sample, index=False)

    output = write_packed_image_submission(
        packed_predictions_path=packed_path,
        sample_submission_path=sample,
        output_path=tmp_path / "submission.csv",
        target_cols=["value"],
        id_col="id",
        chunk_rows=2,
    )

    written = pd.read_csv(
        output,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
    )
    assert written["id"].tolist() == ids
    assert written["echo"].tolist() == ["NA"] * len(ids)
    assert written["value"].astype(float).tolist() == pytest.approx(
        [0.6, 0.1, 0.4, 0.3, 0.2, 0.5, 0.8, 0.7]
    )


def test_packed_submission_maps_zero_based_prediction_first_template(
    tmp_path: Path,
) -> None:
    packed_path = save_packed_images(
        tmp_path / "test_candidate.npz",
        [np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)],
        image_ids=["page.png"],
    )
    sample = tmp_path / "sample_submission.csv"
    ids = ["page_0_0", "page_0_1", "page_1_0", "page_1_1"]
    pd.DataFrame({"value": [0.0] * 4, "id": ids}).to_csv(
        sample,
        index=False,
    )

    output = write_packed_image_submission(
        packed_predictions_path=packed_path,
        sample_submission_path=sample,
        output_path=tmp_path / "submission.csv",
        target_cols=["value"],
        chunk_rows=2,
    )

    written = pd.read_csv(output)
    assert written["value"].tolist() == pytest.approx([0.1, 0.2, 0.3, 0.4])


def test_packed_submission_rejects_duplicate_pixel_without_replacing_output(
    tmp_path: Path,
) -> None:
    packed_path = save_packed_images(
        tmp_path / "test_candidate.npz",
        [np.asarray([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)],
        image_ids=["page.png"],
    )
    sample = tmp_path / "sample_submission.csv"
    pd.DataFrame(
        {
            "id": ["page_1_1", "page_1_1", "page_2_1", "page_2_2"],
            "value": [0.0] * 4,
        }
    ).to_csv(sample, index=False)
    output = tmp_path / "submission.csv"
    output.write_text("previous-valid-result\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Duplicate pixel ID"):
        write_packed_image_submission(
            packed_predictions_path=packed_path,
            sample_submission_path=sample,
            output_path=output,
            target_cols=["value"],
            id_col="id",
            chunk_rows=2,
        )

    assert output.read_text(encoding="utf-8") == "previous-valid-result\n"


def test_packed_image_rmse_is_host_computed_and_id_aligned(
    tmp_path: Path,
) -> None:
    targets = save_packed_images(
        tmp_path / "targets.npz",
        [
            np.array([[0.0, 2.0]], dtype=np.float32),
            np.array([[4.0]], dtype=np.float32),
        ],
        image_ids=["a", "b"],
    )
    predictions = save_packed_images(
        tmp_path / "predictions.npz",
        [
            np.array([[1.0, 4.0]], dtype=np.float32),
            np.array([[2.0]], dtype=np.float32),
        ],
        image_ids=["a", "b"],
    )

    assert packed_image_rmse(predictions, targets) == pytest.approx(np.sqrt(3.0), rel=1e-7)


def test_fold_assignment_length_reports_image_id_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="Fold assignment count 5 does not match image ID count 115",
    ):
        validate_image_fold_assignments(
            np.arange(5, dtype=np.int64),
            np.array([f"image-{index}" for index in range(115)], dtype=str),
        )


def test_prepare_image_pairs_supports_relative_paths_and_variable_sizes(
    tmp_path: Path,
) -> None:
    noisy = tmp_path / "train"
    clean = tmp_path / "train_cleaned"
    first = np.full((2, 3, 3), 10, dtype=np.uint8)
    second = np.full((4, 2, 3), 20, dtype=np.uint8)
    _write_rgb(noisy / "café" / "imagem.png", first + 1)
    _write_rgb(clean / "café" / "imagem.png", first)
    _write_rgb(noisy / "犬.png", second + 1)
    _write_rgb(clean / "犬.png", second)

    result = prepare_image_to_image_canonical_data(
        noisy_dir=noisy,
        clean_dir=clean,
        output_dir=tmp_path,
        n_folds=5,
    )

    ids = np.load(result["train_ids_path"], allow_pickle=False)
    folds = np.load(result["folds_path"], allow_pickle=False)
    targets = load_packed_images(result["y_path"])
    assert ids.tolist() == ["café/imagem.png", "犬.png"]
    assert ids.dtype.kind == "U"
    assert folds.shape == (2,)
    assert targets.image_ids.tolist() == ids.tolist()
    assert targets.shapes.tolist() == [[2, 3, 3], [4, 2, 3]]
    assert targets.image(0)[0, 0, 0] == pytest.approx(10.0 / 255.0)
    assert result["metadata"]["target_value_range"] == [0.0, 1.0]
    assert result["metadata"]["integer_pixel_normalization"] == "dtype_max"
    assert result["metadata"]["canonical_rows"] == 2


def test_prepare_image_contract_discovers_test_images_and_persists_ids(
    tmp_path: Path,
) -> None:
    noisy = tmp_path / "train"
    clean = tmp_path / "train_cleaned"
    test = tmp_path / "test"
    _write_rgb(noisy / "train.png", np.ones((2, 2, 3), dtype=np.uint8))
    _write_rgb(clean / "train.png", np.zeros((2, 2, 3), dtype=np.uint8))
    _write_rgb(test / "nested" / "z.png", np.ones((3, 2, 3), dtype=np.uint8))
    _write_rgb(test / "á.png", np.ones((2, 4, 3), dtype=np.uint8))

    result = prepare_image_to_image_canonical_data(
        noisy_dir=noisy,
        clean_dir=clean,
        test_dir=test,
        output_dir=tmp_path,
    )

    test_ids = np.load(result["test_ids_path"], allow_pickle=False)
    test_paths = np.load(result["image_test_input_paths_path"], allow_pickle=False)
    assert test_ids.tolist() == ["nested/z.png", "á.png"]
    assert test_ids.dtype.kind == "U"
    assert len(test_paths) == 2
    assert result["metadata"]["n_test"] == 2


def test_prepare_image_pairs_rejects_shape_mismatch(tmp_path: Path) -> None:
    noisy = tmp_path / "train"
    clean = tmp_path / "train_cleaned"
    _write_rgb(noisy / "a.png", np.zeros((2, 2, 3), dtype=np.uint8))
    _write_rgb(clean / "a.png", np.zeros((3, 2, 3), dtype=np.uint8))

    with pytest.raises(ValueError, match=r"Paired image shape mismatch.*a.png"):
        prepare_image_to_image_canonical_data(
            noisy_dir=noisy,
            clean_dir=clean,
            output_dir=tmp_path,
        )


def test_prepare_image_pairs_rejects_missing_pair_in_either_directory(
    tmp_path: Path,
) -> None:
    noisy = tmp_path / "train"
    clean = tmp_path / "train_cleaned"
    _write_rgb(noisy / "only-noisy.png", np.zeros((2, 2, 3), dtype=np.uint8))
    _write_rgb(clean / "only-clean.png", np.zeros((2, 2, 3), dtype=np.uint8))

    with pytest.raises(
        ValueError,
        match=r"Image pair coverage mismatch.*only-noisy.png.*only-clean.png",
    ):
        prepare_image_to_image_canonical_data(
            noisy_dir=noisy,
            clean_dir=clean,
            output_dir=tmp_path,
        )


def test_image_to_image_node_prepares_canonical_without_train_csv(
    tmp_path: Path,
) -> None:
    noisy = tmp_path / "train"
    clean = tmp_path / "train_cleaned"
    test = tmp_path / "test"
    _write_rgb(noisy / "a.png", np.ones((2, 2, 3), dtype=np.uint8))
    _write_rgb(clean / "a.png", np.zeros((2, 2, 3), dtype=np.uint8))
    _write_rgb(test / "nested" / "b.png", np.ones((3, 2, 3), dtype=np.uint8))
    _write_rgb(test / "c.png", np.ones((2, 3, 3), dtype=np.uint8))

    updates = canonical_data_preparation_node(
        {
            "working_directory": str(tmp_path),
            "domain_detected": "image_to_image",
            "data_files": {
                "data_type": "image",
                "train": str(noisy),
                "clean_train": str(clean),
                "test": str(test),
            },
        }
    )

    assert updates["canonical_data_prepared"] is True
    assert updates["canonical_metadata"]["task_type"] == "image_to_image"
    assert updates["canonical_metadata"]["n_targets"] == 1
    assert updates["canonical_metadata"]["target_type"] == "multi_target"
    assert updates["canonical_metadata"]["packed_image_contract"] is True
    assert Path(updates["canonical_contract"]["y_path"]).suffix == ".npz"
    assert updates["expected_train_rows"] == 1
    assert updates["expected_test_rows"] == 2
    assert updates["test_rec_ids"] == ["c.png", "nested/b.png"]
    assert Path(updates["canonical_contract"]["test_ids_path"]).is_file()
    contract = CanonicalDataContract.from_dict(updates["canonical_contract"])
    assert contract.validate() == (True, [])


def test_packed_canonical_contract_requires_image_alignment_files(
    tmp_path: Path,
) -> None:
    noisy = tmp_path / "train"
    clean = tmp_path / "train_cleaned"
    test = tmp_path / "test"
    _write_rgb(noisy / "a.png", np.ones((2, 2, 3), dtype=np.uint8))
    _write_rgb(clean / "a.png", np.zeros((2, 2, 3), dtype=np.uint8))
    _write_rgb(test / "b.png", np.ones((2, 2, 3), dtype=np.uint8))
    updates = canonical_data_preparation_node(
        {
            "working_directory": str(tmp_path),
            "domain_detected": "image_to_image",
            "data_files": {
                "data_type": "image",
                "train": str(noisy),
                "clean_train": str(clean),
                "test": str(test),
            },
        }
    )
    contract = CanonicalDataContract.from_dict(updates["canonical_contract"])
    Path(contract.test_ids_path).unlink()

    valid, violations = contract.validate()

    assert valid is False
    assert any("test_ids_path" in violation for violation in violations)


def test_image_to_image_node_and_trusted_scorer_end_to_end(
    tmp_path: Path,
) -> None:
    noisy = tmp_path / "train"
    clean = tmp_path / "train_cleaned"
    _write_rgb(noisy / "a.png", np.full((2, 2, 3), 3, dtype=np.uint8))
    _write_rgb(clean / "a.png", np.full((2, 2, 3), 2, dtype=np.uint8))
    updates = canonical_data_preparation_node(
        {
            "working_directory": str(tmp_path),
            "domain_detected": "image_to_image",
            "data_files": {
                "data_type": "image",
                "train": str(noisy),
                "clean_train": str(clean),
            },
        }
    )
    canonical_targets = load_packed_images(updates["canonical_contract"]["y_path"])
    (tmp_path / "models").mkdir()
    save_packed_images(
        tmp_path / "models" / "oof_model.npz",
        [canonical_targets.image(0) + 2.0],
        image_ids=canonical_targets.image_ids,
    )

    class _Validator(ValidationMixin):
        pass

    score = _Validator()._compute_trusted_oof_score(
        AblationComponent("model", "model", "train"),
        {
            **updates,
            "working_directory": str(tmp_path),
            "domain_detected": "image_to_image",
            "competition_info": CompetitionInfo("demo", "", "rmse", "image_to_image"),
        },
    )

    assert score == pytest.approx(2.0)
