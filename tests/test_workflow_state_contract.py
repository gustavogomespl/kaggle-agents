"""Tests that workflow contract fields survive LangGraph state projection."""

from langgraph.graph import END, START, StateGraph

from kaggle_agents.core.state import KaggleState, create_initial_state


def test_contract_fields_survive_langgraph_nodes(tmp_path):
    def prepare_data(_state):
        return {
            "canonical_data_prepared": True,
            "canonical_data_skipped_reason": None,
            "train_rec_ids": ["train-a", "train-b"],
            "test_rec_ids": ["test-a"],
            "train_file_paths": ["/data/train-a.wav", "/data/train-b.wav"],
            "test_file_paths": ["/data/test-a.wav"],
            "cv_folds_used": True,
            "submission_format_info": {"format_type": "wide"},
            "canonical_dir": str(tmp_path / "canonical"),
            "canonical_train_ids_path": str(tmp_path / "canonical" / "train_ids.npy"),
            "sota_retrieval_k": 5,
            "last_sota_update_iteration": 0,
            "search_attempted": True,
            "search_effective": True,
            "search_failure_reason": None,
        }

    def accept_first_component(state):
        return {
            "oof_availability": {
                **state["oof_availability"],
                "model-a": True,
            },
            "component_results": {
                **state["component_results"],
                "model-a": {"success": True},
            },
            "trusted_component_scores": {
                **state["trusted_component_scores"],
                "model-a": 0.7,
            },
            "robustness_approved_components": {
                **state["robustness_approved_components"],
                "model-a": True,
            },
        }

    def accept_second_component(state):
        return {
            "oof_availability": {
                **state["oof_availability"],
                "model-b": True,
            },
            "component_results": {
                **state["component_results"],
                "model-b": {"success": True},
            },
            "trusted_component_scores": {
                **state["trusted_component_scores"],
                "model-b": 0.8,
            },
            "robustness_approved_components": {
                **state["robustness_approved_components"],
                "model-b": False,
            },
        }

    graph = StateGraph(KaggleState)
    graph.add_node("prepare", prepare_data)
    graph.add_node("first", accept_first_component)
    graph.add_node("second", accept_second_component)
    graph.add_edge(START, "prepare")
    graph.add_edge("prepare", "first")
    graph.add_edge("first", "second")
    graph.add_edge("second", END)

    initial = create_initial_state("demo", str(tmp_path))
    result = graph.compile().invoke(initial)

    assert result["canonical_data_prepared"] is True
    assert result["train_rec_ids"] == ["train-a", "train-b"]
    assert result["test_rec_ids"] == ["test-a"]
    assert result["train_file_paths"] == [
        "/data/train-a.wav",
        "/data/train-b.wav",
    ]
    assert result["test_file_paths"] == ["/data/test-a.wav"]
    assert result["cv_folds_used"] is True
    assert result["submission_format_info"] == {"format_type": "wide"}
    assert result["canonical_dir"] == str(tmp_path / "canonical")
    assert result["canonical_train_ids_path"].endswith("canonical/train_ids.npy")
    assert result["sota_retrieval_k"] == 5
    assert result["last_sota_update_iteration"] == 0
    assert result["search_attempted"] is True
    assert result["search_effective"] is True
    assert result["search_failure_reason"] is None
    assert result["oof_availability"] == {"model-a": True, "model-b": True}
    assert set(result["component_results"]) == {"model-a", "model-b"}
    assert result["trusted_component_scores"] == {
        "model-a": 0.7,
        "model-b": 0.8,
    }
    assert result["robustness_approved_components"] == {
        "model-a": True,
        "model-b": False,
    }


def test_initial_state_has_unique_artifact_lifecycle_fields(tmp_path):
    first = create_initial_state("demo", str(tmp_path / "first"))
    second = create_initial_state("demo", str(tmp_path / "second"))

    assert first["run_id"]
    assert first["run_id"] != second["run_id"]
    assert first["accepted_submission_path"] is None
    assert first["accepted_submission_sha256"] is None
    assert first["accepted_submission_snapshot_path"] is None
    assert first["accepted_submission_cv_score"] is None
    assert first["accepted_submission_score_owner"] is None
    assert first["accepted_submission_score_source"] is None
    assert first["best_candidate_submission_snapshot_path"] is None
    assert first["best_candidate_submission_sha256"] is None
    assert first["best_candidate_submission_component_name"] is None
    assert first["ensemble_oof_score"] is None
    assert first["ensemble_submission_sha256"] is None
    assert first["ensemble_submission_owner"] is None
    assert first["ensemble_score_source"] is None
    assert first["train_file_paths"] == []
    assert first["test_file_paths"] == []
    assert first["search_attempted"] is False
    assert first["search_effective"] is False
    assert first["search_failure_reason"] is None
    assert first["trusted_component_scores"] == {}
