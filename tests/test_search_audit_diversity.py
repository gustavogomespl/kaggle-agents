"""Focused tests for auditable adaptive external retrieval."""

from types import SimpleNamespace

from kaggle_agents.core.state import SOTASolution
from kaggle_agents.core.state.base import merge_sota_solutions
from kaggle_agents.utils.telemetry import summarize_run_telemetry


def _solution(
    source: str,
    source_sha256: str,
    *,
    title: str = "solution",
) -> SOTASolution:
    return SOTASolution(
        source=source,
        title=title,
        score=0.0,
        votes=1,
        source_sha256=source_sha256,
    )


def test_stagnation_sources_are_fresh_first_and_distinct_history_survives():
    """A recovery batch must be visible before the initial top-K."""
    initial = [
        _solution("author/old-a", "a" * 64),
        _solution("author/old-b", "b" * 64),
    ]
    recovery = [
        # Updated download of the same stable source replaces the old entry.
        _solution("author/old-a", "c" * 64, title="fresh a"),
        _solution("author/new-c", "d" * 64),
    ]

    merged = merge_sota_solutions(initial, recovery)

    assert [solution.source for solution in merged] == [
        "author/old-a",
        "author/new-c",
        "author/old-b",
    ]
    assert merged[0].title == "fresh a"


def test_retrieval_merge_deduplicates_alias_refs_by_complete_source_hash():
    initial = [_solution("first/ref", "e" * 64)]
    recovery = [_solution("mirror/ref", "e" * 64)]

    merged = merge_sota_solutions(initial, recovery)

    assert [solution.source for solution in merged] == ["mirror/ref"]


def test_every_provider_candidate_gets_query_and_attempt_audit(monkeypatch):
    from kaggle_agents.tools import kaggle_search

    good = SimpleNamespace(
        ref="author/good",
        title="Generic model",
        author="author",
        total_votes=12,
        language="python",
        competition_data_sources=["other-public-task"],
    )
    low_votes = SimpleNamespace(
        ref="author/low",
        title="Low vote model",
        author="author",
        total_votes=2,
        language="python",
        competition_data_sources=["other-public-task"],
    )
    missing_ref = SimpleNamespace(
        title="Malformed provider object",
        total_votes=20,
    )

    class FakeApi:
        def kernels_list(self, **kwargs):
            if kwargs["search"] == "first generic query":
                return [good, low_votes, missing_ref]
            return [good]

    class FakeSearcher:
        _get_kernel_attr = staticmethod(kaggle_search.KaggleSearcher._get_kernel_attr)
        _get_kernel_competitions = classmethod(
            kaggle_search.KaggleSearcher._get_kernel_competitions.__func__
        )

        def __init__(self):
            self.api = FakeApi()

    monkeypatch.setattr(kaggle_search, "KaggleSearcher", FakeSearcher)

    solutions, audit = kaggle_search.search_notebooks_cross_competition(
        "private-target",
        ["first generic query", "second generic query"],
        max_notebooks=0,
        min_votes=5,
        iteration=4,
        search_attempt_id="recovery:iteration-4",
    )

    provider_records = [record for record in audit if record["stage"] == "provider_candidate"]
    assert solutions == []
    assert len(provider_records) == 4
    assert {record["provider_decision"] for record in provider_records} == {
        "queued_for_metadata_guard",
        "below_min_votes",
        "parse_error",
        "duplicate",
    }
    assert all(record["query"] for record in provider_records)
    assert all(record["iteration"] == 4 for record in provider_records)
    assert all(record["search_attempt_id"] == "recovery:iteration-4" for record in provider_records)
    duplicate = next(
        record for record in provider_records if record["provider_decision"] == "duplicate"
    )
    assert duplicate["query"] == "second generic query"
    assert duplicate["duplicate_of_query"] == "first generic query"
    budget_terminal = [
        record
        for record in audit
        if record.get("filter_reason") == "not_selected_top_k_budget"
    ]
    assert len(budget_terminal) == 1
    assert budget_terminal[0]["ref"] == "author/good"
    assert budget_terminal[0]["query"] == "first generic query"
    assert budget_terminal[0]["search_attempt_id"] == "recovery:iteration-4"


def test_search_telemetry_does_not_claim_unmeasured_downstream_gain():
    state = {
        "search_attempted": True,
        "search_eligible_retrieved": True,
        "search_last_attempt_eligible_retrieved": True,
        "search_eligibility_reason": None,
        "search_downstream_gain": None,
        "search_downstream_gain_status": "unknown_not_measured",
        "search_audit": [
            {
                "stage": "provider_candidate",
                "provider_decision": "queued_for_metadata_guard",
                "ref": "author/good",
                "query": "generic query",
                "iteration": 2,
                "search_attempt_id": "initial:iteration-2",
                "filtered": False,
            },
            {
                "stage": "provider_candidate",
                "provider_decision": "duplicate",
                "ref": "author/good",
                "query": "generic recovery query",
                "iteration": 4,
                "search_attempt_id": "recovery:iteration-4",
                "filtered": True,
            },
            {
                "stage": "code_scan",
                "ref": "author/good",
                "source_sha256": "f" * 64,
                "filtered": False,
                "same_competition": False,
            },
        ],
    }

    search = summarize_run_telemetry(state)["search"]

    assert search["eligible_retrieved"] is True
    assert search["retrieval_treatment_eligible"] is True
    assert search["downstream_gain"] is None
    assert search["downstream_gain_status"] == "unknown_not_measured"
    assert search["causal_effect_estimated"] is False
    assert search["provider_candidates_audited"] == 2
    assert search["provider_candidate_context_complete"] is True
    assert search["provider_duplicates"] == 1
    assert search["eligible_external_sources_unique"] == 1
    assert search["search_attempt_ids"] == [
        "initial:iteration-2",
        "recovery:iteration-4",
    ]
    assert "effective" not in search
