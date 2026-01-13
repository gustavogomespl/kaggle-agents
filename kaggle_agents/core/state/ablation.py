"""
Ablation History for Kaggle Agents.

Tracks all ablation experiments with:
- Standardized delta_score (>0 always means improvement)
- Evaluation fidelity for valid comparisons
- Contract compliance tracking
- Evidence/artifact references

MLE-STAR uses ablation history to learn which changes are effective
and avoid repeating unsuccessful strategies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal


@dataclass
class AblationExecution:
    """Record of an executed ablation experiment.

    CRITICAL: delta_score is computed via MetricContract.compute_delta()
    so that delta > 0 ALWAYS means improvement (regardless of metric direction).

    Attributes:
        abl_id: Unique identifier for this ablation
        base_script_hash: Hash of script before ablation
        ablated_block_id: Which code block was modified
        component_name: Name of the ablation component

        delta_score: ALWAYS >0 means better (via MetricContract.compute_delta)
        delta_time: Time change in seconds (positive = slower)
        pass_contract: Whether it passed canonical contract validation
        eval_fidelity: Fidelity level for comparison validity

        old_score: Raw score before ablation
        new_score: Raw score after ablation

        summary: Short description for planner context
        log_path: Path to execution log
        diff_path: Path to code diff (optional)
        timestamp: When the ablation was executed
    """

    abl_id: str
    base_script_hash: str
    ablated_block_id: str
    component_name: str

    # Results (delta_score standardized: >0 = improvement)
    delta_score: float
    delta_time: float
    pass_contract: bool
    eval_fidelity: Literal["debug", "fast_cv", "full_cv", "train_all"]

    # Scores (raw)
    old_score: float
    new_score: float

    # Evidence
    summary: str
    log_path: str
    diff_path: str | None = None

    timestamp: str = ""

    def __post_init__(self):
        """Set timestamp if not provided."""
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()

    @property
    def is_effective(self) -> bool:
        """Check if this ablation was effective (improved score and passed contract)."""
        return self.delta_score > 0 and self.pass_contract

    @property
    def is_regression(self) -> bool:
        """Check if this ablation caused a regression."""
        return self.delta_score < 0

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        sign = "+" if self.delta_score >= 0 else ""
        status = "PASS" if self.pass_contract else "FAIL"
        return (
            f"AblationExecution({self.component_name}, "
            f"delta={sign}{self.delta_score:.4f}, {status}, {self.eval_fidelity})"
        )

    def to_dict(self) -> dict:
        """Serialize for checkpointing."""
        return {
            "abl_id": self.abl_id,
            "base_script_hash": self.base_script_hash,
            "ablated_block_id": self.ablated_block_id,
            "component_name": self.component_name,
            "delta_score": self.delta_score,
            "delta_time": self.delta_time,
            "pass_contract": self.pass_contract,
            "eval_fidelity": self.eval_fidelity,
            "old_score": self.old_score,
            "new_score": self.new_score,
            "summary": self.summary,
            "log_path": self.log_path,
            "diff_path": self.diff_path,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict) -> AblationExecution:
        """Deserialize from checkpoint."""
        return cls(
            abl_id=data["abl_id"],
            base_script_hash=data["base_script_hash"],
            ablated_block_id=data["ablated_block_id"],
            component_name=data["component_name"],
            delta_score=data["delta_score"],
            delta_time=data["delta_time"],
            pass_contract=data["pass_contract"],
            eval_fidelity=data["eval_fidelity"],
            old_score=data["old_score"],
            new_score=data["new_score"],
            summary=data["summary"],
            log_path=data["log_path"],
            diff_path=data.get("diff_path"),
            timestamp=data.get("timestamp", ""),
        )


@dataclass
class AblationHistory:
    """History of all ablation executions.

    Provides:
    - Recording of ablation experiments
    - Filtering by component, fidelity, effectiveness
    - Learning from past experiments
    """

    executions: list[AblationExecution] = field(default_factory=list)

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        n_effective = len(self.get_effective_ablations())
        n_regressions = len(self.get_regressions())
        return (
            f"AblationHistory({len(self.executions)} executions, "
            f"{n_effective} effective, {n_regressions} regressions)"
        )

    def add(self, execution: AblationExecution) -> None:
        """Add an ablation execution to history."""
        self.executions.append(execution)

    def get_by_component(self, component_name: str) -> list[AblationExecution]:
        """Get all ablations for a specific component."""
        return [e for e in self.executions if e.component_name == component_name]

    def get_effective_ablations(self) -> list[AblationExecution]:
        """Get ablations that improved score and passed contract."""
        return [e for e in self.executions if e.is_effective]

    def get_regressions(self) -> list[AblationExecution]:
        """Get ablations that caused score regressions."""
        return [e for e in self.executions if e.is_regression]

    def get_by_fidelity(self, fidelity: str) -> list[AblationExecution]:
        """Get ablations at a specific fidelity level.

        Only compare scores with same fidelity for valid comparisons.
        """
        return [e for e in self.executions if e.eval_fidelity == fidelity]

    def get_contract_failures(self) -> list[AblationExecution]:
        """Get ablations that failed contract validation."""
        return [e for e in self.executions if not e.pass_contract]

    def get_best_ablation(
        self,
        component_name: str | None = None,
        fidelity: str | None = None,
    ) -> AblationExecution | None:
        """Get the ablation with highest delta_score.

        Args:
            component_name: Filter by component (optional)
            fidelity: Filter by fidelity level (optional)

        Returns:
            Best ablation or None if no matching ablations
        """
        candidates = self.executions

        if component_name:
            candidates = [e for e in candidates if e.component_name == component_name]

        if fidelity:
            candidates = [e for e in candidates if e.eval_fidelity == fidelity]

        # Only consider effective ablations
        candidates = [e for e in candidates if e.is_effective]

        if not candidates:
            return None

        return max(candidates, key=lambda e: e.delta_score)

    def get_summary_for_planner(self, max_entries: int = 10) -> str:
        """Get a summary of ablation history for the planner.

        Returns:
            Formatted string with effective and failed ablations
        """
        effective = self.get_effective_ablations()
        regressions = self.get_regressions()
        contract_failures = self.get_contract_failures()

        lines = ["## Ablation History Summary"]

        if effective:
            lines.append(f"\n### Effective Ablations ({len(effective)} total)")
            for e in effective[-max_entries:]:
                lines.append(f"- {e.component_name}: +{e.delta_score:.4f} ({e.summary})")

        if regressions:
            lines.append(f"\n### Regressions ({len(regressions)} total) - AVOID THESE")
            for e in regressions[-max_entries:]:
                lines.append(f"- {e.component_name}: {e.delta_score:.4f} ({e.summary})")

        if contract_failures:
            lines.append(f"\n### Contract Failures ({len(contract_failures)} total)")
            for e in contract_failures[-max_entries:]:
                lines.append(f"- {e.component_name}: {e.summary}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize for checkpointing."""
        return {"executions": [e.to_dict() for e in self.executions]}

    @classmethod
    def from_dict(cls, data: dict) -> AblationHistory:
        """Deserialize from checkpoint."""
        history = cls()
        for e in data.get("executions", []):
            history.executions.append(AblationExecution.from_dict(e))
        return history
