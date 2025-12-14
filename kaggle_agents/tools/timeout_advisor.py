"""
Timeout Advisor - LLM-based intelligent timeout decision making.

This module provides an LLM-powered advisor that analyzes execution progress
and makes intelligent decisions about whether to extend timeouts or abort
with partial results.
"""

from dataclasses import dataclass
from typing import Literal

from ..core.config import get_llm_for_role
from .code_executor import ExecutionProgress


@dataclass
class TimeoutDecision:
    """Result from timeout advisor."""

    action: Literal["extend", "abort", "continue"]
    additional_seconds: int = 0
    reasoning: str = ""
    confidence: float = 0.0
    use_partial_results: bool = False


class TimeoutAdvisor:
    """
    LLM-based advisor for timeout decisions.

    This class analyzes execution progress and uses an LLM to make intelligent
    decisions about timeout handling:

    - EXTEND: Give more time if completion is likely soon
    - ABORT: Stop and use partial results if available
    - CONTINUE: Keep running within current timeout
    """

    # Default thresholds
    SOFT_TIMEOUT_PERCENT = 0.80  # Start checking at 80% of timeout
    MIN_FOLDS_FOR_PARTIAL = 2  # Minimum folds needed for partial submission
    MAX_EXTENSIONS = 2  # Maximum number of timeout extensions

    def __init__(self, use_llm: bool = True):
        """
        Initialize timeout advisor.

        Args:
            use_llm: Whether to use LLM for decisions (False uses heuristics only)
        """
        self.use_llm = use_llm
        self.extensions_granted = 0

    def _calculate_heuristic_decision(
        self,
        progress: ExecutionProgress,
        elapsed: float,
        timeout: float,
    ) -> TimeoutDecision:
        """
        Calculate decision using simple heuristics (no LLM).

        This provides a fast fallback when LLM is not available or for
        simple cases that don't need LLM reasoning.
        """
        remaining = timeout - elapsed
        progress_pct = progress.progress_percent

        # Estimate remaining time
        estimated_remaining = progress.estimate_remaining_time()

        # Case 1: Very close to completion (>80% done)
        if progress_pct >= 80:
            if estimated_remaining and estimated_remaining < remaining:
                return TimeoutDecision(
                    action="continue",
                    reasoning=f"{progress_pct:.0f}% complete, estimated {estimated_remaining:.0f}s remaining fits within timeout",
                    confidence=0.9,
                )

            # Extend if we have margin
            if self.extensions_granted < self.MAX_EXTENSIONS:
                extension = min(int(estimated_remaining * 1.2), 600) if estimated_remaining else 300
                return TimeoutDecision(
                    action="extend",
                    additional_seconds=extension,
                    reasoning=f"{progress_pct:.0f}% complete, extending to allow completion",
                    confidence=0.8,
                )

        # Case 2: Moderate progress (40-80% done)
        if 40 <= progress_pct < 80:
            if estimated_remaining and estimated_remaining > remaining * 2:
                # Will take way too long - abort with partial
                if progress.can_create_partial_submission:
                    return TimeoutDecision(
                        action="abort",
                        reasoning=f"Only {progress_pct:.0f}% complete, estimated {estimated_remaining:.0f}s remaining exceeds timeout significantly",
                        confidence=0.7,
                        use_partial_results=True,
                    )

            # Otherwise extend
            if self.extensions_granted < self.MAX_EXTENSIONS:
                return TimeoutDecision(
                    action="extend",
                    additional_seconds=600,  # 10 minutes
                    reasoning=f"{progress_pct:.0f}% complete, extending to gather more data",
                    confidence=0.6,
                )

        # Case 3: Low progress (<40% done)
        if progress_pct < 40:
            # If we have some partial results, abort
            if progress.folds_completed >= self.MIN_FOLDS_FOR_PARTIAL:
                return TimeoutDecision(
                    action="abort",
                    reasoning=f"Only {progress_pct:.0f}% complete with {progress.folds_completed} folds, aborting with partial results",
                    confidence=0.6,
                    use_partial_results=True,
                )

            # No partial results - one extension attempt
            if self.extensions_granted == 0:
                return TimeoutDecision(
                    action="extend",
                    additional_seconds=300,  # 5 minutes
                    reasoning=f"Low progress ({progress_pct:.0f}%), granting single extension",
                    confidence=0.5,
                )

        # Default: abort
        return TimeoutDecision(
            action="abort",
            reasoning="Timeout reached with insufficient progress",
            confidence=0.5,
            use_partial_results=progress.can_create_partial_submission,
        )

    def _create_llm_prompt(
        self,
        progress: ExecutionProgress,
        elapsed: float,
        timeout: float,
        stdout_tail: str,
    ) -> str:
        """Create prompt for LLM decision."""
        estimated_remaining = progress.estimate_remaining_time()

        return f"""You are an ML execution timeout advisor. Analyze the execution progress and decide whether to:
1. EXTEND: Grant more time if completion is likely soon
2. ABORT: Stop execution and use partial results if available
3. CONTINUE: Keep running within current timeout

## Execution Progress
- Folds completed: {progress.folds_completed}/{progress.total_folds} ({progress.progress_percent:.1f}%)
- Fold scores: {progress.fold_scores if progress.fold_scores else 'N/A'}
- Current CV score: {progress.current_cv_score if progress.current_cv_score else 'N/A'}
- OOF predictions saved: {progress.oof_predictions_saved}
- Test predictions saved: {progress.test_predictions_saved}
- Current phase: {progress.current_phase}

## Time Analysis
- Elapsed time: {elapsed:.0f}s ({elapsed/60:.1f} minutes)
- Timeout: {timeout:.0f}s ({timeout/60:.1f} minutes)
- Remaining: {timeout - elapsed:.0f}s
- Avg time per fold: {progress.avg_fold_time:.0f}s if progress.avg_fold_time else 'Unknown'
- Estimated remaining: {estimated_remaining:.0f}s if estimated_remaining else 'Unknown'
- Extensions already granted: {self.extensions_granted}

## Recent Output
```
{stdout_tail[-500:] if stdout_tail else 'No output'}
```

## Decision Criteria
- If >80% complete and estimated remaining < remaining timeout: CONTINUE
- If >60% complete and close to finishing: EXTEND (max +600s)
- If <40% complete with no good partial results: ABORT
- If partial results available (OOF saved): consider ABORT with partial

Respond in this exact format:
ACTION: [EXTEND|ABORT|CONTINUE]
SECONDS: [additional seconds if EXTEND, 0 otherwise]
USE_PARTIAL: [true|false]
CONFIDENCE: [0.0-1.0]
REASONING: [one line explanation]
"""

    def _parse_llm_response(self, response: str) -> TimeoutDecision:
        """Parse LLM response into TimeoutDecision."""
        import re

        # Default values
        action = "abort"
        additional_seconds = 0
        use_partial = False
        confidence = 0.5
        reasoning = "Could not parse LLM response"

        # Parse action
        action_match = re.search(r'ACTION:\s*(EXTEND|ABORT|CONTINUE)', response, re.IGNORECASE)
        if action_match:
            action = action_match.group(1).lower()

        # Parse seconds
        seconds_match = re.search(r'SECONDS:\s*(\d+)', response)
        if seconds_match:
            additional_seconds = int(seconds_match.group(1))

        # Parse use_partial
        partial_match = re.search(r'USE_PARTIAL:\s*(true|false)', response, re.IGNORECASE)
        if partial_match:
            use_partial = partial_match.group(1).lower() == 'true'

        # Parse confidence
        conf_match = re.search(r'CONFIDENCE:\s*([\d.]+)', response)
        if conf_match:
            confidence = float(conf_match.group(1))

        # Parse reasoning
        reason_match = re.search(r'REASONING:\s*(.+?)(?:\n|$)', response)
        if reason_match:
            reasoning = reason_match.group(1).strip()

        return TimeoutDecision(
            action=action,
            additional_seconds=additional_seconds,
            reasoning=reasoning,
            confidence=confidence,
            use_partial_results=use_partial,
        )

    def should_extend_timeout(
        self,
        progress: ExecutionProgress,
        elapsed: float,
        timeout: float,
        stdout: str = "",
    ) -> TimeoutDecision:
        """
        Decide whether to extend timeout or abort.

        Args:
            progress: Current execution progress
            elapsed: Elapsed time in seconds
            timeout: Original timeout in seconds
            stdout: Standard output for context

        Returns:
            TimeoutDecision with action and reasoning
        """
        # Update progress timing
        progress.elapsed_seconds = elapsed
        progress.estimate_remaining_time()

        # Check if we're within soft timeout (not yet critical)
        if elapsed < timeout * self.SOFT_TIMEOUT_PERCENT:
            return TimeoutDecision(
                action="continue",
                reasoning="Within normal execution time",
                confidence=1.0,
            )

        # Use heuristics if LLM disabled or for simple cases
        if not self.use_llm:
            return self._calculate_heuristic_decision(progress, elapsed, timeout)

        # Use LLM for complex decisions
        try:
            llm = get_llm_for_role("evaluator", temperature=0.1, max_tokens=500)

            prompt = self._create_llm_prompt(progress, elapsed, timeout, stdout)

            response = llm.invoke(prompt)
            response_text = response.content if hasattr(response, 'content') else str(response)

            decision = self._parse_llm_response(response_text)

            # Track extensions
            if decision.action == "extend":
                self.extensions_granted += 1

            return decision

        except Exception as e:
            # Fallback to heuristics on LLM failure
            print(f"   Warning: LLM timeout advisor failed ({e}), using heuristics")
            return self._calculate_heuristic_decision(progress, elapsed, timeout)

    def reset(self):
        """Reset advisor state for new execution."""
        self.extensions_granted = 0
