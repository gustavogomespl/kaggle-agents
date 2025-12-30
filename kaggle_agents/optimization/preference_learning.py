"""
DPO-style Preference Learning for Code Generation.

This module implements preference-based learning where the agent learns from
pairs of (chosen, rejected) code examples to improve code generation quality.

Based on: Direct Preference Optimization (DPO) for LLM alignment.

Key concepts:
- PreferencePair: A pair of code (chosen=successful, rejected=failed)
- PreferenceCollector: Collects pairs during code generation/fixing
- PreferenceRewardModel: Scores code based on learned preferences
"""

from datetime import datetime
from typing import Any

from langchain_core.messages import HumanMessage

from ..core.config import get_config, get_llm_for_role
from ..core.state import PreferencePair
from ..utils.llm_utils import get_text_content


# ==================== Preference Collector ====================


class PreferenceCollector:
    """
    Collects preference pairs from code generation and fixing.

    When code fails and is subsequently fixed:
    - The fixed code becomes 'chosen' (good example)
    - The original failed code becomes 'rejected' (bad example)

    This creates training signal for improving future generations.
    """

    def __init__(self):
        """Initialize the preference collector."""
        self.config = get_config()
        self.pairs: list[PreferencePair] = []
        self._pending_rejected: dict[
            str, tuple[str, str, str]
        ] = {}  # component -> (code, context, type)

    def mark_as_rejected(
        self,
        component_name: str,
        component_type: str,
        code: str,
        context: str,
        error: str | None = None,
    ) -> None:
        """
        Mark a code attempt as rejected (failed).

        This code will be paired with a successful fix later.

        Args:
            component_name: Name of the component
            component_type: Type of component (model, feature_engineering, etc.)
            code: The failed code
            context: Context/prompt used to generate the code
            error: Error message (optional, for margin calculation)
        """
        key = f"{component_name}_{component_type}"
        self._pending_rejected[key] = (code, context, component_type, error)

    def mark_as_chosen(
        self,
        component_name: str,
        component_type: str,
        code: str,
        score: float | None = None,
    ) -> PreferencePair | None:
        """
        Mark a code attempt as chosen (successful).

        If there's a pending rejected code for this component, creates a pair.

        Args:
            component_name: Name of the component
            component_type: Type of component
            code: The successful code
            score: CV score achieved (optional, for margin calculation)

        Returns:
            PreferencePair if a pair was created, None otherwise
        """
        key = f"{component_name}_{component_type}"

        if key not in self._pending_rejected:
            # No rejected code to pair with - this was first-try success
            return None

        rejected_code, context, comp_type, error = self._pending_rejected.pop(key)

        # Calculate margin based on success (chosen) vs failure (rejected)
        margin = self._calculate_margin(
            chosen_code=code,
            rejected_code=rejected_code,
            chosen_score=score,
            error=error,
        )

        pair = PreferencePair(
            context=context,
            chosen=code,
            rejected=rejected_code,
            margin=margin,
            component_type=comp_type,
            timestamp=datetime.now(),
        )

        self.pairs.append(pair)
        return pair

    def _calculate_margin(
        self,
        chosen_code: str,
        rejected_code: str,
        chosen_score: float | None = None,
        error: str | None = None,
    ) -> float:
        """
        Calculate preference margin between chosen and rejected code.

        Higher margin = stronger preference for chosen over rejected.

        Args:
            chosen_code: The successful code
            rejected_code: The failed code
            chosen_score: CV score of chosen (if available)
            error: Error from rejected code (if available)

        Returns:
            Margin score between 0.0 and 1.0
        """
        margin = 0.5  # Base margin

        # Adjust based on error severity
        if error:
            error_lower = error.lower()
            if "syntaxerror" in error_lower:
                margin += 0.3  # Strong preference - syntax errors are basic
            elif "importerror" in error_lower or "modulenotfounderror" in error_lower:
                margin += 0.2
            elif "memoryerror" in error_lower or "timeout" in error_lower:
                margin += 0.15
            elif "keyerror" in error_lower or "valueerror" in error_lower:
                margin += 0.1

        # Adjust based on code quality differences
        chosen_lines = len(chosen_code.split("\n"))
        rejected_lines = len(rejected_code.split("\n"))

        # If fix required significant changes, stronger preference
        change_ratio = abs(chosen_lines - rejected_lines) / max(rejected_lines, 1)
        if change_ratio > 0.3:
            margin += 0.1

        # Adjust based on score if available
        if chosen_score is not None and chosen_score > 0:
            margin += min(chosen_score * 0.1, 0.15)  # Bonus for high score

        return min(margin, 1.0)

    def collect_from_fix_cycle(
        self,
        component_name: str,
        component_type: str,
        original_code: str,
        fixed_code: str,
        context: str,
        error: str,
        cv_score: float | None = None,
    ) -> PreferencePair:
        """
        Collect a preference pair from a fix cycle in one step.

        This is a convenience method when you have both codes at once.

        Args:
            component_name: Name of the component
            component_type: Type of component
            original_code: Failed code (rejected)
            fixed_code: Fixed code (chosen)
            context: Context/prompt used
            error: Error from original code
            cv_score: CV score of fixed code

        Returns:
            Created PreferencePair
        """
        margin = self._calculate_margin(
            chosen_code=fixed_code,
            rejected_code=original_code,
            chosen_score=cv_score,
            error=error,
        )

        pair = PreferencePair(
            context=context,
            chosen=fixed_code,
            rejected=original_code,
            margin=margin,
            component_type=component_type,
            timestamp=datetime.now(),
        )

        self.pairs.append(pair)
        return pair

    def get_pairs(self) -> list[PreferencePair]:
        """Get all collected preference pairs."""
        return self.pairs.copy()

    def get_pairs_for_state(self) -> list[PreferencePair]:
        """
        Get pairs ready to be added to state and clear internal buffer.

        Returns:
            List of PreferencePairs to add to state
        """
        pairs = self.pairs.copy()
        self.pairs = []
        return pairs

    def clear(self) -> None:
        """Clear all collected pairs and pending rejections."""
        self.pairs = []
        self._pending_rejected = {}

    def __len__(self) -> int:
        """Return number of collected pairs."""
        return len(self.pairs)


# ==================== Preference Reward Model ====================


class PreferenceRewardModel:
    """
    Reward model that uses preference pairs to score code quality.

    This model learns to distinguish good code from bad code based on
    historical preference pairs, and can be used to:
    1. Score new code before execution
    2. Provide additional reward signal for meta-evaluation
    3. Guide code generation via prompt injection
    """

    def __init__(self, use_llm: bool = True):
        """
        Initialize the preference reward model.

        Args:
            use_llm: Whether to use LLM for sophisticated scoring
        """
        self.config = get_config()
        self.use_llm = use_llm

        if use_llm:
            self.llm = get_llm_for_role("evaluator")

        # Cache of learned patterns from preferences
        self._learned_patterns: dict[str, list[str]] = {
            "good_patterns": [],
            "bad_patterns": [],
        }

        # Statistics
        self.pairs_analyzed = 0

    def learn_from_pairs(self, pairs: list[PreferencePair]) -> dict[str, Any]:
        """
        Learn patterns from preference pairs.

        Analyzes pairs to extract common patterns in good vs bad code.

        Args:
            pairs: List of preference pairs to learn from

        Returns:
            Dictionary with learned patterns and statistics
        """
        if not pairs:
            return {"patterns": self._learned_patterns, "pairs_analyzed": 0}

        good_patterns = []
        bad_patterns = []

        for pair in pairs:
            # Extract patterns from chosen (good) code
            good_patterns.extend(self._extract_patterns(pair.chosen, is_good=True))

            # Extract patterns from rejected (bad) code
            bad_patterns.extend(self._extract_patterns(pair.rejected, is_good=False))

        # Deduplicate and store
        self._learned_patterns["good_patterns"] = list(set(good_patterns))[:20]
        self._learned_patterns["bad_patterns"] = list(set(bad_patterns))[:20]
        self.pairs_analyzed += len(pairs)

        return {
            "patterns": self._learned_patterns,
            "pairs_analyzed": len(pairs),
            "total_analyzed": self.pairs_analyzed,
        }

    def _extract_patterns(self, code: str, is_good: bool) -> list[str]:
        """
        Extract code patterns from a code sample.

        Args:
            code: Code to analyze
            is_good: Whether this is good (chosen) or bad (rejected) code

        Returns:
            List of pattern strings
        """
        import re

        patterns = []

        # Check for common good/bad patterns
        if is_good:
            # Good patterns
            if "try:" in code and "except" in code:
                patterns.append("has_error_handling")
            if "gc.collect()" in code:
                patterns.append("has_memory_cleanup")
            if "early_stopping" in code.lower():
                patterns.append("has_early_stopping")
            if re.search(r"cv.*=.*\d+", code):
                patterns.append("explicit_cv_folds")
            if "submission.csv" in code and "to_csv" in code:
                patterns.append("saves_submission")
            if "Final Validation Performance" in code:
                patterns.append("prints_cv_score")
        else:
            # Bad patterns (things to avoid)
            if code.count("import") > 20:
                patterns.append("too_many_imports")
            if "raise" in code and "except" not in code:
                patterns.append("unhandled_raise")
            if code.count("\n") > 500:
                patterns.append("excessive_length")
            if "# TODO" in code or "# FIXME" in code:
                patterns.append("has_unfinished_code")

        return patterns

    def score_code(
        self,
        code: str,
        component_type: str,
        context: str | None = None,
    ) -> dict[str, float]:
        """
        Score code based on learned preferences.

        Args:
            code: Code to score
            component_type: Type of component
            context: Optional context for LLM scoring

        Returns:
            Dictionary with score components
        """
        scores = {}

        # Pattern-based scoring
        good_count = 0
        bad_count = 0

        for pattern in self._learned_patterns["good_patterns"]:
            if self._check_pattern(code, pattern):
                good_count += 1

        for pattern in self._learned_patterns["bad_patterns"]:
            if self._check_pattern(code, pattern):
                bad_count += 1

        total_patterns = len(self._learned_patterns["good_patterns"]) + len(
            self._learned_patterns["bad_patterns"]
        )

        if total_patterns > 0:
            # Score: good patterns increase, bad patterns decrease
            pattern_score = (good_count - bad_count * 0.5) / max(total_patterns, 1)
            scores["pattern_score"] = max(0.0, min(pattern_score + 0.5, 1.0))
        else:
            scores["pattern_score"] = 0.5  # Neutral

        # Structure-based scoring
        scores["structure_score"] = self._score_structure(code)

        # LLM-based scoring (if enabled and we have context)
        if self.use_llm and context and self.pairs_analyzed >= 3:
            scores["llm_score"] = self._score_with_llm(code, component_type, context)
        else:
            scores["llm_score"] = scores["pattern_score"]

        # Combined score
        scores["combined"] = (
            0.3 * scores["pattern_score"]
            + 0.3 * scores["structure_score"]
            + 0.4 * scores["llm_score"]
        )

        return scores

    def _check_pattern(self, code: str, pattern: str) -> bool:
        """Check if a pattern exists in code."""
        pattern_checks = {
            "has_error_handling": lambda c: "try:" in c and "except" in c,
            "has_memory_cleanup": lambda c: "gc.collect()" in c,
            "has_early_stopping": lambda c: "early_stopping" in c.lower(),
            "explicit_cv_folds": lambda c: "cv" in c.lower() and "fold" in c.lower(),
            "saves_submission": lambda c: "submission" in c and "to_csv" in c,
            "prints_cv_score": lambda c: "Final Validation Performance" in c,
            "too_many_imports": lambda c: c.count("import") > 20,
            "unhandled_raise": lambda c: "raise" in c and "except" not in c,
            "excessive_length": lambda c: c.count("\n") > 500,
            "has_unfinished_code": lambda c: "# TODO" in c or "# FIXME" in c,
        }

        check_func = pattern_checks.get(pattern)
        if check_func:
            return check_func(code)
        return False

    def _score_structure(self, code: str) -> float:
        """Score code structure quality."""
        score = 0.5  # Base score

        # Positive signals
        if "def " in code:
            score += 0.1  # Has functions
        if "import pandas" in code or "import numpy" in code:
            score += 0.05  # Standard imports
        if "submission" in code.lower():
            score += 0.1  # Creates submission
        if "print(" in code:
            score += 0.05  # Has output for debugging

        # Negative signals
        lines = code.split("\n")
        if len(lines) > 800:
            score -= 0.15  # Too long
        if code.count("pass") > 5:
            score -= 0.1  # Too many empty blocks

        # Check for common issues
        try:
            compile(code, "<string>", "exec")
        except SyntaxError:
            score -= 0.3  # Syntax error

        return max(0.0, min(score, 1.0))

    def _score_with_llm(
        self,
        code: str,
        component_type: str,
        context: str,
    ) -> float:
        """Score code using LLM evaluation."""
        # Truncate code for LLM
        code_preview = code[:2000] + "..." if len(code) > 2000 else code

        prompt = f"""Rate this {component_type} code on a scale of 0.0 to 1.0.

CONTEXT: {context[:500]}

CODE:
```python
{code_preview}
```

Consider:
- Will it execute without errors?
- Does it follow ML best practices?
- Will it produce valid outputs?

Return ONLY a number between 0.0 and 1.0."""

        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            content = get_text_content(response.content).strip()

            # Extract number from response
            import re

            match = re.search(r"(\d+\.?\d*)", content)
            if match:
                score = float(match.group(1))
                return max(0.0, min(score, 1.0))
        except Exception:
            pass

        return 0.5  # Neutral on failure

    def get_guidance_for_prompt(self) -> str:
        """
        Generate guidance text to inject into code generation prompts.

        Returns:
            Guidance string based on learned patterns
        """
        if (
            not self._learned_patterns["good_patterns"]
            and not self._learned_patterns["bad_patterns"]
        ):
            return ""

        guidance_parts = []

        if self._learned_patterns["good_patterns"]:
            good_list = ", ".join(self._learned_patterns["good_patterns"][:5])
            guidance_parts.append(f"✅ INCLUDE: {good_list}")

        if self._learned_patterns["bad_patterns"]:
            bad_list = ", ".join(self._learned_patterns["bad_patterns"][:5])
            guidance_parts.append(f"❌ AVOID: {bad_list}")

        if guidance_parts:
            return "\n## Learned Preferences (DPO)\n" + "\n".join(guidance_parts)

        return ""


# ==================== Factory Functions ====================


def create_preference_collector() -> PreferenceCollector:
    """Create a new preference collector instance."""
    return PreferenceCollector()


def create_preference_reward_model(use_llm: bool = True) -> PreferenceRewardModel:
    """
    Create a new preference reward model.

    Args:
        use_llm: Whether to use LLM for sophisticated scoring

    Returns:
        PreferenceRewardModel instance
    """
    return PreferenceRewardModel(use_llm=use_llm)
