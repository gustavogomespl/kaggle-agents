"""
Robustness Checks for Kaggle Agents.

Provides structured validation results with:
- Detection/correction separation (MLE-STAR pattern)
- Evidence tracking for each check
- Fix tracking for auto-corrected issues
- Blocking vs non-blocking issue classification

MLE-STAR separates detection from correction. We track both:
what was detected AND what fix was applied.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class CheckResult:
    """Result of a single validation check.

    Attributes:
        passed: Whether the check passed
        evidence_ref: Path to evidence/log file
        fix_applied: Whether an auto-fix was applied
        fix_patch_ref: Path to patch/diff that was applied
    """

    passed: bool
    evidence_ref: str | None = None
    fix_applied: bool = False
    fix_patch_ref: str | None = None

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        if self.passed:
            return "CheckResult(PASS)"
        elif self.fix_applied:
            return "CheckResult(FIXED)"
        else:
            return "CheckResult(FAIL)"

    def to_dict(self) -> dict:
        """Serialize for checkpointing."""
        return {
            "passed": self.passed,
            "evidence_ref": self.evidence_ref,
            "fix_applied": self.fix_applied,
            "fix_patch_ref": self.fix_patch_ref,
        }

    @classmethod
    def from_dict(cls, data: dict) -> CheckResult:
        """Deserialize from checkpoint."""
        return cls(
            passed=data["passed"],
            evidence_ref=data.get("evidence_ref"),
            fix_applied=data.get("fix_applied", False),
            fix_patch_ref=data.get("fix_patch_ref"),
        )


@dataclass
class RobustnessChecks:
    """Structured robustness validation results.

    Each check has:
    - Detection (passed/failed)
    - Evidence (log path)
    - Fix tracking (was it auto-corrected?)

    Attributes:
        canonical_contract: Canonical data contract validation
        metric_contract: Metric contract validation
        data_leakage: Data leakage detection
        data_usage: Data usage contract validation
        submission_format: Submission format validation

        canonical_violations: List of canonical contract violations
        metric_violations: List of metric contract violations
        leakage_type: Type of leakage detected (if any)
        leakage_details: Details about the leakage
        missing_columns: Columns missing in predictions
        suspicious_drops: Suspicious row drops detected
        unused_assets: Data assets not used (from DataUsageContract)
    """

    # Check results
    canonical_contract: CheckResult = field(default_factory=lambda: CheckResult(passed=False))
    metric_contract: CheckResult = field(default_factory=lambda: CheckResult(passed=False))
    data_leakage: CheckResult = field(default_factory=lambda: CheckResult(passed=False))
    data_usage: CheckResult = field(default_factory=lambda: CheckResult(passed=False))
    submission_format: CheckResult = field(default_factory=lambda: CheckResult(passed=False))

    # Detailed violations
    canonical_violations: list[str] = field(default_factory=list)
    metric_violations: list[str] = field(default_factory=list)
    leakage_type: str | None = None
    leakage_details: str | None = None
    missing_columns: list[str] = field(default_factory=list)
    suspicious_drops: list[str] = field(default_factory=list)
    unused_assets: list[str] = field(default_factory=list)

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        if self.all_passed():
            return "RobustnessChecks(ALL_PASS)"
        elif self.critical_passed():
            n_issues = len(self.blocking_issues())
            return f"RobustnessChecks(CRITICAL_PASS, {n_issues} warnings)"
        else:
            n_blocking = len(self.blocking_issues())
            return f"RobustnessChecks(BLOCKED, {n_blocking} issues)"

    def all_passed(self) -> bool:
        """Check if all validations passed."""
        return all([
            self.canonical_contract.passed,
            self.metric_contract.passed,
            self.data_leakage.passed,
            self.data_usage.passed,
            self.submission_format.passed,
        ])

    def critical_passed(self) -> bool:
        """Check if critical validations passed (canonical, leakage, submission)."""
        return all([
            self.canonical_contract.passed or self.canonical_contract.fix_applied,
            self.data_leakage.passed or self.data_leakage.fix_applied,
            self.submission_format.passed,
        ])

    def blocking_issues(self) -> list[str]:
        """Get issues that should block progression (not auto-fixable).

        Returns:
            List of blocking issue descriptions
        """
        issues = []

        if not self.canonical_contract.passed and not self.canonical_contract.fix_applied:
            if self.canonical_violations:
                issues.append(f"Canonical contract violated (unfixed): {self.canonical_violations}")
            else:
                issues.append("Canonical contract violated (unfixed)")

        if not self.metric_contract.passed:
            if self.metric_violations:
                issues.append(f"Metric contract violated: {self.metric_violations}")
            else:
                issues.append("Metric contract violated")

        if not self.data_leakage.passed and not self.data_leakage.fix_applied:
            details = f": {self.leakage_type}" if self.leakage_type else ""
            issues.append(f"Data leakage detected (unfixed){details}")

        if not self.submission_format.passed:
            issues.append("Submission format validation failed")

        return issues

    def auto_fixed_issues(self) -> list[str]:
        """Get issues that were detected AND auto-fixed.

        Returns:
            List of auto-fixed issue descriptions
        """
        fixed = []

        if self.canonical_contract.fix_applied:
            ref = f" via {self.canonical_contract.fix_patch_ref}" if self.canonical_contract.fix_patch_ref else ""
            fixed.append(f"Canonical contract{ref}")

        if self.data_leakage.fix_applied:
            ref = f" via {self.data_leakage.fix_patch_ref}" if self.data_leakage.fix_patch_ref else ""
            fixed.append(f"Data leakage{ref}")

        return fixed

    def warnings(self) -> list[str]:
        """Get non-blocking warnings.

        Returns:
            List of warning descriptions
        """
        warnings = []

        if not self.data_usage.passed and self.unused_assets:
            warnings.append(f"Unused data assets: {self.unused_assets}")

        if self.suspicious_drops:
            warnings.append(f"Suspicious row drops: {self.suspicious_drops}")

        return warnings

    def get_summary(self) -> str:
        """Get a human-readable summary of validation results."""
        lines = ["## Robustness Validation Summary"]

        # Overall status
        if self.all_passed():
            lines.append("\nStatus: ALL CHECKS PASSED")
        elif self.critical_passed():
            lines.append("\nStatus: CRITICAL CHECKS PASSED (some warnings)")
        else:
            lines.append("\nStatus: BLOCKING ISSUES DETECTED")

        # Check details
        checks = [
            ("Canonical Contract", self.canonical_contract),
            ("Metric Contract", self.metric_contract),
            ("Data Leakage", self.data_leakage),
            ("Data Usage", self.data_usage),
            ("Submission Format", self.submission_format),
        ]

        lines.append("\n### Check Results")
        for name, check in checks:
            status = "PASS" if check.passed else ("FIXED" if check.fix_applied else "FAIL")
            lines.append(f"- {name}: {status}")

        # Blocking issues
        blocking = self.blocking_issues()
        if blocking:
            lines.append("\n### Blocking Issues")
            for issue in blocking:
                lines.append(f"- {issue}")

        # Auto-fixed
        fixed = self.auto_fixed_issues()
        if fixed:
            lines.append("\n### Auto-Fixed Issues")
            for issue in fixed:
                lines.append(f"- {issue}")

        # Warnings
        warns = self.warnings()
        if warns:
            lines.append("\n### Warnings")
            for warn in warns:
                lines.append(f"- {warn}")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize for checkpointing."""
        return {
            "canonical_contract": self.canonical_contract.to_dict(),
            "metric_contract": self.metric_contract.to_dict(),
            "data_leakage": self.data_leakage.to_dict(),
            "data_usage": self.data_usage.to_dict(),
            "submission_format": self.submission_format.to_dict(),
            "canonical_violations": self.canonical_violations,
            "metric_violations": self.metric_violations,
            "leakage_type": self.leakage_type,
            "leakage_details": self.leakage_details,
            "missing_columns": self.missing_columns,
            "suspicious_drops": self.suspicious_drops,
            "unused_assets": self.unused_assets,
        }

    @classmethod
    def from_dict(cls, data: dict) -> RobustnessChecks:
        """Deserialize from checkpoint."""
        return cls(
            canonical_contract=CheckResult.from_dict(data.get("canonical_contract", {"passed": False})),
            metric_contract=CheckResult.from_dict(data.get("metric_contract", {"passed": False})),
            data_leakage=CheckResult.from_dict(data.get("data_leakage", {"passed": False})),
            data_usage=CheckResult.from_dict(data.get("data_usage", {"passed": False})),
            submission_format=CheckResult.from_dict(data.get("submission_format", {"passed": False})),
            canonical_violations=data.get("canonical_violations", []),
            metric_violations=data.get("metric_violations", []),
            leakage_type=data.get("leakage_type"),
            leakage_details=data.get("leakage_details"),
            missing_columns=data.get("missing_columns", []),
            suspicious_drops=data.get("suspicious_drops", []),
            unused_assets=data.get("unused_assets", []),
        )
