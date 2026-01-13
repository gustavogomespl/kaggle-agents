"""
Time Budget Management for Kaggle Agents.

Provides time-aware training utilities to prevent timeouts on large datasets:
- Pre-training time estimation via calibration samples
- Progressive training schedules (10% -> 25% -> 50% -> 100%)
- Time-aware early stopping callbacks for LightGBM/XGBoost
- Safe sample fraction calculation given time constraints

Research: FLAML achieves same performance with 10% computation via cost-frugal optimization.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass
class TimeEstimate:
    """Result of time estimation calibration."""

    calibration_time: float  # Time to train on sample
    sample_size: int
    full_size: int
    estimated_full_time: float
    buffer_factor: float = 1.2
    model_type: str = "unknown"

    @property
    def scale_factor(self) -> float:
        """Estimated scaling factor from sample to full dataset."""
        return self.full_size / self.sample_size

    def get_safe_sample_fraction(self, time_budget: float, safety_margin: float = 0.8) -> float:
        """Calculate safe sample fraction given time budget.

        Args:
            time_budget: Available time in seconds
            safety_margin: Fraction of budget to use (default 80%)

        Returns:
            Fraction of data to use (0.0 to 1.0)
        """
        available_time = time_budget * safety_margin
        if self.estimated_full_time <= available_time:
            return 1.0
        return available_time / self.estimated_full_time


class TimeBudgetManager:
    """Manage training time budget with estimation and progressive scheduling.

    Usage:
        manager = TimeBudgetManager(time_budget=600)  # 10 minutes
        estimate = manager.estimate_training_time(X, y, model_type="lightgbm")

        if estimate.estimated_full_time > manager.time_budget * 0.8:
            sample_frac = estimate.get_safe_sample_fraction(manager.time_budget)
            X = X.sample(frac=sample_frac, random_state=42)
    """

    # Scaling factors for different model types (empirically determined)
    MODEL_SCALING_FACTORS = {
        "lightgbm": 1.0,  # LightGBM scales linearly due to histogram binning
        "xgboost": 1.0,  # XGBoost also roughly linear with histograms
        "catboost": 1.2,  # CatBoost slightly slower scaling
        "random_forest": 1.5,  # RF can be quadratic in some cases
        "neural_network": 0.8,  # NN batched, sublinear scaling
        "linear": 0.5,  # Linear models very efficient
        "unknown": 1.2,  # Conservative default
    }

    def __init__(
        self,
        time_budget: float,
        calibration_sample_pct: float = 0.01,
        min_calibration_samples: int = 1000,
        max_calibration_samples: int = 10000,
    ):
        """Initialize time budget manager.

        Args:
            time_budget: Total time budget in seconds
            calibration_sample_pct: Percentage of data for calibration (default 1%)
            min_calibration_samples: Minimum calibration samples
            max_calibration_samples: Maximum calibration samples
        """
        self.time_budget = time_budget
        self.calibration_sample_pct = calibration_sample_pct
        self.min_calibration_samples = min_calibration_samples
        self.max_calibration_samples = max_calibration_samples
        self.start_time = time.time()
        self._estimates: list[TimeEstimate] = []

    @property
    def elapsed_time(self) -> float:
        """Time elapsed since manager was created."""
        return time.time() - self.start_time

    @property
    def remaining_time(self) -> float:
        """Remaining time in budget."""
        return max(0, self.time_budget - self.elapsed_time)

    def get_calibration_size(self, n_samples: int) -> int:
        """Calculate calibration sample size.

        Args:
            n_samples: Total number of samples

        Returns:
            Number of samples for calibration
        """
        size = int(n_samples * self.calibration_sample_pct)
        return max(self.min_calibration_samples, min(size, self.max_calibration_samples))

    def estimate_training_time(
        self,
        X: np.ndarray,
        y: np.ndarray,
        model_type: str = "unknown",
        train_fn: Callable[[np.ndarray, np.ndarray], Any] | None = None,
    ) -> TimeEstimate:
        """Estimate full training time by running on calibration sample.

        Args:
            X: Feature matrix
            y: Target vector
            model_type: Type of model (lightgbm, xgboost, etc.)
            train_fn: Optional training function for actual calibration

        Returns:
            TimeEstimate with calibration results
        """
        n_samples = len(X)
        calibration_size = self.get_calibration_size(n_samples)

        # Use indices for sampling to avoid copying large arrays
        indices = np.random.choice(n_samples, size=calibration_size, replace=False)
        X_sample = X[indices]
        y_sample = y[indices]

        if train_fn is not None:
            # Actual calibration with provided training function
            start = time.time()
            train_fn(X_sample, y_sample)
            calibration_time = time.time() - start
        else:
            # Heuristic estimation based on model type
            # Simulate with simple operation
            start = time.time()
            # Simple operation to estimate overhead
            _ = np.dot(X_sample.T, X_sample) if X_sample.ndim > 1 else np.sum(X_sample)
            calibration_time = (time.time() - start) * 10  # Rough scaling

        # Calculate estimated full training time
        scale_factor = n_samples / calibration_size
        model_factor = self.MODEL_SCALING_FACTORS.get(model_type.lower(), 1.2)
        buffer_factor = 1.2  # 20% buffer for safety

        estimated_full = calibration_time * scale_factor * model_factor * buffer_factor

        estimate = TimeEstimate(
            calibration_time=calibration_time,
            sample_size=calibration_size,
            full_size=n_samples,
            estimated_full_time=estimated_full,
            buffer_factor=buffer_factor,
            model_type=model_type,
        )

        self._estimates.append(estimate)
        return estimate

    def create_progressive_schedule(
        self,
        min_fraction: float = 0.1,
        max_fraction: float = 1.0,
        n_stages: int = 4,
    ) -> list[float]:
        """Create progressive training schedule.

        Research shows 30% samples can retain 95% of accuracy with 100x speedup.

        Args:
            min_fraction: Minimum sample fraction (default 10%)
            max_fraction: Maximum sample fraction (default 100%)
            n_stages: Number of stages in schedule

        Returns:
            List of sample fractions [0.1, 0.25, 0.5, 1.0]
        """
        if n_stages <= 1:
            return [max_fraction]

        # Logarithmic spacing for progressive training
        log_min = np.log(min_fraction)
        log_max = np.log(max_fraction)
        fractions = np.exp(np.linspace(log_min, log_max, n_stages))

        return fractions.tolist()

    def get_adaptive_schedule(
        self,
        estimated_full_time: float,
        safety_margin: float = 0.8,
    ) -> list[float]:
        """Get adaptive schedule based on time budget.

        Args:
            estimated_full_time: Estimated time for full training
            safety_margin: Fraction of budget to use

        Returns:
            List of sample fractions that fit within budget
        """
        available_time = self.remaining_time * safety_margin

        if estimated_full_time <= available_time:
            # Full training fits in budget
            return [1.0]

        # Calculate how many stages we can fit
        base_fractions = self.create_progressive_schedule()
        feasible = []

        cumulative_time = 0
        for frac in base_fractions:
            stage_time = estimated_full_time * frac
            if cumulative_time + stage_time <= available_time:
                feasible.append(frac)
                cumulative_time += stage_time
            else:
                # Calculate maximum fraction that fits
                remaining = available_time - cumulative_time
                max_frac = remaining / estimated_full_time
                if max_frac > 0.05:  # At least 5% to be useful
                    feasible.append(max_frac)
                break

        return feasible if feasible else [0.1]  # Always train on at least 10%

    def should_continue(self, buffer_seconds: float = 60) -> bool:
        """Check if there's enough time to continue training.

        Args:
            buffer_seconds: Buffer time to keep for saving/cleanup

        Returns:
            True if training should continue
        """
        return self.remaining_time > buffer_seconds

    def get_remaining_iterations(
        self,
        avg_iteration_time: float,
        buffer_pct: float = 0.15,
    ) -> int:
        """Calculate how many more iterations can safely run.

        Args:
            avg_iteration_time: Average time per iteration
            buffer_pct: Percentage of remaining time to keep as buffer

        Returns:
            Number of safe iterations
        """
        available = self.remaining_time * (1 - buffer_pct)
        if avg_iteration_time <= 0:
            return 1000  # Fallback
        return max(0, int(available / avg_iteration_time))

    def log_estimate(self, estimate: TimeEstimate) -> None:
        """Log time estimation for debugging."""
        print("\n   TIME BUDGET ESTIMATION:")
        print(f"      Calibration: {estimate.calibration_time:.2f}s on {estimate.sample_size:,} samples")
        print(f"      Estimated full: {estimate.estimated_full_time:.2f}s on {estimate.full_size:,} samples")
        print(f"      Time budget: {self.time_budget:.2f}s, remaining: {self.remaining_time:.2f}s")

        if estimate.estimated_full_time > self.time_budget * 0.8:
            safe_frac = estimate.get_safe_sample_fraction(self.time_budget)
            print("      WARNING: Full training exceeds budget!")
            print(f"      Recommended sample fraction: {safe_frac:.1%}")


class TimeAwareEarlyStopping:
    """Early stopping callback that respects time budget.

    Compatible with LightGBM and XGBoost callback interfaces.

    Usage:
        callback = TimeAwareEarlyStopping(time_budget=300, buffer_pct=0.15)
        model = lgb.train(params, train_data, callbacks=[callback])
    """

    def __init__(
        self,
        time_budget: float,
        buffer_pct: float = 0.15,
        min_iterations: int = 10,
        verbose: bool = True,
    ):
        """Initialize time-aware early stopping.

        Args:
            time_budget: Total time budget in seconds
            buffer_pct: Percentage of budget to keep as buffer (default 15%)
            min_iterations: Minimum iterations before checking time
            verbose: Whether to print messages
        """
        self.time_budget = time_budget
        self.buffer_pct = buffer_pct
        self.min_iterations = min_iterations
        self.verbose = verbose
        self.start_time = time.time()
        self.iteration_times: list[float] = []
        self.last_iteration_time = self.start_time

    @property
    def elapsed_time(self) -> float:
        """Time elapsed since training started."""
        return time.time() - self.start_time

    @property
    def remaining_time(self) -> float:
        """Remaining time in budget."""
        return max(0, self.time_budget - self.elapsed_time)

    def _should_stop(self, iteration: int, total_iterations: int) -> bool:
        """Check if training should stop due to time constraints."""
        if iteration < self.min_iterations:
            return False

        current_time = time.time()
        iter_time = current_time - self.last_iteration_time
        self.iteration_times.append(iter_time)
        self.last_iteration_time = current_time

        # Calculate average iteration time (use recent iterations for better estimate)
        recent_times = self.iteration_times[-10:]
        avg_iter_time = sum(recent_times) / len(recent_times)

        # Calculate remaining iterations
        remaining_iters = total_iterations - iteration
        estimated_remaining_time = avg_iter_time * remaining_iters

        # Check if we have enough time with buffer
        available_time = self.remaining_time * (1 - self.buffer_pct)

        if estimated_remaining_time > available_time:
            if self.verbose:
                print(f"\n   TIME-AWARE EARLY STOPPING at iteration {iteration}")
                print(f"      Elapsed: {self.elapsed_time:.1f}s, Remaining budget: {self.remaining_time:.1f}s")
                print(f"      Avg iter time: {avg_iter_time:.3f}s, Remaining iters: {remaining_iters}")
                print(f"      Would need: {estimated_remaining_time:.1f}s, Available: {available_time:.1f}s")
            return True

        return False

    # LightGBM callback interface
    def __call__(self, env):
        """LightGBM callback interface."""
        if self._should_stop(env.iteration, env.end_iteration):
            raise EarlyStopException(env.iteration, env.evaluation_result_list)

    # XGBoost callback interface
    def after_iteration(self, model, epoch: int, evals_log: dict) -> bool:
        """XGBoost callback interface.

        Returns:
            True to stop training, False to continue
        """
        # XGBoost doesn't provide total iterations, estimate from current progress
        # Assume we're in the middle if no other info available
        estimated_total = epoch * 2 if epoch < 100 else epoch + 100
        return self._should_stop(epoch, estimated_total)


class EarlyStopException(Exception):
    """Exception raised for time-aware early stopping."""

    def __init__(self, iteration: int, best_score: Any = None):
        self.iteration = iteration
        self.best_score = best_score
        super().__init__(f"Early stopping at iteration {iteration}")


def create_time_aware_training_code(time_budget: float) -> str:
    """Generate code snippet for time-aware training.

    This can be injected into generated model training code.

    Args:
        time_budget: Time budget in seconds

    Returns:
        Python code snippet as string
    """
    return f'''
# === TIME-AWARE TRAINING (AUTO-INJECTED) ===
import time
TIME_BUDGET = {time_budget}
_training_start_time = time.time()

def _check_time_budget(buffer_seconds=60):
    """Check if there's enough time to continue."""
    elapsed = time.time() - _training_start_time
    remaining = TIME_BUDGET - elapsed
    if remaining < buffer_seconds:
        print(f"[LOG:WARNING] Time budget nearly exhausted: {{remaining:.1f}}s remaining")
        return False
    return True

def _estimate_training_time(X, y, model, sample_pct=0.01):
    """Estimate training time on 1% sample."""
    sample_size = max(1000, int(len(X) * sample_pct))
    X_sample, y_sample = X[:sample_size], y[:sample_size]

    start = time.time()
    model.fit(X_sample, y_sample)
    calibration_time = time.time() - start

    scale_factor = len(X) / sample_size
    estimated_full = calibration_time * scale_factor * 1.2  # 20% buffer
    return estimated_full, calibration_time

def _get_safe_sample_fraction(estimated_time, safety_margin=0.8):
    """Calculate safe sample fraction given time estimate."""
    remaining = TIME_BUDGET - (time.time() - _training_start_time)
    available = remaining * safety_margin
    if estimated_time <= available:
        return 1.0
    return available / estimated_time
# === END TIME-AWARE TRAINING ===
'''
