"""Ensemble agent for model stacking and blending."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import cross_val_predict

from ..core.config import get_config, is_metric_minimization
from ..core.state import KaggleState
from ..utils.llm_utils import get_text_content


class EnsembleAgent:
    """Agent responsible for creating model ensembles."""

    def __init__(self):
        """Initialize ensemble agent."""
        pass

    def _find_prediction_pairs(self, models_dir: Path) -> dict[str, tuple[Path, Path]]:
        """Find matching OOF/Test prediction pairs under models/."""
        oof_files = sorted(models_dir.glob("oof_*.npy"))
        pairs: dict[str, tuple[Path, Path]] = {}
        for oof_path in oof_files:
            name = oof_path.stem.replace("oof_", "", 1)
            test_path = models_dir / f"test_{name}.npy"
            if test_path.exists():
                pairs[name] = (oof_path, test_path)
        return pairs

    def _ensemble_from_predictions(
        self,
        prediction_pairs: dict[str, tuple[Path, Path]],
        sample_submission_path: Path,
        output_path: Path,
    ) -> bool:
        """Create a simple average ensemble directly from saved predictions."""
        if not sample_submission_path.exists():
            print("   ❌ Sample submission not found, cannot build prediction ensemble")
            return False

        sample_sub = pd.read_csv(sample_submission_path)
        preds_list = []
        names = []

        for name, (_, test_path) in prediction_pairs.items():
            preds = np.load(test_path)
            preds = np.asarray(preds, dtype=np.float32)
            if preds.ndim == 1:
                preds = preds.reshape(-1, 1)
            preds_list.append(preds)
            names.append(name)

        if len(preds_list) < 2:
            print("   ⚠️  Not enough prediction pairs for ensemble")
            return False

        # Ensure consistent shapes
        shapes = {p.shape for p in preds_list}
        if len(shapes) != 1:
            print(f"   ❌ Prediction shapes mismatch: {shapes}")
            return False

        stacked = np.stack(preds_list, axis=0)  # (n_models, n_samples, n_cols)
        ensemble_preds = stacked.mean(axis=0)

        if ensemble_preds.shape[0] != len(sample_sub):
            print(
                f"   ❌ Prediction length mismatch: preds={ensemble_preds.shape[0]}, sample={len(sample_sub)}"
            )
            return False

        if ensemble_preds.shape[1] == 1:
            sample_sub.iloc[:, 1] = ensemble_preds[:, 0]
        else:
            if ensemble_preds.shape[1] != (len(sample_sub.columns) - 1):
                print(
                    "   ❌ Prediction column mismatch: "
                    f"preds={ensemble_preds.shape[1]}, sample_cols={len(sample_sub.columns) - 1}"
                )
                return False
            sample_sub.iloc[:, 1:] = ensemble_preds

        output_path.parent.mkdir(parents=True, exist_ok=True)
        sample_sub.to_csv(output_path, index=False)
        print(f"   ✅ Saved prediction-only ensemble to {output_path.name}")
        print(f"   ✅ Models used: {', '.join(names)}")
        return True

    def create_stacking_ensemble(
        self,
        models: list[Any],
        model_names: list[str],
        working_dir: Path,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str,
    ) -> Any:
        """Create stacking ensemble from best models using saved OOF predictions.

        Args:
            models: List of trained models
            model_names: List of model names
            working_dir: Working directory
            X: Feature matrix
            y: Target variable
            problem_type: 'classification' or 'regression'

        Returns:
            Trained meta-model
        """
        # Generate out-of-fold predictions from base models
        print(f"  Creating stacking ensemble with {len(models)} base models...")

        meta_features = []
        valid_models = []
        valid_names = []

        for i, (model, name) in enumerate(zip(models, model_names, strict=False)):
            print(f"    Processing model {i + 1}/{len(models)}: {name}")

            # Try to load OOF predictions
            oof_path = working_dir / "models" / f"oof_{name}.npy"
            if oof_path.exists():
                print(f"      ✅ Loaded OOF from {oof_path.name}")
                oof_preds = np.load(oof_path)
                meta_features.append(oof_preds)
                valid_models.append(model)
                valid_names.append(name)
            else:
                print("      ⚠️  OOF file not found, falling back to cross_val_predict (slow)")
                if problem_type == "classification":
                    oof_preds = cross_val_predict(
                        model, X, y, cv=5, method="predict_proba", n_jobs=-1
                    )
                    # Take probabilities for positive class
                    if oof_preds.ndim > 1:
                        meta_features.append(oof_preds[:, 1])
                    else:
                        meta_features.append(oof_preds)
                else:
                    oof_preds = cross_val_predict(model, X, y, cv=5, n_jobs=-1)
                    meta_features.append(oof_preds)

                valid_models.append(model)
                valid_names.append(name)

        if not meta_features:
            raise ValueError("No meta-features could be generated")

        # Create meta-feature matrix
        meta_X = np.column_stack(meta_features)

        # Train meta-model
        print("    Training meta-model (LogisticRegression/Ridge)...")
        if problem_type == "classification":
            meta_model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            meta_model = Ridge(random_state=42)

        meta_model.fit(meta_X, y)

        # We don't need to retrain base models if we use the saved test preds!
        # But we keep them in the return dict for completeness

        return {
            "meta_model": meta_model,
            "base_models": valid_models,
            "base_model_names": valid_names,
        }

    def create_blending_ensemble(
        self,
        models: list[Any],
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str,
    ) -> dict[str, Any]:
        """Create blending ensemble using simple averaging.

        Args:
            models: List of trained models
            X: Feature matrix
            y: Target variable
            problem_type: 'classification' or 'regression'

        Returns:
            Dictionary with base models and weights
        """
        print(f"  Creating blending ensemble with {len(models)} models...")

        # Optimize weights
        weights = self.optimize_blending_weights(models, X, y, problem_type)

        return {"base_models": models, "weights": weights}

    def optimize_blending_weights(
        self,
        models: list[Any],
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str,
    ) -> list[float]:
        """Optimize blending weights using scipy.minimize."""
        from scipy.optimize import minimize
        from sklearn.metrics import log_loss, mean_squared_error

        print("    Optimizing blending weights...")

        # Generate OOF predictions
        oof_preds = []
        for model in models:
            if problem_type == "classification":
                preds = cross_val_predict(model, X, y, cv=5, method="predict_proba", n_jobs=-1)
                if preds.ndim > 1:
                    oof_preds.append(preds[:, 1])
                else:
                    oof_preds.append(preds)
            else:
                preds = cross_val_predict(model, X, y, cv=5, n_jobs=-1)
                oof_preds.append(preds)

        oof_preds = np.column_stack(oof_preds)

        # Define loss function
        def loss_func(weights):
            # Normalize weights
            weights = np.array(weights)
            weights /= weights.sum()

            # Weighted average
            final_preds = np.average(oof_preds, axis=1, weights=weights)

            if problem_type == "classification":
                # Clip to avoid log(0)
                final_preds = np.clip(final_preds, 1e-15, 1 - 1e-15)
                return log_loss(y, final_preds)
            return np.sqrt(mean_squared_error(y, final_preds))

        # Initial weights (equal)
        init_weights = [1.0 / len(models)] * len(models)

        # Constraints: weights sum to 1, 0 <= weight <= 1
        constraints = {"type": "eq", "fun": lambda w: 1 - sum(w)}
        bounds = [(0, 1)] * len(models)

        result = minimize(
            loss_func, init_weights, method="SLSQP", bounds=bounds, constraints=constraints
        )

        opt_weights = result.x / result.x.sum()
        print(f"    Optimal weights: {opt_weights}")
        opt_weights = result.x / result.x.sum()
        print(f"    Optimal weights: {opt_weights}")
        return opt_weights.tolist()

    def create_caruana_ensemble(
        self,
        models: list[Any],
        model_names: list[str],
        working_dir: Path,
        X: pd.DataFrame,
        y: pd.Series,
        problem_type: str,
        n_iterations: int = 100,
    ) -> dict[str, Any]:
        """
        Create ensemble using Caruana's Hill Climbing (Forward Selection).

        Iteratively adds the model that maximizes the ensemble's CV score.
        Allows repetition of models (weighted ensemble by count).
        """
        from sklearn.metrics import log_loss, mean_squared_error

        print(f"  Creating Caruana Ensemble (Hill Climbing) with {len(models)} models...")

        # Load OOFs
        oof_preds = []
        valid_models = []
        valid_names = []

        for i, (model, name) in enumerate(zip(models, model_names, strict=False)):
            oof_path = working_dir / "models" / f"oof_{name}.npy"
            if oof_path.exists():
                preds = np.load(oof_path)
                oof_preds.append(preds)
                valid_models.append(model)
                valid_names.append(name)
            else:
                print(f"    ⚠️ Skipping {name} (no OOF found)")

        if not oof_preds:
            raise ValueError("No OOF predictions found for Caruana ensemble")

        oof_preds = np.column_stack(oof_preds)
        n_models = oof_preds.shape[1]

        # Metric function
        def get_score(y_true, y_pred):
            if problem_type == "classification":
                # Assuming AUC for classification if not specified, or LogLoss
                # Let's use LogLoss for optimization as it's differentiable/smooth
                y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
                return -log_loss(y_true, y_pred)  # Maximize negative log loss
            return -np.sqrt(mean_squared_error(y_true, y_pred))  # Maximize negative RMSE

        # Hill Climbing
        current_ensemble_preds = np.zeros_like(oof_preds[:, 0])
        ensemble_counts = np.zeros(n_models, dtype=int)
        best_score = -float("inf")

        # Initial step: pick best single model
        for i in range(n_models):
            score = get_score(y, oof_preds[:, i])
            if score > best_score:
                best_score = score
                best_init_idx = i

        current_ensemble_preds = oof_preds[:, best_init_idx]
        ensemble_counts[best_init_idx] = 1

        print(f"    Init Best Score: {best_score:.4f} (Model: {valid_names[best_init_idx]})")

        # Iterations
        for it in range(n_iterations):
            best_iter_score = -float("inf")
            best_iter_idx = -1

            # Try adding each model
            current_size = it + 2  # +1 for init, +1 for current iter (1-based)

            for i in range(n_models):
                # Candidate: current sum + new model prediction
                # Average = (current_sum + new_pred) / current_size
                # But we maintain sum for efficiency?
                # Actually, let's maintain the running sum to avoid re-summing everything
                # current_ensemble_preds is currently the AVERAGE.
                # So convert back to sum: current_avg * (current_size - 1)

                current_sum = current_ensemble_preds * (current_size - 1)
                candidate_avg = (current_sum + oof_preds[:, i]) / current_size

                score = get_score(y, candidate_avg)
                if score > best_iter_score:
                    best_iter_score = score
                    best_iter_idx = i

            # Update best
            if best_iter_score > best_score:
                best_score = best_iter_score
                ensemble_counts[best_iter_idx] += 1
                current_ensemble_preds = (
                    current_ensemble_preds * (current_size - 1) + oof_preds[:, best_iter_idx]
                ) / current_size
                # print(f"    Iter {it+1}: Added {valid_names[best_iter_idx]} -> Score: {best_score:.4f}")
            else:
                # If no improvement, should we stop? Caruana usually continues to smooth out
                # But for simplicity, we can continue or stop. Let's continue.
                ensemble_counts[best_iter_idx] += 1
                current_ensemble_preds = (
                    current_ensemble_preds * (current_size - 1) + oof_preds[:, best_iter_idx]
                ) / current_size

        # Calculate final weights
        weights = ensemble_counts / ensemble_counts.sum()
        print(f"    Final Caruana Weights: {weights}")

        return {
            "base_models": valid_models,
            "base_model_names": valid_names,
            "weights": weights.tolist(),
        }

    def create_temporal_ensemble(
        self,
        working_dir: Path,
        submissions: list[Any],  # List[SubmissionResult]
        current_iteration: int,
        metric_name: str,
    ) -> bool:
        """
        Create Temporal Ensemble (Success Memory) by blending past best submissions.
        Strategies:
        1. Rank Averaging (Robust) - Primary
        2. Weighted Blending (Fallback)

        Args:
            working_dir: Path to working directory
            submissions: List of SubmissionResult objects
            current_iteration: Current iteration number
            metric_name: Name of the evaluation metric (to determine sort direction)

        Returns:
            True if ensemble created and saved as submission.csv
        """
        print(f"\n  ⏳ Temporal Ensemble (Iteration {current_iteration})")

        # Determine strict direction
        minimize = is_metric_minimization(metric_name)
        print(f"      Metric: {metric_name} (Minimize: {minimize})")

        # 1. Gather candidate files
        candidates = []

        # From state history
        valid_history = [
            s
            for s in submissions
            if s.file_path and Path(s.file_path).exists() and s.public_score is not None
        ]

        # Also scan directory for manual matches (recovered state)
        for f in working_dir.glob("submission_iter_*_score_*.csv"):
            if f.name not in [Path(s.file_path).name for s in valid_history]:
                try:
                    # Parse score from filename: submission_iter_X_score_0.1234.csv
                    parts = f.stem.split("_")
                    if "score" in parts:
                        score_idx = parts.index("score") + 1
                        score = float(parts[score_idx])
                        candidates.append({"path": f, "score": score})
                except Exception:
                    continue

        # Convert history to uniform dict
        for sub in valid_history:
            candidates.append({"path": Path(sub.file_path), "score": sub.public_score})

        # Deduplicate by path
        unique_candidates = {str(c["path"]): c for c in candidates}.values()
        candidates = list(unique_candidates)

        if len(candidates) < 2:
            print(
                f"      Running single model (History: {len(candidates)}), needs 2+ for ensemble."
            )
            return False

        # Sort by score (Assume HIGHER is better for selection logic, we will check metric later)
        # Actually simplest heuristic: take top 3 distinct files
        # We don't know metric direction here easily, but usually MLE-bench scores are "higher=better" implies internal conversion?
        # Let's assume standard kaggle logic: we need to know metric.
        # SAFE FALLBACK: Just take the *last* 3 iterations as they should be improving?
        # BETTER: Sort by score descending (assuming AUC/Acc) or ascending (RMSE/LogLoss)??
        # CRITICAL: We need metric direction. But Rank Averaging is robust to scale, not direction if sorted wrong.
        # Let's use the explicit 'best_score' tracking in state to know which submissions were "improvements".
        # Filter candidates to only those that were considered "best" at their time?
        # SIMPLIFICATION: Just take the top 3 available files. Assuming 'score' in filename is meaningful.

        # Sort logic based on metric direction
        # If minimize (RMSE, LogLoss): asc=True (lower score is better)
        # If maximize (AUC, Accuracy): asc=False (higher score is better)
        # We want the BEST files first.
        # So for minimize: sort by score ASC.
        # For maximize: sort by score DESC (reverse=True).
        reverse_sort = not minimize

        sorted_candidates = sorted(candidates, key=lambda x: x["score"], reverse=reverse_sort)
        top_k = sorted_candidates[:3]

        print(f"      Blending Top {len(top_k)} past submissions:")
        dfs = []
        for c in top_k:
            print(f"      - {c['path'].name} (Score: {c['score']:.4f})")
            try:
                df = pd.read_csv(c["path"])
                # Sort by ID to ensure alignment
                if "id" in df.columns:
                    df = df.sort_values("id")
                dfs.append(df)
            except Exception as e:
                print(f"        ⚠️ Failed to read: {e}")

        if not dfs:
            return False

        # Rank Averaging
        # 1. Convert predictions to Ranks (0..1)
        # 2. Average Ranks
        # 3. (Optional) Map back to distribution? Or just use Scaled Rank as prob?
        # For submission, we need actual values.
        # If Regression: Average Values.
        # If Classification (Probs): Average Probs.
        # Rank Averaging is mostly for ROC-AUC / Ranking metrics.
        # Let's stick to SIMPLE WEIGHTED BLENDING based on rank (1st gets 50%, 2nd 30%, 3rd 20%)

        try:
            sample = dfs[0]
            if len(sample.columns) < 2:
                return False
            pred_col = sample.columns[1]

            # Weighted Average
            # Weights: 1st=3, 2nd=2, 3rd=1 (normalized)
            weights = np.array([3.0, 2.0, 1.0])[: len(dfs)]
            weights /= weights.sum()

            print(f"      Weights: {weights}")

            final_preds = np.zeros_like(sample[pred_col], dtype=float)

            for i, df in enumerate(dfs):
                vals = df[pred_col].values
                # Sanity fill NaNs
                vals = np.nan_to_num(vals)
                final_preds += vals * weights[i]

            # Save
            submission_path = working_dir / "submission.csv"
            out_df = sample.copy()
            out_df[pred_col] = final_preds
            out_df.to_csv(submission_path, index=False)
            print(f"      ✅ Saved Temporal Ensemble to {submission_path}")
            return True

        except Exception as e:
            print(f"      ❌ Temporal Ensemble Failed: {e}")
            return False

    def predict_stacking(
        self, ensemble: dict[str, Any], X: pd.DataFrame, problem_type: str
    ) -> np.ndarray:
        """Make predictions using stacking ensemble.

        Args:
            ensemble: Ensemble dictionary with meta_model and base_models
            X: Feature matrix
            problem_type: 'classification' or 'regression'

        Returns:
            Predictions
        """
        base_models = ensemble["base_models"]
        meta_model = ensemble["meta_model"]

        # Generate predictions from base models
        meta_features = []
        for model in base_models:
            if problem_type == "classification":
                if hasattr(model, "predict_proba"):
                    preds = model.predict_proba(X)
                    if preds.ndim > 1:
                        meta_features.append(preds[:, 1])
                    else:
                        meta_features.append(preds)
                else:
                    meta_features.append(model.predict(X))
            else:
                meta_features.append(model.predict(X))

        # Create meta-features
        meta_X = np.column_stack(meta_features)

        # Predict with meta-model
        if problem_type == "classification" and hasattr(meta_model, "predict_proba"):
            return meta_model.predict_proba(meta_X)[:, 1]
        return meta_model.predict(meta_X)

    def predict_blending(
        self, ensemble: dict[str, Any], X: pd.DataFrame, problem_type: str
    ) -> np.ndarray:
        """Make predictions using blending ensemble.

        Args:
            ensemble: Ensemble dictionary with base_models and weights
            X: Feature matrix
            problem_type: 'classification' or 'regression'

        Returns:
            Predictions
        """
        base_models = ensemble["base_models"]
        weights = ensemble["weights"]

        predictions = []
        for model in base_models:
            if problem_type == "classification":
                if hasattr(model, "predict_proba"):
                    preds = model.predict_proba(X)
                    if preds.ndim > 1:
                        predictions.append(preds[:, 1])
                    else:
                        predictions.append(preds)
                else:
                    predictions.append(model.predict(X))
            else:
                predictions.append(model.predict(X))

        # Weighted average
        return np.average(predictions, axis=0, weights=weights)

    def plan_ensemble_strategy(
        self, models: list[Any], problem_type: str, eda_summary: dict[str, Any]
    ) -> dict[str, Any]:
        """Plan ensemble strategy using LLM."""
        import json

        from langchain_core.messages import HumanMessage

        from ..core.config import get_llm

        llm = get_llm()

        model_descriptions = []
        for i, m in enumerate(models):
            model_descriptions.append(f"Model {i + 1}: {type(m).__name__}")

        prompt = f"""# Introduction
- You are a Kaggle grandmaster attending a competition.
- We have {len(models)} trained models: {", ".join(model_descriptions)}.
- Problem Type: {problem_type}
- EDA Insights: {str(eda_summary)[:500]}...

# Your task
- Suggest a plan to ensemble these solutions.
- The suggested plan should be novel, effective, and easy to implement.
        - Consider:
            1. "caruana_ensemble": Hill Climbing / Forward Selection (State of the Art).
            2. "stacking": Train a meta-model (LogisticRegression) on OOF predictions.
            3. "weighted_blending": Simple optimized weights.

# Response format
Return a JSON object:
{{
    "strategy_name": "caruana_ensemble" or "stacking_xgboost_meta" or "weighted_blending",
    "description": "Brief description of strategy",
    "meta_learner_config": {{ ... }} (if applicable)
}}
"""
        try:
            response = llm.invoke([HumanMessage(content=prompt)])
            content = get_text_content(response.content).strip()
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()
            return json.loads(content)
        except:
            return {
                "strategy_name": "weighted_blending",
                "description": "Fallback to weighted blending",
            }

    def __call__(self, state: KaggleState) -> KaggleState:
        """Create ensemble from trained models.

        Args:
            state: Current workflow state

        Returns:
            Updated state with ensemble model
        """
        print("\n" + "=" * 60)
        print("ENSEMBLE AGENT: Creating Model Ensemble")
        print("=" * 60)

        # check for temporal ensemble opportunity first if in later iterations
        current_iteration = state.get("current_iteration", 0)
        submissions = state.get("submissions", []) if isinstance(state, dict) else state.submissions
        working_dir_value = (
            state.get("working_directory", "")
            if isinstance(state, dict)
            else state.working_directory
        )
        working_dir = Path(working_dir_value) if working_dir_value else Path()

        # If we have history, try temporal ensemble first (it's cheap and robust)
        # We do this at the END of the standard ensemble logic?
        # Actually, the user requirement is: "Create a final/iterative ensemble step"
        # Since EnsembleAgent runs *before* Submission, we replace the model's output with the blend.
        # BUT: The current iteration's model hasn't been submitted yet!
        # SO: We should blend [Current Model Predictions] + [Past Best Submissions].
        # The current code generates "submission.csv" from the current ensemble/model structure.
        # We can wrap that.

        try:
            # Handle both dict and dataclass state access
            models_trained = (
                state.get("models_trained", []) if isinstance(state, dict) else state.models_trained
            )
            train_data_path = (
                state.get("train_data_path", "")
                if isinstance(state, dict)
                else state.train_data_path
            )
            current_train_path = (
                state.get("current_train_path", "")
                if isinstance(state, dict)
                else getattr(state, "current_train_path", "")
            )
            current_test_path = (
                state.get("current_test_path", "")
                if isinstance(state, dict)
                else getattr(state, "current_test_path", "")
            )
            competition_name = (
                state.get("competition_name", "")
                if isinstance(state, dict)
                else state.competition_name
            )
            eda_summary = (
                state.get("eda_summary", {}) if isinstance(state, dict) else state.eda_summary
            )
            best_model = (
                state.get("best_model", {}) if isinstance(state, dict) else state.best_model
            )
            working_dir_value = (
                state.get("working_directory", "")
                if isinstance(state, dict)
                else state.working_directory
            )
            working_dir = Path(working_dir_value) if working_dir_value else Path()
            working_dir = Path(working_dir_value) if working_dir_value else Path()
            test_data_path = (
                state.get("test_data_path", "") if isinstance(state, dict) else state.test_data_path
            )
            sample_submission_path = (
                state.get("sample_submission_path", "")
                if isinstance(state, dict)
                else state.sample_submission_path
            )
            models_dir = working_dir / "models"

            # Access metric name safely from competition_info
            comp_info = (
                state.get("competition_info")
                if isinstance(state, dict)
                else getattr(state, "competition_info", None)
            )
            metric_name = getattr(comp_info, "evaluation_metric", "") if comp_info else ""

            # DEBUG: Detailed information about available models
            dev_results = state.get("development_results", [])
            successful_results = [r for r in dev_results if r.success] if dev_results else []

            print("\n   📊 Ensemble Prerequisites Check:")
            print(f"      Total development results: {len(dev_results)}")
            print(f"      Successful results: {len(successful_results)}")
            print(f"      Models trained count: {len(models_trained)}")

            if successful_results:
                print("\n   ✅ Successful components:")
                for i, result in enumerate(successful_results[-5:], 1):  # Last 5
                    artifacts_str = (
                        ", ".join(result.artifacts_created[:3])
                        if result.artifacts_created
                        else "none"
                    )
                    print(f"      {i}. {artifacts_str}")

            # Check if we have multiple models
            if len(models_trained) < 2:
                prediction_pairs = self._find_prediction_pairs(models_dir)
                if prediction_pairs:
                    missing_tests = [
                        p.stem.replace("oof_", "", 1)
                        for p in models_dir.glob("oof_*.npy")
                        if not (models_dir / f"test_{p.stem.replace('oof_', '', 1)}.npy").exists()
                    ]
                    if missing_tests:
                        print(f"   ⚠️ Missing test_* for: {', '.join(missing_tests[:5])}")
                    if len(prediction_pairs) >= 2:
                        print("\n   ✅ Using prediction-only ensemble from OOF/Test pairs")
                        output_path = working_dir / "submission.csv"
                        sample_path = (
                            Path(sample_submission_path)
                            if sample_submission_path
                            else working_dir / "sample_submission.csv"
                        )
                        if self._ensemble_from_predictions(
                            prediction_pairs, sample_path, output_path
                        ):
                            return {
                                "ensemble_created": True,
                                "ensemble_method": "prediction_average",
                            }

                print(
                    f"\n   ⚠️  Not enough models for ensemble (need 2+, have {len(models_trained)})"
                )
                print(
                    "      Reason: Ensemble requires at least 2 trained models or 2 OOF/Test pairs"
                )
                print("      Skipping ensemble step")
                return {
                    "ensemble_skipped": True,
                    "skip_reason": f"insufficient_models (have {len(models_trained)}, need 2+)",
                }

            # Resolve train/test paths (prefer engineered data if available)
            resolved_train_path = (
                Path(current_train_path)
                if current_train_path
                else Path(train_data_path)
                if train_data_path
                else working_dir / "train.csv"
            )
            resolved_test_path = (
                Path(current_test_path)
                if current_test_path
                else Path(test_data_path)
                if test_data_path
                else working_dir / "test.csv"
            )

            print("\n   📂 Data Paths:")
            print(f"      Train: {resolved_train_path.name}")
            print(f"      Test:  {resolved_test_path.name}")
            if current_train_path:
                print("      ✅ Using engineered features (from feature_engineering component)")
            else:
                print("      📊 Using original raw features")

            if not resolved_train_path.exists():
                print(f"  ❌ Train data not found at {resolved_train_path}, skipping ensemble")
                return state

            # Load processed data
            train_df = pd.read_csv(resolved_train_path)

            # Identify target
            potential_targets = ["target", "label", train_df.columns[-1]]
            target_col = None
            for col in potential_targets:
                if col in train_df.columns:
                    target_col = col
                    break

            if not target_col:
                print("  Could not identify target column, skipping ensemble")
                return state

            # Separate features and target
            X = train_df.drop(columns=[target_col])
            y = train_df[target_col]

            # Determine problem type
            problem_type = "classification" if y.nunique() < 20 else "regression"

            # Prepare test features for submission generation
            test_features = None
            test_path = resolved_test_path
            if test_path.exists():
                try:
                    test_df = pd.read_csv(test_path)
                    missing_cols = [col for col in X.columns if col not in test_df.columns]
                    if missing_cols:
                        print(f"   ⚠️ Test data missing columns: {missing_cols} (filled with 0)")
                        for col in missing_cols:
                            test_df[col] = 0
                    test_features = test_df[X.columns]
                except Exception as e:
                    print(f"   ⚠️ Failed to load test data for ensemble predictions: {e}")
            else:
                print(f"   ⚠️ Test data not found at {test_path}, skipping submission generation")

            sample_sub_path = (
                Path(sample_submission_path)
                if sample_submission_path
                else working_dir / "sample_submission.csv"
            )

            # Load top models (top 3 by CV score)
            print("\n   🔍 Loading top models for ensemble...")
            sorted_models = sorted(models_trained, key=lambda x: x["mean_cv_score"], reverse=True)[
                :3
            ]

            top_models = []
            top_model_names = []
            for i, model_info in enumerate(sorted_models, 1):
                model_path = f"{get_config().paths.models_dir}/{model_info['name']}_{competition_name}.joblib"
                print(
                    f"      Model {i}: {model_info['name']} (CV: {model_info['mean_cv_score']:.4f})"
                )

                if Path(model_path).exists():
                    model = joblib.load(model_path)
                    top_models.append(model)
                    top_model_names.append(model_info["name"])
                    print(f"         ✅ Loaded from {model_path}")
                else:
                    print(f"         ❌ Model file not found: {model_path}")

            if len(top_models) < 2:
                print("\n   ❌ Not enough trained models loaded for ensemble")
                print(f"      Required: 2+, Found: {len(top_models)}")
                return {
                    "ensemble_skipped": True,
                    "skip_reason": f"models_not_found (loaded {len(top_models)}, need 2+)",
                }

            # PLAN ENSEMBLE STRATEGY
            print("\n   🎯 Planning ensemble strategy...")
            plan = self.plan_ensemble_strategy(top_models, problem_type, eda_summary)
            ensemble_strategy = plan.get("strategy_name", "weighted_blending")
            print(f"      Strategy: {ensemble_strategy}")
            print(f"      Description: {plan.get('description', '')}")
            print(f"      Combining {len(top_models)} models using {ensemble_strategy}")

            # Create ensemble
            if "stack" in ensemble_strategy.lower():
                ensemble = self.create_stacking_ensemble(
                    top_models, top_model_names, working_dir, X, y, problem_type
                )
                if test_features is None:
                    print("  ⚠️ Skipping stacking submission because test features are unavailable")
                else:
                    final_preds = self.predict_stacking(ensemble, test_features, problem_type)
                    if sample_sub_path.exists():
                        sub_df = pd.read_csv(sample_sub_path)
                        sub_df.iloc[:, 1] = final_preds
                        submission_path = working_dir / "submission.csv"
                        sub_df.to_csv(submission_path, index=False)
                        print(f"  ✅ Saved ensemble submission to {submission_path}")
                    else:
                        print(
                            f"  ⚠️ Sample submission not found at {sample_sub_path}, skipping submission save"
                        )

            elif "caruana" in ensemble_strategy.lower():
                ensemble = self.create_caruana_ensemble(
                    top_models, top_model_names, working_dir, X, y, problem_type
                )

                # Generate Final Submission using Test Preds (Weighted Average)
                print("  Generating final submission from Caruana Ensemble...")
                valid_names = ensemble["base_model_names"]
                weights = np.array(ensemble["weights"], dtype=float)
                base_models = ensemble.get("base_models", [])
                test_meta_features = []
                used_weights = []
                missing_test_models = []

                for idx, (name, weight) in enumerate(zip(valid_names, weights, strict=False)):
                    preds = None
                    test_pred_path = working_dir / "models" / f"test_{name}.npy"
                    if test_pred_path.exists():
                        preds = np.load(test_pred_path)
                    elif test_features is not None:
                        model = base_models[idx] if idx < len(base_models) else None
                        if model is not None:
                            try:
                                if problem_type == "classification" and hasattr(
                                    model, "predict_proba"
                                ):
                                    model_preds = model.predict_proba(test_features)
                                    preds = (
                                        model_preds[:, 1] if model_preds.ndim > 1 else model_preds
                                    )
                                else:
                                    preds = model.predict(test_features)
                            except Exception as e:
                                print(f"  ⚠️ Failed to generate test preds for {name}: {e}")
                    if preds is not None:
                        test_meta_features.append(preds)
                        used_weights.append(weight)
                    else:
                        missing_test_models.append(name)

                if missing_test_models:
                    print(f"  ⚠️ Missing test predictions for: {', '.join(missing_test_models)}")

                if test_meta_features:
                    weight_array = np.array(used_weights, dtype=float)
                    weight_sum = weight_array.sum()
                    if weight_sum <= 0:
                        weight_array = np.ones_like(weight_array) / len(weight_array)
                    else:
                        weight_array = weight_array / weight_sum
                    final_preds = np.average(test_meta_features, axis=0, weights=weight_array)

                    # Save submission
                    if sample_sub_path.exists():
                        sub_df = pd.read_csv(sample_sub_path)
                        sub_df.iloc[:, 1] = final_preds
                        submission_path = working_dir / "submission.csv"
                        sub_df.to_csv(submission_path, index=False)
                        print(f"  ✅ Saved ensemble submission to {submission_path}")
                    else:
                        print(
                            f"  ⚠️ Sample submission not found at {sample_sub_path}, skipping submission save"
                        )
                else:
                    print(
                        "  ❌ No test predictions available for Caruana ensemble, skipping submission."
                    )

            else:
                ensemble = self.create_blending_ensemble(top_models, X, y, problem_type)

            # Evaluate ensemble
            print("  Evaluating ensemble performance...")
            if "stack" in ensemble_strategy.lower():
                # Optimistic estimate for stacking
                best_model.get("mean_cv_score", 0.0) * 1.01
            else:
                # For blending, we already calculated OOF loss during optimization
                pass

            # Save ensemble
            ensemble_path = f"{get_config().paths.models_dir}/ensemble_{competition_name}.joblib"
            joblib.dump(
                {
                    "ensemble": ensemble,
                    "problem_type": problem_type,
                    "strategy": ensemble_strategy,
                    "plan": plan,
                },
                ensemble_path,
            )

            print(
                f"Ensemble Agent: Created {ensemble_strategy} ensemble with {len(top_models)} models"
            )

            return {
                "best_model": {
                    "name": f"ensemble_{ensemble_strategy}",
                    "path": ensemble_path,
                    "mean_cv_score": best_model.get("mean_cv_score", 0.0),
                    "is_ensemble": True,
                }
            }

        except Exception as e:
            error_msg = f"Ensemble creation failed: {e!s}"
            print(f"Ensemble Agent ERROR: {error_msg}")
            # Return state with error appended, don't lose existing state
            # TEMPORAL ENSEMBLE STEP (Final Boost)
            # After generating the current iteration's "submission.csv" (either from stacking, caruana, or just best model)
            # We explicitly check if we can improve it by blending with history.
            if current_iteration > 1:
                # Ensure current submission is considered as a candidate
                # We temporarily save it as "current_candidate.csv" to be picked up?
                # Or we just pass the path.
                # Actually, create_temporal_ensemble scans for submission_iter_*.
                # The current one is just "submission.csv".
                # Let's simple copy current submission.csv to a temp name so it's included in the blend logic
                # as a "candidate" (maybe with assumed high score?).
                # Actually, simpler: Just run temporal ensemble. If it finds enough history, it overwrites submission.csv.
                # The newly generated submission.csv effectively becomes valid for this iteration.
                current_sub = working_dir / "submission.csv"
                if current_sub.exists():
                    # Give it a temp name to be picked up by the scanner?
                    # Scanner looks for "submission_iter_*.csv".
                    # We create a fake one representing "current"
                    temp_current = working_dir / f"submission_iter_{current_iteration}_current.csv"
                    import shutil

                    shutil.copy2(current_sub, temp_current)

                self.create_temporal_ensemble(
                    working_dir, submissions, current_iteration, metric_name
                )

            return (
                {"errors": [*errors, error_msg]}
                if errors
                else {
                    "best_model": {
                        "name": f"ensemble_{ensemble_strategy}",
                        "path": ensemble_path,
                        "mean_cv_score": best_model.get("mean_cv_score", 0.0),
                        "is_ensemble": True,
                    }
                }
            )


def ensemble_agent_node(state: KaggleState) -> dict[str, Any]:
    """
    LangGraph node function for ensemble agent.

    Args:
        state: Current workflow state

    Returns:
        State updates
    """
    agent = EnsembleAgent()
    return agent(state)
