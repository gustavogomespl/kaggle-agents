"""
Code sanitization and pre-execution validation.

Contains methods for sanitizing and validating code before execution.
"""

from __future__ import annotations

import re


class CodeSanitizerMixin:
    """Mixin providing code sanitization and validation methods."""

    def sanitize_code(self, code: str) -> tuple[str, list[str]]:
        """
        Automatically sanitize code by removing/replacing prohibited patterns.

        Args:
            code: Python code to sanitize

        Returns:
            Tuple of (sanitized_code, list_of_fixes_applied)
        """
        fixes_applied = []

        # Auto-fix sys.exit() calls
        if "sys.exit(1)" in code:
            code = code.replace(
                "sys.exit(1)", 'raise ValueError("Missing required data or configuration")'
            )
            fixes_applied.append("Replaced sys.exit(1) with ValueError")

        if "sys.exit(0)" in code:
            code = code.replace("sys.exit(0)", "pass  # Replaced sys.exit(0)")
            fixes_applied.append("Replaced sys.exit(0) with pass")

        # Generic sys.exit() with variable
        if "sys.exit(" in code:
            code = re.sub(r"sys\.exit\([^)]*\)", 'raise RuntimeError("Execution terminated")', code)
            fixes_applied.append("Replaced remaining sys.exit() calls with RuntimeError")

        # Auto-fix other termination calls
        if re.search(r"(?<!\\w)exit\\(\\)", code) and "sys.exit" not in code:
            code = re.sub(r"(?<!\\w)exit\\(\\)", "pass  # Replaced exit()", code)
            fixes_applied.append("Replaced exit() with pass")

        if re.search(r"(?<!\\w)quit\\(\\)", code):
            code = re.sub(r"(?<!\\w)quit\\(\\)", "pass  # Replaced quit()", code)
            fixes_applied.append("Replaced quit() with pass")

        if "os._exit(" in code:
            code = re.sub(r"os\._exit\([^)]*\)", 'raise RuntimeError("Forced exit")', code)
            fixes_applied.append("Replaced os._exit() with RuntimeError")

        if fixes_applied:
            print(f"   🔧 Auto-sanitized code: {', '.join(fixes_applied)}")

        return code, fixes_applied

    def detect_model_training(self, code: str) -> list[str]:
        """
        Detect model training patterns in code.

        Used to validate that preprocessing/feature_engineering components
        do not contain model training, which violates their contract.

        Args:
            code: Python code to analyze

        Returns:
            List of detected training patterns (empty if none found)
        """
        training_patterns = [
            # Sklearn-style fit patterns
            (r"\.fit\s*\(", ".fit() call"),
            (r"\.fit_transform\s*\(", ".fit_transform() call"),

            # Tree-based models
            (r"RandomForest(?:Classifier|Regressor)\s*\(", "RandomForest model"),
            (r"(?:LGBM|LightGBM)(?:Classifier|Regressor)\s*\(", "LightGBM model"),
            (r"(?:XGB|XGBoost)(?:Classifier|Regressor)\s*\(", "XGBoost model"),
            (r"CatBoost(?:Classifier|Regressor)\s*\(", "CatBoost model"),
            (r"GradientBoosting(?:Classifier|Regressor)\s*\(", "GradientBoosting model"),
            (r"ExtraTrees(?:Classifier|Regressor)\s*\(", "ExtraTrees model"),

            # Linear models
            (r"LogisticRegression\s*\(", "LogisticRegression model"),
            (r"(?:Linear|Ridge|Lasso|ElasticNet)(?:Regression|Classifier)\s*\(", "Linear model"),
            (r"(?:SVC|SVR)\s*\(", "SVM model"),

            # Neural network training
            (r"model\.train\s*\(", "PyTorch model.train()"),
            (r"\.fit\s*\([^)]*epochs", "Keras/TF model.fit() with epochs"),
            (r"optimizer\.step\s*\(", "PyTorch optimizer.step()"),
            (r"loss\.backward\s*\(", "PyTorch loss.backward()"),

            # Cross-validation (implies training)
            (r"cross_val_predict\s*\(", "cross_val_predict()"),
            (r"cross_val_score\s*\(", "cross_val_score()"),
        ]

        detected = []
        for pattern, description in training_patterns:
            if re.search(pattern, code):
                detected.append(description)

        return detected

    def validate_code_before_execution(
        self, code: str, component_type: str | None = None
    ) -> tuple[bool, str]:
        """
        Validates code meets requirements before execution (MLE-STAR pattern).

        Args:
            code: Python code to validate
            component_type: Type of component ('preprocessing', 'feature_engineering', 'model', etc.)
                           Used to validate that preprocessing doesn't train models.

        Returns:
            Tuple of (is_valid, message)
        """
        # Check 1: Has required output format (only for model/ensemble components)
        # Feature engineering and preprocessing components don't train models,
        # so they can't produce a meaningful CV score - they should print "1.0" as placeholder
        if component_type in ("model", "ensemble", None):
            if "Final Validation Performance" not in code:
                return False, "Missing required output: 'Final Validation Performance: {score}'"
        # For preprocessing/feature_engineering, validation happens post-execution via artifacts

        # Check 2: No prohibited exit() calls (enhanced check)
        # Note: These should have been sanitized by sanitize_code() before validation
        prohibited_calls = ["sys.exit(", "quit()", "raise SystemExit", "os._exit("]
        for call in prohibited_calls:
            if call in code:
                return False, f"Code contains prohibited termination call: {call}"

        # Check 3: Early stopping rounds misuse (common API error)
        if "early_stopping_rounds=" in code and ".fit(" in code:
            # Check if it's being passed as a parameter to fit()
            if re.search(r"\.fit\([^)]*early_stopping_rounds=", code):
                return False, (
                    "API Error: early_stopping_rounds cannot be passed to fit(). "
                    "Use callbacks=[lgb.early_stopping(100)] for LightGBM, "
                    "callbacks=[xgb.callback.EarlyStopping(100)] for XGBoost <2, "
                    "or pass early_stopping_rounds in the XGBoost 2.0+ constructor."
                )

        # Check 4: Has basic structure for ML code
        required_patterns = [
            (
                "import pandas" in code or "import numpy" in code,
                "Missing required imports (pandas or numpy)",
            ),
        ]

        for has_pattern, error_msg in required_patterns:
            if not has_pattern:
                return False, error_msg

        # Check 5: Block model training in preprocessing/feature_engineering components
        # These component types should ONLY transform data, not train ML models
        if component_type in ("preprocessing", "feature_engineering"):
            training_patterns = self.detect_model_training(code)

            if training_patterns:
                # Allow certain patterns for legitimate preprocessing uses
                # e.g., LabelEncoder.fit_transform for encoding, StandardScaler.fit for scaling
                allowed_patterns = {
                    ".fit() call",  # Could be scaler.fit() which is allowed
                    ".fit_transform() call",  # Could be encoder.fit_transform() which is allowed
                }

                # Check if patterns are for feature importance-based selection (allowed)
                is_feature_selection = (
                    "feature_importances_" in code
                    or "get_score(" in code  # XGBoost feature importance
                    or "feature_importance(" in code  # LightGBM
                    or "SelectFromModel" in code
                )

                # Filter out allowed patterns
                blocked_patterns = [
                    p for p in training_patterns
                    if p not in allowed_patterns or any(
                        model in p for model in [
                            "RandomForest", "LightGBM", "XGBoost", "CatBoost",
                            "LogisticRegression", "GradientBoosting", "SVM",
                            "Linear model", "ExtraTrees", "cross_val"
                        ]
                    )
                ]

                if blocked_patterns and not is_feature_selection:
                    return False, (
                        f"Model training detected in {component_type} component: "
                        f"{', '.join(blocked_patterns)}. "
                        f"Preprocessing/feature_engineering components MUST NOT train models. "
                        f"Move model training to a 'model' component instead."
                    )
                if blocked_patterns:
                    # It's feature selection - just warn but allow
                    print(
                        f"   ⚠️  {component_type} uses models for feature selection: "
                        f"{', '.join(blocked_patterns)}"
                    )

        # Check 6: Warning about categorical features (informational only)
        # This is a soft check - we warn but don't fail
        has_categorical_check = (
            "select_dtypes" in code
            or "LabelEncoder" in code
            or "OneHotEncoder" in code
            or "TargetEncoder" in code
            or "CatBoost" in code
        )

        if (not has_categorical_check and "LGBMClassifier" in code) or "XGBClassifier" in code:
            # This is just a warning, not a failure
            print("   ⚠️  Warning: LightGBM/XGBoost code without categorical encoding detected")

        # Check 7: Optuna pruning contract (only for model/ensemble with Hyperband)
        # This is CONDITIONAL - only validates when a pruner is active in the code
        is_valid, pruning_error = self._validate_optuna_pruning_contract(code)
        if not is_valid:
            return False, f"HPO Contract Violation: {pruning_error}"

        return True, "Validation passed"

    def _validate_optuna_pruning_contract(self, code: str) -> tuple[bool, str]:
        """
        Validate that generated code follows the Optuna pruning contract.

        The contract requires (when a pruner is active):
        1. trial.report(score, step) called at each iteration
        2. trial.should_prune() checked and TrialPruned raised if True

        This validation is CONDITIONAL - only applies when a pruner is active.

        Args:
            code: Generated Python code string

        Returns:
            Tuple of (is_valid, error_message)
        """
        # Check if Optuna is being used
        uses_optuna = any(pattern in code for pattern in [
            "import optuna",
            "from optuna",
            "optuna.create_study",
            "optuna.Study",
        ])

        if not uses_optuna:
            return True, ""  # No Optuna = no validation needed

        # Check if a pruner is being used (not NopPruner)
        pruner_patterns = [
            ("HyperbandPruner", "Hyperband"),
            ("MedianPruner", "Median"),
            ("SuccessiveHalvingPruner", "SuccessiveHalving"),
            ("ThresholdPruner", "Threshold"),
            ("PercentilePruner", "Percentile"),
        ]

        active_pruner = None
        for pattern, name in pruner_patterns:
            if pattern in code:
                active_pruner = name
                break

        if active_pruner is None:
            return True, ""  # No active pruner = no validation needed

        # Pruner is active - check contract
        has_report = "trial.report" in code
        has_prune_check = "should_prune" in code or "TrialPruned" in code

        if not has_report:
            return False, (
                f"Code uses {active_pruner}Pruner but does not call trial.report(). "
                "The pruner cannot work without intermediate score reporting. "
                "Add: trial.report(score, step) inside your training loop."
            )

        if not has_prune_check:
            return False, (
                f"Code uses {active_pruner}Pruner but does not check trial.should_prune(). "
                "Trials will never be pruned, wasting compute. "
                "Add: if trial.should_prune(): raise optuna.TrialPruned()"
            )

        return True, ""
