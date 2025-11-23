"""
Developer Agent with Code Generation and Auto-Retry.

This agent generates Python code to implement ablation components,
with automatic retry and debugging capabilities.
"""

import json
from typing import List, Dict, Any, Optional
from datetime import datetime
from pathlib import Path

import dspy
from langchain_core.messages import HumanMessage, SystemMessage
from ..core.state import KaggleState, AblationComponent, DevelopmentResult
from ..core.config import get_config, get_llm_for_role
from ..tools.code_executor import CodeExecutor, ArtifactValidator, ExecutionResult
from ..prompts.templates.developer_prompts import (
    DEVELOPER_SYSTEM_PROMPT,
    GENERATE_CODE_PROMPT,
    FIX_CODE_PROMPT,
    DEBUG_CODE_PROMPT,
    format_component_details,
    format_error_info,
    get_domain_template,
)
from ..optimization import create_optimizer, create_developer_metric


# ==================== DSPy Signatures ====================

class CodeGeneratorSignature(dspy.Signature):
    """Signature for code generation."""

    component_details: str = dspy.InputField(desc="Component to implement")
    competition_context: str = dspy.InputField(desc="Competition metadata")
    data_paths: str = dspy.InputField(desc="Paths to data files")
    requirements: str = dspy.InputField(desc="Implementation requirements")

    code: str = dspy.OutputField(desc="Complete Python code")
    explanation: str = dspy.OutputField(desc="Brief explanation of implementation")


class CodeFixerSignature(dspy.Signature):
    """Signature for code fixing."""

    code: str = dspy.InputField(desc="Code with errors")
    error: str = dspy.InputField(desc="Error message")
    error_type: str = dspy.InputField(desc="Type of error")

    fixed_code: str = dspy.OutputField(desc="Fixed Python code")
    changes_made: str = dspy.OutputField(desc="Description of fixes")


# ==================== DSPy Modules ====================

class CodeGeneratorModule(dspy.Module):
    """DSPy module for code generation."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(CodeGeneratorSignature)

    def forward(self, component_details, competition_context, data_paths, requirements):
        """Generate code."""
        result = self.generate(
            component_details=component_details,
            competition_context=competition_context,
            data_paths=data_paths,
            requirements=requirements,
        )
        return result


class CodeFixerModule(dspy.Module):
    """DSPy module for code fixing."""

    def __init__(self):
        super().__init__()
        self.fix = dspy.ChainOfThought(CodeFixerSignature)

    def forward(self, code, error, error_type):
        """Fix code."""
        result = self.fix(code=code, error=error, error_type=error_type)
        return result


# ==================== Developer Agent ====================

class DeveloperAgent:
    """
    Agent responsible for code generation and execution.

    Features:
    - Generate code from ablation components
    - Execute code in sandbox
    - Automatic retry on failure (5 attempts)
    - Debug iterations (10 max)
    - Artifact validation
    - DSPy optimization support
    """

    def __init__(self, use_dspy: bool = True):
        """
        Initialize the developer agent.

        Args:
            use_dspy: Whether to use DSPy modules
        """
        self.config = get_config()
        self.use_dspy = use_dspy and self.config.dspy.enabled

        # Code executor (use configured timeout)
        timeout = self.config.ablation.testing_timeout
        self.executor = CodeExecutor(timeout=timeout)
        self.validator = ArtifactValidator()

        print(f"   ⏱️  Component timeout set to: {timeout}s ({timeout/60:.1f} min)")

        # Always create LLM client (used for debugging even with DSPy)
        # MLE-STAR Pattern: Lower temperature for implementation tasks (0.3)
        implementation_temperature = 0.3

        self.llm = get_llm_for_role(
            role="developer",
            temperature=implementation_temperature,
            max_tokens=self.config.llm.max_tokens,
        )

        if self.use_dspy:
            # Try to load optimized modules
            optimizer = create_optimizer()
            self.generator_module = optimizer.load_optimized_prompt("developer_generator")
            self.fixer_module = optimizer.load_optimized_prompt("developer_fixer")

            if self.generator_module is None:
                print("   Using base (unoptimized) generator module")
                self.generator_module = CodeGeneratorModule()

            if self.fixer_module is None:
                print("   Using base (unoptimized) fixer module")
                self.fixer_module = CodeFixerModule()

    def __call__(self, state: KaggleState) -> Dict[str, Any]:
        """
        Execute the developer agent.

        Args:
            state: Current workflow state

        Returns:
            State updates with development results
        """
        print("\n" + "="*60)
        print("= DEVELOPER AGENT: Implementing Components")
        print("="*60)

        ablation_plan = state.get("ablation_plan", [])
        current_index = state.get("current_component_index", 0)

        if not ablation_plan:
            print("  No ablation plan found. Run Planner Agent first.")
            return {}

        if current_index >= len(ablation_plan):
            print(" All components implemented!")
            return {"current_component_index": current_index}

        # Implement current component
        component = ablation_plan[current_index]
        print(f"\n= Implementing: {component.name} ({component.component_type})")
        print(f"   Estimated Impact: {component.estimated_impact:.1%}")

        # Generate and execute code
        result = self._implement_component(component, state)

        # Determine if we should move to next component
        # Always move forward if it's a critical error (data files missing)
        should_advance = result.success or (
            not result.success and "Data files not found" in (result.stderr or "")
        )

        # Cache successful results for skip logic (MLE-STAR Pattern)
        state_updates = {
            "development_results": [result],
            "current_code": result.code,
            "code_retry_count": 0,
            "current_component_index": current_index + 1 if should_advance else current_index,
            "last_updated": datetime.now(),
        }

        # If successful, cache for future iterations
        if result.success:
            cache_key = f"component_result_{component.name}"
            state_updates[cache_key] = result
            print(f"  💾 Cached successful result for: {component.name}")

        return state_updates

    def _implement_component(
        self,
        component: AblationComponent,
        state: KaggleState,
    ) -> DevelopmentResult:
        """
        Implement a single component with retry and debug.

        Args:
            component: Component to implement
            state: Current state

        Returns:
            DevelopmentResult
        """
        competition_info = state["competition_info"]
        working_dir = Path(state["working_directory"])
        domain = state.get("domain_detected", "tabular")

        # Verify data files exist before proceeding
        train_path = working_dir / 'train.csv'
        test_path = working_dir / 'test.csv'

        if not train_path.exists() or not test_path.exists():
            error_msg = f"Data files not found in {working_dir}\n"
            error_msg += f"  Expected: {train_path.name}, {test_path.name}\n"

            # Check what files are actually present
            if working_dir.exists():
                existing_files = [f.name for f in working_dir.iterdir() if f.is_file()]
                error_msg += f"  Found: {existing_files if existing_files else 'No files'}\n"
            else:
                error_msg += f"  Working directory doesn't exist\n"

            error_msg += "\n💡 Possible causes:\n"
            error_msg += "  - Data download failed (check Kaggle credentials)\n"
            error_msg += "  - Competition data not downloaded yet\n"
            error_msg += "  - Wrong working directory path\n"

            print(f"\n❌ {error_msg}")

            return DevelopmentResult(
                code="",
                success=False,
                stdout="",
                stderr=error_msg,
                execution_time=0.0,
                artifacts_created=[],
                errors=[error_msg],
            )

        # SKIP LOGIC (MLE-STAR Pattern): Check if component already done
        skip_result = self._should_skip_component(component, state)
        if skip_result is not None:
            return skip_result

        # Generate initial code
        print("\n   =' Generating code...")
        code = self._generate_code(component, competition_info, working_dir, domain, state)

        # Preview generated code
        if self.config.ablation.enable_code_preview if hasattr(self.config.ablation, 'enable_code_preview') else True:
            print("\n   📝 Generated code preview:")
            code_lines = code.split('\n')
            preview_lines = min(500, len(code_lines))  # Show first 500 lines
            for i, line in enumerate(code_lines[:preview_lines], 1):
                print(f"      {i:3d} | {line}")
            if len(code_lines) > preview_lines:
                print(f"      ... ({len(code_lines) - preview_lines} more lines)")
            print()

        # Save code to file for inspection
        if self.config.ablation.save_generated_code if hasattr(self.config.ablation, 'save_generated_code') else True:
            code_file = working_dir / f"generated_code_{component.name}.py"
            try:
                code_file.write_text(code)
                print(f"   💾 Code saved to: {code_file.name}")
            except Exception as e:
                print(f"   ⚠️ Could not save code: {e}")

        # Validate syntax
        is_valid, syntax_error = self.executor.validate_syntax(code)
        if not is_valid:
            print(f"     Syntax error detected: {syntax_error}")
            code = self._fix_syntax_error(code, syntax_error)

        # Execute with retry
        print("\n     Executing code...")
        max_retries = 3
        for attempt in range(max_retries):
            print(f"\n   Attempt {attempt + 1}/{max_retries}")

            exec_result = self.executor.execute(
                code=code,
                working_dir=working_dir,
            )

            if exec_result.success:
                print(f"    Execution successful ({exec_result.execution_time:.2f}s)")

                return DevelopmentResult(
                    code=code,
                    success=True,
                    stdout=exec_result.stdout,
                    stderr=exec_result.stderr,
                    execution_time=exec_result.execution_time,
                    artifacts_created=exec_result.artifacts_created,
                    errors=[],
                )

            print(f"   L Execution failed: {exec_result.errors[0] if exec_result.errors else 'Unknown'}")

            # Get meta-evaluator feedback on first failure (Phase 4: Mini Meta-Evaluator)
            if attempt == 0:
                print("\n   🧠 Getting meta-evaluator feedback...")
                error_msg = exec_result.errors[0] if exec_result.errors else exec_result.stderr
                feedback = self._get_meta_feedback(code, error_msg, component.name)
                print(f"   📋 Meta-Feedback:\n   {feedback}\n")

            # Try to fix
            if attempt < max_retries - 1:
                print("   =' Attempting to fix...")
                code = self._fix_code_error(code, exec_result.errors[0] if exec_result.errors else exec_result.stderr)

        # If all retries failed, try debug iterations
        print("\n   = Entering debug mode...")
        code, exec_result, debug_success = self._debug_code(
            code, exec_result, working_dir, max_iterations=5
        )

        # Return final result
        return DevelopmentResult(
            code=code,
            success=exec_result.success if debug_success else False,
            stdout=exec_result.stdout,
            stderr=exec_result.stderr,
            execution_time=exec_result.execution_time,
            artifacts_created=exec_result.artifacts_created,
            errors=exec_result.errors,
        )

    def _execute_with_multi_level_retry_v2(
        self,
        component: AblationComponent,
        initial_code: str,
        working_dir: Path,
        competition_info,
        domain: str,
        state: KaggleState,
    ) -> tuple[str, bool]:
        """
        Multi-level retry with rollback (MLE-STAR pattern).

        This wraps the existing retry logic and adds Level 3: simplified rollback.
        Returns (code, success) tuple.
        """
        # Try normal execution first (Level 1 + Level 2 handled by existing code)
        # We already have the result from the normal path
        # If we get here, both Level 1 and 2 failed

        # LEVEL 3: Rollback to simplified version
        print("\n   ⚠️  LEVEL 3: Attempting simplified version...")
        simplified_component = self._create_simplified_component(component)
        print(f"   📝 Simplified: {simplified_component.name}")

        # Generate code for simplified version
        simplified_code = self._generate_code(
            simplified_component,
            competition_info,
            working_dir,
            domain,
            state,
        )

        # Validate syntax
        is_valid, syntax_error = self.executor.validate_syntax(simplified_code)
        if not is_valid:
            print(f"     Syntax error in simplified code: {syntax_error}")
            simplified_code = self._fix_syntax_error(simplified_code, syntax_error)

        # Try simplified version with quick retries only
        print("   Executing simplified version...")
        for attempt in range(3):  # Fewer attempts for simplified version
            print(f"   Simplified attempt {attempt + 1}/3")

            exec_result = self.executor.execute(
                code=simplified_code,
                working_dir=working_dir,
            )

            if exec_result.success:
                print(f"   ✅ Simplified version successful!")
                return simplified_code, True

            print(f"   L Simplified attempt failed: {exec_result.errors[0] if exec_result.errors else 'Unknown'}")

            if attempt < 2:
                simplified_code = self._fix_code_error(
                    simplified_code,
                    exec_result.errors[0] if exec_result.errors else exec_result.stderr
                )

        # All levels exhausted
        print("\n   ❌ All retry levels exhausted (original + debug + simplified)")
        return simplified_code, False

    def _should_skip_component(
        self,
        component: AblationComponent,
        state: KaggleState,
    ) -> Optional[DevelopmentResult]:
        """
        Check if component should be skipped (MLE-STAR pattern).

        This implements callback-based skip logic to avoid redundant work:
        - Skip if code already generated and successfully executed
        - Skip if this is a refinement iteration and component worked before

        Args:
            component: Component to check
            state: Current workflow state

        Returns:
            DevelopmentResult if should skip (reuse previous result), None otherwise
        """
        # Check if we have previous development results for this component
        dev_results = state.get("development_results", [])

        # Look for existing successful result for this component
        for result in dev_results:
            # Match by checking if code contains component name
            if result.success and component.name in result.code:
                print(f"  ⏭️  Skipping {component.name} - already implemented successfully")
                print(f"     Reusing previous execution ({result.execution_time:.2f}s)")
                return result

        # Check state for explicitly cached results (for refinement iterations)
        cached_result_key = f"component_result_{component.name}"
        if cached_result_key in state:
            cached_result = state[cached_result_key]
            if cached_result.success:
                print(f"  ⏭️  Skipping {component.name} - found in cache")
                print(f"     Reusing cached execution ({cached_result.execution_time:.2f}s)")
                return cached_result

        # Don't skip - component needs to be implemented
        return None

    def _create_simplified_component(
        self,
        component: AblationComponent,
    ) -> AblationComponent:
        """
        Create a simplified version of component for rollback (MLE-STAR pattern).

        Simplification strategies:
        - Model: Use simpler hyperparameters, fewer estimators
        - Feature engineering: Reduce complexity of features
        - Ensemble: Use simple averaging instead of stacking

        Args:
            component: Original component

        Returns:
            Simplified component
        """
        simplified_desc = ""

        if component.component_type == "model":
            # Simplify model to basic configuration
            model_name = component.name.split("_")[0]  # e.g., "lightgbm" from "lightgbm_tuned"
            simplified_desc = f"Simple {model_name} model with basic hyperparameters: n_estimators=100, max_depth=5, learning_rate=0.1. Use default class_weight='balanced' and 5-fold StratifiedKFold."

        elif component.component_type == "feature_engineering":
            simplified_desc = f"Basic feature engineering: simple polynomial features (degree 2) and basic statistical aggregations (mean, std, min, max). Avoid complex transformations."

        elif component.component_type == "ensemble":
            simplified_desc = f"Simple ensemble: weighted average of model predictions with equal weights. Load predictions from submission files and average them."

        else:
            simplified_desc = f"Simplified version of {component.name}"

        # Create new component with simplified code outline
        from dataclasses import replace
        simplified_component = replace(
            component,
            name=f"{component.name}_simplified",
            code=simplified_desc,  # Use code field for description
            estimated_impact=component.estimated_impact * 0.7,  # Lower expected impact
        )

        return simplified_component

    def _build_dynamic_instructions(
        self,
        component: AblationComponent,
        state: KaggleState,
    ) -> str:
        """
        Build dynamic instructions based on current state (MLE-STAR pattern).

        Creates context-aware guidance by analyzing:
        - Previous component results (what worked/failed)
        - Current iteration number (more specific in later iterations)
        - Performance trends
        - Common error patterns

        Args:
            component: Component being implemented
            state: Current workflow state

        Returns:
            Dynamic instructions string
        """
        instructions = []

        # Base instruction
        instructions.append(f"Implement {component.component_type}: {component.name}")

        # Add iteration-specific guidance
        current_iteration = state.get("current_iteration", 0)
        if current_iteration > 0:
            instructions.append(f"\n⚡ REFINEMENT ITERATION {current_iteration}")
            instructions.append("Focus on improvements that address previous shortcomings.")

        # INJECT META-EVALUATOR GUIDANCE (RL Pattern)
        refinement_guidance = state.get("refinement_guidance", {})
        if refinement_guidance and refinement_guidance.get("developer_guidance"):
            instructions.append(f"\n🧠 META-EVALUATOR GUIDANCE:")
            instructions.append(f"  {refinement_guidance['developer_guidance']}")

        # Component-specific guidance from meta-evaluator
        if refinement_guidance and "component_type_guidance" in refinement_guidance:
            comp_guidance = refinement_guidance["component_type_guidance"].get(component.component_type)
            if comp_guidance:
                instructions.append(f"\n🎯 {component.component_type.upper()} SPECIFIC GUIDANCE:")
                instructions.append(f"  {comp_guidance}")

        # Priority fixes from error analysis
        if refinement_guidance and refinement_guidance.get("priority_fixes"):
            instructions.append("\n⚠️  AVOID THESE ERROR PATTERNS:")
            for error in refinement_guidance["priority_fixes"][:3]:  # Top 3
                instructions.append(f"  - {error}")

        # Analyze previous results for lessons learned
        dev_results = state.get("development_results", [])
        if dev_results:
            successful_components = [r for r in dev_results if r.success]
            failed_components = [r for r in dev_results if not r.success]

            if successful_components:
                instructions.append("\n✅ SUCCESSFUL PATTERNS FROM PREVIOUS COMPONENTS:")
                # Extract common patterns from successful code
                for i, result in enumerate(successful_components[-2:], 1):  # Last 2 successes
                    if "LightGBM" in result.code:
                        instructions.append(f"  - LightGBM implementation worked well")
                    if "StratifiedKFold" in result.code:
                        instructions.append(f"  - StratifiedKFold cross-validation successful")
                    if "predict_proba" in result.code:
                        instructions.append(f"  - predict_proba() for probabilities confirmed working")

            if failed_components:
                instructions.append("\n⚠️  AVOID THESE ERRORS FROM PREVIOUS ATTEMPTS:")
                # Extract common error patterns
                for i, result in enumerate(failed_components[-2:], 1):  # Last 2 failures
                    if result.errors:
                        error_msg = result.errors[0][:300]  # First 300 chars
                        instructions.append(f"  - {error_msg}")

        # Add performance-based guidance
        current_score = state.get("current_performance_score", 0.0)
        target_score = 0.9238  # Target for top 20%
        if current_score > 0:
            gap = target_score - current_score
            instructions.append(f"\n📊 PERFORMANCE GAP: {gap:.4f} to reach target ({target_score:.4f})")
            if gap < 0.01:
                instructions.append("  - Small gap: Focus on fine-tuning hyperparameters")
            elif gap < 0.05:
                instructions.append("  - Medium gap: Consider feature engineering or ensemble methods")
            else:
                instructions.append("  - Large gap: May need different model architecture or approach")

        # Component-type specific instructions
        if component.component_type == "model":
            instructions.append("\n🎯 MODEL COMPONENT REQUIREMENTS:")
            instructions.append("  - MUST train a model and generate predictions")
            instructions.append("  - MUST create submission.csv with probability predictions (0.0-1.0)")
            instructions.append(f"  - CRITICAL: Use target_col from dataset info (target_col='{state.get('target_col', 'target')}' if available)")
            instructions.append("  - CRITICAL: submission column name MUST match sample_submission.columns[1] (DO NOT hardcode 'target')")
            instructions.append("  - CRITICAL: MUST encode categorical features (object/category dtypes) using ColumnTransformer + OneHotEncoder")
            instructions.append("  - CRITICAL: Never pass raw categorical strings to LightGBM/XGBoost/sklearn (will fail with 'could not convert string to float')")
            instructions.append("  - CatBoost is the ONLY exception that handles categorical features natively")
            instructions.append("  - MUST use StratifiedKFold for cross-validation")
            instructions.append("  - CRITICAL: MUST save Out-of-Fold (OOF) predictions during CV to models/oof_{component_name}.npy")
            instructions.append("  - OOF predictions enable proper stacking ensemble (meta-model trained on OOF)")
            instructions.append("  - MUST print 'Final Validation Performance: {score}'")
            instructions.append("  - MUST handle class imbalance with class_weight='balanced'")

            # OPTUNA SPECIFIC GUIDANCE
            if "optuna" in component.name.lower() or "tuned" in component.name.lower() or "optimized" in component.name.lower():
                n_trials = self.config.ablation.optuna_trials
                timeout = self.config.ablation.testing_timeout - 60  # Leave 60s buffer
                
                instructions.append("\n🔍 HYPERPARAMETER OPTIMIZATION (OPTUNA) REQUIRED:")
                instructions.append("  - MUST use 'optuna' library for hyperparameter search")
                instructions.append(f"  - Run AT MOST {n_trials} trials (n_trials={n_trials}) and timeout={timeout}s to prevent timeouts")
                instructions.append("  - CRITICAL: Check if 'optuna-integration' is available with try/except:")
                instructions.append("    try:")
                instructions.append("        from optuna.integration import OptunaSearchCV")
                instructions.append("    except ImportError:")
                instructions.append("        # Use manual Optuna with study.optimize() instead")
                instructions.append("  - If optuna-integration is missing, use manual Optuna tuning with study.optimize()")
                instructions.append("  - Use 'TPESampler' for efficient sampling")
                instructions.append("  - CRITICAL: Do NOT pass 'callbacks' or 'early_stopping_rounds' to .fit() for XGBoost/LightGBM/CatBoost sklearn API; use fixed n_estimators")
                instructions.append("  - Optimize for the competition metric (minimize RMSE/LogLoss or maximize AUC/Accuracy)")
                instructions.append("  - Print the best parameters found")
                instructions.append("  - Train final model with best parameters")

        elif component.component_type == "feature_engineering":
            instructions.append("\n🔧 FEATURE ENGINEERING REQUIREMENTS:")
            instructions.append("  - Create NEW features from existing ones")
            instructions.append("  - IMPLEMENT SOTA TECHNIQUES:")
            instructions.append("    - Target Encoding: MUST be done inside Cross-Validation (fit on train folds, transform val fold) to prevent leakage.")
            instructions.append("    - Frequency Encoding: Map categorical features to their frequency/count.")
            instructions.append("    - Aggregations: Mean/Count of numeric features grouped by categorical features.")
            instructions.append("  - Save engineered features to file for model components")
            instructions.append("  - NO model training in this component")
            instructions.append("  - Print feature importance or correlation metrics")
        elif component.component_type == "ensemble":
            instructions.append("\n🎭 ENSEMBLE REQUIREMENTS:")
            instructions.append("  - Combine predictions from multiple models")
            instructions.append("  - PREFERRED STRATEGY: Stacking Ensemble (best performance)")
            instructions.append("    - Load OOF predictions from models/oof_*.npy files")
            instructions.append("    - Stack OOF predictions: oof_stack = np.column_stack([oof1, oof2, ...])")
            instructions.append("    - Train meta-model (LogisticRegression/Ridge) on stacked OOF")
            instructions.append("    - Load test predictions from each model and stack them")
            instructions.append("    - Use meta-model to predict on stacked test predictions")
            instructions.append("  - FALLBACK: Weighted average if OOF files missing")
            instructions.append("    - Load submission files from each model")
            instructions.append("    - Combine with weights: final = w1*pred1 + w2*pred2 + ...")
            instructions.append("  - Generate final submission.csv")
            instructions.append(f"  - CRITICAL: Use target_col from dataset info (target_col='{state.get('target_col', 'target')}' if available)")
            instructions.append("  - CRITICAL: submission column name MUST match sample_submission.columns[1] (DO NOT hardcode 'target' or 'prediction')")
            instructions.append("  - Print which models were used and their contribution/weights")

        # Standard requirements
        instructions.append("\n📋 STANDARD REQUIREMENTS:")
        instructions.append("  - Save models to models/ directory")
        instructions.append("  - Print progress and metrics throughout execution")
        instructions.append("  - NO sys.exit() or exit() calls")
        instructions.append("  - CRITICAL: Do NOT use deprecated 'pandas.append()'. Use 'pd.concat()' instead.")
        instructions.append("  - Complete, executable single-file Python program")

        return "\n".join(instructions)

    def _get_dataset_info(self, working_dir: Path, state: KaggleState = None) -> str:
        """
        Read dataset columns and basic info to provide to LLM.

        Args:
            working_dir: Working directory containing train.csv
            state: Current state (optional)

        Returns:
            Formatted string with dataset information
        """
        try:
            import pandas as pd
            train_path = working_dir / 'train.csv'

            if not train_path.exists():
                return "Dataset info not available (file not found)"

            # Read just first few rows to get columns
            df = pd.read_csv(train_path, nrows=5)

            columns = df.columns.tolist()
            dtypes = df.dtypes.to_dict()

            # Identify likely target column
            target_col = "UNKNOWN"
            
            if state and state.get("target_col"):
                target_col = state["target_col"]
            else:
                target_candidates = [c for c in columns if c.lower() in ['target', 'label', 'y', 'class', 'loan_paid_back', 'survived', 'price', 'sales']]
                target_col = target_candidates[0] if target_candidates else "UNKNOWN"

            # Format column info
            numeric_cols = [c for c, dtype in dtypes.items() if dtype in ['int64', 'float64']]
            categorical_cols = [c for c, dtype in dtypes.items() if dtype == 'object']

            info = f"""
**CRITICAL**: Use these EXACT column names from the dataset:

Target Column: {target_col}
Total Columns: {len(columns)}

Numeric Columns ({len(numeric_cols)}): {', '.join(numeric_cols[:10])}{'...' if len(numeric_cols) > 10 else ''}
Categorical Columns ({len(categorical_cols)}): {', '.join(categorical_cols[:10])}{'...' if len(categorical_cols) > 10 else ''}

All Columns: {', '.join(columns)}

IMPORTANT: Always use target_col='{target_col}' in your code!
"""
            return info

        except Exception as e:
            return f"Dataset info not available (error: {str(e)})"

    def _generate_code(
        self,
        component: AblationComponent,
        competition_info,
        working_dir: Path,
        domain: str,
        state: KaggleState = None,
    ) -> str:
        """Generate code for a component."""
        component_details = format_component_details(component)

        # Get dataset information
        dataset_info = self._get_dataset_info(working_dir, state)

        competition_context = f"""
Name: {competition_info.name}
Domain: {domain}
Problem Type: {competition_info.problem_type}
Metric: {competition_info.evaluation_metric}
"""

        data_paths = f"""
Train: {working_dir / 'train.csv'}
Test: {working_dir / 'test.csv'}
Models: {working_dir / 'models'}
Submission: {working_dir / 'submission.csv'}
"""

        # DYNAMIC INSTRUCTION GENERATION (MLE-STAR Pattern)
        # Build context-aware requirements based on state
        if state is not None:
            requirements = self._build_dynamic_instructions(component, state)
        else:
            # Fallback to basic requirements if no state
            requirements = f"""
1. Implement {component.component_type}: {component.name}
2. Save models to models/ directory
3. Print progress and metrics
4. Handle errors gracefully
"""

        if self.use_dspy:
            # Use DSPy module
            result = self.generator_module(
                component_details=component_details,
                competition_context=competition_context,
                data_paths=data_paths,
                requirements=requirements,
            )
            # Extract code from markdown if present
            code = self._extract_code_from_response(result.code)
        else:
            # Use direct LLM call
            prompt = GENERATE_CODE_PROMPT.format(
                component_details=component_details,
                competition_name=competition_info.name,
                domain=domain,
                problem_type=competition_info.problem_type,
                metric=competition_info.evaluation_metric,
                train_data_path=str(working_dir / 'train.csv'),
                test_data_path=str(working_dir / 'test.csv'),
                models_dir=str(working_dir / 'models'),
                submission_path=str(working_dir / 'submission.csv'),
                dataset_info=dataset_info,
                component_name=component.name,
            )

            messages = [
                SystemMessage(content=DEVELOPER_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]

            response = self.llm.invoke(messages)
            code = self._extract_code_from_response(response.content)

        return code

    def _fix_syntax_error(self, code: str, error: str) -> str:
        """Fix syntax error in code."""
        return self._fix_code_error(code, f"SyntaxError: {error}")

    def _get_meta_feedback(self, code: str, error: str, component_name: str) -> str:
        """
        Get quick meta-evaluator feedback on failure (Phase 4: Mini Meta-Evaluator).

        Provides immediate strategic guidance to improve code quality.

        Args:
            code: Failed code
            error: Error message
            component_name: Name of component

        Returns:
            Strategic feedback string
        """
        # Quick analysis prompt
        prompt = f"""You are a Meta-Evaluator analyzing code failure.

Component: {component_name}
Error: {error[:500]}

Code Summary (first 500 lines):
```python
{chr(10).join(code.split(chr(10))[:500])}
```

Provide 2-3 specific, actionable suggestions to fix this error.
Focus on:
1. Root cause of the error
2. Specific code changes needed
3. Best practices to avoid similar errors

Keep response under 150 words."""

        try:
            messages = [
                SystemMessage(content="You are an expert code reviewer and meta-evaluator."),
                HumanMessage(content=prompt),
            ]

            response = self.llm.invoke(messages)
            return response.content.strip()
        except Exception as e:
            return f"Meta-feedback unavailable: {str(e)}"

    def _fix_code_error(self, code: str, error: str) -> str:
        """Fix code based on error."""
        error_info = format_error_info(error)

        if self.use_dspy:
            result = self.fixer_module(
                code=code,
                error=error_info["error"],
                error_type=error_info["error_type"],
            )
            # Extract code from markdown if present
            fixed_code = self._extract_code_from_response(result.fixed_code)
        else:
            prompt = FIX_CODE_PROMPT.format(
                code=code,
                error=error_info["error"],
                error_type=error_info["error_type"],
            )

            messages = [
                SystemMessage(content=DEVELOPER_SYSTEM_PROMPT),
                HumanMessage(content=prompt),
            ]

            response = self.llm.invoke(messages)
            fixed_code = self._extract_code_from_response(response.content)

        return fixed_code

    def _debug_code(
        self,
        code: str,
        exec_result: ExecutionResult,
        working_dir: Path,
        max_iterations: int = 10,
    ) -> tuple[str, ExecutionResult, bool]:
        """Debug code iteratively with loop-safety and shorter timeouts."""
        # Use a shorter timeout during debug to avoid long-running loops
        original_timeout = getattr(self.executor, "timeout", None)
        if original_timeout is not None:
            self.executor.timeout = min(original_timeout, 180)

        last_error_sig = None

        for iteration in range(max_iterations):
            print(f"   Debug iteration {iteration + 1}/{max_iterations}")

            # Prepare debug prompt
            issue = f"Code failed after {iteration + 1} attempts. Errors: {', '.join(exec_result.errors)}"

            prompt = DEBUG_CODE_PROMPT.format(
                code=code,
                issue=issue,
                stdout=exec_result.stdout[-2000:] if exec_result.stdout else "",  # Last 2000 chars
                stderr=exec_result.stderr[-2000:] if exec_result.stderr else "",
            )

            messages = [
                SystemMessage(content=DEVELOPER_SYSTEM_PROMPT + "\n\nYou are in DEBUG MODE. Fix the code carefully."),
                HumanMessage(content=prompt),
            ]

            response = self.llm.invoke(messages)
            debugged_code = self._extract_code_from_response(response.content)

            # Test the debugged code
            test_result = self.executor.execute(debugged_code, working_dir)

            if test_result.success:
                print(f"    Debug successful!")
                # Restore timeout before returning
                if original_timeout is not None:
                    self.executor.timeout = original_timeout
                return debugged_code, test_result, True

            # Detect stagnation (same error repeating) to break out of the loop
            error_sig = "|".join(test_result.errors) if test_result.errors else test_result.stderr.strip()
            if error_sig and error_sig == last_error_sig:
                print("   ⚠️  Debug halted: same error persists; stopping to avoid infinite loop")
                if original_timeout is not None:
                    self.executor.timeout = original_timeout
                return debugged_code, test_result, False

            # Stop early on repeated timeouts
            if any("Timeout" in e for e in test_result.errors):
                print("   ⚠️  Debug halted: repeated timeout during debug")
                if original_timeout is not None:
                    self.executor.timeout = original_timeout
                return debugged_code, test_result, False

            code = debugged_code
            exec_result = test_result
            last_error_sig = error_sig

        print("   L Debug failed after max iterations")
        if original_timeout is not None:
            self.executor.timeout = original_timeout
        return code, exec_result, False

    def _extract_code_from_response(self, response: str) -> str:
        """Extract Python code from LLM response."""
        # Try to extract from markdown code block
        if "```python" in response:
            code = response.split("```python")[1].split("```")[0]
        elif "```" in response:
            code = response.split("```")[1].split("```")[0]
        else:
            code = response

        return code.strip()


# ==================== LangGraph Node Function ====================

def developer_agent_node(state: KaggleState) -> Dict[str, Any]:
    """
    LangGraph node function for the developer agent.

    Args:
        state: Current workflow state

    Returns:
        State updates
    """
    agent = DeveloperAgent()
    return agent(state)
