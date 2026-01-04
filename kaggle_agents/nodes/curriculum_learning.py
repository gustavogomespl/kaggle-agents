"""
WEBRL-inspired Curriculum Learning Node.

Implements self-evolving curriculum that creates sub-tasks from failures.
When the agent fails, it generates specific sub-tasks to resolve the
problem before proceeding, improving resilience and learning.

Based on: WEBRL - Training LLM Web Agents via Self-Evolving Curriculum
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal

from langchain_core.messages import HumanMessage

from ..core.config import get_llm_for_role
from ..core.state import KaggleState
from ..utils.llm_utils import get_text_content


@dataclass
class SubTask:
    """Sub-task generated from a failure to resolve before proceeding."""

    parent_component: str
    failure_type: str  # "memory", "timeout", "syntax", "validation", etc.
    task_description: str
    priority: int  # 1 (highest) to 5 (lowest)
    status: Literal["pending", "in_progress", "resolved", "skipped"] = "pending"
    resolution_code: str | None = None
    resolution_guidance: str | None = None
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for state storage."""
        return {
            "parent_component": self.parent_component,
            "failure_type": self.failure_type,
            "task_description": self.task_description,
            "priority": self.priority,
            "status": self.status,
            "resolution_code": self.resolution_code,
            "resolution_guidance": self.resolution_guidance,
            "created_at": self.created_at.isoformat(),
        }


# ==================== Error to SubTask Mapping ====================

ERROR_TO_SUBTASK_TEMPLATE = {
    "memory_error": {
        "task_description": "Implement memory optimization: sampling, chunk processing, or reduced batch sizes",
        "priority": 1,
        "guidance": """
        Solutions to try:
        1. Use df.sample(frac=0.3) for initial development
        2. Process data in chunks with pd.read_csv(chunksize=10000)
        3. Reduce batch_size in model training
        4. Use float32 instead of float64: df = df.astype('float32')
        5. Delete unused variables with del and gc.collect()
        """,
    },
    "timeout_error": {
        "task_description": "Optimize code for speed: early stopping, simpler model, reduced CV folds",
        "priority": 1,
        "guidance": """
        Solutions to try:
        1. Add early stopping (LightGBM callbacks; XGBoost 2.0+ constructor or callbacks for <2)
        2. Reduce n_estimators (try 100 instead of 1000)
        3. Reduce CV folds from 5 to 3
        4. Use simpler model (e.g., LogisticRegression as baseline)
        5. Limit hyperparameter search iterations
        """,
    },
    "dimension_mismatch": {
        "task_description": "Add shape validation and fix dimension mismatches in data/model pipeline",
        "priority": 2,
        "guidance": """
        Solutions to try:
        1. Add print(X.shape, y.shape) before model.fit()
        2. Ensure train/test have same columns after feature engineering
        3. Use sklearn ColumnTransformer for consistent preprocessing
        4. Check for NaN rows that might be dropped inconsistently
        5. Verify target column is not included in features
        """,
    },
    "import_error": {
        "task_description": "Handle missing dependencies with fallback imports or pip install",
        "priority": 1,
        "guidance": """
        Solutions to try:
        1. Add try/except with fallback: try: import xgboost except: from sklearn.ensemble import GradientBoostingClassifier
        2. Use subprocess.run(['pip', 'install', 'package']) at runtime
        3. Check if package is available with importlib.util.find_spec()
        4. Use lighter alternatives (e.g., sklearn instead of catboost)
        """,
    },
    "data_contains_nans": {
        "task_description": "Implement robust NaN handling with imputation or removal strategy",
        "priority": 2,
        "guidance": """
        Solutions to try:
        1. Use SimpleImputer(strategy='median') for numeric columns
        2. Use SimpleImputer(strategy='most_frequent') for categorical
        3. Add df.dropna() for rows with critical missing values
        4. Create missing indicator features: df['col_missing'] = df['col'].isna()
        5. Use IterativeImputer for sophisticated imputation
        """,
    },
    "key_error": {
        "task_description": "Fix column name mismatches between train and test data",
        "priority": 2,
        "guidance": """
        Solutions to try:
        1. Print df.columns to verify column names
        2. Use df.columns.str.lower().str.strip() for normalization
        3. Check sample_submission.csv for expected output columns
        4. Ensure feature engineering produces same columns for train/test
        5. Use df.get('column_name', default_value) for safe access
        """,
    },
    "type_error": {
        "task_description": "Fix data type mismatches in pipeline (string vs numeric, etc.)",
        "priority": 2,
        "guidance": """
        Solutions to try:
        1. Convert columns explicitly: df['col'] = pd.to_numeric(df['col'], errors='coerce')
        2. Handle mixed types with df.infer_objects()
        3. Check for object dtype columns that should be numeric
        4. Use LabelEncoder for string categorical columns before model
        5. Ensure consistent dtypes between train and test
        """,
    },
    "syntax_error": {
        "task_description": "Fix Python syntax errors in generated code",
        "priority": 1,
        "guidance": """
        Common fixes:
        1. Check for unmatched parentheses, brackets, quotes
        2. Verify indentation is consistent (use 4 spaces)
        3. Check for missing colons after if/for/def
        4. Verify string formatting (f-strings, .format())
        5. Check for Python 2 vs 3 compatibility issues
        """,
    },
    "validation_error": {
        "task_description": "Fix validation/output format issues for submission",
        "priority": 1,
        "guidance": """
        Solutions to try:
        1. Match sample_submission.csv format exactly
        2. Ensure submission has correct ID column name
        3. Check prediction column dtype matches expected
        4. Verify no NaN values in predictions
        5. Ensure correct number of rows (len(test))
        """,
    },
    "runtime_error": {
        "task_description": "Debug and fix runtime errors in code execution",
        "priority": 2,
        "guidance": """
        Debugging steps:
        1. Add try/except blocks around risky operations
        2. Print intermediate values to identify failure point
        3. Check for division by zero, empty arrays
        4. Verify file paths exist before reading
        5. Add input validation at function entry points
        """,
    },
    "attribute_error": {
        "task_description": "Fix missing attribute/method calls on objects",
        "priority": 2,
        "guidance": """
        Solutions to try:
        1. Check object type with type(obj) and dir(obj)
        2. Verify sklearn version compatibility (fit_transform vs fit().transform())
        3. Check if model is fitted before calling predict()
        4. Use hasattr(obj, 'method') before calling
        5. Verify pandas Series vs DataFrame methods
        """,
    },
    "index_error": {
        "task_description": "Fix array/list indexing issues",
        "priority": 2,
        "guidance": """
        Solutions to try:
        1. Check array length before indexing: if len(arr) > idx
        2. Use .iloc for positional indexing in pandas
        3. Verify loop bounds are correct
        4. Handle empty arrays/DataFrames explicitly
        5. Use .get() for safe dictionary access
        """,
    },
}


def generate_subtask_from_error(
    error_type: str,
    parent_component: str,
    error_message: str = "",
    state: KaggleState | None = None,
) -> SubTask:
    """
    Generate a SubTask from an error type with context-aware guidance.

    Args:
        error_type: Type of error (e.g., "memory_error", "timeout_error")
        parent_component: Name of the component that failed
        error_message: Actual error message for context
        state: Current workflow state for additional context

    Returns:
        SubTask with resolution guidance
    """
    template = ERROR_TO_SUBTASK_TEMPLATE.get(
        error_type,
        {
            "task_description": f"Fix {error_type} in code generation",
            "priority": 3,
            "guidance": "Debug the error and implement a fix.",
        },
    )

    return SubTask(
        parent_component=parent_component,
        failure_type=error_type,
        task_description=template["task_description"],
        priority=template["priority"],
        resolution_guidance=template["guidance"],
    )


def generate_subtask_with_llm(
    error_type: str,
    parent_component: str,
    error_message: str,
    state: KaggleState,
) -> SubTask:
    """
    Generate a SubTask using LLM for sophisticated error analysis.

    Uses the LLM to analyze the error context and generate targeted
    resolution guidance, including code snippets when applicable.

    Args:
        error_type: Type of error
        parent_component: Component that failed
        error_message: Full error message
        state: Current workflow state

    Returns:
        SubTask with LLM-generated guidance
    """
    llm = get_llm_for_role("evaluator")

    # Get context from state
    domain = state.get("domain_detected", "unknown")
    competition_info = state.get("competition_info")
    comp_name = competition_info.name if competition_info else "unknown"
    metric = competition_info.evaluation_metric if competition_info else "unknown"

    # Get recent code if available
    dev_results = state.get("development_results", [])
    recent_code = ""
    if dev_results:
        last_result = dev_results[-1]
        code_lines = (last_result.code or "").split("\n")
        if len(code_lines) > 40:
            recent_code = "\n".join(code_lines[:20]) + "\n...\n" + "\n".join(code_lines[-10:])
        else:
            recent_code = last_result.code or ""
        recent_code = recent_code[:2000]

    prompt = f"""You are an expert ML engineer analyzing a failure in a Kaggle competition pipeline.

## Context
- **Competition**: {comp_name}
- **Domain**: {domain}
- **Metric**: {metric}
- **Failed Component**: {parent_component}
- **Error Type**: {error_type}

## Error Message
```
{error_message[:1000]}
```

## Recent Code (if available)
```python
{recent_code}
```

## Your Task
Generate a specific sub-task to resolve this error. Consider:
1. The root cause of the error
2. The competition context and domain
3. Best practices for the specific ML framework involved
4. Concrete code changes needed

## Response Format
Return a JSON object:
{{
    "task_description": "Clear, actionable description of what needs to be done (1-2 sentences)",
    "priority": 1-5 (1 = critical/blocking, 2 = high, 3 = medium, 4 = low, 5 = minor),
    "resolution_steps": [
        "Step 1: Specific action",
        "Step 2: Another action",
        "Step 3: Verification"
    ],
    "code_snippet": "```python\\n# Example fix code here\\n```",
    "rationale": "Brief explanation of why this fix works"
}}

Be specific and actionable. Include parameter values, function names, and code patterns."""

    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        content = get_text_content(response.content).strip()

        # Parse JSON response
        if "```json" in content:
            content = content.split("```json")[1].split("```")[0].strip()
        elif "```" in content:
            # Try to find JSON block
            parts = content.split("```")
            for part in parts:
                if part.strip().startswith("{"):
                    content = part.strip()
                    break

        import json

        result = json.loads(content)

        # Build guidance from steps and rationale
        steps = result.get("resolution_steps", [])
        rationale = result.get("rationale", "")
        guidance = "\n".join(f"- {step}" for step in steps)
        if rationale:
            guidance += f"\n\nRationale: {rationale}"

        return SubTask(
            parent_component=parent_component,
            failure_type=error_type,
            task_description=result.get("task_description", f"Fix {error_type}"),
            priority=result.get("priority", 2),
            resolution_guidance=guidance,
            resolution_code=result.get("code_snippet"),
        )
    except Exception as e:
        # Fallback to simple description
        print(f"   LLM subtask generation failed: {e}, using fallback")
        return SubTask(
            parent_component=parent_component,
            failure_type=error_type,
            task_description=f"Fix {error_type} in {parent_component}",
            priority=2,
            resolution_guidance=f"Error: {error_message[:200]}",
        )


# ==================== Curriculum Learning Node ====================


def curriculum_learning_node(state: KaggleState) -> dict[str, Any]:
    """
    WEBRL-inspired curriculum learning node.

    Analyzes failures and generates sub-tasks to resolve them before proceeding.
    This creates a self-evolving curriculum where the agent learns to overcome
    specific challenges.

    Args:
        state: Current workflow state

    Returns:
        State updates with curriculum subtasks
    """
    print("\n" + "=" * 60)
    print("= CURRICULUM LEARNING: Generating Sub-tasks from Failures")
    print("=" * 60)

    failure_analysis = state.get("failure_analysis", {})
    error_patterns = failure_analysis.get("error_patterns", [])
    failed_components = failure_analysis.get("failed_components", [])

    if not error_patterns and not failed_components:
        print("\n   No failures to analyze - skipping curriculum generation")
        return {
            "curriculum_subtasks": [],
            "needs_subtask_resolution": False,
        }

    print(f"\n   Found {len(error_patterns)} error patterns to address")

    subtasks = []

    # Generate subtasks from error patterns
    for error_type in error_patterns:
        # Find the component that had this error
        parent_component = "unknown"
        error_message = ""

        for failed in failed_components:
            if failed.get("error_type") == error_type:
                parent_component = failed.get("name", "unknown")
                error_message = failed.get("error", "")
                break

        # Check if we should use LLM for sophisticated analysis
        use_llm = state.get("fast_mode", False) is False  # Only use LLM in non-fast mode

        if use_llm and error_message:
            subtask = generate_subtask_with_llm(error_type, parent_component, error_message, state)
        else:
            subtask = generate_subtask_from_error(
                error_type, parent_component, error_message, state
            )

        subtasks.append(subtask)
        print(f"   + SubTask: {subtask.task_description[:60]}... (priority={subtask.priority})")

    # Sort by priority (highest first)
    subtasks.sort(key=lambda s: s.priority)

    # Limit to top 3 most critical subtasks to avoid overwhelming
    subtasks = subtasks[:3]

    print(f"\n   Generated {len(subtasks)} priority sub-tasks")

    # Convert to dict for state storage
    subtask_dicts = [s.to_dict() for s in subtasks]

    return {
        "curriculum_subtasks": subtask_dicts,
        "needs_subtask_resolution": len(subtasks) > 0,
        "last_updated": datetime.now(),
    }


def route_after_curriculum(state: KaggleState) -> Literal["resolve", "continue"]:
    """
    Route after curriculum learning - decide if subtasks need resolution.

    Args:
        state: Current state

    Returns:
        "resolve" if subtasks need resolution, "continue" otherwise
    """
    needs_resolution = state.get("needs_subtask_resolution", False)
    subtasks = state.get("curriculum_subtasks", [])

    # Check if there are pending subtasks
    pending = [s for s in subtasks if s.get("status") == "pending"]

    if needs_resolution and pending:
        print(f"\n   {len(pending)} subtasks need resolution before continuing")
        return "resolve"

    return "continue"


def inject_subtask_guidance(state: KaggleState) -> dict[str, Any]:
    """
    Inject subtask resolution guidance into the developer prompt.

    This modifies the state to include guidance from curriculum subtasks
    so the developer agent can use it when regenerating code.

    Args:
        state: Current workflow state

    Returns:
        State updates with injected guidance
    """
    subtasks = state.get("curriculum_subtasks", [])

    if not subtasks:
        return {}

    # Build guidance string from pending subtasks
    guidance_parts = []

    for subtask in subtasks:
        if subtask.get("status") in ["pending", "in_progress"]:
            # Build code example section separately to avoid f-string backslash issue
            code_section = ""
            if subtask.get("resolution_code"):
                code_section = f"**Example Code**:\n```python\n{subtask['resolution_code']}\n```"

            guidance_parts.append(f"""
### CRITICAL FIX REQUIRED: {subtask["failure_type"].upper()}

**Problem**: {subtask["task_description"]}

**Resolution Steps**:
{subtask.get("resolution_guidance", "Apply standard debugging techniques.")}

{code_section}
""")

    if not guidance_parts:
        return {}

    # Combine with existing refinement guidance
    existing_guidance = state.get("refinement_guidance", {})
    curriculum_guidance = "\n---\n".join(guidance_parts)

    # Merge guidance
    updated_guidance = {
        **existing_guidance,
        "curriculum_fixes": curriculum_guidance,
        "priority_errors": [s["failure_type"] for s in subtasks if s.get("status") == "pending"],
    }

    # Update developer guidance with curriculum context
    if "developer_guidance" in updated_guidance:
        updated_guidance["developer_guidance"] = (
            updated_guidance["developer_guidance"]
            + "\n\n## CURRICULUM FIXES (from previous failures):\n"
            + curriculum_guidance
        )
    else:
        updated_guidance["developer_guidance"] = curriculum_guidance

    print("\n   Injected curriculum guidance into developer prompt")

    return {
        "refinement_guidance": updated_guidance,
    }


def mark_subtask_resolved(
    state: KaggleState,
    error_type: str,
    resolution_code: str | None = None,
) -> dict[str, Any]:
    """
    Mark a subtask as resolved after successful execution.

    Args:
        state: Current workflow state
        error_type: The error type that was resolved
        resolution_code: The code that resolved the issue

    Returns:
        State updates with resolved subtask
    """
    subtasks = state.get("curriculum_subtasks", [])

    for subtask in subtasks:
        if subtask.get("failure_type") == error_type and subtask.get("status") == "pending":
            subtask["status"] = "resolved"
            if resolution_code:
                subtask["resolution_code"] = resolution_code
            print(f"   Resolved subtask: {error_type}")
            break

    return {
        "curriculum_subtasks": subtasks,
        "needs_subtask_resolution": any(s.get("status") == "pending" for s in subtasks),
    }
