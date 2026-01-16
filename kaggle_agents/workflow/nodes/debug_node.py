"""
PiML-style Debug Chain Node for the Kaggle Agents workflow.

Implements a 3-attempt debug loop before escalating to the planner.
Based on PiML (Persistent Iterative Machine Learning) debug chain approach.
"""

from datetime import datetime
from typing import Any

from ...core.state import KaggleState

# Maximum debug attempts before escalating to planner
DEBUG_MAX_ATTEMPTS = 3


def _summarize_error(state: KaggleState) -> str:
    """
    Summarize the error context for escalation to planner.

    Args:
        state: Current state with error information

    Returns:
        A summary diagnosis string for the planner
    """
    development_results = state.get("development_results", [])
    errors = state.get("errors", [])
    debug_history = state.get("debug_history", [])

    summary_parts = []

    # Analyze recent development results for patterns
    if development_results:
        recent_failures = [r for r in development_results[-3:] if not r.success]
        if recent_failures:
            error_types = set()
            for result in recent_failures:
                stderr = result.stderr or ""
                if "MemoryError" in stderr or "out of memory" in stderr.lower():
                    error_types.add("memory_exhaustion")
                elif "TimeoutError" in stderr or "timeout" in stderr.lower():
                    error_types.add("execution_timeout")
                elif "ImportError" in stderr or "ModuleNotFoundError" in stderr:
                    error_types.add("missing_dependency")
                elif "SyntaxError" in stderr:
                    error_types.add("syntax_error")
                elif "RuntimeError" in stderr:
                    error_types.add("runtime_error")
                elif "FileNotFoundError" in stderr:
                    error_types.add("file_not_found")
                elif "KeyError" in stderr or "IndexError" in stderr:
                    error_types.add("data_access_error")
                elif "ValueError" in stderr or "TypeError" in stderr:
                    error_types.add("type_or_value_error")
                else:
                    error_types.add("unknown_error")

            summary_parts.append(f"Error types: {', '.join(error_types)}")

            # Extract last error message
            last_failure = recent_failures[-1]
            if last_failure.stderr:
                # Get last 500 chars of stderr
                last_error = last_failure.stderr[-500:]
                summary_parts.append(f"Last error: {last_error}")

    # Include debug attempt history
    if debug_history:
        summary_parts.append(f"Debug attempts: {len(debug_history)}")
        for i, attempt in enumerate(debug_history[-3:], 1):
            summary_parts.append(f"  Attempt {i}: {attempt.get('fix_strategy', 'unknown')}")

    # Include global errors
    if errors:
        recent_errors = errors[-3:]
        summary_parts.append(f"Recent errors: {'; '.join(recent_errors)}")

    return "\n".join(summary_parts) if summary_parts else "Unknown error - no diagnostic information available"


def _analyze_error(state: KaggleState) -> dict[str, Any]:
    """
    Analyze the error to determine fix strategy.

    Args:
        state: Current state with error information

    Returns:
        Dict with error analysis including type, cause, and suggested fix strategy
    """
    development_results = state.get("development_results", [])

    if not development_results:
        return {"error_type": "unknown", "fix_strategy": "retry_clean", "confidence": 0.0}

    last_result = development_results[-1]
    stderr = last_result.stderr or ""
    stdout = last_result.stdout or ""
    current_code = state.get("current_code", "")

    analysis = {
        "error_type": "unknown",
        "fix_strategy": "retry_clean",
        "confidence": 0.5,
        "specific_issue": None,
        "suggested_fix": None,
    }

    # Memory errors
    if "MemoryError" in stderr or "out of memory" in stderr.lower() or "CUDA out of memory" in stderr:
        analysis["error_type"] = "memory_exhaustion"
        analysis["fix_strategy"] = "reduce_batch_size"
        analysis["confidence"] = 0.9
        analysis["suggested_fix"] = "Reduce batch_size, use gradient accumulation, or process in chunks"

    # Timeout errors
    elif "TimeoutError" in stderr or "timeout" in stderr.lower():
        analysis["error_type"] = "execution_timeout"
        analysis["fix_strategy"] = "optimize_performance"
        analysis["confidence"] = 0.8
        analysis["suggested_fix"] = "Reduce data size, use vectorized operations, or limit iterations"

    # Import errors
    elif "ImportError" in stderr or "ModuleNotFoundError" in stderr:
        analysis["error_type"] = "missing_dependency"
        analysis["fix_strategy"] = "fix_imports"
        analysis["confidence"] = 0.95
        # Extract module name
        import re
        match = re.search(r"No module named ['\"](\w+)['\"]", stderr)
        if match:
            analysis["specific_issue"] = match.group(1)
            analysis["suggested_fix"] = f"Install or import {match.group(1)} correctly"

    # Syntax errors
    elif "SyntaxError" in stderr:
        analysis["error_type"] = "syntax_error"
        analysis["fix_strategy"] = "fix_syntax"
        analysis["confidence"] = 0.95
        analysis["suggested_fix"] = "Fix the syntax error in the code"

    # File not found
    elif "FileNotFoundError" in stderr:
        analysis["error_type"] = "file_not_found"
        analysis["fix_strategy"] = "fix_file_paths"
        analysis["confidence"] = 0.9
        analysis["suggested_fix"] = "Use correct file paths from injected variables"

    # Data type mismatches (common in PyTorch)
    elif "RuntimeError" in stderr and ("type" in stderr.lower() or "dtype" in stderr.lower()):
        analysis["error_type"] = "type_mismatch"
        analysis["fix_strategy"] = "fix_tensor_types"
        analysis["confidence"] = 0.85
        analysis["suggested_fix"] = "Add .float() conversion or ensure tensor dtype consistency"

    # Shape mismatches
    elif "shape" in stderr.lower() or "dimension" in stderr.lower():
        analysis["error_type"] = "shape_mismatch"
        analysis["fix_strategy"] = "fix_tensor_shapes"
        analysis["confidence"] = 0.8
        analysis["suggested_fix"] = "Check input/output dimensions and reshape as needed"

    # Data access errors
    elif "KeyError" in stderr or "IndexError" in stderr:
        analysis["error_type"] = "data_access_error"
        analysis["fix_strategy"] = "fix_data_access"
        analysis["confidence"] = 0.85
        analysis["suggested_fix"] = "Use correct keys/indices, check data structure"

    # Value/Type errors
    elif "ValueError" in stderr or "TypeError" in stderr:
        analysis["error_type"] = "type_or_value_error"
        analysis["fix_strategy"] = "fix_types_values"
        analysis["confidence"] = 0.75
        analysis["suggested_fix"] = "Check function arguments and data types"

    return analysis


def _generate_fix_guidance(state: KaggleState, error_analysis: dict[str, Any]) -> str:
    """
    Generate specific fix guidance based on error analysis.

    Args:
        state: Current state
        error_analysis: Analysis from _analyze_error

    Returns:
        Guidance string for the developer
    """
    fix_strategy = error_analysis.get("fix_strategy", "retry_clean")
    suggested_fix = error_analysis.get("suggested_fix", "")
    specific_issue = error_analysis.get("specific_issue")

    guidance_parts = [
        f"Debug Chain Attempt - Fix Strategy: {fix_strategy}",
        f"Error Type: {error_analysis.get('error_type', 'unknown')}",
    ]

    if specific_issue:
        guidance_parts.append(f"Specific Issue: {specific_issue}")

    if suggested_fix:
        guidance_parts.append(f"Suggested Fix: {suggested_fix}")

    # Add strategy-specific guidance
    strategy_guidance = {
        "reduce_batch_size": [
            "- Reduce batch_size by 50% (e.g., 32 -> 16)",
            "- Add gradient accumulation if needed",
            "- Consider processing data in smaller chunks",
        ],
        "optimize_performance": [
            "- Use vectorized numpy/pandas operations",
            "- Limit number of iterations or epochs",
            "- Consider sampling a subset of data for initial testing",
        ],
        "fix_imports": [
            "- Check that all imports are available in the environment",
            "- Use alternative libraries if module not available",
            "- Verify spelling of module names",
        ],
        "fix_syntax": [
            "- Check for missing colons, parentheses, brackets",
            "- Verify indentation is consistent",
            "- Look for incomplete statements",
        ],
        "fix_file_paths": [
            "- Use TRAIN_DATA_PATH, TEST_DATA_PATH, MODELS_DIR variables",
            "- Check if file exists before reading",
            "- Use Path objects for cross-platform compatibility",
        ],
        "fix_tensor_types": [
            "- Add .float() after tensor operations",
            "- Ensure model and data use same dtype",
            "- Convert numpy arrays correctly: torch.from_numpy(arr).float()",
        ],
        "fix_tensor_shapes": [
            "- Print shapes before operations for debugging",
            "- Use .view() or .reshape() to fix dimensions",
            "- Check batch dimension handling",
        ],
        "fix_data_access": [
            "- Verify keys exist before accessing",
            "- Check array bounds before indexing",
            "- Use .get() for dict access with defaults",
        ],
        "fix_types_values": [
            "- Check function signatures for expected types",
            "- Validate data before passing to functions",
            "- Add type conversions where needed",
        ],
    }

    if fix_strategy in strategy_guidance:
        guidance_parts.append("\nSpecific Actions:")
        guidance_parts.extend(strategy_guidance[fix_strategy])

    return "\n".join(guidance_parts)


def debug_node(state: KaggleState) -> dict[str, Any]:
    """
    PiML Debug Chain: 3-attempt fix loop before escalation.

    This node attempts to diagnose and guide fixes for developer errors
    before escalating to the planner for a new approach.

    Args:
        state: Current state with error information

    Returns:
        State updates with debug guidance or escalation
    """
    print("\n" + "=" * 60)
    print("= DEBUG CHAIN (PiML)")
    print("=" * 60)

    debug_attempt = state.get("debug_attempt", 0)
    debug_history = state.get("debug_history", [])

    print(f"\nDebug Attempt: {debug_attempt + 1}/{DEBUG_MAX_ATTEMPTS}")

    # Check if we've exceeded max attempts
    if debug_attempt >= DEBUG_MAX_ATTEMPTS:
        print(f"\n⚠️  Max debug attempts ({DEBUG_MAX_ATTEMPTS}) reached - escalating to planner")
        diagnosis = _summarize_error(state)
        print(f"\nDiagnosis Summary:\n{diagnosis[:500]}...")

        return {
            "debug_escalate": True,
            "debug_diagnosis": diagnosis,
            "debug_attempt": 0,  # Reset for next component
            "debug_history": [],  # Clear history
            "debug_guidance": None,
            "last_updated": datetime.now(),
        }

    # Analyze the error
    error_analysis = _analyze_error(state)
    print(f"\nError Analysis:")
    print(f"   Type: {error_analysis['error_type']}")
    print(f"   Strategy: {error_analysis['fix_strategy']}")
    print(f"   Confidence: {error_analysis['confidence']:.0%}")

    # Generate fix guidance
    fix_guidance = _generate_fix_guidance(state, error_analysis)
    print(f"\nFix Guidance Generated ({len(fix_guidance)} chars)")

    # Record this debug attempt
    new_history_entry = {
        "attempt": debug_attempt + 1,
        "error_type": error_analysis["error_type"],
        "fix_strategy": error_analysis["fix_strategy"],
        "timestamp": datetime.now().isoformat(),
    }

    updated_history = list(debug_history) + [new_history_entry]

    print(f"\n🔧 Sending debug guidance to developer (attempt {debug_attempt + 1}/{DEBUG_MAX_ATTEMPTS})")

    return {
        "debug_attempt": debug_attempt + 1,
        "debug_escalate": False,
        "debug_guidance": fix_guidance,
        "debug_history": updated_history,
        "debug_diagnosis": None,
        "last_updated": datetime.now(),
    }
