"""
Utility functions for the Developer Agent prompts.

Contains helper functions for formatting and error categorization.
"""


def format_component_details(component) -> str:
    """Format component details for prompts."""
    name = getattr(component, "name", "Unknown")
    component_type = getattr(component, "component_type", "model")
    estimated_impact = getattr(component, "estimated_impact", 0.0)
    code = getattr(component, "code", "No description")

    return f"""Name: {name}
Type: {component_type}
Estimated Impact: {estimated_impact:.1%}
Description: {code}"""


def format_error_info(error: str) -> dict[str, str]:
    """Categorize and format error information."""
    error_types = {
        "ModuleNotFoundError": "missing_import",
        "FileNotFoundError": "missing_file",
        "KeyError": "missing_key",
        "ValueError": "value_error",
        "TypeError": "type_error",
        "SyntaxError": "syntax_error",
        "MemoryError": "memory_error",
        "Timeout": "timeout",
        "TimeoutError": "timeout",
    }

    error_type = "unknown_error"
    for key, value in error_types.items():
        if key in error:
            error_type = value
            break

    return {
        "error_type": error_type,
        "error": error,
    }
