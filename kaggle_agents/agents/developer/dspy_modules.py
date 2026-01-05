"""
DSPy Signatures and Modules for code generation and fixing.

PICKLE-SAFE VERSION:
- Removed explicit type annotations that cause StringSignature pickle errors
- Using string-based signature definitions for better serialization
- Added fallback handling for optimizer persistence
"""

import dspy


# ==================== Simplified Signatures (Pickle-Safe) ====================
# Using simple field definitions without type annotations to avoid pickle errors

class CodeGeneratorSignature(dspy.Signature):
    """Generate ML code for a Kaggle competition component."""

    # Input fields - no type annotations for pickle compatibility
    component_details = dspy.InputField()
    competition_context = dspy.InputField()
    data_paths = dspy.InputField()
    requirements = dspy.InputField()

    # Output fields
    code = dspy.OutputField()


class CodeFixerSignature(dspy.Signature):
    """Fix errors in ML code."""

    # Input fields
    code = dspy.InputField()
    error = dspy.InputField()
    error_type = dspy.InputField()

    # Output field
    fixed_code = dspy.OutputField()


# ==================== Pickle-Safe Modules ====================

class CodeGeneratorModule(dspy.Module):
    """DSPy module for code generation with pickle-safe state handling."""

    def __init__(self):
        super().__init__()
        self.generate = dspy.ChainOfThought(CodeGeneratorSignature)

    def forward(self, component_details, competition_context, data_paths, requirements):
        """Generate code."""
        return self.generate(
            component_details=component_details,
            competition_context=competition_context,
            data_paths=data_paths,
            requirements=requirements,
        )

    def __getstate__(self):
        """Custom pickle serialization - exclude unpicklable ChainOfThought."""
        state = self.__dict__.copy()
        # Remove the unpicklable ChainOfThought object
        state.pop("generate", None)
        # Mark that we need to reinitialize on unpickle
        state["_needs_reinit"] = True
        return state

    def __setstate__(self, state):
        """Custom pickle deserialization - reinitialize ChainOfThought."""
        self.__dict__.update(state)
        # Reinitialize the ChainOfThought that was excluded from pickle
        self.generate = dspy.ChainOfThought(CodeGeneratorSignature)
        self.__dict__.pop("_needs_reinit", None)


class CodeFixerModule(dspy.Module):
    """DSPy module for code fixing with pickle-safe state handling."""

    def __init__(self):
        super().__init__()
        self.fix = dspy.ChainOfThought(CodeFixerSignature)

    def forward(self, code, error, error_type):
        """Fix code."""
        return self.fix(code=code, error=error, error_type=error_type)

    def __getstate__(self):
        """Custom pickle serialization - exclude unpicklable ChainOfThought."""
        state = self.__dict__.copy()
        # Remove the unpicklable ChainOfThought object
        state.pop("fix", None)
        # Mark that we need to reinitialize on unpickle
        state["_needs_reinit"] = True
        return state

    def __setstate__(self, state):
        """Custom pickle deserialization - reinitialize ChainOfThought."""
        self.__dict__.update(state)
        # Reinitialize the ChainOfThought that was excluded from pickle
        self.fix = dspy.ChainOfThought(CodeFixerSignature)
        self.__dict__.pop("_needs_reinit", None)


# ==================== Safe Save/Load Utilities ====================

def save_module_safe(module: dspy.Module, path: str) -> bool:
    """
    Safely save a DSPy module, handling pickle errors gracefully.

    Returns True if saved successfully, False otherwise.
    """
    import pickle
    from pathlib import Path

    try:
        with open(path, "wb") as f:
            pickle.dump(module, f)
        print(f"[LOG:INFO] Successfully saved DSPy module to {path}")
        return True
    except (pickle.PicklingError, TypeError, AttributeError) as e:
        print(f"[LOG:WARNING] Could not pickle DSPy module: {e}")
        print("[LOG:INFO] Using in-memory optimization only (state not persisted)")

        # Try saving just the module parameters as JSON fallback
        try:
            import json
            params_path = Path(path).with_suffix(".json")
            # Extract any learnable parameters
            params = {}
            if hasattr(module, "named_parameters"):
                for name, param in module.named_parameters():
                    if hasattr(param, "tolist"):
                        params[name] = param.tolist()
                    else:
                        params[name] = str(param)

            with open(params_path, "w") as f:
                json.dump(params, f, indent=2)
            print(f"[LOG:INFO] Saved module parameters to {params_path}")
        except Exception as json_err:
            print(f"[LOG:DEBUG] JSON fallback also failed: {json_err}")

        return False


def load_module_safe(path: str, module_class: type) -> dspy.Module:
    """
    Safely load a DSPy module, falling back to fresh initialization if needed.

    Args:
        path: Path to saved module
        module_class: Class to instantiate if loading fails

    Returns:
        Loaded or fresh module instance
    """
    import pickle

    try:
        with open(path, "rb") as f:
            module = pickle.load(f)
        print(f"[LOG:INFO] Successfully loaded DSPy module from {path}")
        return module
    except (FileNotFoundError, pickle.UnpicklingError, Exception) as e:
        print(f"[LOG:WARNING] Could not load DSPy module: {e}")
        print("[LOG:INFO] Initializing fresh module instance")
        return module_class()
