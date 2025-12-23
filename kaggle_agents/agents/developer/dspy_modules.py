"""
DSPy Signatures and Modules for code generation and fixing.
"""

import dspy


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


class CodeGeneratorModule(dspy.Module):
    """DSPy module for code generation."""

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


class CodeFixerModule(dspy.Module):
    """DSPy module for code fixing."""

    def __init__(self):
        super().__init__()
        self.fix = dspy.ChainOfThought(CodeFixerSignature)

    def forward(self, code, error, error_type):
        """Fix code."""
        return self.fix(code=code, error=error, error_type=error_type)
