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
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic

from ..core.state import KaggleState, AblationComponent, DevelopmentResult
from ..core.config import get_config
from ..tools.code_executor import CodeExecutor, ArtifactValidator
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

        # Code executor
        self.executor = CodeExecutor(timeout=600)  # 10 minutes
        self.validator = ArtifactValidator()

        # Always create LLM client (used for debugging even with DSPy)
        if self.config.llm.provider == "openai":
            self.llm = ChatOpenAI(
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
                max_tokens=self.config.llm.max_tokens,
            )
        else:
            self.llm = ChatAnthropic(
                model=self.config.llm.model,
                temperature=self.config.llm.temperature,
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

        # Update state
        return {
            "development_results": [result],
            "current_code": result.code,
            "code_retry_count": 0,
            "current_component_index": current_index + 1 if result.success else current_index,
            "last_updated": datetime.now(),
        }

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

        # Generate initial code
        print("\n   =' Generating code...")
        code = self._generate_code(component, competition_info, working_dir, domain)

        # Validate syntax
        is_valid, syntax_error = self.executor.validate_syntax(code)
        if not is_valid:
            print(f"     Syntax error detected: {syntax_error}")
            code = self._fix_syntax_error(code, syntax_error)

        # Execute with retry
        print("\n     Executing code...")
        max_retries = 5
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

            # Try to fix
            if attempt < max_retries - 1:
                print("   =' Attempting to fix...")
                code = self._fix_code_error(code, exec_result.errors[0] if exec_result.errors else exec_result.stderr)

        # If all retries failed, try debug iterations
        print("\n   = Entering debug mode...")
        code, debug_success = self._debug_code(code, exec_result, working_dir, max_iterations=10)

        if debug_success:
            exec_result = self.executor.execute(code=code, working_dir=working_dir)

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

    def _generate_code(
        self,
        component: AblationComponent,
        competition_info,
        working_dir: Path,
        domain: str,
    ) -> str:
        """Generate code for a component."""
        component_details = format_component_details(component)

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
        exec_result,
        working_dir: Path,
        max_iterations: int = 10,
    ) -> tuple[str, bool]:
        """Debug code iteratively."""
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
                return debugged_code, True

            code = debugged_code
            exec_result = test_result

        print("   L Debug failed after max iterations")
        return code, False

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
