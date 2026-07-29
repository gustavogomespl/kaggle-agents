"""Static contract checks applied to generated code before it is executed.

Both checks answer the same question cheaply: will this program produce the
evidence the pipeline needs? Answering it after execution costs a full training
budget and then throws the result away, which is exactly how a smoke run spent
its GPU hours re-running work that had already succeeded.

They live here rather than in the agent because the retry loop and the debug
loop both need them, and importing across those two would be circular.
"""

from __future__ import annotations

import ast

# Injected into every model script by the code generator; one call writes all
# four evidence artifacts under the correct component name.
ARTIFACT_HELPER = "save_component_artifacts"

# Also injected into every model script; fills the template's prediction
# columns without the caller having to identify them.
SUBMISSION_HELPER = "write_submission"


def requires_submission_helper(component_type: str) -> bool:
    """Whether a generated component owns a final submission CSV."""
    return str(component_type).lower() in {"model", "ensemble"}


def _helper_body_nodes(tree: ast.AST, helper_name: str) -> set[int]:
    """Node ids belonging to the injected helper's own definition.

    The helper is defined in every model script, so calls inside its body prove
    nothing about the program that was generated around it.
    """
    return {
        id(sub)
        for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == helper_name
        for sub in ast.walk(node)
    }


def handwritten_submission_write(code: str) -> str | None:
    """Report a submission written by hand instead of through the helper.

    Generated code reaches for ``sample_sub[sample_sub.columns[1]] = preds``,
    which on a template whose first column is the prediction writes the model's
    output into an input column and leaves the graded column holding its
    placeholder. The result is structurally valid and scores nothing, and it is
    only caught after training has already been paid for.

    Reading is conservative: an unparseable program, or one that writes no
    submission at all, yields no finding.

    Returns:
        The offending expression, or None when the contract is satisfied.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    helper_body_nodes = _helper_body_nodes(tree, SUBMISSION_HELPER)
    for node in ast.walk(tree):
        if id(node) in helper_body_nodes or not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "attr", "") != "to_csv":
            continue
        destination_node = node.args[0] if node.args else next(
            (
                keyword.value
                for keyword in node.keywords
                if keyword.arg in {"path_or_buf", "path"}
            ),
            None,
        )
        if destination_node is None:
            continue
        destination = ast.unparse(destination_node)
        if "SUBMISSION_PATH" in destination or "submission.csv" in destination:
            return destination
    return None


def missing_submission_helper_call(code: str) -> bool:
    """Whether model code omits a call to the injected submission helper."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    helper_body_nodes = _helper_body_nodes(tree, SUBMISSION_HELPER)
    return not any(
        isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == SUBMISSION_HELPER
        and id(node) not in helper_body_nodes
        for node in ast.walk(tree)
    )


def missing_class_order_helper_argument(code: str) -> bool:
    """Whether every evidence-helper call omits a concrete class order."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    helper_body_nodes = _helper_body_nodes(tree, ARTIFACT_HELPER)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", "") == ARTIFACT_HELPER
        and id(node) not in helper_body_nodes
    ]
    for call in calls:
        class_order_node: ast.expr | None = (
            call.args[4] if len(call.args) >= 5 else None
        )
        for keyword in call.keywords:
            if keyword.arg == "class_order":
                class_order_node = keyword.value
                break
        if class_order_node is not None and not (
            isinstance(class_order_node, ast.Constant)
            and class_order_node.value is None
        ):
            return False
    return True


def untrusted_contract_helper_import(code: str) -> str | None:
    """Report code that shadows the helpers injected into model scripts.

    The static artifact checks recognize calls by helper name.  Reimporting one
    or assigning/redefining one makes a call look compliant even though
    execution bypasses the injected contract implementation.
    """
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None

    helper_names = {ARTIFACT_HELPER, SUBMISSION_HELPER}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)) and any(
            alias.asname in helper_names or alias.name in helper_names
            for alias in node.names
        ):
            return ast.unparse(node)

    helper_definitions: dict[str, list[ast.FunctionDef | ast.AsyncFunctionDef]] = {
        name: [] for name in helper_names
    }
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name in helper_definitions:
                helper_definitions[node.name].append(node)
    for definitions in helper_definitions.values():
        if len(definitions) > 1:
            return ast.unparse(definitions[1])

    for node in ast.walk(tree):
        targets: list[ast.expr] = []
        if isinstance(node, ast.Assign):
            targets = list(node.targets)
        elif isinstance(node, (ast.AnnAssign, ast.AugAssign, ast.NamedExpr)):
            targets = [node.target]
        for target in targets:
            global_key = None
            if (
                isinstance(target, ast.Subscript)
                and isinstance(target.value, ast.Call)
                and isinstance(target.value.func, ast.Name)
                and target.value.func.id == "globals"
                and not target.value.args
                and not target.value.keywords
                and isinstance(target.slice, ast.Constant)
                and isinstance(target.slice.value, str)
            ):
                global_key = target.slice.value
            if global_key in helper_names:
                return ast.unparse(node)
            names = {
                child.id
                for child in ast.walk(target)
                if isinstance(child, ast.Name)
            }
            if names & helper_names:
                return ast.unparse(node)
    return None


SUBMISSION_CONTRACT_ERROR = (
    f"Submission must be written with {SUBMISSION_HELPER}(test_preds): picking "
    "submission columns by position writes predictions into an input column "
    "and leaves the graded column unfilled"
)

HELPER_IMPORT_CONTRACT_ERROR = (
    f"Do not import {SUBMISSION_HELPER} or {ARTIFACT_HELPER}: both helpers are "
    "injected into the script and imports can shadow the validated implementation"
)

MISSING_SUBMISSION_HELPER_ERROR = (
    f"Model code must call the injected {SUBMISSION_HELPER}(test_preds) helper "
    "before it can execute"
)

MISSING_CLASS_ORDER_ERROR = (
    "Multiclass model code must pass the canonical probability-column order "
    "as class_order= to save_component_artifacts(...)"
)
