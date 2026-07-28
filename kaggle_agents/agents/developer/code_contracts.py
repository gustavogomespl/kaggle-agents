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
        if (
            isinstance(node, ast.Call)
            and getattr(node.func, "id", "") == SUBMISSION_HELPER
            and id(node) not in helper_body_nodes
        ):
            return None

    for node in ast.walk(tree):
        if id(node) in helper_body_nodes or not isinstance(node, ast.Call):
            continue
        if getattr(node.func, "attr", "") != "to_csv" or not node.args:
            continue
        destination = ast.unparse(node.args[0])
        if "SUBMISSION_PATH" in destination or "submission.csv" in destination:
            return destination
    return None


SUBMISSION_CONTRACT_ERROR = (
    f"Submission must be written with {SUBMISSION_HELPER}(test_preds): picking "
    "submission columns by position writes predictions into an input column "
    "and leaves the graded column unfilled"
)
