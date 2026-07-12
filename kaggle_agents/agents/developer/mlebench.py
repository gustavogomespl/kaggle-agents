"""Legacy compatibility shim for the former in-workflow MLE-bench grader.

MLE-bench grading is deliberately owned by :mod:`kaggle_agents.mlebench.runner`
and only runs after the workflow has finished.  Keeping this empty mixin avoids
breaking third-party imports while preventing agents from observing test-set
scores during model development.
"""


class MLEBenchMixin:
    """Deprecated no-op mixin retained for import compatibility."""
