"""
MLE-bench Data Adapter Module.

Provides utilities to adapt MLE-bench prepared data to kaggle-agents expected format.
"""

from .adapter import MLEBenchDataAdapter
from .artifact_roles import (
    PRIMARY_TABLE_ALIASES,
    TableCandidate,
    build_auxiliary_artifact,
    one_artifact_path,
    resolve_public_artifacts,
)
from .dataclasses import (
    ArchiveMemberProvenance,
    ArchiveProvenance,
    ArtifactLayout,
    ArtifactRole,
    MLEBenchDataInfo,
    PublicArtifact,
)


__all__ = [
    "PRIMARY_TABLE_ALIASES",
    "ArchiveMemberProvenance",
    "ArchiveProvenance",
    "ArtifactLayout",
    "ArtifactRole",
    "MLEBenchDataAdapter",
    "MLEBenchDataInfo",
    "PublicArtifact",
    "TableCandidate",
    "build_auxiliary_artifact",
    "one_artifact_path",
    "resolve_public_artifacts",
]
