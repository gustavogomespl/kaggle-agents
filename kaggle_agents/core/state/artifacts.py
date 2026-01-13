"""
Artifact Index for Kaggle Agents.

Provides structured tracking of artifacts stored on disk:
- Replaces storing content in state with storing references
- Reduces state bloat
- Improves auditability with hash verification
- Enables efficient artifact lookup by type or component

MLE-STAR stores artifacts on disk and references them in state,
rather than embedding large content directly in the state object.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal


ArtifactType = Literal[
    "log",
    "csv",
    "npy",
    "json",
    "model",
    "code",
    "diff",
    "report",
    "config",
    "other",
]


@dataclass
class ArtifactRef:
    """Reference to an artifact stored on disk.

    Instead of storing content in state, store paths + metadata.
    This reduces bloat and improves auditability.

    Attributes:
        path: Path to the artifact file
        artifact_type: Type of artifact (log, csv, npy, etc.)
        size_bytes: Size of the artifact in bytes
        hash: MD5 hash for integrity verification
        created_at: When the artifact was created (ISO format)
        component_name: Which component created this artifact
        description: Brief description of the artifact
    """

    path: str
    artifact_type: ArtifactType
    size_bytes: int
    hash: str
    created_at: str
    component_name: str | None = None
    description: str = ""

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        size_kb = self.size_bytes / 1024
        if size_kb >= 1024:
            size_str = f"{size_kb / 1024:.1f} MB"
        else:
            size_str = f"{size_kb:.1f} KB"
        name = Path(self.path).name
        return f"ArtifactRef({self.artifact_type}, {name}, {size_str})"

    def verify_integrity(self) -> bool:
        """Verify the artifact file matches the stored hash.

        Uses chunked reading for memory efficiency with large files.

        Returns:
            True if hash matches, False otherwise
        """
        artifact_path = Path(self.path)
        if not artifact_path.exists():
            return False

        # Use chunked reading to handle large files without memory issues
        md5_hash = hashlib.md5()
        with open(artifact_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                md5_hash.update(chunk)

        return md5_hash.hexdigest() == self.hash

    def exists(self) -> bool:
        """Check if the artifact file exists."""
        return Path(self.path).exists()

    def to_dict(self) -> dict:
        """Serialize for checkpointing."""
        return {
            "path": self.path,
            "artifact_type": self.artifact_type,
            "size_bytes": self.size_bytes,
            "hash": self.hash,
            "created_at": self.created_at,
            "component_name": self.component_name,
            "description": self.description,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ArtifactRef:
        """Deserialize from checkpoint."""
        return cls(
            path=data["path"],
            artifact_type=data["artifact_type"],
            size_bytes=data["size_bytes"],
            hash=data["hash"],
            created_at=data["created_at"],
            component_name=data.get("component_name"),
            description=data.get("description", ""),
        )

    @classmethod
    def from_file(
        cls,
        path: str | Path,
        artifact_type: ArtifactType,
        component_name: str | None = None,
        description: str = "",
    ) -> ArtifactRef:
        """Create an ArtifactRef from an existing file.

        Args:
            path: Path to the artifact file (converted to str for JSON serialization)
            artifact_type: Type of artifact
            component_name: Which component created this
            description: Brief description

        Returns:
            ArtifactRef with computed hash and size

        Note:
            The path is always stored as str (not Path) for JSON serialization
            compatibility with LangGraph checkpointing.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Artifact file not found: {path}")

        # Compute hash and size
        with open(path, "rb") as f:
            content = f.read()
            file_hash = hashlib.md5(content).hexdigest()
            size_bytes = len(content)

        return cls(
            path=str(path),
            artifact_type=artifact_type,
            size_bytes=size_bytes,
            hash=file_hash,
            created_at=datetime.now().isoformat(),
            component_name=component_name,
            description=description,
        )


@dataclass
class ArtifactIndex:
    """Index of all artifacts created during workflow.

    Replaces storing content in state with storing references.
    Provides efficient lookup by name, type, or component.

    Attributes:
        artifacts: Dictionary mapping artifact names to references
    """

    artifacts: dict[str, ArtifactRef] = field(default_factory=dict)

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return f"ArtifactIndex({len(self.artifacts)} artifacts, {self.total_size_mb():.1f} MB)"

    def add(self, name: str, ref: ArtifactRef) -> None:
        """Add an artifact reference.

        Args:
            name: Unique name for the artifact
            ref: The ArtifactRef to add
        """
        self.artifacts[name] = ref

    def add_from_file(
        self,
        name: str,
        path: str | Path,
        artifact_type: ArtifactType,
        component_name: str | None = None,
        description: str = "",
    ) -> ArtifactRef:
        """Add an artifact from a file path.

        Args:
            name: Unique name for the artifact
            path: Path to the artifact file
            artifact_type: Type of artifact
            component_name: Which component created this
            description: Brief description

        Returns:
            The created ArtifactRef
        """
        ref = ArtifactRef.from_file(path, artifact_type, component_name, description)
        self.add(name, ref)
        return ref

    def get(self, name: str) -> ArtifactRef | None:
        """Get an artifact by name."""
        return self.artifacts.get(name)

    def get_by_type(self, artifact_type: ArtifactType) -> list[ArtifactRef]:
        """Get all artifacts of a specific type."""
        return [a for a in self.artifacts.values() if a.artifact_type == artifact_type]

    def get_by_component(self, component_name: str) -> list[ArtifactRef]:
        """Get all artifacts created by a specific component."""
        return [a for a in self.artifacts.values() if a.component_name == component_name]

    def get_logs(self) -> list[ArtifactRef]:
        """Get all log artifacts."""
        return self.get_by_type("log")

    def get_models(self) -> list[ArtifactRef]:
        """Get all model artifacts."""
        return self.get_by_type("model")

    def get_predictions(self) -> list[ArtifactRef]:
        """Get all prediction artifacts (npy files)."""
        return self.get_by_type("npy")

    def verify_all(self) -> tuple[list[str], list[str]]:
        """Verify all artifacts exist and have correct hashes.

        Returns:
            Tuple of (valid_names, invalid_names)
        """
        valid = []
        invalid = []

        for name, ref in self.artifacts.items():
            if ref.verify_integrity():
                valid.append(name)
            else:
                invalid.append(name)

        return valid, invalid

    def total_size_bytes(self) -> int:
        """Get total size of all artifacts in bytes."""
        return sum(a.size_bytes for a in self.artifacts.values())

    def total_size_mb(self) -> float:
        """Get total size of all artifacts in megabytes."""
        return self.total_size_bytes() / (1024 * 1024)

    def get_summary(self) -> str:
        """Get a summary of indexed artifacts."""
        lines = [f"## Artifact Index ({len(self.artifacts)} artifacts, {self.total_size_mb():.1f} MB)"]

        # Group by type
        by_type: dict[str, list[tuple[str, ArtifactRef]]] = {}
        for name, ref in self.artifacts.items():
            by_type.setdefault(ref.artifact_type, []).append((name, ref))

        for atype, items in sorted(by_type.items()):
            total_size = sum(ref.size_bytes for _, ref in items)
            lines.append(f"\n### {atype} ({len(items)} files, {total_size / 1024 / 1024:.1f} MB)")
            for name, ref in items[:5]:  # Show first 5
                size_kb = ref.size_bytes / 1024
                lines.append(f"- {name}: {size_kb:.1f} KB")
            if len(items) > 5:
                lines.append(f"  ... and {len(items) - 5} more")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize for checkpointing."""
        return {"artifacts": {k: v.to_dict() for k, v in self.artifacts.items()}}

    @classmethod
    def from_dict(cls, data: dict) -> ArtifactIndex:
        """Deserialize from checkpoint."""
        index = cls()
        for k, v in data.get("artifacts", {}).items():
            index.artifacts[k] = ArtifactRef.from_dict(v)
        return index
