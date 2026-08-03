"""
MLE-bench Data Adapter Dataclasses.

Contains data structures for MLE-bench competition data information.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Literal


ArtifactRole = Literal["train", "test", "submission", "auxiliary"]
ArtifactLayout = Literal[
    "rectangular_table",
    "sparse_labels",
    "id_mapping",
    "media",
    "unknown",
]


@dataclass(frozen=True)
class ArchiveMemberProvenance:
    """Provenance for one role-inspectable member retained from a ZIP archive."""

    member_name: str
    extracted_path: Path
    crc: int
    file_size: int
    at_archive_root: bool


@dataclass(frozen=True)
class ArchiveProvenance:
    """Bounded, idempotent provenance for one extracted (or already-extracted) ZIP.

    Built every time `_extract_zips()` runs, whether or not extraction is
    actually performed on that call: the archive's central directory is
    always opened, so repeated calls against an unchanged directory return
    equal provenance.
    """

    archive_path: Path
    extraction_root: Path
    member_count: int
    supported_tabular_member_count: int
    supported_tabular_root_member_count: int
    supported_members: tuple[ArchiveMemberProvenance, ...]

    @property
    def single_supported_tabular_root_member(self) -> bool:
        return (
            self.supported_tabular_member_count == 1
            and self.supported_tabular_root_member_count == 1
        )


@dataclass(frozen=True)
class PublicArtifact:
    """A typed, role-tagged public data artifact discovered from MLE-bench data."""

    path: Path
    role: ArtifactRole
    layout: ArtifactLayout
    source_archive: Path | None = None
    evidence: tuple[str, ...] = ()
    fingerprint: str = ""

    def with_staged_paths(
        self,
        path: Path,
        source_archive: Path | None,
    ) -> PublicArtifact:
        return replace(
            self,
            path=Path(path),
            source_archive=(
                Path(source_archive) if source_archive is not None else None
            ),
        )

    def to_state(self) -> dict[str, object]:
        return {
            "path": str(self.path),
            "role": self.role,
            "layout": self.layout,
            "source_archive": (
                str(self.source_archive) if self.source_archive else ""
            ),
            "evidence": list(self.evidence),
            "fingerprint": self.fingerprint,
        }


@dataclass
class MLEBenchDataInfo:
    """Information about MLE-bench competition data."""

    competition_id: str
    workspace: Path
    train_path: Path | None = None
    test_path: Path | None = None
    clean_train_path: Path | None = None
    sample_submission_path: Path | None = None
    train_csv_path: Path | None = None  # For image competitions with labels CSV
    test_csv_path: Path | None = None
    ground_truth_path: Path | None = None  # Private test labels
    description_path: Path | None = None
    data_type: str = "tabular"  # tabular, image, audio, text
    target_column: str = "target"
    target_columns: list[str] = field(default_factory=lambda: ["target"])
    id_column: str = "id"
    extra_files: list[Path] = field(default_factory=list)
    # Label and split-metadata files discovered from the public data.
    label_files: list[Path] = field(default_factory=list)
    # Audio source directory inferred from local file extensions.
    audio_source_path: Path | None = None
    # Typed, role-tagged public artifacts backed by bounded ZIP provenance.
    public_artifacts: list[PublicArtifact] = field(default_factory=list)
