"""Exclusive role resolution for the public delimited tables of a competition.

The adapter used to recognize its primary tables by globbing a handful of
literal filenames (``train.csv``, ``test*.csv``, ``*sample*.csv``). Any
competition that prefixes or camel-cases those names -- or ships them inside
per-table archives -- lost its train and test tables entirely, and the train
table was then re-discovered by a filename heuristic as a "label file".

This module replaces that with one bounded, auditable pass:

* enumerate a *bounded* candidate pool (retained ZIP members plus the loose
  delimited files directly beneath ``public/``), never reading a path outside
  ``public/``;
* score every candidate per role from generic, domain-agnostic evidence
  (exact standard name, role token in the archive stem, role token in the
  member stem, a weak train synonym, schema corroboration);
* solve train/test/submission as *one exclusive assignment*: a candidate may
  occupy at most one role, a role may stay unresolved, and two equally ranked
  maximum assignments are an error rather than a lexical coin flip;
* inspect everything that stays unassigned with the bounded label-layout
  inspector and record it as an auxiliary artifact carrying its evidence.

Nothing here knows a competition, a language prefix or a benchmark name: the
only inputs are file names, ZIP provenance and bounded schema samples.
"""

from __future__ import annotations

import csv
import hashlib
import io
import itertools
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from kaggle_agents.utils.label_parser import (
    inspect_label_layout,
    split_semantic_tokens,
)

from .dataclasses import ArchiveProvenance, ArtifactLayout, PublicArtifact


PrimaryRole = Literal["train", "test", "submission"]

PRIMARY_ROLES: tuple[PrimaryRole, ...] = ("train", "test", "submission")

EXACT_STANDARD_NAMES: dict[str, str] = {
    "train": "train.csv",
    "test": "test.csv",
    "submission": "sample_submission.csv",
}

ROLE_TOKENS: dict[str, str] = {
    "train": "train",
    "test": "test",
    "submission": "submission",
}

# Workspace alias every resolved primary table is staged under.
PRIMARY_TABLE_ALIASES: dict[str, str] = {
    "train": "train.csv",
    "test": "test.csv",
    "submission": "sample_submission.csv",
}

# Only comma-separated tables may occupy a primary role: staging a delimited
# source under a `.csv` alias does not convert its delimiter, and every
# downstream reader assumes the alias really is a CSV.
_PRIMARY_SUFFIXES = frozenset({".csv"})
_AUXILIARY_ONLY_SUFFIXES = frozenset({".tsv", ".txt"})
_INSPECTABLE_SUFFIXES = _PRIMARY_SUFFIXES | _AUXILIARY_ONLY_SUFFIXES

# Audit ruling M2: a stem that is exactly this generic word is weak train
# evidence, and only when no candidate carries real train evidence.
_WEAK_TRAIN_STEMS = frozenset({"label", "labels"})
_ID_LIKE_TOKENS = frozenset({"id", "uid"})

# Layouts whose content can execute inside the generated preamble, so their
# fingerprint pins the whole file rather than a bounded sample.
_FULL_CONTENT_LAYOUTS = frozenset({"sparse_labels", "id_mapping"})

_MAX_SAMPLE_ROWS = 5
_MAX_HEADER_BYTES = 64 * 1024
_CONTENT_CHUNK_BYTES = 1 << 20

# Streaming digests are cached by (path, size, mtime) so the same public file
# is never re-read once per generated component.
_content_digest_cache: dict[tuple[str, int, int], str] = {}


@dataclass(frozen=True)
class TableCandidate:
    """One bounded, public delimited table considered for a role."""

    path: Path
    source_archive: Path | None
    columns: tuple[str, ...]
    evidence: tuple[str, ...]
    fingerprint: str


@dataclass(frozen=True)
class _RoleScore:
    """Lexicographic rank plus the evidence strings that produced it."""

    rank: tuple[int, int, int, int, int]
    evidence: tuple[str, ...]

    @property
    def eligible(self) -> bool:
        """Name-level evidence is what makes a candidate eligible at all.

        Schema compatibility (the last rank element) is corroborating
        evidence only: it breaks ties between candidates that already name a
        role, and never promotes an unrelated table into a primary role.
        """
        return any(self.rank[:4])


@dataclass(frozen=True)
class _ResolutionContext:
    public_root: Path
    candidates: tuple[TableCandidate, ...]
    archives: dict[str, ArchiveProvenance]
    weak_train_available: bool


def one_artifact_path(
    artifacts: Sequence[PublicArtifact],
    role: Literal["train", "test", "submission"],
) -> Path | None:
    matches = [artifact.path for artifact in artifacts if artifact.role == role]
    if len(matches) > 1:
        raise ValueError(f"Multiple resolved {role} artifacts: {matches}")
    return matches[0] if matches else None


# --- Bounded reading and fingerprints -----------------------------------


def _read_bounded_text(path: Path) -> str:
    """Read at most ``_MAX_HEADER_BYTES``, dropping a truncated final line."""
    try:
        with path.open("rb") as handle:
            raw = handle.read(_MAX_HEADER_BYTES)
            truncated = bool(handle.read(1))
    except OSError:
        return ""
    text = raw.decode("utf-8", errors="replace")
    if truncated:
        cut = text.rfind("\n")
        if cut != -1:
            text = text[: cut + 1]
    return text


def _read_columns(path: Path) -> tuple[str, ...]:
    """Header columns from the header plus at most five sampled rows.

    Returns ``()`` when the sample is not rectangular: a header whose width
    the data does not corroborate is not schema evidence about anything.
    """
    text = _read_bounded_text(path)
    if not text.strip():
        return ()
    try:
        delimiter = csv.Sniffer().sniff(text, delimiters=",\t;|").delimiter
    except csv.Error:
        delimiter = ","
    rows: list[list[str]] = []
    try:
        for row in csv.reader(io.StringIO(text), delimiter=delimiter):
            if not row:
                continue
            rows.append(row)
            if len(rows) > _MAX_SAMPLE_ROWS:
                break
    except csv.Error:
        return ()
    if not rows:
        return ()
    header = tuple(field.strip() for field in rows[0])
    if any(len(row) != len(header) for row in rows[1:]):
        return ()
    return header


def _bounded_fingerprint(
    relative_path: str,
    crc: int | None,
    archive_size: int | None,
    table_size: int,
    columns: Sequence[str],
) -> str:
    """Digest of public-relative path, archive CRC/size, size and schema."""
    schema_digest = hashlib.sha256("\x1f".join(columns).encode("utf-8")).hexdigest()
    payload = "|".join(
        (
            f"path={relative_path}",
            f"crc={'' if crc is None else crc}",
            f"archive_size={'' if archive_size is None else archive_size}",
            f"size={table_size}",
            f"schema={schema_digest}",
        )
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _content_digest(path: Path) -> str:
    """Streaming SHA-256 of a public file, cached by path/size/mtime."""
    try:
        stat = path.stat()
    except OSError:
        return ""
    key = (str(path), stat.st_size, stat.st_mtime_ns)
    cached = _content_digest_cache.get(key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    try:
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(_CONTENT_CHUNK_BYTES), b""):
                digest.update(chunk)
    except OSError:
        return ""
    value = digest.hexdigest()
    _content_digest_cache[key] = value
    return value


# --- Bounded candidate enumeration --------------------------------------


def _public_relative(path: Path, public_root: Path) -> str | None:
    """Normalized public-relative path, or ``None`` outside the boundary."""
    try:
        resolved = Path(path).resolve()
    except OSError:
        return None
    if not resolved.is_relative_to(public_root):
        return None
    return resolved.relative_to(public_root).as_posix()


def _build_candidate(
    path: Path,
    public_root: Path,
    source_archive: Path | None,
    crc: int | None,
    archive_size: int | None,
    evidence: tuple[str, ...],
) -> TableCandidate | None:
    relative = _public_relative(path, public_root)
    if relative is None or not path.is_file():
        return None
    try:
        table_size = path.stat().st_size
    except OSError:
        return None
    columns = _read_columns(path)
    return TableCandidate(
        path=path,
        source_archive=source_archive,
        columns=columns,
        evidence=evidence,
        fingerprint=_bounded_fingerprint(
            relative, crc, archive_size, table_size, columns
        ),
    )


def _enumerate_candidates(
    public_dir: Path,
    provenance: Sequence[ArchiveProvenance],
    public_root: Path,
) -> list[TableCandidate]:
    """Retained archive members plus loose delimited files under ``public/``.

    Sibling roots (notably ``prepared/private/``) are never enumerated, and
    any path that resolves outside ``public/`` is rejected before it is read.
    """
    found: dict[str, TableCandidate] = {}
    for record in provenance:
        archive = (
            record.archive_path
            if _public_relative(record.archive_path, public_root) is not None
            else None
        )
        for member in record.supported_members:
            member_path = Path(member.extracted_path)
            if member_path.suffix.lower() not in _INSPECTABLE_SUFFIXES:
                continue
            relative = _public_relative(member_path, public_root)
            if relative is None or relative in found:
                continue
            candidate = _build_candidate(
                member_path,
                public_root,
                archive,
                member.crc,
                member.file_size,
                (f"archive_member:{member.member_name}",),
            )
            if candidate is not None:
                found[relative] = candidate

    for path in sorted(public_dir.glob("*")):
        if path.suffix.lower() not in _INSPECTABLE_SUFFIXES or not path.is_file():
            continue
        relative = _public_relative(path, public_root)
        if relative is None or relative in found:
            continue
        candidate = _build_candidate(
            path, public_root, None, None, None, ("loose_public_table",)
        )
        if candidate is not None:
            found[relative] = candidate

    return [found[relative] for relative in sorted(found)]


# --- Evidence and ranking ------------------------------------------------


def _supports_primary_role(candidate: TableCandidate) -> bool:
    return candidate.path.suffix.lower() in _PRIMARY_SUFFIXES


def _member_tokens(candidate: TableCandidate) -> tuple[str, ...]:
    return split_semantic_tokens(candidate.path.stem)


def _archive_tokens(
    candidate: TableCandidate, archives: dict[str, ArchiveProvenance]
) -> tuple[str, ...]:
    """Archive stem tokens, only for an archive holding one root table.

    An archive shipping several delimited members cannot lend its own name to
    any single member without inventing evidence.
    """
    if candidate.source_archive is None:
        return ()
    record = archives.get(str(Path(candidate.source_archive).resolve()))
    if record is None or not record.single_supported_tabular_root_member:
        return ()
    return split_semantic_tokens(Path(candidate.source_archive).stem)


def _is_id_like(column: str) -> bool:
    return bool(_ID_LIKE_TOKENS.intersection(split_semantic_tokens(column)))


def _schema_corroborates(
    role: str, candidate: TableCandidate, candidates: Sequence[TableCandidate]
) -> bool:
    """Corroborating structural evidence only -- never role-establishing.

    A train table is wider than the matching test table (it carries the
    targets), a test table is narrower than some other table, and a
    submission template starts with an identifier column.
    """
    columns = set(candidate.columns)
    if not columns:
        return False
    if role == "submission":
        return len(candidate.columns) >= 2 and _is_id_like(candidate.columns[0])
    others = [
        set(other.columns)
        for other in candidates
        if other.columns and other.path != candidate.path
    ]
    if role == "train":
        return any(other < columns for other in others)
    return any(columns < other for other in others)


def _weak_train_available(
    candidates: Sequence[TableCandidate], archives: dict[str, ArchiveProvenance]
) -> bool:
    """Audit ruling M2, gate (a): nothing stronger competes for train."""
    for candidate in candidates:
        if not _supports_primary_role(candidate):
            continue
        if candidate.path.name.lower() == EXACT_STANDARD_NAMES["train"]:
            return False
        tokens = set(_member_tokens(candidate)) | set(
            _archive_tokens(candidate, archives)
        )
        if ROLE_TOKENS["train"] in tokens:
            return False
    return True


def _is_weak_train_evidence(
    role: str,
    member_tokens: Sequence[str],
    archive_tokens: Sequence[str],
    context: _ResolutionContext,
) -> bool:
    """Audit ruling M2: a bare ``label``/``labels`` stem is the train table.

    Two real competitions publish their entire train table under this generic
    one-word stem with no train token anywhere. This only fires when nothing
    stronger competes (gate a) and the same stem carries no test/submission
    token (gate b), and it ranks below every explicit role token.
    """
    if role != "train" or not context.weak_train_available:
        return False
    if len(member_tokens) != 1 or member_tokens[0] not in _WEAK_TRAIN_STEMS:
        return False
    blocked = {ROLE_TOKENS["test"], ROLE_TOKENS["submission"]}
    return not blocked.intersection(member_tokens) and not blocked.intersection(
        archive_tokens
    )


def _score_role(
    role: str, candidate: TableCandidate, context: _ResolutionContext
) -> _RoleScore:
    """Lexicographic rank of ``candidate`` for ``role``."""
    if not _supports_primary_role(candidate):
        return _RoleScore((0, 0, 0, 0, 0), ())

    member_tokens = _member_tokens(candidate)
    archive_tokens = _archive_tokens(candidate, context.archives)
    token = ROLE_TOKENS[role]
    evidence: list[str] = []

    exact_standard_name = int(
        candidate.path.name.lower() == EXACT_STANDARD_NAMES[role]
    )
    if exact_standard_name:
        evidence.append("exact_standard_name")
    role_token_in_archive = int(token in archive_tokens)
    if role_token_in_archive:
        evidence.append(f"role_token_in_archive:{token}")
    role_token_in_member = int(token in member_tokens)
    if role_token_in_member:
        evidence.append(f"role_token_in_member:{token}")
    weak_train_synonym = int(
        _is_weak_train_evidence(role, member_tokens, archive_tokens, context)
    )
    if weak_train_synonym:
        evidence.append(f"weak_train_synonym:{member_tokens[0]}")
    schema_corroborates_role = int(
        _schema_corroborates(role, candidate, context.candidates)
    )
    if schema_corroborates_role:
        evidence.append("schema_corroborates_role")

    return _RoleScore(
        (
            exact_standard_name,
            role_token_in_archive,
            role_token_in_member,
            weak_train_synonym,
            schema_corroborates_role,
        ),
        tuple(evidence),
    )


# --- Exclusive assignment ------------------------------------------------

_Entry = tuple[TableCandidate, _RoleScore]
_Assignment = dict[str, _Entry]


def _iter_assignments(
    options: dict[str, list[_Entry]],
) -> Iterator[_Assignment]:
    """Every injective partial assignment of candidates to the three roles."""
    choices = [[None, *options[role]] for role in PRIMARY_ROLES]
    for combination in itertools.product(*choices):
        chosen = [entry for entry in combination if entry is not None]
        if len({entry[0].path for entry in chosen}) != len(chosen):
            continue
        yield {
            role: entry
            for role, entry in zip(PRIMARY_ROLES, combination, strict=True)
            if entry is not None
        }


def _assignment_score(
    assignment: _Assignment,
) -> tuple[int, tuple[tuple[int, ...], ...]]:
    """Maximum cardinality first, then the per-role ranks in role order."""
    ranks = tuple(
        assignment[role][1].rank if role in assignment else (0, 0, 0, 0, 0)
        for role in PRIMARY_ROLES
    )
    return (len(assignment), ranks)


def _describe(candidate: TableCandidate, score: _RoleScore, public_root: Path) -> str:
    relative = _public_relative(candidate.path, public_root) or candidate.path.name
    evidence = ", ".join(candidate.evidence + score.evidence) or "no_evidence"
    return f"{relative} [{evidence}]"


def _raise_ambiguity(
    winners: Sequence[_Assignment], context: _ResolutionContext
) -> None:
    lines: list[str] = []
    for role in PRIMARY_ROLES:
        distinct: dict[str, _Entry] = {}
        for winner in winners:
            entry = winner.get(role)
            if entry is not None:
                distinct[str(entry[0].path)] = entry
        if len(distinct) < 2:
            continue
        for candidate, score in distinct.values():
            lines.append(f"  {role}: {_describe(candidate, score, context.public_root)}")
    if not lines:
        for winner in winners:
            for role, (candidate, score) in sorted(winner.items()):
                lines.append(
                    f"  {role}: {_describe(candidate, score, context.public_root)}"
                )
    raise ValueError(
        "Ambiguous public artifact roles; refusing to guess between equally "
        "ranked candidates:\n" + "\n".join(lines)
    )


def _assign_primary_roles(context: _ResolutionContext) -> _Assignment:
    """Solve train/test/submission as one exclusive assignment."""
    options: dict[str, list[_Entry]] = {}
    for role in PRIMARY_ROLES:
        entries: list[_Entry] = []
        for candidate in context.candidates:
            score = _score_role(role, candidate, context)
            if score.eligible:
                entries.append((candidate, score))
        options[role] = entries

    best_score: tuple[int, tuple[tuple[int, ...], ...]] | None = None
    winners: list[_Assignment] = []
    for assignment in _iter_assignments(options):
        score = _assignment_score(assignment)
        if best_score is None or score > best_score:
            best_score, winners = score, [assignment]
        elif score == best_score:
            winners.append(assignment)

    if len(winners) > 1:
        _raise_ambiguity(winners, context)
    return winners[0] if winners else {}


# --- Artifact construction ----------------------------------------------


def _auxiliary_artifact(candidate: TableCandidate) -> PublicArtifact:
    """Inspect an unassigned candidate and type it as an auxiliary record.

    A candidate the inspector cannot describe is recorded as
    ``auxiliary``/``unknown`` carrying its rejection evidence: staging must
    never abort over a file nothing may ever consume. Selection time (never
    staging time) is where an unusable target candidate fails closed.
    """
    inspection = inspect_label_layout(candidate.path)
    layout: ArtifactLayout = inspection.layout
    fingerprint = candidate.fingerprint
    if layout in _FULL_CONTENT_LAYOUTS:
        content = _content_digest(candidate.path)
        if content:
            fingerprint = f"{fingerprint}:sha256={content}"
    return PublicArtifact(
        path=candidate.path,
        role="auxiliary",
        layout=layout,
        source_archive=candidate.source_archive,
        evidence=(
            *candidate.evidence,
            f"layout:{layout}",
            *inspection.evidence,
        ),
        fingerprint=fingerprint,
    )


def build_auxiliary_artifact(
    public_dir: Path,
    path: Path,
    source_archive: Path | None = None,
    evidence: tuple[str, ...] = ("recursive_public_scan",),
) -> PublicArtifact | None:
    """Type one recursively discovered public file as an auxiliary record."""
    public_root = Path(public_dir).resolve()
    if Path(path).suffix.lower() not in _INSPECTABLE_SUFFIXES:
        return None
    candidate = _build_candidate(
        Path(path), public_root, source_archive, None, None, evidence
    )
    if candidate is None:
        return None
    return _auxiliary_artifact(candidate)


def resolve_public_artifacts(
    public_dir: Path,
    provenance: Sequence[ArchiveProvenance],
    data_type: str,
) -> list[PublicArtifact]:
    """Type every public delimited table, at most one per primary role.

    ``data_type`` is reported in the resolution diagnostics only: role
    resolution itself is deliberately domain-independent, so an image or
    audio competition resolves its metadata tables by exactly the same
    evidence a tabular one does.
    """
    public_dir = Path(public_dir)
    public_root = public_dir.resolve()
    candidates = _enumerate_candidates(public_dir, provenance, public_root)
    if not candidates:
        print(
            f"   Typed artifacts: no public delimited tables ({data_type})",
            flush=True,
        )
        return []

    archives = {
        str(Path(record.archive_path).resolve()): record for record in provenance
    }
    context = _ResolutionContext(
        public_root=public_root,
        candidates=tuple(candidates),
        archives=archives,
        weak_train_available=_weak_train_available(candidates, archives),
    )
    assignment = _assign_primary_roles(context)

    artifacts: list[PublicArtifact] = []
    assigned: set[Path] = set()
    for role in PRIMARY_ROLES:
        entry = assignment.get(role)
        if entry is None:
            print(f"   Typed role {role}: unresolved", flush=True)
            continue
        candidate, score = entry
        assigned.add(candidate.path)
        artifacts.append(
            PublicArtifact(
                path=candidate.path,
                role=role,
                layout="rectangular_table",
                source_archive=candidate.source_archive,
                evidence=candidate.evidence + score.evidence,
                fingerprint=candidate.fingerprint,
            )
        )
        print(
            f"   Typed role {role}: {_describe(candidate, score, public_root)}",
            flush=True,
        )

    for candidate in candidates:
        if candidate.path in assigned:
            continue
        artifact = _auxiliary_artifact(candidate)
        artifacts.append(artifact)
        relative = _public_relative(artifact.path, public_root)
        print(
            f"   Typed auxiliary: {relative} [{artifact.layout}]",
            flush=True,
        )
    return artifacts
