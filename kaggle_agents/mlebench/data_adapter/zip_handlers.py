"""
ZIP file handling for MLE-bench data adapter.

Contains methods for extracting ZIPs and populating from fallback sources.
"""

from __future__ import annotations

import zipfile
from pathlib import Path

from .dataclasses import ArchiveMemberProvenance, ArchiveProvenance


# Delimited files whose rows are cheap to inspect for a role (train/test/
# submission/auxiliary) without knowing the archive's competition-specific
# meaning. `.json`/`.jsonl` are intentionally excluded: their bounded readers
# differ by JSON shape and belong to the existing JSON materialization path.
_ROLE_INSPECTABLE_SUFFIXES = {".csv", ".tsv", ".txt"}


def _should_extract_to_subdir(z: zipfile.ZipFile, sample_limit: int = 2000) -> bool:
    """Heuristic: extract to subdir if there are files at zip root."""
    seen = 0
    for name in z.namelist():
        if not name or name.endswith("/"):
            continue
        seen += 1
        # Root-level files -> no directory structure
        if "/" not in name:
            return True
        if seen >= sample_limit:
            break
    return False


def _already_extracted(
    z: zipfile.ZipFile, destination_root: Path, sample_limit: int = 50
) -> bool:
    """Best-effort check to avoid re-extracting large archives."""
    checked = 0
    for name in z.namelist():
        if not name or name.endswith("/"):
            continue
        checked += 1
        if (destination_root / name).exists():
            return True
        if checked >= sample_limit:
            break
    return False


def _build_archive_provenance(
    z: zipfile.ZipFile, zip_file: Path, destination_root: Path
) -> ArchiveProvenance:
    """Open the central directory and record bounded, idempotent provenance.

    Counts every non-directory member (without retaining media names) and
    retains per-member details only for role-inspectable delimited files, so
    the record stays cheap even for large media archives.
    """
    member_count = 0
    supported_tabular_member_count = 0
    supported_tabular_root_member_count = 0
    supported_members: list[ArchiveMemberProvenance] = []
    for member in z.infolist():
        name = member.filename
        if not name or name.endswith("/"):
            continue
        member_count += 1
        if Path(name).suffix.lower() not in _ROLE_INSPECTABLE_SUFFIXES:
            continue
        at_archive_root = "/" not in name
        supported_tabular_member_count += 1
        if at_archive_root:
            supported_tabular_root_member_count += 1
        supported_members.append(
            ArchiveMemberProvenance(
                member_name=name,
                extracted_path=destination_root / name,
                crc=member.CRC,
                file_size=member.file_size,
                at_archive_root=at_archive_root,
            )
        )
    return ArchiveProvenance(
        archive_path=zip_file,
        extraction_root=destination_root,
        member_count=member_count,
        supported_tabular_member_count=supported_tabular_member_count,
        supported_tabular_root_member_count=supported_tabular_root_member_count,
        supported_members=tuple(supported_members),
    )


class ZipHandlerMixin:
    """Mixin providing ZIP file handling methods."""

    # mle_cache attribute will be provided by the main class
    mle_cache: Path

    def _extract_zips(self, directory: Path) -> list[ArchiveProvenance]:
        """Extract all ZIP files in directory and return their provenance.

        Notes:
            Some competitions ship flat zips (files at root). For those, we extract into
            a subdirectory named after the zip stem to avoid polluting `directory/` and
            to create stable `train/` / `test/` folders when the zip is named similarly.

            Provenance is additive observation, not a behavior change: every call
            opens each ZIP's central directory -- even when extraction itself is
            skipped because the archive was already extracted -- and returns one
            aggregate `ArchiveProvenance` record per valid ZIP. Repeated calls
            against an unchanged directory therefore return equal provenance.
        """
        provenance: list[ArchiveProvenance] = []
        for zip_file in sorted(directory.glob("*.zip")):
            extract_dir = directory / zip_file.stem
            try:
                with zipfile.ZipFile(zip_file, "r") as z:
                    extract_to_subdir = _should_extract_to_subdir(z)
                    destination_root = extract_dir if extract_to_subdir else directory

                    provenance.append(
                        _build_archive_provenance(z, zip_file, destination_root)
                    )

                    if destination_root.exists() and _already_extracted(z, destination_root):
                        continue

                    print(f"   Extracting: {zip_file.name}")
                    if extract_to_subdir:
                        extract_dir.mkdir(parents=True, exist_ok=True)
                        z.extractall(extract_dir)
                    else:
                        z.extractall(directory)
            except zipfile.BadZipFile:
                print(f"   Warning: {zip_file.name} is not a valid zip")
        return provenance

    def _populate_from_fallback(self, competition_id: str, public_dir: Path) -> bool:
        """
        Attempt to populate empty public_dir from raw data or competition ZIP.

        This is a fallback mechanism when MLE-bench's prepare step didn't populate
        the public/ directory correctly.

        Returns True if successful, False otherwise.
        """
        import shutil

        base_dir = self.mle_cache / competition_id
        raw_dir = base_dir / "raw"
        comp_zip = base_dir / f"{competition_id}.zip"

        # Strategy 1: Copy from raw/ if it exists and has contents
        if raw_dir.exists():
            try:
                raw_contents = list(raw_dir.iterdir())
                if raw_contents:
                    print(f"   📂 Populating from raw/: {raw_dir}", flush=True)
                    for item in raw_contents:
                        dest = public_dir / item.name
                        if dest.exists():
                            continue  # Don't overwrite existing files
                        if item.is_file():
                            shutil.copy2(item, dest)
                        else:
                            shutil.copytree(item, dest, symlinks=True)
                    print(f"   ✅ Copied {len(raw_contents)} items from raw/", flush=True)
                    return True
            except Exception as e:
                print(f"   ⚠️ Failed to copy from raw/: {e}", flush=True)

        # Strategy 2: Extract competition ZIP directly to public/
        if comp_zip.exists():
            print(f"   📦 Extracting from competition ZIP: {comp_zip}", flush=True)
            try:
                with zipfile.ZipFile(comp_zip, "r") as z:
                    z.extractall(public_dir)
                print("   ✅ Extracted competition ZIP to public/", flush=True)
                return True
            except Exception as e:
                print(f"   ⚠️ Failed to extract competition ZIP: {e}", flush=True)

        # No fallback available
        print("   ❌ No fallback data source found", flush=True)
        return False

    def _auto_prepare_via_kaggle_api(self, competition_id: str) -> bool:
        """
        Auto-prepare competition data by downloading from Kaggle API.

        This is called when MLE-bench cache doesn't exist but Kaggle credentials are available.

        Returns True if successful, False otherwise.
        """
        comp_path = self.get_competition_path(competition_id)
        public_dir = comp_path / "public"

        print("   🌐 Attempting auto-download from Kaggle API...", flush=True)

        try:
            from ..tools.kaggle_api import KaggleAPIClient

            client = KaggleAPIClient()  # Uses existing credentials

            # Create public directory
            public_dir.mkdir(parents=True, exist_ok=True)

            # Download directly to public/
            print(f"   📥 Downloading competition data: {competition_id}", flush=True)
            client.download_competition_data(
                competition_id,
                path=str(public_dir),
                quiet=False,
            )

            # Verify we got data
            public_contents = list(public_dir.glob("*"))
            if public_contents:
                print(f"   ✅ Downloaded {len(public_contents)} items to {public_dir}", flush=True)
                return True
            print("   ⚠️ Download completed but no files found", flush=True)
            return False

        except ImportError:
            print("   ⚠️ Kaggle API client not available", flush=True)
            return False
        except Exception as e:
            print(f"   ⚠️ Auto-download failed: {e}", flush=True)
            return False
