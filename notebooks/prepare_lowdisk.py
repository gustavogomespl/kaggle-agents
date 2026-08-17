#!/usr/bin/env python
"""Prepare huge MLE-bench competitions on small disks (Colab-friendly).

The standard ``mlebench prepare -c <id>`` flow needs, at its peak, the
competition zip + the fully extracted ``raw/`` + a full copy of the images in
``prepared/public`` all on disk at once (~294 GB for
siim-isic-melanoma-classification, which no Colab VM has). This script
produces the same ``prepared/`` directory with a far lower peak:

1. verify the zip MD5 against the checksums pinned in the mle-bench repo
   (same check ``mlebench prepare`` performs);
2. extract ONLY the zip members the competition's own ``prepare_fn`` reads
   (per-competition rules below, fail-closed: unknown layouts abort);
3. delete the zip BEFORE preparing (the single biggest peak-disk win) —
   only when the zip lives in the mle-bench cache; a ``--zip`` you point at
   (e.g. on Google Drive) is never deleted;
4. run the competition's unmodified ``prepare_fn`` with ``shutil.copy``
   routed to hardlinks, so ``prepared/public`` shares blocks with ``raw/``
   instead of duplicating ~80 GB of images;
5. write ``public/description.md`` and verify every checksum mle-bench pins
   for the prepared data (the public/private CSVs);
6. remove ``raw/`` — the hardlinked image payloads survive in ``prepared/``.

Crash-safe: a completeness manifest is written after extraction, so if a later
step dies (the zip is already gone by then) a re-run resumes from the verified
``raw/`` without re-downloading. An already-existing ``prepared/`` is verified
(CSV checksums + image counts) instead of trusted blindly — the official
preparer writes the CSVs before the images, so a crashed official run can look
"prepared" while missing most of its images.

The result is byte-identical to the official flow for everything mle-bench
checksums, and the images are bit-identical extractions from the verified
zip (hardlinks re-use the exact extracted bytes; nothing is re-encoded).

Usage on Colab (after ``pip install -e /content/mle-bench``):

    python notebooks/prepare_lowdisk.py -c siim-isic-melanoma-classification --wipe-raw

    # zip staged on Google Drive to keep the local peak at ~subset size:
    python notebooks/prepare_lowdisk.py -c siim-isic-melanoma-classification \
        --zip /content/drive/MyDrive/mlebench/siim-isic-melanoma-classification.zip

Peak-disk math for siim-isic-melanoma-classification (~106 GB zip):
  official flow  : zip 106 + raw ~108 + public copy ~80  = ~294 GB
  this script    : zip 106 + needed subset ~81           = ~187 GB peak, ~81 GB final
  with --zip on Drive: needed subset only                = ~81 GB peak
"""

# ruff: noqa: PLC0415 - mlebench comes from an external editable install
# (`pip install -e /content/mle-bench`), not from this repo's dependencies;
# importing it lazily keeps `--help` and the error message usable without it.
from __future__ import annotations

import argparse
import contextlib
import json
import os
import shutil
import sys
import zipfile
from collections.abc import Callable, Iterator
from pathlib import Path


def _siim_isic_needed(name: str) -> bool:
    """Members of the siim-isic zip that prepare.py actually opens.

    mlebench/competitions/siim-isic-melanoma-classification/prepare.py reads
    ONLY: train.csv, tfrecords/train*.tfrec, train/<img>.dcm and
    jpeg/train/<img>.jpg (it re-splits the original train). The original-test
    artifacts (test/, jpeg/test/, tfrecords/test*, test.csv,
    sample_submission.csv) are never opened, so they never need to exist in
    raw/. Skipping them drops ~25% of the 108 GB extraction.
    """
    if name == "train.csv":
        return True
    if name.startswith(("train/", "jpeg/train/")):
        return True
    if name.startswith("tfrecords/"):
        return Path(name).name.startswith("train")
    return False


# Fail-closed: selective extraction only runs for competitions listed here,
# and only when every `must_have` prefix matches at least one zip member
# (guards against upstream layout changes / nested-zip archives).
SELECTIVE_EXTRACT_RULES: dict[str, dict] = {
    "siim-isic-melanoma-classification": {
        "needed": _siim_isic_needed,
        "must_have": ("train.csv", "train/", "jpeg/train/", "tfrecords/train"),
        # public subdirs that hold modalities the generated pipelines do not
        # read (models train on jpeg/); used only with --prune-public.
        "prunable": {"dicom": ("train", "test"), "tfrecords": ("tfrecords",)},
        # completeness probes for an already-existing prepared/: one image per
        # row of the named CSV, plus fixed-count globs.
        "public_image_dirs": (
            ("train.csv", "train", ".dcm"),
            ("train.csv", "jpeg/train", ".jpg"),
            ("test.csv", "test", ".dcm"),
            ("test.csv", "jpeg/test", ".jpg"),
        ),
        "public_fixed_counts": (("tfrecords", "*.tfrec", 16),),
    },
}

_MANIFEST_NAME = ".lowdisk_manifest.json"
_PRUNED_MARKER = ".lowdisk_pruned"

_GIB = 1024**3


def _human(n_bytes: float) -> str:
    return f"{n_bytes / _GIB:.1f} GiB"


@contextlib.contextmanager
def _copies_as_hardlinks() -> Iterator[None]:
    """Route shutil.copy/copy2 to os.link (with copy fallback) during prepare.

    prepare_fn only uses shutil.copy* to duplicate raw images into
    prepared/public. Hardlinking keeps those bytes shared with raw/, so the
    prepare step costs ~0 extra disk; once raw/ is rmtree'd the public entry
    remains the sole owner. Falls back to a real copy across filesystems.
    """
    real_copy, real_copy2 = shutil.copy, shutil.copy2

    def _link(src, dst, **_kwargs):
        src_path, dst_path = Path(src), Path(dst)
        if dst_path.is_dir():
            dst_path = dst_path / src_path.name
        try:
            if dst_path.exists():
                dst_path.unlink()
            os.link(src_path, dst_path)
            return str(dst_path)
        except OSError:
            return real_copy2(src_path, dst_path)

    shutil.copy = _link
    shutil.copy2 = _link
    try:
        yield
    finally:
        shutil.copy = real_copy
        shutil.copy2 = real_copy2


def _resolve_zip(comp_dir: Path, explicit_zip: str | None, competition_id: str) -> Path:
    if explicit_zip:
        zip_path = Path(explicit_zip)
        if not zip_path.is_file():
            sys.exit(f"--zip does not exist: {zip_path}")
        return zip_path
    existing = sorted(comp_dir.glob("*.zip"))
    if len(existing) == 1:
        print(f"Using already-downloaded zip: {existing[0]}")
        return existing[0]
    if len(existing) > 1:
        sys.exit(f"Multiple zips in {comp_dir}; pass --zip to disambiguate: {existing}")
    from mlebench.data import download_dataset

    comp_dir.mkdir(parents=True, exist_ok=True)
    return download_dataset(competition_id=competition_id, download_dir=comp_dir, force=False)


def _verify_zip_checksum(zip_path: Path, checksums_file: Path) -> None:
    from mlebench.data import get_checksum
    from mlebench.utils import load_yaml

    if not checksums_file.is_file():
        sys.exit(f"No pinned checksums for this competition ({checksums_file}); refusing to skip")
    expected = load_yaml(checksums_file)["zip"]
    print(f"Verifying zip MD5 (large file, several minutes): {zip_path}")
    actual = get_checksum(zip_path)
    if actual != expected:
        sys.exit(f"Zip checksum mismatch! expected {expected}, got {actual}. Re-download the zip.")
    print("Zip checksum matches the pinned mle-bench checksum.")


def _select_members(
    zf: zipfile.ZipFile, rule: dict | None
) -> tuple[list[zipfile.ZipInfo], int, int]:
    infos = zf.infolist()
    if rule is None:
        needed = infos
    else:
        names = [i.filename for i in infos]
        missing = [p for p in rule["must_have"] if not any(n.startswith(p) for n in names)]
        if missing:
            listing = "\n  ".join(sorted({n.split("/")[0] for n in names}))
            sys.exit(
                "Zip layout does not match the selective-extraction rule "
                f"(missing prefixes: {missing}). Top-level entries:\n  {listing}\n"
                "Aborting instead of guessing; run the standard `mlebench prepare` "
                "on a machine with enough disk."
            )
        needed_fn: Callable[[str], bool] = rule["needed"]
        needed = [i for i in infos if needed_fn(i.filename)]
    needed_bytes = sum(i.file_size for i in needed)
    skipped_bytes = sum(i.file_size for i in infos) - needed_bytes
    return needed, needed_bytes, skipped_bytes


def _extract_needed(zip_path: Path, raw_dir: Path, rule: dict | None) -> None:
    from tqdm.auto import tqdm

    with zipfile.ZipFile(zip_path) as zf:
        needed, needed_bytes, skipped_bytes = _select_members(zf, rule)
        free = shutil.disk_usage(raw_dir.parent).free
        print(
            f"Extracting {len(needed)} members ({_human(needed_bytes)}); "
            f"skipping {_human(skipped_bytes)} the preparer never reads. "
            f"Free disk: {_human(free)}"
        )
        if free < needed_bytes + 2 * _GIB:
            sys.exit(
                f"Not enough free disk: need ~{_human(needed_bytes)} + margin, "
                f"have {_human(free)}. Free space, or stage the zip on Google Drive "
                "and re-run with --zip so the local peak is just the extracted subset."
            )
        raw_dir.mkdir(parents=True, exist_ok=True)
        with tqdm(total=needed_bytes, unit="B", unit_scale=True, desc="extract") as bar:
            for info in needed:
                zf.extract(info, raw_dir)
                bar.update(info.file_size)
        manifest = {i.filename: i.file_size for i in needed if not i.is_dir()}
        (raw_dir / _MANIFEST_NAME).write_text(json.dumps(manifest))


def _require_manifest_match(raw_dir: Path) -> None:
    """Allow resuming from raw/ only when its completeness is provable."""
    manifest_path = raw_dir / _MANIFEST_NAME
    if not manifest_path.is_file():
        sys.exit(
            f"raw/ exists but has no completeness manifest: {raw_dir}\n"
            "It is probably a partial extraction from a crashed `mlebench prepare` "
            "(truncated files cannot be detected without the zip). Re-run with "
            "--wipe-raw to re-extract from the zip."
        )
    manifest: dict[str, int] = json.loads(manifest_path.read_text())
    problems = [
        name
        for name, size in manifest.items()
        if not (raw_dir / name).is_file() or (raw_dir / name).stat().st_size != size
    ]
    if problems:
        sys.exit(
            f"raw/ does not match its completeness manifest ({len(problems)} files "
            f"missing or truncated, e.g. {problems[:3]}). Re-run with --wipe-raw."
        )
    print(f"Resuming from verified raw/ ({len(manifest)} files match the manifest).")


def _verify_prepared_checksums(competition) -> None:
    from mlebench.data import generate_checksums
    from mlebench.utils import load_yaml

    expected = load_yaml(competition.checksums)
    actual = {
        "public": generate_checksums(competition.public_dir),
        "private": generate_checksums(competition.private_dir),
    }
    for scope in ("public", "private"):
        if actual[scope] != expected.get(scope):
            sys.exit(
                f"Prepared {scope} checksums do not match the pinned mle-bench values!\n"
                f"expected: {expected.get(scope)}\nactual:   {actual[scope]}"
            )
    print("Prepared public/private checksums match the pinned mle-bench values.")


def _verify_public_counts(public_dir: Path, rule: dict) -> None:
    """Probe an existing prepared/public for the image files the CSVs promise."""
    marker = public_dir / _PRUNED_MARKER
    pruned = set(marker.read_text().split()) if marker.is_file() else set()

    def _rows(csv_name: str) -> int:
        with (public_dir / csv_name).open(encoding="utf-8") as f:
            return sum(1 for _ in f) - 1

    checks = [(c, s, f"*{ext}", None) for c, s, ext in rule.get("public_image_dirs", ())]
    checks += [(None, s, glob, count) for s, glob, count in rule.get("public_fixed_counts", ())]
    for csv_name, subdir, glob, fixed in checks:
        if subdir in pruned:
            continue
        expected = fixed if fixed is not None else _rows(csv_name)
        actual = len(list((public_dir / subdir).glob(glob)))
        if actual != expected:
            sys.exit(
                f"prepared/public/{subdir} has {actual} files but {expected} were expected — "
                "an earlier prepare run probably crashed after writing the CSVs. Remove "
                f"`{public_dir.parent}` and re-run this script."
            )
    print("Prepared image counts match the public CSVs.")


def _confirm_existing_prepared(competition, rule: dict | None) -> None:
    _verify_prepared_checksums(competition)
    if rule is not None:
        _verify_public_counts(competition.public_dir, rule)
    description = competition.public_dir / "description.md"
    if not description.is_file():
        description.write_text(competition.description)
    print(f"`{competition.id}` is already prepared and verified; nothing to do.")


def _prune_public(public_dir: Path, rule: dict, targets: list[str]) -> None:
    prunable = rule.get("prunable", {})
    pruned_subdirs: list[str] = []
    for target in targets:
        if target not in prunable:
            sys.exit(f"--prune-public {target} is not supported for this competition")
        for subdir in prunable[target]:
            victim = public_dir / subdir
            pruned_subdirs.append(subdir)
            if victim.exists():
                size = sum(f.stat().st_size for f in victim.rglob("*") if f.is_file())
                shutil.rmtree(victim)
                print(f"Pruned public/{subdir} ({_human(size)})")
    marker = public_dir / _PRUNED_MARKER
    already = set(marker.read_text().split()) if marker.is_file() else set()
    marker.write_text("\n".join(sorted(already | set(pruned_subdirs))))
    print(
        "NOTE: --prune-public removed unused modalities from prepared/public. "
        "The pinned CSV checksums (train/test/sample_submission + private answers) "
        "are untouched and grading is unaffected, but public/ is no longer the "
        "byte-complete official layout. Record this deviation in your run notes."
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("-c", "--competition", required=True, help="competition id (slug)")
    parser.add_argument("--data-dir", default=None, help="mle-bench data dir (default: cache dir)")
    parser.add_argument(
        "--zip",
        default=None,
        help="path to an already-downloaded competition zip (e.g. on Google Drive); "
        "never deleted when passed explicitly",
    )
    parser.add_argument(
        "--wipe-raw",
        action="store_true",
        help="delete an existing (e.g. partially extracted) raw/ before starting",
    )
    parser.add_argument(
        "--keep-zip",
        action="store_true",
        help="keep the cache-local zip after extraction (default: delete it to free ~zip-size "
        "before preparing)",
    )
    parser.add_argument(
        "--skip-zip-checksum",
        action="store_true",
        help="skip the zip MD5 verification (only if this exact zip already passed it)",
    )
    parser.add_argument(
        "--prune-public",
        default=None,
        help="comma list of unused modalities to drop from prepared/public after "
        "verification (siim-isic: dicom,tfrecords). Off by default.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()

    try:
        from mlebench import registry as registry_module
        from mlebench.data import create_prepared_dir, is_dataset_prepared
        from mlebench.utils import is_empty
    except ImportError as exc:
        sys.exit(f"mlebench is not importable ({exc}). Run `pip install -e /content/mle-bench`.")

    registry = registry_module.registry
    if args.data_dir:
        registry = registry.set_data_dir(Path(args.data_dir))
    competition = registry.get_competition(args.competition)
    rule = SELECTIVE_EXTRACT_RULES.get(args.competition)

    if is_dataset_prepared(competition):
        _confirm_existing_prepared(competition, rule)
        return

    raw_dir = competition.raw_dir
    comp_dir = raw_dir.parent
    if args.wipe_raw and raw_dir.exists():
        print(f"Wiping existing raw/: {raw_dir}")
        shutil.rmtree(raw_dir)

    if raw_dir.exists() and not is_empty(raw_dir):
        # e.g. a previous run of this script crashed after deleting the zip.
        _require_manifest_match(raw_dir)
    else:
        zip_path = _resolve_zip(comp_dir, args.zip, args.competition)
        if args.skip_zip_checksum:
            print("Skipping zip checksum verification (--skip-zip-checksum).")
        else:
            _verify_zip_checksum(zip_path, competition.checksums)

        _extract_needed(zip_path, raw_dir, rule)

        zip_in_cache = zip_path.parent.resolve() == comp_dir.resolve()
        if zip_in_cache and not args.keep_zip:
            print(f"Deleting zip to free {_human(zip_path.stat().st_size)} before preparing.")
            zip_path.unlink()
        elif not zip_in_cache:
            print("Zip was supplied via --zip; leaving it untouched.")

    create_prepared_dir(competition)
    print(f"Running the competition's own preparer with hardlinked copies: {competition.id}")
    with _copies_as_hardlinks():
        competition.prepare_fn(
            raw=competition.raw_dir,
            public=competition.public_dir,
            private=competition.private_dir,
        )

    (competition.public_dir / "description.md").write_text(competition.description)
    _verify_prepared_checksums(competition)

    print(f"Removing raw/ (hardlinked payloads live on in prepared/): {raw_dir}")
    shutil.rmtree(raw_dir)

    if args.prune_public:
        if rule is None:
            sys.exit("--prune-public is only supported for competitions with selective rules")
        _prune_public(competition.public_dir, rule, args.prune_public.split(","))

    free = shutil.disk_usage(comp_dir).free
    print(
        f"Done. `{args.competition}` is prepared at {competition.public_dir.parent} "
        f"(free disk now: {_human(free)}). The runner will detect it as prepared."
    )


if __name__ == "__main__":
    main()
