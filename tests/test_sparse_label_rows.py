"""RED/GREEN coverage for bounded, quote-aware sparse-label inspection and
parsing.

Exercises the two public helpers added to ``kaggle_agents.utils.label_parser``
- ``inspect_label_layout`` and ``parse_sparse_label_rows`` - plus the shared
``split_semantic_tokens`` tokenizer they (and, later, the Task 3 role
resolver) both rely on.

Several tests are direct regressions for the Task 0 audit findings in the
implementation plan (referenced inline as M1/M3/M4/M5); those docstrings
name the audit item so the intent survives independent of this file.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from kaggle_agents.utils.label_parser import (
    inspect_label_layout,
    parse_sparse_label_rows,
    split_semantic_tokens,
)


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


# ---------------------------------------------------------------------------
# split_semantic_tokens: generic text-in/tokens-out tokenizer (audit M1)
# ---------------------------------------------------------------------------


def test_split_semantic_tokens_splits_camel_case_boundary() -> None:
    assert split_semantic_tokens("sampleSubmission") == ("sample", "submission")


def test_split_semantic_tokens_no_false_split_on_plain_word() -> None:
    """``contest`` has no case transition, so it must not collapse toward a
    false ``test`` match - the exact regression the audit called out."""
    assert split_semantic_tokens("contest") == ("contest",)


def test_split_semantic_tokens_splits_trailing_uppercase_run() -> None:
    """Audit M4: a trailing uppercase run (``UID``) becomes its own token
    instead of collapsing into the previous word."""
    assert split_semantic_tokens("StudyInstanceUID") == ("study", "instance", "uid")


def test_split_semantic_tokens_splits_underscore_and_lowercases() -> None:
    assert split_semantic_tokens("id_code") == ("id", "code")
    assert split_semantic_tokens("record_id") == ("record", "id")


def test_split_semantic_tokens_splits_leading_uppercase_run() -> None:
    """General camelCase robustness: an uppercase run followed by a
    capitalized word splits before the capitalized word."""
    assert split_semantic_tokens("HTMLParser") == ("html", "parser")


# ---------------------------------------------------------------------------
# inspect_label_layout: id_mapping
# ---------------------------------------------------------------------------


def test_id_and_file_path_header_is_id_mapping(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "mapping.csv",
        "record_id,file_path\na,clip_alpha.wav\nb,clip_beta.wav\n",
    )

    inspection = inspect_label_layout(path)

    assert inspection.layout == "id_mapping"
    assert inspection.has_header is True


def test_leading_id_prefix_header_is_id_like_for_id_mapping(tmp_path: Path) -> None:
    """Audit M4: a *leading* ``id_`` prefix (``id_code``) is ID-like too,
    not only a trailing ``_id``."""
    path = _write(
        tmp_path,
        "mapping.csv",
        "id_code,file_path\nabc123,scan1.dcm\ndef456,scan2.dcm\n",
    )

    inspection = inspect_label_layout(path)

    assert inspection.layout == "id_mapping"


def test_camel_case_uid_header_is_id_like_for_id_mapping(tmp_path: Path) -> None:
    """Audit M1 + M4 together: camelCase splitting plus treating standalone
    ``uid`` as a synonym of ``id`` (``StudyInstanceUID``)."""
    path = _write(
        tmp_path,
        "mapping.csv",
        "StudyInstanceUID,file_path\nabc123,scan1.dcm\ndef456,scan2.dcm\n",
    )

    inspection = inspect_label_layout(path)

    assert inspection.layout == "id_mapping"


# ---------------------------------------------------------------------------
# inspect_label_layout: fixed two-column tables that must stay `unknown`
# ---------------------------------------------------------------------------


def test_two_column_id_feature_header_is_unknown(tmp_path: Path) -> None:
    path = _write(tmp_path, "table.csv", "id,feature\n1,x\n2,y\n")

    inspection = inspect_label_layout(path)

    assert inspection.layout == "unknown"


def test_two_column_id_timestamp_header_is_unknown(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "table.csv",
        "id,timestamp\n1,2020-01-01\n2,2020-01-02\n",
    )

    inspection = inspect_label_layout(path)

    assert inspection.layout == "unknown"


def test_two_column_id_value_header_is_unknown_not_rescued(tmp_path: Path) -> None:
    """Controller ruling: audit M5 (treat any 2-col ID+X header's ``X`` as
    a target purely by structural elimination) is explicitly NOT adopted.
    ``value`` must stay unknown exactly like ``feature``/``timestamp``."""
    path = _write(tmp_path, "table.csv", "id,value\n1,10\n2,20\n")

    inspection = inspect_label_layout(path)

    assert inspection.layout == "unknown"


def test_two_column_headerless_table_is_unknown(tmp_path: Path) -> None:
    path = _write(tmp_path, "table.csv", "1,alpha\n2,beta\n3,gamma\n")

    inspection = inspect_label_layout(path)

    assert inspection.layout == "unknown"
    assert inspection.has_header is False


# ---------------------------------------------------------------------------
# inspect_label_layout: fixed two-column sparse_labels (both roles explicit)
# ---------------------------------------------------------------------------


def test_record_id_target_two_column_is_sparse_labels(tmp_path: Path) -> None:
    path = _write(tmp_path, "labels.csv", "record_id,target\n1,alpha\n2,beta\n")

    inspection = inspect_label_layout(path)

    assert inspection.layout == "sparse_labels"
    assert inspection.has_header is True
    assert inspection.delimiter == ","


# ---------------------------------------------------------------------------
# inspect_label_layout / parse_sparse_label_rows: rectangular_table
# ---------------------------------------------------------------------------


def test_five_column_table_is_rectangular_table(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "tokens.csv",
        "sentence_id,token_id,class,before,after\n"
        "1,1,PLAIN,hello,hello\n"
        "1,2,PUNCT,.,.\n",
    )

    inspection = inspect_label_layout(path)

    assert inspection.layout == "rectangular_table"


def test_parser_rejects_rectangular_table_layout(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "tokens.csv",
        "sentence_id,token_id,class,before,after\n"
        "1,1,PLAIN,hello,hello\n"
        "1,2,PUNCT,.,.\n",
    )

    with pytest.raises(ValueError):
        parse_sparse_label_rows(path)


# ---------------------------------------------------------------------------
# inspect_label_layout / parse_sparse_label_rows: genuinely variable width
# (audit ruling M3, including the width-1 bare-ID floor)
# ---------------------------------------------------------------------------


@pytest.fixture
def variable_width_with_bare_id_path(tmp_path: Path) -> Path:
    """A genuinely variable-width ID-plus-label file: id ``1`` has two
    labels, id ``2`` has one label, and id ``3`` is a bare width-1 record
    with zero labels and no delimiter at all."""
    return _write(
        tmp_path,
        "variable_labels.csv",
        "rec_id,labels\n1,alpha,beta\n2,gamma\n3\n",
    )


def test_variable_width_id_plus_label_file_is_sparse_labels(
    variable_width_with_bare_id_path: Path,
) -> None:
    inspection = inspect_label_layout(variable_width_with_bare_id_path)

    assert inspection.layout == "sparse_labels"


def test_variable_width_bare_id_row_parses_to_zero_label_rows(
    variable_width_with_bare_id_path: Path,
) -> None:
    """The width-1 record (id ``3``) contributes no target rows but must
    not break parsing or classification."""
    df = parse_sparse_label_rows(variable_width_with_bare_id_path)

    assert df["record_id"].tolist() == ["1", "1", "2"]
    assert df["target"].tolist() == ["alpha", "beta", "gamma"]
    assert "3" not in df["record_id"].tolist()


# ---------------------------------------------------------------------------
# inspect_label_layout: bounded reading (max 64 KiB, stop after sample_rows)
# ---------------------------------------------------------------------------


def test_inspect_label_layout_stops_after_sample_rows(tmp_path: Path) -> None:
    """The default ``sample_rows=50`` budget must be honored: genuinely
    variable-width rows planted far past row 50 must never be consulted,
    or this fixture would (wrongly) flip from `unknown` to `sparse_labels`."""
    lines = [f"{i},{i * 10}" for i in range(1, 51)]
    for i in range(51, 5000):
        width = 2 + (i % 4)
        lines.append(",".join(str(i) for _ in range(width)))

    path = _write(tmp_path, "bounded_rows.csv", "\n".join(lines) + "\n")

    inspection = inspect_label_layout(path)

    assert inspection.layout == "unknown"


def test_inspect_label_layout_is_bounded_by_bytes(tmp_path: Path) -> None:
    """The 64 KiB read cap must be honored on its own: ``sample_rows`` is
    set far higher than the row budget could ever reach first, isolating
    the byte cap as the only thing standing between this fixture's clean
    prefix and the variable-width tail beyond it."""
    fixed_row = "1,10\n"
    prefix = fixed_row * ((70 * 1024) // len(fixed_row))

    tail_lines = []
    for i in range(1, 500):
        width = 2 + (i % 4)
        tail_lines.append(",".join(str(i) for _ in range(width)))
    tail = "\n".join(tail_lines) + "\n"

    path = _write(tmp_path, "bounded_bytes.csv", prefix + tail)

    inspection = inspect_label_layout(path, sample_rows=1_000_000)

    assert inspection.layout == "unknown"


def test_empty_file_is_unknown_with_evidence(tmp_path: Path) -> None:
    path = _write(tmp_path, "empty.csv", "")

    inspection = inspect_label_layout(path)

    assert inspection.layout == "unknown"
    assert len(inspection.evidence) > 0


# ---------------------------------------------------------------------------
# inspect_label_layout: delimiter-sniffing fallback (fix round 1, IMPORTANT)
# ---------------------------------------------------------------------------


def test_sniff_delimiter_fallback_on_undetectable_delimiter(tmp_path: Path) -> None:
    """A single-column file with no delimiter character anywhere in the
    sample makes ``csv.Sniffer().sniff()`` raise ``csv.Error``. The
    inspector must fall back to "," instead of propagating the exception,
    and still reach a safe, deterministic verdict."""
    path = _write(tmp_path, "single_column.csv", "alpha\nbeta\ngamma\n")

    inspection = inspect_label_layout(path)

    assert inspection.delimiter == ","
    assert inspection.layout == "unknown"


# ---------------------------------------------------------------------------
# inspect_label_layout / parse_sparse_label_rows: header detection requires
# semantic corroboration (fix round 1, CRITICAL)
# ---------------------------------------------------------------------------


def test_headerless_variable_width_file_with_non_numeric_first_row_keeps_all_records(
    tmp_path: Path,
) -> None:
    """Regression: a genuinely headerless file whose first record's fields
    are all non-numeric strings must not be misdetected as a header.
    ``_looks_like_header``'s all-non-numeric guess alone is not enough
    evidence - without a corroborating id/target/file-like token, treating
    row 0 as a header silently drops record "abc" and both its labels when
    parsing skips it. Every record - including the trailing bare-ID "ghi",
    which legitimately carries zero labels - must survive."""
    path = _write(tmp_path, "headerless_strings.csv", "abc,alpha,beta\ndef,gamma\nghi\n")

    inspection = inspect_label_layout(path)

    assert inspection.has_header is False
    assert inspection.layout == "sparse_labels"

    df = parse_sparse_label_rows(path)

    assert df["record_id"].tolist() == ["abc", "abc", "def"]
    assert df["target"].tolist() == ["alpha", "beta", "gamma"]
    assert "ghi" not in df["record_id"].tolist()


# ---------------------------------------------------------------------------
# parse_sparse_label_rows: quote-awareness and lexical preservation
# ---------------------------------------------------------------------------


def test_quoted_delimiter_produces_single_label(tmp_path: Path) -> None:
    path = _write(tmp_path, "quoted.csv", 'record_id,target\n1,"alpha,beta"\n')

    df = parse_sparse_label_rows(path)

    assert df["record_id"].tolist() == ["1"]
    assert df["target"].tolist() == ["alpha,beta"]


def test_mixed_lexical_values_are_strings(tmp_path: Path) -> None:
    path = _write(tmp_path, "labels.csv", "record_id,target\n1,42\n2,species\n")

    df = parse_sparse_label_rows(path)
    values = df["target"].tolist()

    assert values == ["42", "species"]
    assert all(isinstance(value, str) for value in values)


def test_all_numeric_looking_values_remain_strings_by_default(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, "labels.csv", "record_id,target\n1,10\n2,20\n3,30\n")

    df = parse_sparse_label_rows(path)
    values = df["target"].tolist()

    assert values == ["10", "20", "30"]
    assert all(isinstance(value, str) for value in values)


def test_leading_zero_and_sign_preserved_distinct(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "labels.csv",
        "record_id,target\n1,001\n2,1\n3,+1\n",
    )

    df = parse_sparse_label_rows(path)
    values = df["target"].tolist()

    assert values == ["001", "1", "+1"]
    assert len(set(values)) == 3


def test_record_ids_preserved_as_strings_even_when_numeric_looking(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path,
        "labels.csv",
        "record_id,target\n007,alpha\n1,beta\n+2,gamma\n",
    )

    df = parse_sparse_label_rows(path)
    record_ids = df["record_id"].tolist()

    assert record_ids == ["007", "1", "+2"]
    assert all(isinstance(value, str) for value in record_ids)


# ---------------------------------------------------------------------------
# parse_sparse_label_rows: explicit integer target_scalar_type
# ---------------------------------------------------------------------------


def test_explicit_integer_target_scalar_type_converts_numeric_fixture(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, "labels.csv", "record_id,target\n1,10\n2,20\n3,30\n")

    df = parse_sparse_label_rows(path, target_scalar_type="integer")
    values = df["target"].tolist()

    assert values == [10, 20, 30]
    assert all(isinstance(value, int) for value in values)
    assert pd.api.types.is_integer_dtype(df["target"])


def test_explicit_integer_target_scalar_type_rejects_non_numeric_value(
    tmp_path: Path,
) -> None:
    path = _write(
        tmp_path,
        "labels.csv",
        "record_id,target\n1,10\n2,notanumber\n",
    )

    with pytest.raises(ValueError):
        parse_sparse_label_rows(path, target_scalar_type="integer")


# ---------------------------------------------------------------------------
# parse_sparse_label_rows: hidden marker filtering and layout rejection
# ---------------------------------------------------------------------------


def test_hidden_marker_values_are_filtered_out(tmp_path: Path) -> None:
    path = _write(
        tmp_path,
        "labels.csv",
        "record_id,target\n1,alpha\n2,?\n3,beta\n",
    )

    df = parse_sparse_label_rows(path)

    assert df["record_id"].tolist() == ["1", "3"]
    assert df["target"].tolist() == ["alpha", "beta"]


def test_parser_rejects_unknown_layout(tmp_path: Path) -> None:
    path = _write(tmp_path, "table.csv", "id,feature\n1,x\n2,y\n")

    with pytest.raises(ValueError):
        parse_sparse_label_rows(path)
