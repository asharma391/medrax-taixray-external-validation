from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from medrax_taixray_audit.pediatric import (
    PEDIATRIC_EXPECTED_LABEL_COUNTS,
    PEDIATRIC_EXPECTED_ROWS,
    iter_pediatric_test_records,
)


def _write_pinned_shape_parquet(path: Path, labels: list[int]) -> None:
    image_type = pa.struct([pa.field("bytes", pa.binary()), pa.field("path", pa.string())])
    images = [{"bytes": f"image-{index}".encode(), "path": f"ignored-{index}.jpeg"} for index in range(len(labels))]
    table = pa.table(
        {
            "image": pa.array(images, type=image_type),
            "label": pa.array(labels, type=pa.int64()),
        }
    )
    pq.write_table(table, path)


def test_pediatric_iterator_enforces_pinned_rows_and_label_totals(tmp_path):
    labels = [0] * PEDIATRIC_EXPECTED_LABEL_COUNTS[0] + [1] * PEDIATRIC_EXPECTED_LABEL_COUNTS[1]
    assert len(labels) == PEDIATRIC_EXPECTED_ROWS
    parquet_path = tmp_path / "test.parquet"
    _write_pinned_shape_parquet(parquet_path, labels)

    records = list(iter_pediatric_test_records(parquet_path))

    assert len(records) == PEDIATRIC_EXPECTED_ROWS
    assert records[0] == (0, b"image-0")
    assert records[-1] == (1, f"image-{PEDIATRIC_EXPECTED_ROWS - 1}".encode())


def test_pediatric_iterator_rejects_changed_label_total(tmp_path):
    labels = [0] * (PEDIATRIC_EXPECTED_LABEL_COUNTS[0] - 1) + [1] * (
        PEDIATRIC_EXPECTED_LABEL_COUNTS[1] + 1
    )
    parquet_path = tmp_path / "changed.parquet"
    _write_pinned_shape_parquet(parquet_path, labels)

    with pytest.raises(RuntimeError, match="label totals changed"):
        list(iter_pediatric_test_records(parquet_path))
