import pytest

from medrax_taixray_audit.chestdr import (
    CHESTDR_LABEL_COLUMNS,
    parse_published_csv,
    parse_test_manifest,
    validate_manifest_labels,
)


def _labels(first: int, second: int) -> str:
    values = ["0"] * len(CHESTDR_LABEL_COLUMNS)
    values[0] = str(first)
    values[1] = str(second)
    return ",".join(values)


def test_manifest_and_published_csv_are_cross_checked():
    manifest = parse_test_manifest(f"A001 {_labels(1, 0)}\nB002 {_labels(0, 1)}\n")
    headings = ",".join(["img_id", *CHESTDR_LABEL_COLUMNS])
    csv_text = f"{headings}\nA001.png,{_labels(1, 0)}\nB002.png,{_labels(0, 1)}\n"
    published = parse_published_csv(csv_text)
    validate_manifest_labels(manifest, published)
    assert manifest[0][1]["pleural_effusion"] == 1
    assert manifest[1][1]["nodule"] == 1


def test_manifest_validation_rejects_mismatched_labels():
    manifest = parse_test_manifest(f"A001 {_labels(1, 0)}\n")
    headings = ",".join(["img_id", *CHESTDR_LABEL_COLUMNS])
    published = parse_published_csv(f"{headings}\nA001,{_labels(0, 1)}\n")
    with pytest.raises(ValueError, match="disagreement"):
        validate_manifest_labels(manifest, published)
