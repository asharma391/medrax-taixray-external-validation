from medrax_taixray_audit.dataset import LABEL_COLUMNS, TAIXRecord
from medrax_taixray_audit.labels import PRIMARY_ENDPOINTS, binary_target
from medrax_taixray_audit.sampling import (
    select_two_stage_patient_pilot,
    select_unique_patient_records,
)


def _labels(value: int) -> dict[str, int]:
    return {label: value for label in LABEL_COLUMNS}


def test_primary_target_excludes_questionable_grade():
    endpoint = PRIMARY_ENDPOINTS[0]
    assert binary_target(_labels(0), endpoint) == 0
    assert binary_target(_labels(1), endpoint) is None
    assert binary_target(_labels(2), endpoint) == 1


def test_sampler_keeps_one_in_memory_record_per_patient_and_is_deterministic():
    records = {
        index: TAIXRecord(
            row_idx=index,
            patient_id=f"patient-{index // 2}",
            image_url=f"https://example.invalid/{index}.png",
            labels=_labels(index % 5),
        )
        for index in range(12)
    }
    fetch = records.__getitem__
    first = select_unique_patient_records(fetch, total_rows=12, sample_size=5, seed=11)
    second = select_unique_patient_records(fetch, total_rows=12, sample_size=5, seed=11)
    assert [record.row_idx for record in first] == [record.row_idx for record in second]
    assert len({record.patient_id for record in first}) == 5


def test_two_stage_sampler_is_deterministic_and_reports_design_details():
    records = [
        TAIXRecord(
            row_idx=index,
            patient_id=f"patient-{index // 2}",
            image_url=f"https://example.invalid/{index}.png",
            labels=_labels(index % 5),
        )
        for index in range(40)
    ]

    def fetch_page(offset: int):
        return list(records[offset : offset + 10])

    first, details = select_two_stage_patient_pilot(
        fetch_page, total_rows=40, sample_size=8, seed=7, page_size=10, page_budget=2
    )
    second, _ = select_two_stage_patient_pilot(
        fetch_page, total_rows=40, sample_size=8, seed=7, page_size=10, page_budget=2
    )
    assert [record.row_idx for record in first] == [record.row_idx for record in second]
    assert len({record.patient_id for record in first}) == 8
    assert details["pages_examined"] == 2
    assert details["candidate_rows"] == 20
