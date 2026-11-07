from medrax_taixray_audit.dataset import LABEL_COLUMNS, TAIXRayClient, TAIXRecord


def test_manifest_entry_omits_identifiers_and_urls():
    record = TAIXRecord(
        row_idx=17,
        patient_id="not-for-publication",
        image_url="https://signed.example.invalid/image.png",
        labels={label: 0 for label in LABEL_COLUMNS},
    )
    entry = record.manifest_entry("a" * 64)
    assert entry == {
        "row_idx": 17,
        "labels": {label: 0 for label in LABEL_COLUMNS},
        "image_sha256": "a" * 64,
    }


def test_parse_row_retains_only_required_values():
    source_row = {
        "row_idx": 5,
        "row": {
            "PatientID": "pseudonym",
            "Image": {"src": "https://signed.example.invalid/image.png"},
            **{label: 2 for label in LABEL_COLUMNS},
        },
    }
    record = TAIXRayClient()._parse_row(source_row)
    assert record.row_idx == 5
    assert record.patient_id == "pseudonym"
    assert set(record.labels) == set(LABEL_COLUMNS)


def test_fetch_row_caches_the_containing_server_page(monkeypatch):
    calls = []
    client = TAIXRayClient(min_request_interval_seconds=0)

    def fake_request(params):
        calls.append(params)
        return {
            "rows": [
                {
                    "row_idx": index,
                    "row": {
                        "PatientID": f"patient-{index}",
                        "Image": {"src": f"https://example.invalid/{index}.png"},
                        **{label: 0 for label in LABEL_COLUMNS},
                    },
                }
                for index in range(100, 200)
            ]
        }

    monkeypatch.setattr(client, "_request_json", fake_request)
    assert client.fetch_row(123).row_idx == 123
    assert client.fetch_row(156).row_idx == 156
    assert len(calls) == 1
    assert calls[0]["offset"] == 100
    assert calls[0]["length"] == 100


def test_fetch_page_exposes_records_in_server_order(monkeypatch):
    client = TAIXRayClient(min_request_interval_seconds=0)
    expected = [
        TAIXRecord(
            row_idx=index,
            patient_id=f"patient-{index}",
            image_url=f"https://example.invalid/{index}.png",
            labels={label: 0 for label in LABEL_COLUMNS},
        )
        for index in (200, 201)
    ]
    monkeypatch.setattr(client, "_fetch_page", lambda offset, length: {r.row_idx: r for r in expected})
    assert client.fetch_page(200, 2) == expected
