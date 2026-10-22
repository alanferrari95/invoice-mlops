from src.api.main import EXTRACTIONS_TABLE, prompt_version_from_env
from tests.fakes import GOLD_DIR, gold_labels


def _pdf(i=0):
    label = gold_labels()[i]
    return label, (GOLD_DIR / "docs" / f"{label['doc_id']}.pdf").read_bytes()


def test_extract_returns_fields_usage_and_pages(client):
    label, pdf = _pdf(1)
    r = client.post("/extract", files={"file": ("x.pdf", pdf, "application/pdf")})
    assert r.status_code == 200
    body = r.json()
    assert body["fields"]["total_amount"] == label["total_amount"]
    assert body["pages"] == {"total": label["n_pages"], "kept": [label["invoice_page"]]}
    assert body["usage"]["total_tokens"] > 0
    assert body["prompt_version"] == "v2"


def test_extract_logs_metadata_only(client, sink):
    label, pdf = _pdf(2)
    client.post("/extract", files={"file": ("x.pdf", pdf, "application/pdf")})
    table, row = sink.rows[-1]
    assert table == EXTRACTIONS_TABLE
    assert row["prompt_tokens"] > 0 and row["pages_sent"] == 1
    assert label["vendor_name"] not in str(row)  # no document content in the log


def test_extract_rejects_non_pdf(client):
    r = client.post("/extract", files={"file": ("x.txt", b"hello", "text/plain")})
    assert r.status_code == 415


def test_version_reports_prompt(client):
    assert client.get("/version").json()["prompt_version"] == "v2"


def test_prompt_version_comes_from_registry_label(monkeypatch):
    monkeypatch.setenv("EXTRACTOR_REGISTRY_MODEL", "projects/p/locations/europe-west4/models/invoice-extractor")

    def fetch(url):
        return {"versionId": "2", "artifactUri": "gs://b/p", "labels": {"prompt_version": "v2"}}

    assert prompt_version_from_env(fetch=fetch) == "v2"


def test_extraction_can_be_switched_off(tiny_model):
    from fastapi.testclient import TestClient

    from src.api.main import LoadedModel, create_app
    from tests.conftest import MemorySink

    app = create_app(model_loader=lambda: LoadedModel(tiny_model, "t", "memory://"), sink=MemorySink(),
                     extractor_factory=lambda: None)
    with TestClient(app) as c:
        _, pdf = _pdf(0)
        assert c.post("/extract", files={"file": ("x.pdf", pdf, "application/pdf")}).status_code == 503
        assert c.get("/version").json()["prompt_version"] is None
