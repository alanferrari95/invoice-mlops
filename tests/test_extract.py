from pathlib import Path

import pytest

from src.eval.evaluate_extract import evaluate
from src.extract.extractor import Extractor, load_prompt, make_backend
from src.extract.gemini import GeminiBackend
from src.extract.ollama import OllamaBackend
from src.extract.pages import filter_pdf
from tests.fakes import GOLD_DIR, FakeGeminiClient, gold_labels


def test_page_filter_keeps_exactly_the_invoice_page_on_the_gold_set():
    for label in gold_labels():
        doc = filter_pdf((GOLD_DIR / "docs" / f"{label['doc_id']}.pdf").read_bytes())
        assert doc.pages_kept == [label["invoice_page"]], label["doc_id"]
        assert doc.pages_total == label["n_pages"]


def test_extractor_sends_structured_output_request_and_counts_tokens():
    client = FakeGeminiClient()
    extractor = Extractor(GeminiBackend(client), prompt_version="v2")
    label = gold_labels()[0]
    result = extractor.extract((GOLD_DIR / "docs" / f"{label['doc_id']}.pdf").read_bytes())

    call = client.models.calls[0]
    assert call["model"] == "gemini-3.8-flash"
    assert call["config"].response_mime_type == "application/json"
    assert call["config"].system_instruction == load_prompt("v2")
    assert "--- page" in call["contents"]
    assert result.fields["vendor_name"] == label["vendor_name"]
    assert result.usage.output_tokens == 40 and result.usage.thoughts_tokens == 10
    assert len(result.doc_sha256) == 64


def test_unknown_prompt_version_is_rejected():
    with pytest.raises(ValueError):
        load_prompt("v99")


def test_evaluate_on_gold_with_perfect_fake_reports_full_marks_and_cost():
    out = evaluate(Extractor(GeminiBackend(FakeGeminiClient())), GOLD_DIR)
    assert out["report"]["micro"]["precision"] == 1.0
    assert out["report"]["micro"]["recall"] == 1.0
    assert out["usage"]["docs_ok"] == 40
    assert out["usage"]["usd_per_1000_docs_launch"] > 0
    assert out["usage"]["pages_kept_ratio"] < 1.0


def test_evaluate_detects_a_worse_extractor():
    out = evaluate(Extractor(GeminiBackend(FakeGeminiClient(corrupt={"currency": "€"}))), GOLD_DIR)
    assert out["report"]["fields"]["currency"]["precision"] == 0.0
    assert out["report"]["micro"]["precision"] < 0.85


def test_ollama_backend_sends_the_json_schema_and_reads_token_counts():
    import json

    label = gold_labels()[3]
    sent = {}

    def fake_post(url, payload, timeout):
        sent.update(url=url, payload=payload)
        answer = {k: label[k] for k in ("vendor_name", "invoice_number", "invoice_date", "total_amount", "currency")}
        return {"message": {"content": json.dumps(answer)}, "prompt_eval_count": 321, "eval_count": 45}

    extractor = Extractor(OllamaBackend(model="qwen2.5:7b", host="http://gpu-box:11434", post=fake_post))
    result = extractor.extract((GOLD_DIR / "docs" / f"{label['doc_id']}.pdf").read_bytes())

    assert sent["url"] == "http://gpu-box:11434/api/chat"
    assert sent["payload"]["format"]["properties"].keys() >= {"vendor_name", "total_amount"}
    assert sent["payload"]["options"] == {"temperature": 0}
    assert result.fields["total_amount"] == label["total_amount"]
    assert (result.usage.prompt_tokens, result.usage.output_tokens) == (321, 45)
    assert result.model == "qwen2.5:7b"


def test_backend_is_chosen_by_env(monkeypatch):
    monkeypatch.setenv("EXTRACT_BACKEND", "ollama")
    monkeypatch.setenv("OLLAMA_MODEL", "llama3.2:3b")
    assert make_backend().model == "llama3.2:3b"
    with pytest.raises(ValueError):
        make_backend("openai")
