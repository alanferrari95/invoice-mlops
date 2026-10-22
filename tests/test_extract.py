from pathlib import Path

import pytest

from src.eval.evaluate_extract import evaluate
from src.extract.gemini import GeminiExtractor, load_prompt
from src.extract.pages import filter_pdf
from tests.fakes import GOLD_DIR, FakeGeminiClient, gold_labels


def test_page_filter_keeps_exactly_the_invoice_page_on_the_gold_set():
    for label in gold_labels():
        doc = filter_pdf((GOLD_DIR / "docs" / f"{label['doc_id']}.pdf").read_bytes())
        assert doc.pages_kept == [label["invoice_page"]], label["doc_id"]
        assert doc.pages_total == label["n_pages"]


def test_extractor_sends_structured_output_request_and_counts_tokens():
    client = FakeGeminiClient()
    extractor = GeminiExtractor(client, prompt_version="v2")
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
    out = evaluate(GeminiExtractor(FakeGeminiClient()), GOLD_DIR)
    assert out["report"]["micro"]["precision"] == 1.0
    assert out["report"]["micro"]["recall"] == 1.0
    assert out["usage"]["docs_ok"] == 40
    assert out["usage"]["usd_per_1000_docs_launch"] > 0
    assert out["usage"]["pages_kept_ratio"] < 1.0


def test_evaluate_detects_a_worse_extractor():
    out = evaluate(GeminiExtractor(FakeGeminiClient(corrupt={"currency": "€"})), GOLD_DIR)
    assert out["report"]["fields"]["currency"]["precision"] == 0.0
    assert out["report"]["micro"]["precision"] < 0.85
