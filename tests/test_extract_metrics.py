from src.eval.extract_metrics import decide_prompt, field_report, matches

GOLD = {
    "a": {"vendor_name": "Baker, Miller and Smith", "invoice_number": "INV-1", "invoice_date": "2026-02-22",
          "total_amount": 1234.56, "currency": "EUR"},
    "b": {"vendor_name": "Acme S.L.", "invoice_number": None, "invoice_date": "2026-03-01",
          "total_amount": 10.0, "currency": "USD"},
}


def test_normalisation_rules():
    assert matches("vendor_name", "BAKER MILLER & SMITH", "Baker, Miller and Smith") is False
    assert matches("vendor_name", "baker, miller and smith ", "Baker, Miller and Smith")
    assert matches("total_amount", "1234.555", 1234.56)
    assert not matches("total_amount", 1234.0, 1234.56)
    assert not matches("invoice_date", "22/02/2026", "2026-02-22")
    assert matches("currency", "eur", "EUR")


def test_perfect_predictions():
    report = field_report(GOLD, GOLD)
    assert report["micro"]["precision"] == 1.0 and report["micro"]["recall"] == 1.0
    assert report["fields"]["invoice_number"]["tp"] == 1  # the null in b is neither TP nor FN


def test_wrong_value_hurts_precision_and_recall():
    pred = {**GOLD, "a": {**GOLD["a"], "invoice_date": "22/02/2026"}}
    f = field_report(pred, GOLD)["fields"]["invoice_date"]
    assert (f["tp"], f["fp"], f["fn"]) == (1, 1, 1)
    assert f["precision"] == 0.5 and f["recall"] == 0.5


def test_hallucinated_value_where_gold_is_null_is_a_false_positive():
    pred = {**GOLD, "b": {**GOLD["b"], "invoice_number": "PO-4551164366"}}
    f = field_report(pred, GOLD)["fields"]["invoice_number"]
    assert (f["tp"], f["fp"], f["fn"]) == (1, 1, 0)


def test_missing_document_counts_as_misses():
    f = field_report({"a": GOLD["a"]}, GOLD)["fields"]["currency"]
    assert (f["tp"], f["fp"], f["fn"]) == (1, 0, 1)


def test_gate_thresholds_and_champion():
    good = {"precision": 0.95, "recall": 0.93, "f1": 0.94}
    assert decide_prompt(good, None, 0.85, 0.80, 0.02).promote
    assert not decide_prompt({**good, "precision": 0.80}, None, 0.85, 0.80, 0.02).promote
    assert not decide_prompt({**good, "recall": 0.70}, None, 0.85, 0.80, 0.02).promote
    assert not decide_prompt(good, 0.98, 0.85, 0.80, 0.02).promote
