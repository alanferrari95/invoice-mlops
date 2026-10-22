"""Run the extractor over a labelled document set and report quality, tokens and cost.

    python -m src.eval.evaluate_extract --prompt-version v2 --gold data/gold/extract
    python -m src.eval.evaluate_extract --prompt-version v2 --gold /tmp/sample50 --out report.json

Used by the evaluate_extract pipeline component and for the README numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional

import numpy as np

from src.eval.extract_metrics import field_report
from src.extract.gemini import GeminiExtractor, make_client
from src.extract.pricing import EU_LAUNCH, EU_STANDARD, cost_usd
from src.extract.schema import FIELDS


def load_gold(gold_dir: Path) -> dict[str, dict]:
    labels = {}
    for line in (gold_dir / "labels.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            labels[row["doc_id"]] = row
    return labels


def evaluate(extractor: GeminiExtractor, gold_dir: Path, limit: Optional[int] = None) -> dict:
    gold = load_gold(gold_dir)
    if limit:
        gold = dict(list(gold.items())[:limit])

    predictions, per_doc, errors = {}, [], {}
    for doc_id in gold:
        pdf = (gold_dir / "docs" / f"{doc_id}.pdf").read_bytes()
        try:
            result = extractor.extract(pdf)
        except Exception as err:  # one bad call must not kill the whole evaluation
            errors[doc_id] = repr(err)
            continue
        predictions[doc_id] = result.fields
        per_doc.append({
            "doc_id": doc_id,
            "fields": result.fields,
            "prompt_tokens": result.usage.prompt_tokens,
            "output_tokens": result.usage.output_tokens,
            "thoughts_tokens": result.usage.thoughts_tokens,
            "latency_ms": round(result.latency_ms, 1),
            "pages_total": result.pages_total,
            "pages_kept": result.pages_kept,
            "chars_total": result.extra["chars_total"],
            "chars_kept": result.extra["chars_kept"],
        })

    report = field_report(predictions, {k: {f: v.get(f) for f in FIELDS} for k, v in gold.items()})
    n = len(per_doc)
    totals = {k: int(sum(d[k] for d in per_doc)) for k in ("prompt_tokens", "output_tokens", "thoughts_tokens")}
    latencies = [d["latency_ms"] for d in per_doc]
    usage = {
        "docs_ok": n,
        "docs_failed": len(errors),
        **totals,
        "avg_prompt_tokens": round(totals["prompt_tokens"] / n, 1) if n else None,
        "avg_output_tokens": round((totals["output_tokens"] + totals["thoughts_tokens"]) / n, 1) if n else None,
        "latency_p50_ms": round(float(np.percentile(latencies, 50)), 1) if n else None,
        "latency_p95_ms": round(float(np.percentile(latencies, 95)), 1) if n else None,
        "pages_kept_ratio": round(sum(len(d["pages_kept"]) for d in per_doc) / max(sum(d["pages_total"] for d in per_doc), 1), 4),
        "chars_kept_ratio": round(sum(d["chars_kept"] for d in per_doc) / max(sum(d["chars_total"] for d in per_doc), 1), 4),
    }
    for price, key in ((EU_LAUNCH, "usd_per_1000_docs_launch"), (EU_STANDARD, "usd_per_1000_docs_standard")):
        usage[key] = round(1000 * cost_usd(totals["prompt_tokens"], totals["output_tokens"],
                                           totals["thoughts_tokens"], price) / n, 4) if n else None
    return {
        "model": extractor.model,
        "prompt_version": extractor.prompt_version,
        "thinking_level": extractor.thinking_level,
        "report": report,
        "usage": usage,
        "errors": errors,
        "per_doc": per_doc,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt-version", default="v2")
    parser.add_argument("--gold", default="data/gold/extract")
    parser.add_argument("--model", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    kwargs = {"model": args.model} if args.model else {}
    extractor = GeminiExtractor(make_client(), prompt_version=args.prompt_version, **kwargs)
    result = evaluate(extractor, Path(args.gold), args.limit)
    summary = {k: v for k, v in result.items() if k != "per_doc"}
    print(json.dumps(summary, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
