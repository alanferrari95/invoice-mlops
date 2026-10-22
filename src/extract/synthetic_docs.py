"""Synthetic multi-page invoice PDFs with known field values.

Each document has one invoice page and 1-3 distractor pages (cover letter, terms and
conditions, remittance advice) that also contain numbers and dates, so both the page filter
and the extractor have something to get wrong. Three layouts (English, Spanish, "tax
invoice"), several date and number formats, and ~10% of invoices without an invoice number
(the expected value is then null).

    python -m src.extract.synthetic_docs --n 40 --seed 13 --out data/gold/extract
"""

from __future__ import annotations

import argparse
import json
import random
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

FIELDS = ["vendor_name", "invoice_number", "invoice_date", "total_amount", "currency"]
SPANISH_MONTHS = ["enero", "febrero", "marzo", "abril", "mayo", "junio", "julio", "agosto",
                  "septiembre", "octubre", "noviembre", "diciembre"]
CURRENCY_SYMBOL = {"EUR": "€", "USD": "$", "GBP": "£"}


@dataclass
class InvoiceTruth:
    doc_id: str
    vendor_name: str
    invoice_number: Optional[str]
    invoice_date: str  # ISO 8601
    total_amount: float
    currency: str
    layout: str
    invoice_page: int  # 1-based
    n_pages: int


def _money_en(x: float) -> str:
    return f"{x:,.2f}"


def _money_es(x: float) -> str:
    return f"{x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _lines(rng: random.Random, n: int) -> list[tuple[str, int, float]]:
    items = ["Consulting hours", "Cloud hosting", "Office supplies", "Maintenance", "Software licence",
             "Freight", "Training session", "Hardware rental", "Support plan", "Translation"]
    return [(rng.choice(items), rng.randint(1, 12), round(rng.uniform(15, 900), 2)) for _ in range(n)]


class _Page:
    """Tiny text-layout helper over a reportlab canvas."""

    def __init__(self, c: canvas.Canvas):
        self.c = c
        self.y = A4[1] - 60

    def line(self, text: str, size: int = 10, x: int = 50, gap: int = 16, bold: bool = False):
        self.c.setFont("Helvetica-Bold" if bold else "Helvetica", size)
        self.c.drawString(x, self.y, text)
        self.y -= gap

    def skip(self, n: int = 10):
        self.y -= n

    def end(self):
        self.c.showPage()


def _invoice_page(p: _Page, t: InvoiceTruth, customer: str, lines, vat_rate: float, rng: random.Random):
    d = date.fromisoformat(t.invoice_date)
    subtotal = round(sum(q * u for _, q, u in lines), 2)
    vat = round(t.total_amount - subtotal, 2)
    sym = CURRENCY_SYMBOL[t.currency]

    if t.layout == "en":
        p.line("INVOICE", 20, gap=30, bold=True)
        p.line(t.vendor_name, 12, bold=True)
        p.line(f"{rng.randint(1, 999)} Market Street, Barcelona", 9)
        p.skip()
        if t.invoice_number:
            p.line(f"Invoice No: {t.invoice_number}")
        p.line(f"Invoice date: {d.strftime('%d/%m/%Y')}")
        p.line(f"Due date: {(d + timedelta(days=30)).strftime('%d/%m/%Y')}")
        p.line(f"Bill to: {customer}")
        p.skip()
        for desc, q, u in lines:
            p.line(f"{desc:<24} {q:>3} x {_money_en(u):>10}  = {_money_en(q * u):>12}", 9, gap=13)
        p.skip()
        p.line(f"Subtotal: {_money_en(subtotal)} {t.currency}")
        p.line(f"VAT {int(vat_rate * 100)}%: {_money_en(vat)} {t.currency}")
        p.line(f"TOTAL: {_money_en(t.total_amount)} {t.currency}", 12, bold=True)
    elif t.layout == "es":
        p.line("FACTURA", 20, gap=30, bold=True)
        p.line(f"Emisor: {t.vendor_name}", 12, bold=True)
        p.line(f"NIF: B{rng.randint(10000000, 99999999)}", 9)
        p.skip()
        if t.invoice_number:
            p.line(f"Nº de factura: {t.invoice_number}")
        p.line(f"Fecha de emisión: {d.day} de {SPANISH_MONTHS[d.month - 1]} de {d.year}")
        p.line(f"Cliente: {customer}")
        p.skip()
        for desc, q, u in lines:
            p.line(f"{desc:<24} {q:>3} x {_money_es(u):>10} {sym}", 9, gap=13)
        p.skip()
        p.line(f"Base imponible: {_money_es(subtotal)} {sym}")
        p.line(f"IVA ({int(vat_rate * 100)}%): {_money_es(vat)} {sym}")
        p.line(f"Total factura: {_money_es(t.total_amount)} {sym}", 12, bold=True)
    else:  # "tax"
        p.line("Tax Invoice", 18, gap=28, bold=True)
        if t.invoice_number:
            p.line(f"Reference: {t.invoice_number}")
        p.line(f"Issued {d.strftime('%B')} {d.day}, {d.year}")
        p.line(f"Customer: {customer}")
        p.skip()
        for desc, q, u in lines:
            p.line(f"- {desc}: {q} units @ {sym}{_money_en(u)}", 9, gap=13)
        p.skip()
        p.line(f"Net {sym}{_money_en(subtotal)}   Tax {sym}{_money_en(vat)}")
        p.line(f"Amount due: {t.currency} {_money_en(t.total_amount)}", 12, bold=True)
        p.skip(30)
        p.line(f"{t.vendor_name} - registered office {rng.randint(1, 99)} Harbour Road", 8)
    p.end()


def _cover_letter(p: _Page, t: InvoiceTruth, customer: str, rng: random.Random):
    other = round(rng.uniform(100, 9000), 2)
    p.line(customer, 11, bold=True)
    p.line(f"Accounts payable - PO {rng.randint(4500000000, 4599999999)}", 9)
    p.skip(20)
    p.line("Dear customer,")
    p.line("Please find enclosed our invoice for the services delivered this period.")
    p.line(f"Your previous balance of {_money_en(other)} was settled on "
           f"{(date.fromisoformat(t.invoice_date) - timedelta(days=rng.randint(20, 60))).isoformat()}.")
    p.line("Payment terms are 30 days from the invoice date.")
    p.skip(20)
    p.line("Kind regards,")
    p.line("The billing team")
    p.end()


def _terms(p: _Page, rng: random.Random):
    p.line("GENERAL TERMS AND CONDITIONS", 12, bold=True)
    clauses = [
        "1. Prices exclude taxes unless stated otherwise.",
        f"2. Late payments accrue a fixed fee of {rng.choice([40, 50, 75])} EUR plus statutory interest.",
        "3. Claims must be notified in writing within 8 days of delivery.",
        f"4. These terms were last updated on {rng.randint(1, 28)}/0{rng.randint(1, 9)}/2025.",
        "5. Jurisdiction: courts of Barcelona.",
    ]
    for clause in clauses:
        p.line(clause, 9)
    p.end()


def _remittance(p: _Page, t: InvoiceTruth, rng: random.Random):
    p.line("REMITTANCE ADVICE", 12, bold=True)
    p.line("Detach and return with your payment.")
    p.line(f"IBAN: ES{rng.randint(10, 99)} {rng.randint(1000, 9999)} {rng.randint(1000, 9999)} "
           f"{rng.randint(1000, 9999)} {rng.randint(1000, 9999)}")
    p.line("Amount enclosed: ______________")
    p.end()


def make_document(path: Path, doc_id: str, rng: random.Random, fake) -> InvoiceTruth:
    layout = rng.choice(["en", "es", "tax"])
    currency = "EUR" if layout == "es" else rng.choice(["EUR", "EUR", "USD", "GBP"])
    lines = _lines(rng, rng.randint(2, 6))
    vat_rate = rng.choice([0.21, 0.10, 0.20])
    subtotal = round(sum(q * u for _, q, u in lines), 2)
    total = round(subtotal * (1 + vat_rate), 2)
    number = None if rng.random() < 0.10 else rng.choice(
        [f"INV-2026-{rng.randint(1, 9999):04d}", f"F{rng.randint(100, 999)}/26", f"{rng.randint(100000, 999999)}"]
    )
    issued = date(2026, 1, 1) + timedelta(days=rng.randint(0, 280))

    distractors = rng.sample(["cover", "terms", "remittance"], k=rng.randint(1, 3))
    invoice_page = rng.randint(0, len(distractors))  # position among all pages, 0-based
    truth = InvoiceTruth(
        doc_id=doc_id, vendor_name=fake.company(), invoice_number=number, invoice_date=issued.isoformat(),
        total_amount=total, currency=currency, layout=layout,
        invoice_page=invoice_page + 1, n_pages=len(distractors) + 1,
    )
    customer = fake.company()

    c = canvas.Canvas(str(path), pagesize=A4, invariant=1)  # byte-reproducible
    c.setTitle(f"Document {doc_id}")
    order = distractors[:invoice_page] + ["invoice"] + distractors[invoice_page:]
    for kind in order:
        page = _Page(c)
        if kind == "invoice":
            _invoice_page(page, truth, customer, lines, vat_rate, rng)
        elif kind == "cover":
            _cover_letter(page, truth, customer, rng)
        elif kind == "terms":
            _terms(page, rng)
        else:
            _remittance(page, truth, rng)
    c.save()
    return truth


def generate(out_dir: Path, n: int, seed: int, prefix: str = "doc") -> list[InvoiceTruth]:
    from faker import Faker

    fake = Faker(["en_GB", "es_ES"])
    Faker.seed(seed)
    rng = random.Random(seed)
    docs_dir = out_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)
    truths = []
    for i in range(1, n + 1):
        doc_id = f"{prefix}-{i:03d}"
        truths.append(make_document(docs_dir / f"{doc_id}.pdf", doc_id, rng, fake))
    with (out_dir / "labels.jsonl").open("w", encoding="utf-8", newline="\n") as f:
        for t in truths:
            f.write(json.dumps(asdict(t), ensure_ascii=False) + "\n")
    return truths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=40)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", default="data/gold/extract")
    parser.add_argument("--prefix", default="gold")
    args = parser.parse_args()
    truths = generate(Path(args.out), args.n, args.seed, args.prefix)
    print(f"{len(truths)} documents -> {args.out}")


if __name__ == "__main__":
    main()
