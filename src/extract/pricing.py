"""Gemini list prices used to turn measured tokens into cost.

Source: https://cloud.google.com/vertex-ai/generative-ai/pricing, Gemini 3.8 Flash, checked
2026-10-22. We call the `eu` multi-region endpoint, which is billed at the "non-global" rate:
USD 0.825 / 1M input and USD 4.125 / 1M output tokens until 2026-12-31 (launch price), then
USD 1.65 / 8.25. The global endpoint is 10% cheaper but gives no EU processing guarantee.
Thinking tokens are billed as output tokens.

Only token counts are measured; cost = tokens x these prices.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Price:
    input_per_m: float
    output_per_m: float
    valid: str


EU_LAUNCH = Price(0.825, 4.125, "gemini-3.8-flash, eu multi-region, until 2026-12-31")
EU_STANDARD = Price(1.65, 8.25, "gemini-3.8-flash, eu multi-region, from 2027-01-01")


def cost_usd(prompt_tokens: int, output_tokens: int, thoughts_tokens: int, price: Price) -> float:
    return (prompt_tokens * price.input_per_m + (output_tokens + thoughts_tokens) * price.output_per_m) / 1e6
