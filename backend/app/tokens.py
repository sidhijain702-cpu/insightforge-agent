"""Approximate token counting.

LEARNING NOTE: 1 token is roughly 4 characters of English. That is accurate to
~10-15%, which is fine for *budgeting* (we leave headroom) and needs no network
or tokenizer download. For billing-grade numbers use the provider's usage field.
"""


def count_tokens(text: str) -> int:
    return max(1, len(text) // 4) if text else 0
