from __future__ import annotations

import re
from typing import Iterable


DOMESTIC_PATTERNS = (r"\bunited states\b", r"\bu\.s\.?\b", r"\busa\b", r"\bus\b")


def is_domestic_label(label: object, patterns: Iterable[str] = DOMESTIC_PATTERNS) -> bool:
    if label is None:
        return False
    text = str(label).strip().lower()
    if not text:
        return False
    return any(re.search(pattern, text) for pattern in patterns)


def hhi(weights: Iterable[float]) -> float:
    values = [float(w) for w in weights if w is not None and float(w) >= 0]
    total = sum(values)
    if total <= 0:
        return 0.0
    return sum((value / total) ** 2 for value in values)
