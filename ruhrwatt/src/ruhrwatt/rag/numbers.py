"""Reading numbers out of free text (German and English formats)."""

from __future__ import annotations

import re

# A unit suffix must end the word ("4 Mio", "3.8 million"), so "MWh" is not read as million.
_NUM = re.compile(
    r"(?<![\w.])[-−]?\d[\d.,]*"
    r"(?:\s?(?:milliarden|millionen|million|mrd\.?|mio\.?|mn|m|k|tsd\.?)(?![a-zäöü]))?",
    re.I,
)


def numbers(text: str) -> list[float]:
    """Every number in the text, reading 1,234.5 / 1.234,5 / 99,72 / 3.8 million / 4 Mio."""
    out: list[float] = []
    for raw in _NUM.findall(text):
        s = raw.strip().lower().replace("−", "-")
        mult = 1.0
        m = re.search(r"\s?(milliarden|millionen|million|mrd\.?|mio\.?|mn|m|k|tsd\.?)$", s)
        if m:
            unit = m.group(1).rstrip(".")
            mult = {"k": 1e3, "tsd": 1e3, "mrd": 1e9, "milliarden": 1e9}.get(unit, 1e6)
            s = s[: m.start()].strip()
        s = s.rstrip(".,")
        candidates: set[str] = set()
        if "," in s and "." in s:
            if s.rfind(",") > s.rfind("."):
                candidates.add(s.replace(".", "").replace(",", "."))
            else:
                candidates.add(s.replace(",", ""))
        elif "," in s or "." in s:
            sep = "," if "," in s else "."
            parts = s.split(sep)
            if len(parts) > 2:  # 1,234,567 or 1.234.567
                candidates.add("".join(parts))
            else:
                candidates.add(s.replace(sep, "."))  # decimal
                if len(parts[1]) == 3:  # 3,819 could also be thousands
                    candidates.add("".join(parts))
        else:
            candidates.add(s)
        for c in candidates:
            try:
                out.append(float(c) * mult)
            except ValueError:
                pass
    return out
