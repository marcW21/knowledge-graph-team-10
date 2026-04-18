import re
from datetime import datetime
from dateutil import parser as dtparser


CORRUPT_TOKENS = {
    "#NAME?": None,
    "N/A": None,
    "NA": None,
    "NULL": None,
    "NONE": None,
    "": None,
}

LEGAL_SUFFIXES = {
    "INC", "INC.", "CORP", "CORP.", "CORPORATION", "LLC", "L.L.C.",
    "LTD", "LTD.", "LIMITED", "PLC", "CO", "CO.", "COMPANY",
    "BV", "B.V.", "AB", "AG", "GMBH", "S.A.", "SA", "NV", "N.V."
}


def normalize_whitespace(text: str | None) -> str:
    if not text:
        return ""
    return re.sub(r"\s+", " ", text).strip()


def clean_corrupted_value(value: str | None) -> str | None:
    if value is None:
        return None
    v = normalize_whitespace(str(value))
    if v.upper() in CORRUPT_TOKENS:
        return None

    if v.upper() == "11-JUL":
        return None
    if v.startswith("#"):
        return None
    return v


def normalize_company_seed(value: str | None) -> str | None:
    v = clean_corrupted_value(value)
    if not v:
        return None
    return normalize_whitespace(v)


def canonicalize_company_for_lookup(name: str | None) -> str | None:
    """
    Conservative normalization for API lookup only.
    Do not use this to merge entities yet.
    """
    if not name:
        return None

    s = name.upper()
    s = re.sub(r"[,\.;:()/\-]+", " ", s)
    parts = [p for p in s.split() if p not in LEGAL_SUFFIXES]
    s = " ".join(parts)
    s = normalize_whitespace(s)
    return s or None


def parse_date_to_mon_yy(value: str | None) -> str | None:
    """
    Normalize to format like Jan-23 when possible.
    """
    v = clean_corrupted_value(value)
    if not v:
        return None

    for fmt in ("%b-%y", "%b-%Y", "%Y-%m-%d", "%Y/%m/%d", "%m/%d/%Y"):
        try:
            dt = datetime.strptime(v, fmt)
            return dt.strftime("%b-%y")
        except ValueError:
            pass

    try:
        dt = dtparser.parse(v, fuzzy=True)
        return dt.strftime("%b-%y")
    except Exception:
        return None


def build_raw_text(*parts: str | None, max_len: int = 12000) -> str:
    text = " ".join(normalize_whitespace(p) for p in parts if p)
    text = normalize_whitespace(text)
    return text[:max_len]