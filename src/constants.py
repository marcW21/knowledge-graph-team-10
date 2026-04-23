"""Shared constants for the KG pipeline.

All modules import from here so definitions stay in one place.
"""

# Legal suffixes that are "generic" (safely stripped for base-name comparison).
GENERIC_SUFFIXES: frozenset[str] = frozenset({
    "INC", "INCORPORATED", "CORP", "CORPORATION", "CO", "COMPANY",
})

# Legal suffixes that carry jurisdictional / structural meaning (keep for merge decisions).
REGIONAL_SUFFIXES: frozenset[str] = frozenset({
    "LLC", "LTD", "LIMITED", "PLC", "LP", "LLP",
    "BV", "AB", "GMBH", "AG", "SA", "SAS", "NV", "SPA", "SARL", "KK",
})

ALL_LEGAL_SUFFIXES: frozenset[str] = GENERIC_SUFFIXES | REGIONAL_SUFFIXES

# Values that should never be treated as real company names.
CORRUPT_TOKENS: frozenset[str] = frozenset({
    "#NAME?", "#VALUE!", "#REF!", "#DIV/0!", "#NUM!", "#N/A", "#NULL!",
    "N/A", "NA", "NULL", "NONE", "TBD", "--", "UNKNOWN", "",
})

MONTH_ABBREVS: frozenset[str] = frozenset({
    "JAN", "FEB", "MAR", "APR", "MAY", "JUN",
    "JUL", "AUG", "SEP", "SEPT", "OCT", "NOV", "DEC",
})

VALID_SOURCE_TYPES: frozenset[str] = frozenset({
    "SEC", "USPTO", "PUBMED", "NCBI", "OPENCORPORATES",
})

SOURCE_TYPE_ALIASES: dict[str, str] = {
    "NCBI E-UTILITIES": "PUBMED",
    "NCBI": "PUBMED",
    "PATENTSVIEW": "USPTO",
    "EDGAR": "SEC",
}
