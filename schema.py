"""Data models for Stage 1 (data collection)."""

from dataclasses import asdict, dataclass


@dataclass
class Stage1Record:
    source_id: int
    source_type: str
    raw_text: str
    source_url: str
    date: str
    company_seed: str

    def to_dict(self) -> dict:
        return asdict(self)
