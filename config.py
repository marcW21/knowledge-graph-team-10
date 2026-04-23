"""Runtime settings for Stage 1 data collection.

All flags are read from environment variables (or a .env file via python-dotenv).
The *_enabled properties are the canonical way to check whether a source is active;
they ensure both the feature flag AND any required API key are present.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    return v.strip().lower() in {"1", "true", "yes", "y", "on"} if v else default


@dataclass
class Settings:
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    max_records_per_source: int = int(os.getenv("MAX_RECORDS_PER_SOURCE", "100"))

    # NCBI / PubMed
    ncbi_tool: str = os.getenv("NCBI_TOOL", "knowledge-graph-stage1")
    ncbi_email: str = os.getenv("NCBI_EMAIL", "")
    ncbi_api_key: str | None = os.getenv("NCBI_API_KEY")
    ncbi_enabled: bool = _env_bool("USE_NCBI", True)

    # SEC (public EDGAR)
    sec_user_agent: str = os.getenv("SEC_USER_AGENT", "knowledge-graph-stage1 xh426@cornell.edu")
    sec_enabled: bool = _env_bool("USE_SEC", True)

    # SEC full-text extractor (paid API)
    sec_api_key: str | None = os.getenv("SEC_API_KEY")
    sec_extractor_enabled: bool = _env_bool("USE_SEC_EXTRACTOR", False)

    # OpenCorporates
    opencorporates_api_key: str | None = os.getenv("OPENCORPORATES_API_KEY")
    opencorporates_enabled: bool = _env_bool("USE_OPENCORPORATES", False)

    # USPTO
    uspto_api_key: str | None = os.getenv("USPTO_API_KEY")
    uspto_enabled: bool = _env_bool("USE_USPTO", False)

    def __post_init__(self) -> None:
        # Require API key for sources that need one.
        if self.opencorporates_enabled and not self.opencorporates_api_key:
            self.opencorporates_enabled = False
        if self.uspto_enabled and not self.uspto_api_key:
            self.uspto_enabled = False
        if self.sec_extractor_enabled and not self.sec_api_key:
            self.sec_extractor_enabled = False
