from dataclasses import dataclass
import os


def env_bool(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass
class Settings:
    request_timeout: int = int(os.getenv("REQUEST_TIMEOUT", "30"))
    max_records_per_source: int = int(os.getenv("MAX_RECORDS_PER_SOURCE", "100"))

    # NCBI
    ncbi_tool: str = os.getenv("NCBI_TOOL", "knowledge-graph-stage1")
    ncbi_email: str = os.getenv("NCBI_EMAIL", "xh426@cornell.edu")
    ncbi_api_key: str | None = os.getenv("NCBI_API_KEY")
    use_ncbi: bool = env_bool("USE_NCBI", True)

    # SEC public
    sec_user_agent: str = os.getenv(
        "SEC_USER_AGENT",
        "knowledge-graph-stage1 xh426@cornell.edu"
    )
    use_sec: bool = env_bool("USE_SEC", True)

    # SEC extractor
    sec_api_key: str | None = os.getenv("SEC_API_KEY")
    use_sec_extractor: bool = env_bool("USE_SEC_EXTRACTOR", True)

    # OpenCorporates
    opencorporates_api_key: str | None = os.getenv("OPENCORPORATES_API_KEY")
    use_opencorporates: bool = env_bool("USE_OPENCORPORATES", False)

    # USPTO
    uspto_api_key: str | None = os.getenv("USPTO_API_KEY")
    use_uspto: bool = env_bool("USE_USPTO", False)

    @property
    def ncbi_enabled(self) -> bool:
        return self.use_ncbi

    @property
    def sec_enabled(self) -> bool:
        return self.use_sec

    @property
    def sec_extractor_enabled(self) -> bool:
        return self.use_sec and self.use_sec_extractor and bool(self.sec_api_key)

    @property
    def opencorporates_enabled(self) -> bool:
        return self.use_opencorporates and bool(self.opencorporates_api_key)

    @property
    def uspto_enabled(self) -> bool:
        return self.use_uspto and bool(self.uspto_api_key)