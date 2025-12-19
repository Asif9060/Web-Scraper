"""
Pydantic settings models for the Web Intelligence System.

All configuration is defined here with sensible defaults optimized for
CPU-only, 8GB RAM environments.
"""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class BrowserSettings(BaseModel):
    """Playwright browser configuration."""

    headless: bool = Field(
        default=True,
        description="Run browser in headless mode",
    )
    timeout_ms: int = Field(
        default=30000,
        ge=5000,
        le=120000,
        description="Default timeout for page operations in milliseconds",
    )
    navigation_timeout_ms: int = Field(
        default=60000,
        ge=10000,
        le=180000,
        description="Timeout for page navigation in milliseconds",
    )
    user_agent: str | None = Field(
        default=None,
        description="Custom user agent string. None uses browser default.",
    )
    viewport_width: int = Field(
        default=1280,
        ge=320,
        le=3840,
        description="Browser viewport width in pixels",
    )
    viewport_height: int = Field(
        default=720,
        ge=240,
        le=2160,
        description="Browser viewport height in pixels",
    )
    ignore_https_errors: bool = Field(
        default=False,
        description="Whether to ignore HTTPS certificate errors",
    )
    browser_type: Literal["chromium", "firefox", "webkit"] = Field(
        default="chromium",
        description="Browser engine to use",
    )


class CrawlerSettings(BaseModel):
    """Web crawler configuration."""

    max_pages: int = Field(
        default=500,
        ge=1,
        le=10000,
        description="Maximum number of pages to crawl per website",
    )
    delay_seconds: float = Field(
        default=1.0,
        ge=0.1,
        le=60.0,
        description="Delay between page requests in seconds",
    )
    concurrent_pages: int = Field(
        default=1,
        ge=1,
        le=5,
        description="Number of pages to process concurrently",
    )
    max_depth: int = Field(
        default=10,
        ge=1,
        le=50,
        description="Maximum crawl depth from starting URL",
    )
    respect_robots_txt: bool = Field(
        default=True,
        description="Whether to respect robots.txt directives",
    )
    follow_external_links: bool = Field(
        default=False,
        description="Whether to follow links to external domains",
    )
    url_patterns_include: list[str] = Field(
        default_factory=list,
        description="URL patterns to include (regex). Empty means include all.",
    )
    url_patterns_exclude: list[str] = Field(
        default_factory=lambda: [
            r".*\.(pdf|zip|tar|gz|exe|dmg|pkg|deb|rpm)$",
            r".*\.(jpg|jpeg|png|gif|svg|ico|webp|bmp)$",
            r".*\.(mp3|mp4|avi|mov|wmv|flv|webm)$",
            r".*\?.*utm_",
            r".*/login.*",
            r".*/logout.*",
            r".*/signin.*",
            r".*/signout.*",
        ],
        description="URL patterns to exclude (regex)",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for failed requests",
    )
    retry_delay_seconds: float = Field(
        default=5.0,
        ge=1.0,
        le=60.0,
        description="Delay before retrying failed requests",
    )


class StorageSettings(BaseModel):
    """SQLite storage configuration."""

    database_path: Path = Field(
        default=Path("data/knowledge.db"),
        description="Path to SQLite database file",
    )
    vector_dimensions: int = Field(
        default=384,
        ge=64,
        le=1536,
        description="Dimension of embedding vectors (must match embedding model)",
    )
    max_vectors_in_memory: int = Field(
        default=50000,
        ge=1000,
        le=500000,
        description="Maximum vectors to keep in memory. Older vectors are evicted.",
    )
    vector_eviction_batch_size: int = Field(
        default=5000,
        ge=100,
        le=50000,
        description="Number of vectors to evict when memory limit is reached.",
    )
    wal_mode: bool = Field(
        default=True,
        description="Enable WAL mode for better concurrent access",
    )
    cache_size_mb: int = Field(
        default=64,
        ge=8,
        le=512,
        description="SQLite cache size in megabytes",
    )

    @field_validator("database_path", mode="before")
    @classmethod
    def convert_to_path(cls, v: str | Path) -> Path:
        """Convert string paths to Path objects."""
        return Path(v) if isinstance(v, str) else v


class LocalLLMSettings(BaseModel):
    """Local LLM (HuggingFace Transformers) configuration."""

    enabled: bool = Field(
        default=True,
        description="Whether to use local LLM for extraction",
    )
    model_name: str = Field(
        default="Qwen/Qwen2.5-1.5B-Instruct",
        description="HuggingFace model identifier",
    )
    device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu",
        description="Device to run inference on",
    )
    torch_dtype: Literal["float32", "float16", "bfloat16"] = Field(
        default="float32",
        description="PyTorch dtype for model weights",
    )
    max_new_tokens: int = Field(
        default=512,
        ge=64,
        le=2048,
        description="Maximum tokens to generate per inference",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature (lower = more deterministic)",
    )
    top_p: float = Field(
        default=0.9,
        ge=0.0,
        le=1.0,
        description="Nucleus sampling probability",
    )
    do_sample: bool = Field(
        default=False,
        description="Whether to use sampling (False = greedy decoding)",
    )
    low_cpu_mem_usage: bool = Field(
        default=True,
        description="Load model with reduced CPU memory footprint",
    )
    trust_remote_code: bool = Field(
        default=False,
        description="Whether to trust remote code in model repository",
    )
    cache_dir: Path | None = Field(
        default=None,
        description="Directory to cache downloaded models",
    )

    @field_validator("cache_dir", mode="before")
    @classmethod
    def convert_cache_dir(cls, v: str | Path | None) -> Path | None:
        """Convert string paths to Path objects."""
        if v is None:
            return None
        return Path(v) if isinstance(v, str) else v


class APILLMSettings(BaseModel):
    """API LLM (Anthropic Claude) configuration."""

    enabled: bool = Field(
        default=True,
        description="Whether to use API LLM for query interpretation and answers",
    )
    provider: Literal["anthropic"] = Field(
        default="anthropic",
        description="API provider to use",
    )
    model_name: str = Field(
        default="claude-sonnet-4-20250514",
        description="Model identifier for API calls",
    )
    api_key_env_var: str = Field(
        default="ANTHROPIC_API_KEY",
        description="Environment variable name containing API key",
    )
    max_tokens: int = Field(
        default=1024,
        ge=100,
        le=4096,
        description="Maximum tokens in API response",
    )
    temperature: float = Field(
        default=0.2,
        ge=0.0,
        le=1.0,
        description="Sampling temperature for API calls",
    )
    timeout_seconds: int = Field(
        default=60,
        ge=10,
        le=300,
        description="Timeout for API requests in seconds",
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Maximum retry attempts for API failures",
    )
    retry_delay_seconds: float = Field(
        default=1.0,
        ge=0.5,
        le=30.0,
        description="Initial delay before retrying API calls",
    )


class EmbeddingSettings(BaseModel):
    """Sentence embeddings configuration."""

    model_name: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace sentence-transformer model identifier",
    )
    device: Literal["cpu", "cuda", "mps"] = Field(
        default="cpu",
        description="Device to run embedding model on",
    )
    batch_size: int = Field(
        default=32,
        ge=1,
        le=256,
        description="Batch size for embedding generation",
    )
    normalize_embeddings: bool = Field(
        default=True,
        description="Whether to L2-normalize embeddings",
    )
    show_progress_bar: bool = Field(
        default=False,
        description="Show progress bar during embedding generation",
    )
    cache_dir: Path | None = Field(
        default=None,
        description="Directory to cache downloaded models",
    )

    @field_validator("cache_dir", mode="before")
    @classmethod
    def convert_cache_dir(cls, v: str | Path | None) -> Path | None:
        """Convert string paths to Path objects."""
        if v is None:
            return None
        return Path(v) if isinstance(v, str) else v


class NavigationSettings(BaseModel):
    """Navigation agent configuration with tiered decision system."""

    use_llm_for_navigation: bool = Field(
        default=False,
        description="Whether to use LLM for navigation decisions. Disabled by default due to CPU cost.",
    )
    use_embeddings_for_ranking: bool = Field(
        default=True,
        description="Whether to use embedding similarity as Tier-2 ranking (fast).",
    )
    llm_confidence_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Only use LLM if heuristic confidence is below this threshold.",
    )
    keyword_weight: float = Field(
        default=0.4,
        ge=0.0,
        le=1.0,
        description="Weight for keyword overlap in heuristic scoring.",
    )
    url_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for URL pattern heuristics.",
    )
    embedding_weight: float = Field(
        default=0.3,
        ge=0.0,
        le=1.0,
        description="Weight for embedding similarity in scoring.",
    )
    max_links_to_evaluate: int = Field(
        default=20,
        ge=5,
        le=100,
        description="Maximum links to evaluate per page.",
    )


class LoggingSettings(BaseModel):
    """Logging configuration."""

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = Field(
        default="INFO",
        description="Minimum logging level",
    )
    format: str = Field(
        default="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        description="Log message format string",
    )
    date_format: str = Field(
        default="%Y-%m-%d %H:%M:%S",
        description="Date format for log timestamps",
    )
    file_path: Path | None = Field(
        default=None,
        description="Path to log file. None means console only.",
    )
    max_file_size_mb: int = Field(
        default=10,
        ge=1,
        le=100,
        description="Maximum log file size before rotation",
    )
    backup_count: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Number of backup log files to keep",
    )
    log_to_console: bool = Field(
        default=True,
        description="Whether to output logs to console",
    )

    @field_validator("file_path", mode="before")
    @classmethod
    def convert_file_path(cls, v: str | Path | None) -> Path | None:
        """Convert string paths to Path objects."""
        if v is None:
            return None
        return Path(v) if isinstance(v, str) else v


class Settings(BaseModel):
    """
    Root configuration model containing all subsystem settings.

    Settings are loaded from YAML with environment variable overrides.
    Optimized defaults for CPU-only, 8GB RAM environments.
    """

    browser: BrowserSettings = Field(
        default_factory=BrowserSettings,
        description="Browser/Playwright settings",
    )
    crawler: CrawlerSettings = Field(
        default_factory=CrawlerSettings,
        description="Web crawler settings",
    )
    storage: StorageSettings = Field(
        default_factory=StorageSettings,
        description="Database storage settings",
    )
    local_llm: LocalLLMSettings = Field(
        default_factory=LocalLLMSettings,
        description="Local LLM settings",
    )
    api_llm: APILLMSettings = Field(
        default_factory=APILLMSettings,
        description="API LLM settings",
    )
    embedding: EmbeddingSettings = Field(
        default_factory=EmbeddingSettings,
        description="Embedding model settings",
    )
    navigation: NavigationSettings = Field(
        default_factory=NavigationSettings,
        description="Navigation agent settings",
    )
    logging: LoggingSettings = Field(
        default_factory=LoggingSettings,
        description="Logging configuration",
    )

    model_config = {
        "extra": "forbid",
        "validate_default": True,
    }
