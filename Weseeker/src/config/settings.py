from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict, YamlConfigSettingsSource

ROOT_DIR = Path(__file__).resolve().parents[2]


def _read_dotenv_value(key: str) -> str:
    env_file = ROOT_DIR / ".env"
    if not env_file.exists():
        return ""

    for raw_line in env_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        current_key, value = line.split("=", 1)
        if current_key.strip() == key:
            return value.strip().strip('"').strip("'")
    return ""


class LLMSettings(BaseSettings):
    provider: str = "dashscope"
    model: str = "qwen3.5-plus"
    is_multimodal: bool = False
    api_key: str = ""
    api_base: str = "https://api-inference.modelscope.cn/v1"
    temperature: float = 0.3
    timeout: int = 60
    enable_thinking: str | None = None


class EverythingSettings(BaseSettings):
    host: str = "127.0.0.1"
    port: int = 8080


class MCPSettings(BaseSettings):
    file_tools_port: int = 9100
    rag_tools_port: int = 9200


class PreviewSettings(BaseSettings):
    text_depth_chars: dict[str, int] = {"L1": 2000, "L2": 5000, "L3": 8000}
    excel_depth_rows: dict[str, int] = {"L1": 10, "L2": 50, "L3": 100}
    pdf_depth_pages: dict[str, int] = {"L1": 1, "L2": 2, "L3": 3}
    pdf_render_scale: float = 3.0
    image_max_edge: int = 1600


class PathsSettings(BaseSettings):
    desktop: str = ""
    downloads: str = ""
    documents: str = ""


class StorageSettings(BaseSettings):
    sqlite_path: str = "storage/weseeker.db"


class KBConfig(BaseModel):
    name: str
    root: str
    include_ext: list[str] = Field(
        default_factory=lambda: [".md", ".docx", ".pdf", ".xlsx", ".txt"]
    )
    exclude_glob: list[str] = Field(
        default_factory=lambda: [
            "**/.git/**",
            "**/node_modules/**",
            "**/__pycache__/**",
            "**/dist/**",
            "**/.vite/**",
            "**/.venv/**",
            "**/.idea/**",
        ]
    )
    description: str = ""


class RAGSettings(BaseSettings):
    enabled: bool = False
    chroma_persist_dir: str = "storage/chroma"
    docstore_dir: str = "storage/docstore"
    manifest_dir: str = "storage/manifest"
    embedding_provider: str = "lmstudio"
    embedding_model: str = ""
    embedding_dimension: int = 1024
    embedding_batch_size: int = 8
    embedding_max_retries: int = 3
    lmstudio_embedding_base_url: str = "http://127.0.0.1:1234/v1"
    parent_chunk_size: int = 2000
    parent_chunk_overlap: int = 200
    child_chunk_size: int = 400
    child_chunk_overlap: int = 60
    semantic_breakpoint_percentile: int = 95
    semantic_min_chunk_chars: int = 100
    semantic_max_chunk_chars: int = 800
    max_file_size_mb: int = 50
    knowledge_bases: list[KBConfig] = Field(default_factory=list)
    chunk_size: int = 900
    chunk_overlap: int = 150
    top_k: int = 30


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        env_prefix="WESEEKER_",
        env_nested_delimiter="__",
        extra="ignore",
    )

    debug: bool = False
    llm: LLMSettings = Field(default_factory=LLMSettings)
    everything: EverythingSettings = Field(default_factory=EverythingSettings)
    mcp: MCPSettings = Field(default_factory=MCPSettings)
    preview: PreviewSettings = Field(default_factory=PreviewSettings)
    paths: PathsSettings = Field(default_factory=PathsSettings)
    storage: StorageSettings = Field(default_factory=StorageSettings)
    rag: RAGSettings = Field(default_factory=RAGSettings)
    sender_target: str = "文件传输助手"

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls,
        init_settings,
        env_settings,
        dotenv_settings,
        file_secret_settings,
    ):
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            YamlConfigSettingsSource(
                settings_cls, yaml_file=ROOT_DIR / "settings.yaml", yaml_file_encoding="utf-8"
            ),
            file_secret_settings,
        )

    def model_post_init(self, __context) -> None:
        if not self.llm.api_key:
            # 根据 provider 自动拼 key 名：provider="deepseek" → DEEPSEEK_API_KEY
            env_key = f"{self.llm.provider.upper()}_API_KEY"
            self.llm.api_key = os.getenv(env_key, "") or _read_dotenv_value(env_key)


@lru_cache()
def get_settings() -> AppSettings:
    return AppSettings()
