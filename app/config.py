"""
Configuration management for the Orchestrator Agent.

Priority (highest → lowest):
  1. Process environment variables  (always win)
  2. The env file named in ENV_FILE env var  (e.g. ENV_FILE=.env.stub)
  3. .env.stub  (local stub testing — localhost agent URLs)
  4. .env.free  (Ollama / Docker-compose free mode)
  5. .env.paid  (Claude / production)
  6. .env       (generic fallback)
  7. Field defaults  (Docker service names as last resort)
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


def _active_env_files() -> tuple[str, ...]:
    """Return env-file(s) to load. If ENV_FILE is set, load only that file."""
    explicit = os.environ.get("ENV_FILE", "").strip()
    if explicit:
        return (explicit,)
    candidates = (".env.paid", ".env.free", ".env.stub", ".env")
    return tuple(f for f in candidates if os.path.isfile(f))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_active_env_files(),
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
    )

    # ── Service identity ──────────────────────────────────────────────────────
    app_name: str = "orchestrator-agent"
    port: int = 8000
    log_level: str = "INFO"

    # ── CORS ──────────────────────────────────────────────────────────────────
    cors_origins: str = "*"  # comma-separated list or "*"

    # ── LLM Provider ─────────────────────────────────────────────────────────
    llm_provider: Literal["ollama", "openai", "anthropic", "groq", "grok", "stub"] = "ollama"

    # Ollama (free mode)
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1:8b"

    # Groq
    groq_model: str = "llama-3.1-8b-instant"
    groq_api_key: str = ""

    # xAI / Grok
    xai_api_key: str = ""

    # OpenAI
    openai_api_key: str = ""

    # Claude (paid mode)
    anthropic_api_key: str = ""
    claude_model: str = "claude-3-5-sonnet-20241022"

    # ── LangSmith tracing ────────────────────────────────────────────────────
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""

    # ── Redis ─────────────────────────────────────────────────────────────────
    redis_url: str = "redis://redis:6379"
    session_ttl_seconds: int = 3600  # 1h

    # ── Deployment flags ─────────────────────────────────────────────────────
    use_docker: bool = True
    use_ec2: bool = False

    # ── Agent URLs (Docker service defaults) ─────────────────────────────────
    context_agent_url: str = "http://context-agent:8001"
    sql_agent_url: str = "http://sql-agent:8002"
    viz_agent_url: str = "http://viz-agent:8003"
    ml_agent_url: str = "http://ml-agent:8004"
    nlp_agent_url: str = "http://nlp-agent:8005"
    report_agent_url: str = "http://report-agent:8006"

    # EC2 private-IP overrides (only used when use_ec2=True)
    ec2_context_agent_ip: str = ""
    ec2_sql_agent_ip: str = ""
    ec2_viz_agent_ip: str = ""
    ec2_ml_agent_ip: str = ""
    ec2_nlp_agent_ip: str = ""
    ec2_report_agent_ip: str = ""

    # ── Per-agent HTTP timeout (seconds) ─────────────────────────────────────
    agent_timeout_seconds: float = 30.0

    # ── LLM planning ─────────────────────────────────────────────────────────
    planning_max_retries: int = 1
    planning_temperature: float = 0.2

    @property
    def cors_origins_list(self) -> list[str]:
        if self.cors_origins.strip() == "*":
            return ["*"]
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def agent_registry(self) -> dict[str, str]:
        """Return the effective agent URL map (Docker vs EC2)."""
        registry: dict[str, str] = {}
        agents = {
            "context": ("context_agent_url", "ec2_context_agent_ip", 8001),
            "sql":     ("sql_agent_url",     "ec2_sql_agent_ip",     8002),
            "viz":     ("viz_agent_url",     "ec2_viz_agent_ip",     8003),
            "ml":      ("ml_agent_url",      "ec2_ml_agent_ip",      8004),
            "nlp":     ("nlp_agent_url",     "ec2_nlp_agent_ip",     8005),
            "report":  ("report_agent_url",  "ec2_report_agent_ip",  8006),
        }
        for name, (docker_attr, ec2_attr, port) in agents.items():
            if self.use_ec2:
                ec2_ip = getattr(self, ec2_attr, "")
                if ec2_ip:
                    registry[name] = f"http://{ec2_ip}:{port}"
                    continue
            registry[name] = getattr(self, docker_attr)
        return registry


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def reload_settings() -> Settings:
    """Force-clear lru_cache and reload settings (useful in tests)."""
    get_settings.cache_clear()
    return get_settings()
