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

Why this matters: pydantic-settings loads env files in order and later files
override earlier ones if the SAME key appears in multiple files. By detecting
the ENV_FILE variable and loading only that single file, we avoid Docker
hostnames silently overriding localhost URLs set in the shell.
"""
from __future__ import annotations

import os
from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


def _active_env_files() -> tuple[str, ...]:
    """
    Return the env-file(s) to load, in priority order (last wins in
    pydantic-settings when the same key appears in multiple files).

    If ENV_FILE is set, ONLY that file is loaded — shell env vars still win
    because pydantic-settings always prefers the process environment over
    any file.
    """
    explicit = os.environ.get("ENV_FILE", "").strip()
    if explicit:
        return (explicit,)

    # Auto-detect: load all known files; pydantic-settings merges them and
    # shell env vars always take top priority.
    candidates = (".env.paid", ".env.free", ".env.stub", ".env")
    return tuple(f for f in candidates if os.path.isfile(f))


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=_active_env_files(),
        env_file_encoding="utf-8",
        env_ignore_empty=True,   # empty string in file ≠ override
        extra="ignore",
    )

    # ── Service identity ──────────────────────────────────────────────────────
    app_name: str = "orchestrator-agent"
    port: int = 8000
    log_level: str = "INFO"

    # ── LLM Provider ─────────────────────────────────────────────────────────
    llm_provider: Literal["ollama", "claude", "stub"] = "ollama"

    # Ollama (free mode)
    ollama_base_url: str = "http://ollama:11434"
    ollama_model: str = "qwen3:0.6b"

    # Claude (paid mode)
    anthropic_api_key: str = ""
    claude_model: str = "claude-sonnet-4-20250514"

    # ── LangSmith tracing (paid mode) ────────────────────────────────────────
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
    planning_max_retries: int = 1  # 1 retry on invalid JSON
    planning_temperature: float = 0.2

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
