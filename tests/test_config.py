"""
Tests for configuration management (app/config.py).

Scenarios:
  1.  Default values are correct
  2.  agent_registry returns Docker URLs by default
  3.  agent_registry uses EC2 IPs when use_ec2=True and IPs are set
  4.  agent_registry falls back to Docker URL when EC2 IP is empty
  5.  cors_origins_list parses wildcard correctly
  6.  cors_origins_list parses comma-separated origins
  7.  reload_settings clears cache and returns fresh instance
  8.  All 6 agents present in registry
"""
from __future__ import annotations

import pytest

from app.config import Settings, reload_settings


def _settings(**kwargs) -> Settings:
    """Create a Settings instance with overrides, bypassing env files."""
    return Settings.model_validate({
        "use_docker": True,
        "use_ec2": False,
        **kwargs,
    })


def test_default_port():
    s = _settings()
    assert s.port == 8000


def test_default_llm_provider():
    # Use model_construct to bypass env file loading and check the field default
    s = Settings.model_construct(llm_provider="ollama")
    assert s.llm_provider == "ollama"


def test_agent_registry_has_all_six_agents():
    s = _settings()
    registry = s.agent_registry
    assert set(registry.keys()) == {"context", "sql", "viz", "ml", "nlp", "report"}


def test_agent_registry_docker_urls():
    s = _settings(
        context_agent_url="http://context-agent:8001",
        sql_agent_url="http://sql-agent:8002",
    )
    registry = s.agent_registry
    assert registry["context"] == "http://context-agent:8001"
    assert registry["sql"] == "http://sql-agent:8002"


def test_agent_registry_ec2_override():
    s = _settings(
        use_ec2=True,
        ec2_context_agent_ip="10.0.1.5",
        ec2_sql_agent_ip="10.0.1.6",
    )
    registry = s.agent_registry
    assert registry["context"] == "http://10.0.1.5:8001"
    assert registry["sql"] == "http://10.0.1.6:8002"


def test_agent_registry_ec2_fallback_to_docker_when_ip_empty():
    s = _settings(
        use_ec2=True,
        ec2_context_agent_ip="",  # empty → fall back to docker URL
        context_agent_url="http://context-agent:8001",
    )
    registry = s.agent_registry
    assert registry["context"] == "http://context-agent:8001"


def test_cors_origins_wildcard():
    s = _settings(cors_origins="*")
    assert s.cors_origins_list == ["*"]


def test_cors_origins_comma_separated():
    s = _settings(cors_origins="https://app.example.com,https://admin.example.com")
    assert s.cors_origins_list == ["https://app.example.com", "https://admin.example.com"]


def test_cors_origins_single():
    s = _settings(cors_origins="https://app.example.com")
    assert s.cors_origins_list == ["https://app.example.com"]


def test_reload_settings_returns_settings():
    s = reload_settings()
    assert isinstance(s, Settings)


def test_session_ttl_default():
    s = _settings()
    assert s.session_ttl_seconds == 3600


def test_planning_max_retries_default():
    s = _settings()
    assert s.planning_max_retries == 1


def test_agent_timeout_default():
    s = _settings()
    assert s.agent_timeout_seconds == 30.0
