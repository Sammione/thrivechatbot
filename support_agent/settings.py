"""Agent settings (env-overridable)."""
import os
import re
from dataclasses import dataclass, field

import yaml


def _e(name, default, cast=str):
    return lambda: cast(os.getenv(name, default))


def _default_model():
    if os.getenv("AGENT_MODEL"):
        return os.getenv("AGENT_MODEL")
    provider = os.getenv("AGENT_PROVIDER", "auto")
    if provider == "groq" or os.getenv("GROQ_API_KEY"):
        return "llama-3.3-70b-versatile"
    if provider == "deepseek" or os.getenv("DEEPSEEK_API_KEY"):
        return "deepseek-chat"
    return "claude-3-5-sonnet-latest"


@dataclass
class AgentSettings:
    shop_name: str = field(default_factory=_e("SHOP_NAME", "our store"))
    provider: str = field(default_factory=_e("AGENT_PROVIDER", "auto"))        # auto | groq | anthropic | deepseek | openai
    model: str = field(default_factory=_default_model)
    groq_api_key: str = field(default_factory=_e("GROQ_API_KEY", ""))
    openai_base_url: str = field(default_factory=_e("OPENAI_BASE_URL", ""))
    max_tokens: int = field(default_factory=_e("AGENT_MAX_TOKENS", 1024, int))
    max_steps: int = field(default_factory=_e("AGENT_MAX_STEPS", 8, int))      # tool-call rounds per message
    tool_mode: str = field(default_factory=_e("AGENT_TOOL_MODE", "auto"))       # auto | direct | catalog
    max_direct_tools: int = field(default_factory=_e("AGENT_MAX_DIRECT_TOOLS", 40, int))
    roles: tuple = field(default_factory=lambda: tuple(os.getenv("AGENT_ROLES", "buyer,vendor,admin").split(",")))
    api_key: str = field(default_factory=_e("AGENT_API_KEY", ""))   # shared secret with your backend
    services_file: str = field(default_factory=_e("AGENT_SERVICES", "config/services.yaml"))
    policy_file: str = field(default_factory=_e("AGENT_POLICY", "config/policy.yaml"))
    knowledge_dir: str = field(default_factory=_e("AGENT_KNOWLEDGE", "knowledge"))
    data_dir: str = field(default_factory=_e("AGENT_DATA_DIR", "data/agent"))
    max_history: int = field(default_factory=_e("AGENT_MAX_HISTORY", 40, int))


_VAR = re.compile(r"\$\{(\w+)(?::-([^}]*))?\}")


def load_yaml(path):
    """YAML with ${VAR} and ${VAR:-default} substitution."""
    with open(path, encoding="utf-8") as f:
        text = _VAR.sub(lambda m: os.getenv(m.group(1), m.group(2) or ""), f.read())
    return yaml.safe_load(text) or {}
