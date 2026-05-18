"""加载后端环境配置和共享设置。"""

from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:  # pragma: no cover
    def load_dotenv(*_args, **_kwargs) -> bool:
        return False

# 获取当前文件所在目录下的 .env 文件路径
_ENV_PATH = Path(__file__).resolve().parent / ".env"
# 加载环境变量
load_dotenv(_ENV_PATH)


def get_gemini_api_key() -> str:
    """
    从环境变量或 .env 文件中获取 Gemini API 密钥。
    优先尝试 GEMINI_API_KEY，其次尝试 GOOGLE_API_KEY。
    """
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key or not key.strip():
        raise RuntimeError(
            "缺少 API 密钥：请在环境变量中设置 GEMINI_API_KEY 或 GOOGLE_API_KEY，"
            "或者在 backend/.env 文件中配置（参考 backend/.env.example）。"
        )
    return key.strip()


def get_deepseek_api_key() -> str:
    """
    从环境变量或 .env 文件中获取 DeepSeek API 密钥。
    """
    key = os.environ.get("DEEPSEEK_API_KEY")
    if not key or not key.strip():
        raise RuntimeError(
            "缺少 DeepSeek API 密钥：请在环境变量中设置 DEEPSEEK_API_KEY，"
            "或者在 backend/.env 文件中配置。"
        )
    return key.strip()


def get_gemini_model() -> str:
    """
    从环境变量获取 Gemini 模型名称，默认为 'gemini-2.5-flash'。
    """
    return os.environ.get("GEMINI_MODEL", "gemini-2.5-flash").strip()


def get_llm_provider() -> str:
    """
    从环境变量获取当前的 LLM 提供商。
    """
    return os.environ.get("LLM_PROVIDER", "gemini").lower().strip()


def _get_float_env(name: str, default: float) -> float:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_int_env(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def get_llm_timeout_seconds() -> float:
    return max(1.0, _get_float_env("LLM_TIMEOUT_SECONDS", 30.0))


def get_llm_max_retries() -> int:
    return max(0, _get_int_env("LLM_MAX_RETRIES", 3))


def get_llm_retry_base_delay() -> float:
    return max(0.05, _get_float_env("LLM_RETRY_BASE_DELAY", 0.8))


def get_llm_retry_max_delay() -> float:
    return max(0.1, _get_float_env("LLM_RETRY_MAX_DELAY", 8.0))


def get_llm_concurrency() -> int:
    return max(1, _get_int_env("LLM_MAX_CONCURRENCY", 8))


def get_llm_rate_limit_rps() -> float:
    return max(0.1, _get_float_env("LLM_RATE_LIMIT_RPS", 5.0))


def get_llm_max_input_chars() -> int:
    return max(500, _get_int_env("LLM_MAX_INPUT_CHARS", 6000))


def get_llm_log_path() -> Path:
    p = os.environ.get("LLM_LOG_PATH", "").strip()
    if p:
        return Path(p)
    return Path(__file__).resolve().parent / "logs" / "llm_calls.jsonl"


def get_app_db_path() -> Path:
    p = os.environ.get("APP_DB_PATH", "").strip()
    if p:
        return Path(p)
    return Path(__file__).resolve().parent / "data" / "app_state.json"


def get_app_access_password() -> str:
    return os.environ.get("APP_ACCESS_PASSWORD", "").strip()


def get_app_admin_password() -> str:
    return os.environ.get("APP_ADMIN_PASSWORD", "").strip()
