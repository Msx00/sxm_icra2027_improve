import os

from dotenv import load_dotenv

_ENV_LOADED = False

SENSITIVE_NAMES = ("HIAGENT_SK", "HIAGENT_AGENT_API_KEY")


def load_env():
    """Load .env once. Never prints or returns secret values."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    env_file = os.environ.get("HIAGENT_ENV_FILE") or _default_env_file()
    if env_file:
        load_dotenv(dotenv_path=env_file, override=False)
    _ENV_LOADED = True


def _default_env_file():
    here = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(here)
    candidate = os.path.join(project_root, ".env")
    if os.path.exists(candidate):
        return candidate
    return None


def getenv(name, default=None):
    load_env()
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip() if isinstance(value, str) else value


def mask(value, full=False):
    if not value:
        return "<unset>"
    s = str(value)
    if full or len(s) <= 8:
        return "***"
    return s[:4] + "***" + s[-4:]


def check_config():
    load_env()
    return {
        "platform": {
            "HIAGENT_AK": "configured" if getenv("HIAGENT_AK") else "missing",
            "HIAGENT_SK": "configured" if getenv("HIAGENT_SK") else "missing",
            "HIAGENT_TOP_HOST": "configured" if getenv("HIAGENT_TOP_HOST") else "missing",
        },
        "agent": {
            "HIAGENT_AGENT_API_KEY": "configured" if getenv("HIAGENT_AGENT_API_KEY") else "missing",
            "HIAGENT_AGENT_URL": "configured" if getenv("HIAGENT_AGENT_URL") else "missing",
            "HIAGENT_AGENT_AK": "configured" if getenv("HIAGENT_AGENT_AK") else "missing",
            "HIAGENT_AGENT_SK": "configured" if getenv("HIAGENT_AGENT_SK") else "missing",
            "HIAGENT_AGENT_APP_ID": "configured" if getenv("HIAGENT_AGENT_APP_ID") else "missing",
        },
    }
