"""HiAgent Python API client.

- Platform API: HiAgentPlatformClient (volcengine SignerV4, AK/SK auth)
- Agent API:    HiAgentAgentClient   (Apikey auth)
"""

from hiagent.agent_client import (
    HiAgentAgentAPIError,
    HiAgentAgentClient,
    HiAgentAgentConfigError,
    HiAgentAgentError,
    HiAgentAgentHTTPError,
    HiAgentAgentJSONError,
    HiAgentAgentNetworkError,
    HiAgentAgentSSEError,
)
from hiagent.config import check_config, getenv, mask
from hiagent.platform_client import (
    HiAgentAPIError,
    HiAgentConfigError,
    HiAgentError,
    HiAgentHTTPError,
    HiAgentNetworkError,
    HiAgentPlatformClient,
)

__version__ = "0.1.0"

__all__ = [
    "HiAgentPlatformClient",
    "HiAgentAgentClient",
    "HiAgentError",
    "HiAgentConfigError",
    "HiAgentNetworkError",
    "HiAgentHTTPError",
    "HiAgentAPIError",
    "HiAgentAgentError",
    "HiAgentAgentConfigError",
    "HiAgentAgentNetworkError",
    "HiAgentAgentHTTPError",
    "HiAgentAgentAPIError",
    "HiAgentAgentJSONError",
    "HiAgentAgentSSEError",
    "check_config",
    "getenv",
    "mask",
]
