import json
from collections import OrderedDict
from urllib.parse import urlparse

import requests
from volcengine.auth.SignParam import SignParam
from volcengine.auth.SignerV4 import SignerV4
from volcengine.Credentials import Credentials

from hiagent.config import getenv

DEFAULT_REGION = "cn-north-1"
DEFAULT_SERVICE = "app"
DEFAULT_VERSION = "2023-08-01"
DEFAULT_TIMEOUT = 30

_SUCCESS_CODES = {None, "", "Success", "success", "0", 0, "200", 200}


class HiAgentError(Exception):
    """Base class for all hiagent client errors."""


class HiAgentConfigError(HiAgentError):
    """Missing or invalid configuration."""


class HiAgentNetworkError(HiAgentError):
    """Connection / TLS / timeout / DNS level failure."""


class HiAgentHTTPError(HiAgentError):
    def __init__(self, status_code, body=None):
        self.status_code = status_code
        self.body = body
        super().__init__(f"HTTP {status_code}: {_safe_body(body)}")


class HiAgentAPIError(HiAgentError):
    def __init__(self, code=None, message=None, status_code=None, body=None):
        self.code = code
        self.message = message
        self.status_code = status_code
        self.body = body
        super().__init__(f"HiAgent API error: code={code!r}, message={message!r}")


def _safe_body(body):
    text = body if isinstance(body, str) else json.dumps(body, ensure_ascii=False)
    if len(text) > 2000:
        text = text[:2000] + "...(truncated)"
    return text


def normalize_top_host(top_host):
    """Return (scheme, netloc) for a TopHost like 'x.x.x.x:30040' or 'http://x.x.x.x:30040'.

    Never produces 'http://http://...'.
    """
    if not top_host:
        return None, None
    host = str(top_host).strip().rstrip("/")
    if not host:
        return None, None
    if "://" not in host:
        host = "http://" + host
    parsed = urlparse(host)
    scheme = (parsed.scheme or "http").lower()
    netloc = parsed.netloc or parsed.hostname
    if scheme not in ("http", "https"):
        raise HiAgentConfigError(f"Unsupported TopHost scheme: {top_host!r}")
    if not netloc:
        raise HiAgentConfigError(f"Invalid TopHost: {top_host!r}")
    return scheme, netloc


class HiAgentPlatformClient:
    """Client for the HiAgent Platform API (volcengine SignerV4 signature)."""

    def __init__(
        self,
        ak=None,
        sk=None,
        top_host=None,
        region=None,
        service=None,
        version=None,
        timeout=None,
    ):
        self.ak = ak or getenv("HIAGENT_AK")
        self.sk = sk or getenv("HIAGENT_SK")
        self.top_host = top_host or getenv("HIAGENT_TOP_HOST")
        self.region = region or getenv("HIAGENT_REGION", DEFAULT_REGION)
        self.service = service or getenv("HIAGENT_SERVICE", DEFAULT_SERVICE)
        self.version = version or getenv("HIAGENT_VERSION", DEFAULT_VERSION)
        self.timeout = timeout or int(getenv("HIAGENT_TIMEOUT", DEFAULT_TIMEOUT))

    def _check_config(self):
        if not self.ak:
            raise HiAgentConfigError("Missing HIAGENT_AK. Set it in the environment or .env.")
        if not self.sk:
            raise HiAgentConfigError("Missing HIAGENT_SK. Set it in the environment or .env.")
        scheme, netloc = normalize_top_host(self.top_host)
        if not netloc:
            raise HiAgentConfigError(
                "Missing HIAGENT_TOP_HOST.\n"
                "Please obtain the externally accessible HiAgent TopHost from the administrator."
            )
        return scheme, netloc

    def request(self, action, body=None):
        """Call a Platform API action and return the parsed JSON response as a dict."""
        if not action:
            raise HiAgentConfigError("action must be provided, e.g. 'ListAppCenter'")
        body = body or {}
        scheme, netloc = self._check_config()

        payload = json.dumps(body, ensure_ascii=False)
        query = OrderedDict()
        query["Action"] = action
        query["Version"] = self.version

        param = SignParam()
        param.path = "/"
        param.method = "POST"
        param.host = netloc
        param.body = payload
        param.query = query
        header = OrderedDict()
        header["Host"] = netloc
        param.header_list = header

        credentials = Credentials(self.ak, self.sk, self.service, self.region)
        signed_query = SignerV4().sign_url(param, credentials)
        url = f"{scheme}://{netloc}/?{signed_query}"

        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(
                url,
                data=payload.encode("utf-8"),
                headers=headers,
                timeout=self.timeout,
            )
        except requests.exceptions.SSLError as exc:
            raise HiAgentNetworkError(f"TLS/SSL error connecting to {scheme}://{netloc}: {exc}") from exc
        except requests.exceptions.ConnectionError as exc:
            raise HiAgentNetworkError(f"Connection failed to {scheme}://{netloc}: {exc}") from exc
        except requests.exceptions.Timeout as exc:
            raise HiAgentNetworkError(f"Connection timeout after {self.timeout}s to {scheme}://{netloc}") from exc
        except requests.exceptions.RequestException as exc:
            raise HiAgentNetworkError(f"Request failed: {exc}") from exc

        return self._parse_response(response, netloc)

    @staticmethod
    def _parse_response(response, netloc=None):
        try:
            data = response.json()
        except ValueError:
            data = None

        if response.status_code >= 400:
            raise HiAgentHTTPError(status_code=response.status_code, body=data or response.text)

        if isinstance(data, dict):
            metadata = data.get("ResponseMetadata")
            if isinstance(metadata, dict):
                error = metadata.get("Error")
                if isinstance(error, dict):
                    raise HiAgentAPIError(
                        code=error.get("Code"),
                        message=error.get("Message"),
                        status_code=response.status_code,
                        body=data,
                    )
            code = data.get("Code")
            if code not in _SUCCESS_CODES:
                raise HiAgentAPIError(
                    code=code,
                    message=data.get("Message"),
                    status_code=response.status_code,
                    body=data,
                )
        return data
