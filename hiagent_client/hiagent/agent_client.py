import json
from collections import OrderedDict
from typing import Iterator, Optional
from urllib.parse import urlparse

import requests

from hiagent.config import getenv

DEFAULT_TIMEOUT = 60
DEFAULT_SIGN_REGION = "cn-north-1"
DEFAULT_SIGN_SERVICE = "app"


class _SignRequest:
    """Minimal request object accepted by volcengine SignerV4.sign()."""

    def __init__(self):
        self.method = "POST"
        self.path = "/"
        self.host = ""
        self.body = b""
        self.query = OrderedDict()
        self.headers = OrderedDict()


class HiAgentAgentError(Exception):
    """Base class for Agent API client errors."""


class HiAgentAgentConfigError(HiAgentAgentError):
    """Missing or invalid configuration (e.g. env var not set)."""


class HiAgentAgentNetworkError(HiAgentAgentError):
    """Connection / TLS / timeout / DNS level failure."""


class HiAgentAgentHTTPError(HiAgentAgentError):
    """Non-2xx HTTP response from the Agent API."""

    def __init__(self, status_code, endpoint, body=None):
        self.status_code = status_code
        self.endpoint = endpoint
        self.body = body
        super().__init__(
            f"HTTP {status_code} from POST {endpoint}: {_truncate(body)}"
        )


class HiAgentAgentJSONError(HiAgentAgentError):
    """Response body could not be parsed as JSON."""


class HiAgentAgentAPIError(HiAgentAgentError):
    """API returned 2xx but the payload structure was unexpected."""


class HiAgentAgentSSEError(HiAgentAgentError):
    """Streaming (SSE) response could not be parsed."""


def _truncate(value, limit=1000):
    if value is None:
        return "<empty>"
    if isinstance(value, str):
        text = value
    else:
        try:
            text = json.dumps(value, ensure_ascii=False)
        except (TypeError, ValueError):
            text = str(value)
    if len(text) > limit:
        return text[:limit] + "...(truncated)"
    return text


def _find_first(data, *keys, max_depth=10):
    """Depth-first search for the first occurrence of any of `keys`."""
    if max_depth < 0:
        return None
    if isinstance(data, dict):
        for key in keys:
            if key in data:
                return data[key]
        for value in data.values():
            found = _find_first(value, *keys, max_depth=max_depth - 1)
            if found is not None:
                return found
    elif isinstance(data, list):
        for item in data:
            found = _find_first(item, *keys, max_depth=max_depth - 1)
            if found is not None:
                return found
    return None


_CHAT_FIELD_ALIASES = {
    "answer": ("answer", "Answer"),
    "conversation_id": (
        "conversation_id",
        "conversationId",
        "app_conversation_id",
        "AppConversationID",
    ),
    "task_id": ("task_id", "taskId", "chat_id", "chatId"),
    "latency": ("latency", "Latency"),
    "input_tokens": ("input_tokens", "inputTokens"),
    "output_tokens": ("output_tokens", "outputTokens"),
    "total_tokens": ("total_tokens", "totalTokens"),
}


class HiAgentAgentClient:
    """HiAgent Agent conversation client.

    Supports two auth modes:
    - Apikey mode: header `Apikey: <HIAGENT_AGENT_API_KEY>`.
    - Signed AK/SK mode: volcengine SignerV4 with
      HIAGENT_AGENT_AK + HIAGENT_AGENT_SK + HIAGENT_AGENT_APP_ID (header X-APP-ID).

    Flow: create_conversation -> chat_query_v2 (blocking or streaming).
    """

    def __init__(
        self,
        url=None,
        api_key=None,
        ak=None,
        sk=None,
        app_id=None,
        timeout=None,
    ):
        raw_url = (url or getenv("HIAGENT_AGENT_URL")) or ""
        self.base_url = raw_url.rstrip("/")
        self.api_key = api_key or getenv("HIAGENT_AGENT_API_KEY")
        self.ak = ak or getenv("HIAGENT_AGENT_AK")
        self.sk = sk or getenv("HIAGENT_AGENT_SK")
        self.app_id = app_id or getenv("HIAGENT_AGENT_APP_ID")
        self.sign_region = getenv("HIAGENT_AGENT_REGION", DEFAULT_SIGN_REGION)
        self.sign_service = getenv("HIAGENT_AGENT_SERVICE", DEFAULT_SIGN_SERVICE)
        if timeout is None:
            timeout = int(getenv("HIAGENT_TIMEOUT", DEFAULT_TIMEOUT))
        self.timeout = timeout

    @property
    def _host(self):
        try:
            parsed = urlparse(self.base_url)
            return parsed.netloc or parsed.hostname or ""
        except Exception:
            return ""

    def _check_config(self):
        if not self.base_url:
            raise HiAgentAgentConfigError(
                "Missing HIAGENT_AGENT_URL. "
                "Get it from HiAgent -> Agent -> Publish -> API configuration."
            )
        if self.ak and self.sk and self.app_id:
            return "aksk"
        if self.api_key:
            return "apikey"
        raise HiAgentAgentConfigError(
            "Missing authentication. Provide either "
            "HIAGENT_AGENT_API_KEY (Apikey mode) or "
            "HIAGENT_AGENT_AK + HIAGENT_AGENT_SK + HIAGENT_AGENT_APP_ID "
            "(signed AK/SK mode)."
        )

    def _endpoint(self, path):
        return f"{self.base_url}/{path.lstrip('/')}"

    def _build_auth_headers(self, mode, path, body_bytes, extra_headers=None):
        headers = {"Content-Type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)

        if mode == "aksk":
            from volcengine.auth.SignerV4 import SignerV4
            from volcengine.Credentials import Credentials

            if not self._host:
                raise HiAgentAgentConfigError("Invalid HIAGENT_AGENT_URL: missing host.")

            req = _SignRequest()
            req.method = "POST"
            req.path = path
            req.host = self._host
            req.body = body_bytes
            req.query = OrderedDict()
            req.headers = OrderedDict({"Host": self._host, "Content-Type": "application/json"})
            req.headers["X-APP-ID"] = self.app_id
            if extra_headers:
                for key, value in extra_headers.items():
                    req.headers[key] = value

            credentials = Credentials(
                self.ak, self.sk, self.sign_service, self.sign_region
            )
            SignerV4.sign(req, credentials)
            headers.update(req.headers)
        else:
            headers["Apikey"] = self.api_key
        return headers

    def _post_json(self, path, payload, headers=None, stream=False):
        mode = self._check_config()
        endpoint = self._endpoint(path)
        sign_path = urlparse(endpoint).path or "/"
        body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        request_headers = self._build_auth_headers(
            mode, sign_path, body_bytes, headers
        )

        try:
            response = requests.post(
                endpoint,
                data=body_bytes,
                headers=request_headers,
                timeout=self.timeout,
                stream=stream,
            )
        except requests.exceptions.SSLError as exc:
            raise HiAgentAgentNetworkError(
                f"TLS/SSL error connecting to {endpoint}: {exc}"
            ) from exc
        except requests.exceptions.ConnectionError as exc:
            raise HiAgentAgentNetworkError(
                f"Connection failed to {endpoint}: {exc}"
            ) from exc
        except requests.exceptions.Timeout as exc:
            raise HiAgentAgentNetworkError(
                f"Connection timeout after {self.timeout}s to {endpoint}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise HiAgentAgentNetworkError(f"Request failed: {exc}") from exc

        if response.status_code >= 400:
            raise HiAgentAgentHTTPError(
                status_code=response.status_code,
                endpoint=endpoint,
                body=response.text,
            )
        return response

    def create_conversation(
        self,
        user_id: str,
        inputs: Optional[dict] = None,
        conversation_name: Optional[str] = None,
    ) -> str:
        self._check_config()
        body = {"UserID": user_id}
        if inputs:
            body["Inputs"] = inputs
        if conversation_name:
            body["ConversationName"] = conversation_name

        response = self._post_json("create_conversation", body)
        try:
            data = response.json()
        except ValueError as exc:
            raise HiAgentAgentJSONError(
                f"create_conversation returned non-JSON response: "
                f"{_truncate(response.text)}"
            ) from exc

        conversation_id = _find_first(data, "AppConversationID", "app_conversation_id")
        if not conversation_id:
            raise HiAgentAgentAPIError(
                "create_conversation response did not contain "
                "Conversation.AppConversationID. "
                f"Got: {_truncate(data)}"
            )
        return str(conversation_id)

    def chat(
        self,
        query: str,
        user_id: str = "server001",
        conversation_id: Optional[str] = None,
        response_mode: str = "blocking",
    ) -> dict:
        self._check_config()
        if not conversation_id:
            conversation_id = self.create_conversation(user_id)

        body = {
            "Query": query,
            "AppConversationID": conversation_id,
            "ResponseMode": response_mode,
            "UserID": user_id,
        }
        response = self._post_json("chat_query_v2", body)
        try:
            data = response.json()
        except ValueError as exc:
            raise HiAgentAgentJSONError(
                f"chat_query_v2 returned non-JSON response: "
                f"{_truncate(response.text)}"
            ) from exc

        if not isinstance(data, dict):
            raise HiAgentAgentAPIError(
                f"chat_query_v2 response is not a JSON object: {_truncate(data)}"
            )

        normalized = dict(data)
        for field, aliases in _CHAT_FIELD_ALIASES.items():
            if field in normalized:
                continue
            value = _find_first(data, *aliases)
            if value is not None:
                normalized[field] = value
        return normalized

    def ask(
        self,
        query: str,
        user_id: str = "server001",
        conversation_id: Optional[str] = None,
    ) -> str:
        result = self.chat(query, user_id=user_id, conversation_id=conversation_id)
        answer = result.get("answer")
        if not isinstance(answer, str) or not answer:
            raise HiAgentAgentAPIError(
                "chat_query_v2 returned no answer. "
                f"Response keys: {sorted(result.keys())}"
            )
        return answer

    def chat_stream(
        self,
        query: str,
        user_id: str = "server001",
        conversation_id: Optional[str] = None,
    ) -> Iterator[str]:
        self._check_config()
        if not conversation_id:
            conversation_id = self.create_conversation(user_id)

        body = {
            "Query": query,
            "AppConversationID": conversation_id,
            "ResponseMode": "streaming",
            "UserID": user_id,
        }
        response = self._post_json(
            "chat_query_v2",
            body,
            headers={"Accept": "text/event-stream"},
            stream=True,
        )
        try:
            for line in response.iter_lines(decode_unicode=True):
                line = (line or "").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:"):].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    event = json.loads(payload)
                except ValueError as exc:
                    raise HiAgentAgentSSEError(
                        f"Failed to parse SSE data payload: {_truncate(payload)}"
                    ) from exc
                if not isinstance(event, dict):
                    continue
                answer = event.get("answer")
                if isinstance(answer, str) and answer:
                    yield answer
        finally:
            response.close()
