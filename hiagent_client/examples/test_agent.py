import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hiagent.agent_client import HiAgentAgentClient, HiAgentAgentError  # noqa: E402
from hiagent.config import getenv  # noqa: E402


def _auth_ready():
    if getenv("HIAGENT_AGENT_API_KEY"):
        return True
    return bool(
        getenv("HIAGENT_AGENT_AK")
        and getenv("HIAGENT_AGENT_SK")
        and getenv("HIAGENT_AGENT_APP_ID")
    )


def main():
    if not getenv("HIAGENT_AGENT_URL") or not _auth_ready():
        print("Environment not configured. Please export:")
        print("  export HIAGENT_AGENT_URL='https://agentlab.sdu.edu.cn/api/proxy/api/v1'")
        print("  # Apikey mode:")
        print("  export HIAGENT_AGENT_API_KEY='<your real ApiKey>'")
        print("  # or signed AK/SK mode:")
        print("  export HIAGENT_AGENT_AK='<AccessKey ID>'")
        print("  export HIAGENT_AGENT_SK='<AccessKey Secret>'")
        print("  export HIAGENT_AGENT_APP_ID='<APPID>'")
        return 1

    client = HiAgentAgentClient()
    try:
        conversation_id = client.create_conversation("test001")
    except HiAgentAgentError as exc:
        print(f"[agent error] create_conversation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Conversation created: {conversation_id}")

    try:
        result = client.chat(
            "你好，请简单介绍一下你自己",
            user_id="test001",
            conversation_id=conversation_id,
        )
    except HiAgentAgentError as exc:
        print(f"[agent error] chat failed: {exc}", file=sys.stderr)
        return 1

    print(f"Answer: {result.get('answer')}")
    for label, key in (
        ("Latency", "latency"),
        ("Input tokens", "input_tokens"),
        ("Output tokens", "output_tokens"),
        ("Total tokens", "total_tokens"),
    ):
        if key in result:
            print(f"{label}: {result[key]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
