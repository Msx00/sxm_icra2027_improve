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
        print("Environment not configured. Please export HIAGENT_AGENT_URL and either")
        print("HIAGENT_AGENT_API_KEY or HIAGENT_AGENT_AK/SK/HIAGENT_AGENT_APP_ID.")
        return 1

    client = HiAgentAgentClient()
    try:
        cid = client.create_conversation("test001")
    except HiAgentAgentError as exc:
        print(f"[agent error] create_conversation failed: {exc}", file=sys.stderr)
        return 1
    print(f"Conversation created: {cid}")

    try:
        r1 = client.ask(
            "请记住，我的测试代号是 ALPHA123。",
            user_id="test001",
            conversation_id=cid,
        )
        print(f"Turn 1: {r1}")

        r2 = client.ask(
            "我刚才告诉你的测试代号是什么？",
            user_id="test001",
            conversation_id=cid,
        )
        print(f"Turn 2: {r2}")
    except HiAgentAgentError as exc:
        print(f"[agent error] {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
