import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hiagent.config import getenv  # noqa: E402
from hiagent.platform_client import HiAgentError, HiAgentPlatformClient  # noqa: E402


def main():
    if not getenv("HIAGENT_TOP_HOST"):
        print("Missing HIAGENT_TOP_HOST.")
        print("Please obtain the externally accessible HiAgent TopHost from the administrator.")
        return 1

    client = HiAgentPlatformClient()
    try:
        result = client.request(action="ListAppCenter", body={"App": True})
    except HiAgentError as exc:
        print(f"[platform error] {exc}", file=sys.stderr)
        return 1
    print(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
