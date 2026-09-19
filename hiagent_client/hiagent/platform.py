import argparse
import json
import sys

from hiagent.platform_client import HiAgentError, HiAgentPlatformClient


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m hiagent.platform",
        description="Call a HiAgent Platform API action with volcengine SignerV4 signing.",
    )
    parser.add_argument("--action", required=True, help="API Action, e.g. ListAppCenter")
    parser.add_argument("--json", default="{}", help="Request body as a JSON string, e.g. '{\"App\": true}'")
    parser.add_argument("--region", default=None)
    parser.add_argument("--service", default=None)
    parser.add_argument("--version", default=None)
    parser.add_argument("--timeout", type=int, default=None)
    args = parser.parse_args(argv)

    try:
        body = json.loads(args.json)
    except json.JSONDecodeError as exc:
        parser.error(f"Invalid --json: {exc}")

    client = HiAgentPlatformClient(
        region=args.region,
        service=args.service,
        version=args.version,
        timeout=args.timeout,
    )
    try:
        result = client.request(action=args.action, body=body)
    except HiAgentError as exc:
        print(f"[platform error] {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
