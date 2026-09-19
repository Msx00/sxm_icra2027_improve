import argparse
import json
import sys

from hiagent.agent_client import HiAgentAgentClient, HiAgentAgentError


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="python -m hiagent.agent",
        description="Chat with a published HiAgent agent.",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Query text, e.g. '你好'. If omitted, reads from stdin.",
    )
    parser.add_argument("--user", default="server001", help="UserID (default: server001)")
    parser.add_argument(
        "--conversation",
        default=None,
        help="AppConversationID to reuse (default: create a new conversation)",
    )
    parser.add_argument("--json", action="store_true", help="Print the full JSON response")
    parser.add_argument(
        "--stream",
        action="store_true",
        help="Stream the answer via chat_stream() and print it as it arrives",
    )
    parser.add_argument("--timeout", type=int, default=None)
    args = parser.parse_args(argv)

    query = args.query
    if not query:
        query = sys.stdin.read().strip()
    if not query:
        parser.error('Provide a query, e.g. python -m hiagent.agent "你好"')

    client = HiAgentAgentClient(timeout=args.timeout)
    try:
        if args.stream:
            for chunk in client.chat_stream(
                query, user_id=args.user, conversation_id=args.conversation
            ):
                sys.stdout.write(chunk)
                sys.stdout.flush()
            sys.stdout.write("\n")
            return 0
        if args.json:
            result = client.chat(
                query, user_id=args.user, conversation_id=args.conversation
            )
            print(json.dumps(result, ensure_ascii=False, indent=2))
            return 0
        answer = client.ask(query, user_id=args.user, conversation_id=args.conversation)
        print(answer)
        return 0
    except HiAgentAgentError as exc:
        print(f"[agent error] {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("[interrupted]", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
