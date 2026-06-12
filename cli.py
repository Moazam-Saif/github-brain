"""
cli.py
------
Command-line interface for GitHub Brain.

Two modes:
  python cli.py index --mode full          → index all eligible repos
  python cli.py index --mode repo --name X → re-index one repo
  python cli.py chat                        → start interactive query session

The chat loop maintains session state across turns and passes it to
query/engine.py on every call. Session resets automatically when the
user switches from a repo-specific deep dive to a cross-repo question.
"""

import os
import sys
import argparse
import textwrap
from dotenv import load_dotenv

# Load .env before any module that reads environment variables.
load_dotenv()

from indexer.main  import index_all, index_repo
from query.engine  import query


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

SEPARATOR    = "─" * 60
BOT_PREFIX   = "🤖  "
USER_PREFIX  = "You: "
WRAP_WIDTH   = 80


def _print_answer(answer: str):
    """Print the bot answer, wrapped at WRAP_WIDTH chars."""
    print(f"\n{BOT_PREFIX}")
    for line in answer.splitlines():
        if line.strip():
            print(textwrap.fill(line, width=WRAP_WIDTH, subsequent_indent="    "))
        else:
            print()
    print()


def _print_banner():
    print(f"\n{'='*60}")
    print("  GitHub Brain — Chat with your entire GitHub account")
    print(f"{'='*60}")
    print("  Commands:")
    print("    exit / quit  → end the session")
    print("    reset        → clear active session and start fresh")
    print("    scope        → show which repo is currently in focus")
    print(f"{'='*60}\n")


def _print_session_status(session):
    if session and session.get("active_repo"):
        print(f"  [scope: {session['active_repo']}]\n")
    else:
        print("  [scope: all repos]\n")


# ---------------------------------------------------------------------------
# Index command
# ---------------------------------------------------------------------------

def cmd_index(args):
    """Handle the `index` subcommand."""
    if args.mode == "full":
        print("\n[cli] Starting full index. This may take a while...\n")
        index_all()

    elif args.mode == "repo":
        if not args.name:
            print("[cli] Error: --name is required for --mode repo")
            sys.exit(1)
        print(f"\n[cli] Re-indexing repo: {args.name}\n")
        index_repo(args.name)

    else:
        print(f"[cli] Unknown index mode: {args.mode}")
        sys.exit(1)


# ---------------------------------------------------------------------------
# Chat command
# ---------------------------------------------------------------------------

def cmd_chat(args):
    """Handle the `chat` subcommand — interactive query loop."""
    _print_banner()

    session = None  # No active session at start.

    print(f"{BOT_PREFIX}Hi! Ask me anything about your GitHub repos.")
    print("     Examples:")
    print("       • Do I have any repos using Redis?")
    print("       • What repos do I have deployed?")
    print("       • How does authentication work in claimsense?")
    print()

    while True:
        # Show current scope.
        _print_session_status(session)

        # Get user input.
        try:
            user_input = input(f"{USER_PREFIX}").strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{BOT_PREFIX}Goodbye!")
            break

        if not user_input:
            continue

        # Built-in commands.
        if user_input.lower() in ("exit", "quit"):
            print(f"\n{BOT_PREFIX}Goodbye!")
            break

        if user_input.lower() == "reset":
            session = None
            print(f"\n{BOT_PREFIX}Session cleared. Back to searching all repos.\n")
            continue

        if user_input.lower() == "scope":
            _print_session_status(session)
            continue

        # Query.
        print(f"\n{SEPARATOR}")
        try:
            answer, session = query(user_input, session=session)
            _print_answer(answer)
        except Exception as e:
            print(f"\n{BOT_PREFIX}Something went wrong: {e}")
            print("     Please try again or type 'reset' to clear the session.\n")

        print(SEPARATOR)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        prog="github-brain",
        description="Chat with your entire GitHub account using natural language.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- index subcommand ---
    index_parser = subparsers.add_parser("index", help="Index GitHub repos into Deep Lake")
    index_parser.add_argument(
        "--mode",
        choices=["full", "repo"],
        required=True,
        help="'full' to index all repos, 'repo' to re-index a single repo",
    )
    index_parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="Repo name to re-index (required when --mode repo)",
    )

    # --- chat subcommand ---
    subparsers.add_parser("chat", help="Start an interactive chat session")

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "chat":
        cmd_chat(args)


if __name__ == "__main__":
    main()
