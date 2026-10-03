"""Make an API key for an agent. Standard library only, so it runs anywhere:

    python3 -m nsabot.apikey claude

Prints the key (give it to the agent, shown once) and the line to add to .env. Only the key's
SHA-256 hash goes in .env, so a leaked .env doesn't leak working keys.
"""

import hashlib
import re
import secrets
import sys


def hash_key(key: str) -> str:
    return hashlib.sha256(key.encode()).hexdigest()


def new_key() -> str:
    return "nsa_" + secrets.token_urlsafe(32)  # 256 bits


def main() -> None:
    name = sys.argv[1] if len(sys.argv) > 1 else "agent"
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,32}", name):
        raise SystemExit("name: letters, digits, - and _ only (max 32)")
    key = new_key()
    print(f"API key for {name} (shown once, give it to the agent):\n\n  {key}\n")
    print("Add to NSA_API_KEYS in .env (comma-separate several keys):\n")
    print(f"  {name}:{hash_key(key)}\n")


if __name__ == "__main__":
    main()
