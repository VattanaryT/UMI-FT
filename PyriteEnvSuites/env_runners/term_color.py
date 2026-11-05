"""ANSI colors for operator-facing runner messages.

Failures are red. Warnings, including camera frame drops, are yellow.
"""

from __future__ import annotations

RED = "\033[31m"
YELLOW = "\033[33m"
RESET = "\033[0m"


def red(msg: str) -> str:
    return f"{RED}{msg}{RESET}"


def yellow(msg: str) -> str:
    return f"{YELLOW}{msg}{RESET}"


def print_red(msg: str) -> None:
    print(red(msg), flush=True)


def print_yellow(msg: str) -> None:
    print(yellow(msg), flush=True)
