"""Bring your own keys.

OpenSmoke has no server and no account: every user pays TypeSafe and their LLM
provider directly. Keys are looked up in this order:

  1. the environment (so CI and secret managers win)
  2. ~/.config/opensmoke/keys.env, saved by an earlier prompt or `opensmoke keys`
  3. a hidden prompt, only when a person is at the terminal

Keys are only ever sent to the service they belong to, and never printed.
"""

from __future__ import annotations

import getpass
import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class KeySpec:
    env: str
    label: str
    url: str
    required_for: str


TYPESAFE = KeySpec("TYPESAFE_API_KEY", "TypeSafe API key", "https://console.typesafe.ai", "the Jev judge")
LLM = KeySpec("OPENROUTER_API_KEY", "OpenRouter API key", "https://openrouter.ai/keys", "LLM diagnosis (optional)")
ALL = (TYPESAFE, LLM)


def config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base) / "opensmoke" / "keys.env"


def read_saved() -> dict[str, str]:
    path = config_path()
    if not path.is_file():
        return {}
    saved = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        name, sep, value = line.partition("=")
        if sep and name.strip() and not name.lstrip().startswith("#"):
            saved[name.strip()] = value.strip()
    return saved


def load_saved() -> None:
    """Fill the environment from the saved file without overriding what is set."""
    for name, value in read_saved().items():
        if value and not os.environ.get(name):
            os.environ[name] = value


def save(name: str, value: str) -> Path:
    path = config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    saved = read_saved()
    saved[name] = value
    body = "# OpenSmoke keys. Readable only by you.\n" + "".join(f"{k}={v}\n" for k, v in saved.items() if v)
    # Create with owner-only permissions before any secret is written to it.
    fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(body)
    os.chmod(path, 0o600)
    return path


def interactive() -> bool:
    return sys.stdin.isatty() and sys.stderr.isatty()


def mask(value: str) -> str:
    return f"{value[:3]}…{value[-4:]}" if len(value) > 10 else "…"


def ask(spec: KeySpec, optional: bool = False) -> str | None:
    """Return the key, prompting for it if a person is here to answer."""
    if os.environ.get(spec.env):
        return os.environ[spec.env]
    if not interactive():
        return None
    print(f"\n{spec.label} not found. It is used for {spec.required_for}.", file=sys.stderr)
    print(f"Get one at {spec.url}", file=sys.stderr)
    hint = " (Enter to skip)" if optional else ""
    value = getpass.getpass(f"Paste your {spec.label}{hint}: ").strip()
    if not value:
        return None
    os.environ[spec.env] = value
    answer = input(f"Save it to {config_path()} so you are not asked again? [Y/n] ").strip().lower()
    if answer in ("", "y", "yes"):
        path = save(spec.env, value)
        print(f"Saved to {path} (owner read/write only).", file=sys.stderr)
    return value


def status() -> list[tuple[KeySpec, str]]:
    """Where each key would come from right now, masked."""
    saved = read_saved()
    out = []
    for spec in ALL:
        if os.environ.get(spec.env) and os.environ[spec.env] != saved.get(spec.env):
            out.append((spec, f"environment ({mask(os.environ[spec.env])})"))
        elif saved.get(spec.env):
            out.append((spec, f"{config_path()} ({mask(saved[spec.env])})"))
        else:
            out.append((spec, "not set"))
    return out
