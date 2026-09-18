import os
import stat

import pytest

from opensmoke import keys
from opensmoke.cli import main


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    for spec in keys.ALL:
        monkeypatch.delenv(spec.env, raising=False)
    monkeypatch.delenv("OPENSMOKE_LLM_API_KEY", raising=False)


def test_saved_file_is_owner_only_and_loads(monkeypatch):
    path = keys.save("TYPESAFE_API_KEY", "ts-secret-1234")
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    keys.load_saved()
    assert os.environ["TYPESAFE_API_KEY"] == "ts-secret-1234"


def test_environment_beats_saved_file(monkeypatch):
    keys.save("TYPESAFE_API_KEY", "from-file")
    monkeypatch.setenv("TYPESAFE_API_KEY", "from-env")
    keys.load_saved()
    assert os.environ["TYPESAFE_API_KEY"] == "from-env"
    [(_, where), _] = keys.status()
    assert where.startswith("environment") and "from-env" not in where  # masked


def test_prompts_and_saves_when_a_person_is_there(monkeypatch):
    monkeypatch.setattr(keys, "interactive", lambda: True)
    monkeypatch.setattr(keys.getpass, "getpass", lambda prompt: "ts-typed-5678")
    monkeypatch.setattr("builtins.input", lambda prompt: "")  # Enter = yes, save
    assert keys.ask(keys.TYPESAFE) == "ts-typed-5678"
    assert keys.read_saved()["TYPESAFE_API_KEY"] == "ts-typed-5678"


def test_never_prompts_in_ci(monkeypatch):
    monkeypatch.setattr(keys, "interactive", lambda: False)
    monkeypatch.setattr(keys.getpass, "getpass", lambda prompt: pytest.fail("prompted in CI"))
    assert keys.ask(keys.TYPESAFE) is None


def test_scan_without_key_explains_instead_of_crashing(monkeypatch, capsys):
    monkeypatch.setattr(keys, "interactive", lambda: False)
    assert main(["scan", "examples/traces", "--no-diagnose"]) == 2
    assert "opensmoke keys" in capsys.readouterr().err
