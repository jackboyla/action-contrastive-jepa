from pathlib import Path

from project_paths import DEFAULT_STABLEWM_HOME, configure_stablewm_home


def test_configure_stablewm_home_defaults_to_repo_cache(monkeypatch):
    monkeypatch.delenv("STABLEWM_HOME", raising=False)

    path = configure_stablewm_home()

    assert path == DEFAULT_STABLEWM_HOME.resolve()


def test_configure_stablewm_home_preserves_explicit_override(tmp_path, monkeypatch):
    custom = tmp_path / "stablewm"
    monkeypatch.setenv("STABLEWM_HOME", str(custom))

    path = configure_stablewm_home()

    assert path == custom.resolve()
    assert Path(path).exists()
