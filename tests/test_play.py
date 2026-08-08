import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SPEC = importlib.util.spec_from_file_location("play", ROOT / "play.py")
assert SPEC is not None and SPEC.loader is not None
PLAY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PLAY)
should_handle_keydown = PLAY.should_handle_keydown
is_quit_key = PLAY.is_quit_key


def key_event(key: int, *, repeat: bool):
    return SimpleNamespace(key=key, repeat=repeat)


def test_initial_one_shot_control_keydown_is_handled():
    assert should_handle_keydown(
        key_event(10, repeat=False), action_keys={20, 21}, noop_key=30
    )


def test_repeated_one_shot_control_keydown_is_ignored():
    assert not should_handle_keydown(
        key_event(10, repeat=True), action_keys={20, 21}, noop_key=30
    )


def test_repeated_action_and_noop_keydowns_are_handled():
    assert should_handle_keydown(
        key_event(20, repeat=True), action_keys={20, 21}, noop_key=30
    )
    assert should_handle_keydown(
        key_event(30, repeat=True), action_keys={20, 21}, noop_key=30
    )


def test_escape_always_quits():
    assert is_quit_key(1, action_keys={2}, escape_key=1, q_key=3)


def test_q_quits_only_when_not_bound_to_task_action():
    assert is_quit_key(3, action_keys={2}, escape_key=1, q_key=3)
    assert not is_quit_key(3, action_keys={3}, escape_key=1, q_key=3)
