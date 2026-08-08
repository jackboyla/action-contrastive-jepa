"""Repository-local runtime paths."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_STABLEWM_HOME = PROJECT_ROOT / ".stable_worldmodel"


def configure_stablewm_home() -> Path:
    """Set the repo-local stable_worldmodel cache unless the user overrides it."""

    configured = os.environ.get("STABLEWM_HOME")
    path = Path(configured).expanduser() if configured else DEFAULT_STABLEWM_HOME
    path = path.resolve()
    os.environ["STABLEWM_HOME"] = str(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
