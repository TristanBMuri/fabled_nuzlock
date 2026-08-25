"""Shared fixtures for integration tests. Requires Showdown on :8020 for live cases."""

from __future__ import annotations

import os
import socket
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
_POLICY_DIR = _REPO_ROOT / "no_AI_code_here" / "players" / "policy"

# Parent + spawn children need these on path (SubprocVecEnv start_method=spawn).
for _p in (_REPO_ROOT, _POLICY_DIR):
    s = str(_p)
    if s not in sys.path:
        sys.path.insert(0, s)

_existing = os.environ.get("PYTHONPATH", "")
os.environ["PYTHONPATH"] = os.pathsep.join(
    [str(_REPO_ROOT), str(_POLICY_DIR)] + ([_existing] if _existing else [])
)

import project_config  # noqa: E402

assert project_config.PROJECT_ROOT.resolve() == _REPO_ROOT.resolve()


def _port_open(host: str, port: int, timeout: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


@pytest.fixture(scope="session")
def showdown_8020():
    if not _port_open("127.0.0.1", 8020):
        pytest.skip("Showdown not listening on 8020")
    return "ws://localhost:8020/showdown/websocket"
