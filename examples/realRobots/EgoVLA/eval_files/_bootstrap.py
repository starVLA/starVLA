"""Put the starVLA repo root on ``sys.path``.

These eval servers are standalone entrypoints (``python .../server_*.py``) and the
repo is not pip-installed, so ``deployment`` / ``starVLA`` are not importable by
default when a server is run as a script (only the script's own directory lands on
``sys.path``). Each server does ``import _bootstrap`` at import time; that succeeds
only when this directory is on ``sys.path`` (i.e. run-as-script), and then walks up
to the repo root (marked by ``pyproject.toml``) and inserts it. When a server is
imported as a package module instead, the root is already importable and the bare
``import _bootstrap`` simply raises ModuleNotFoundError (caught by the caller).
"""
import os
import sys

_root = os.path.dirname(os.path.abspath(__file__))
while _root != os.path.dirname(_root) and not os.path.exists(os.path.join(_root, "pyproject.toml")):
    _root = os.path.dirname(_root)
if _root not in sys.path:
    sys.path.insert(0, _root)
