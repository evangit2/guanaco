"""guanaco — maximize your Ollama Cloud subscription."""

# Single source of truth for version.
# importlib.metadata can return stale values after git-pull without re-pip-install,
# so we always use the hardcoded fallback and only override if metadata matches.
__version__ = "0.3.9"

try:
    from importlib.metadata import version as _version
    _pkg_ver = _version("guanaco")
    # Only use pkg version if it's >= our hardcoded version (prevents stale 0.3.0 overrides)
    if _pkg_ver and tuple(int(x) for x in _pkg_ver.split(".") if x.isdigit()) >= (0, 3, 9):
        __version__ = _pkg_ver
except Exception:
    pass