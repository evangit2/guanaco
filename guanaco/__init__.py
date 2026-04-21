"""guanaco — maximize your Ollama Cloud subscription."""

# Single source of truth for version.
# importlib.metadata can return stale values after git-pull without re-pip-install,
# so we always use the hardcoded fallback and only override if metadata matches.
__version__ = "0.4.2-dev"

try:
    from importlib.metadata import version as _version
    _pkg_ver = _version("guanaco")
    # Only use pkg version if it parses as a clean semver >= our hardcoded baseline.
    # This prevents stale/RC versions like "0.4.0rc1" from overriding the hardcoded value.
    import re
    _m = re.match(r"^(\d+)\.(\d+)\.(\d+)$", _pkg_ver or "")
    if _m and tuple(int(x) for x in _m.groups()) >= (0, 4, 1):
        __version__ = _pkg_ver
except Exception:
    pass