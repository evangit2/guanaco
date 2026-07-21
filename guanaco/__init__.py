"""guanaco — multi-provider LLM router."""

# Single source of truth for version.
# importlib.metadata can return stale values after git-pull without re-pip-install,
# so we always use the hardcoded fallback and only override if metadata matches.
__version__ = "0.8.5"

try:
    from importlib.metadata import version as _version
    _pkg_ver = _version("guanaco-llm-proxy")
    # Only override hardcoded if installed metadata is *newer or same* —
    # prevents stale metadata from git-pull without re-pip-install from
    # reverting the version to an old value.
    import re
    _m = re.match(r"^(\d+)\.(\d+)\.(\d+)$", _pkg_ver or "")
    if _m:
        _hardcoded = tuple(int(x) for x in __version__.split("."))
        if tuple(int(x) for x in _m.groups()) >= _hardcoded:
            __version__ = _pkg_ver
except Exception:
    pass
