"""guanaco — maximize your Ollama Cloud subscription."""

try:
    from importlib.metadata import version as _version
    __version__ = _version("guanaco")
except Exception:
    __version__ = "0.3.7"  # fallback when not installed via pip