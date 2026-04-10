#!/usr/bin/env bash
# Guanaco CLI wrapper — matches what install.sh generates
GUANACO_DIR="/opt/guanaco"
VENV_DIR="/opt/guanaco-venv"
exec "$VENV_DIR/bin/python" "-m" "guanaco.cli" "$@"