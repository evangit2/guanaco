#!/usr/bin/env bash
# Guanaco installer
# Usage: curl -sSL https://raw.githubusercontent.com/evangit2/guanaco/main/install.sh | bash
#
# Supports: Linux, macOS, WSL

set -euo pipefail

REPO="evangit2/guanaco"
INSTALL_DIR="$HOME/.guanaco"
VENV_DIR="$INSTALL_DIR/venv"
BIN_DIR="$HOME/.local/bin"
DEFAULT_PORT=8080

# ── Colors (ANSI escape characters, not strings) ──
if [ -t 1 ]; then
    RED=$'\033[0;31m'
    GREEN=$'\033[0;32m'
    YELLOW=$'\033[1;33m'
    CYAN=$'\033[0;36m'
    BOLD=$'\033[1m'
    DIM=$'\033[2m'
    RESET=$'\033[0m'
else
    RED='' GREEN='' YELLOW='' CYAN='' BOLD='' DIM='' RESET=''
fi

# ── Prompt helpers ──
prompt() {
    local var="$1" question="$2" default="$3"
    if [ -n "$default" ]; then
        printf "${CYAN}${question}${RESET} [${DIM}${default}${RESET}]: "
    else
        printf "${CYAN}${question}${RESET}: "
    fi
    read -r value < /dev/tty || true
    declare -g "$var"="${value:-$default}"
}

prompt_yesno() {
    local var="$1" question="$2" default="${3:-n}"
    local indicator="y/N"
    [ "$default" = "y" ] && indicator="Y/n"
    printf "${CYAN}${question}${RESET} [${indicator}]: "
    read -r value < /dev/tty || true
    value="${value:-$default}"
    [[ "$value" =~ ^[Yy] ]] && declare -g "$var"="y" || declare -g "$var"="n"
}

# ── Print helpers ──
step()    { printf "\n  ${BOLD}━━━ %s ━━━${RESET}\n" "$1"; }
info()    { printf "    %s\n" "$1"; }
success() { printf "  ${GREEN}✓${RESET} %s\n" "$1"; }
warn()    { printf "  ${YELLOW}!${RESET} %s\n" "$1"; }
fail()    { printf "  ${RED}✗${RESET} %s\n" "$1"; }

# ── Detect platform ──
detect_platform() {
    local os_name="$(uname -s)"
    case "$os_name" in
        Linux)
            if grep -qi microsoft /proc/version 2>/dev/null; then
                echo "wsl"
            else
                echo "linux"
            fi
            ;;
        Darwin)
            echo "macos"
            ;;
        *)
            echo "unknown"
            ;;
    esac
}

PLATFORM=$(detect_platform)

printf "\n${BOLD}  Guanaco${RESET}\n"
printf "  OpenAI-compatible LLM proxy for Ollama Cloud\n\n"
info "Platform: $PLATFORM"

# ── Auto-install prereqs ──
step "Checking prerequisites"

# git
if ! command -v git &>/dev/null; then
    warn "git not found — installing..."
    case "$PLATFORM" in
        linux|wsl)
            sudo apt update -qq 2>/dev/null && sudo apt install -y -qq git 2>/dev/null || \
            sudo dnf install -y -q git 2>/dev/null || \
            sudo pacman -S --noconfirm git 2>/dev/null || {
                fail "Could not install git automatically. Please install it and re-run."
                exit 1
            }
            ;;
        macos)
            xcode-select --install 2>/dev/null || true
            ;;
    esac
    command -v git &>/dev/null && success "git installed" || { fail "git still not found"; exit 1; }
fi

# python3
if ! command -v python3 &>/dev/null; then
    warn "Python 3.10+ not found — installing..."
    case "$PLATFORM" in
        macos)
            if command -v brew &>/dev/null; then
                brew install python@3.12
            else
                fail "No Homebrew found. Install Python from https://python.org or install Homebrew first."
                exit 1
            fi
            ;;
        linux|wsl)
            sudo apt update -qq 2>/dev/null && sudo apt install -y -qq python3 python3-venv python3-pip 2>/dev/null || \
            sudo dnf install -y -q python3 python3-pip 2>/dev/null || \
            sudo pacman -S --noconfirm python python-pip 2>/dev/null || {
                fail "Could not install Python automatically. Please install it and re-run."
                exit 1
            }
            ;;
        *)
            fail "Please install Python 3.10+ from https://python.org and re-run."
            exit 1
            ;;
    esac
fi

# python3-venv (test actual creation)
if command -v python3 &>/dev/null; then
    VENV_TEST_DIR=$(mktemp -d)
    if ! python3 -m venv "$VENV_TEST_DIR/test_venv" &>/dev/null; then
        warn "python3-venv not working — installing..."
        case "$PLATFORM" in
            linux|wsl)
                sudo apt install -y -qq python3-venv 2>/dev/null || \
                sudo dnf install -y -q python3-venv 2>/dev/null || {
                    fail "Could not install python3-venv. Please install it and re-run."
                    exit 1
                }
                ;;
            macos)
                warn "venv broken — try: brew reinstall python@3.12"
                ;;
        esac
    fi
    rm -rf "$VENV_TEST_DIR"
fi

success "git $(git --version 2>/dev/null | awk '{print $3}')"
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "?")
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    success "python $PYTHON_VERSION"
else
    fail "Python 3.10+ required, found $PYTHON_VERSION"
    exit 1
fi
if python3 -m venv /tmp/guanaco_venv_test &>/dev/null; then
    rm -rf /tmp/guanaco_venv_test
    success "venv ok"
else
    rm -rf /tmp/guanaco_venv_test
    warn "venv missing (will attempt install)"
fi

# ── macOS SSL cert fix ──
if [ "$PLATFORM" = "macos" ]; then
    SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())" 2>/dev/null || echo "")
    if [ -n "$SSL_CERT_FILE" ]; then
        export SSL_CERT_FILE
        mkdir -p "$INSTALL_DIR"
        grep -q "^export SSL_CERT_FILE=" "$INSTALL_DIR/env" 2>/dev/null && \
            sed -i '' "s|^export SSL_CERT_FILE=.*|export SSL_CERT_FILE=$SSL_CERT_FILE|" "$INSTALL_DIR/env" 2>/dev/null || \
            echo "export SSL_CERT_FILE=$SSL_CERT_FILE" >> "$INSTALL_DIR/env"
        success "ssl_certs configured"
    fi
fi

# ── Configuration ──
step "Configuration"

info "You'll need an Ollama Cloud API key."
info "Get one at: ${CYAN}https://ollama.com${RESET}"
echo ""

prompt OLLAMA_API_KEY "Enter your Ollama API key" ""

if [ -z "$OLLAMA_API_KEY" ]; then
    echo ""
    warn "No API key provided. You can set it later with: guanaco setup"
    echo ""
fi

# ── Network configuration ──
step "Network configuration"

printf "  ${YELLOW}Guanaco will start a server on port ${DEFAULT_PORT}.${RESET}\n\n"

printf "  ${RED}  SECURITY WARNING${RESET}\n"
printf "  ${RED}  ─────────────────${RESET}\n"
printf "  ${RED}  If your machine has automatic port forwarding (some VPS${RESET}\n"
printf "  ${RED}  providers, routers with UPnP, Cloudflare tunnels, etc.),${RESET}\n"
printf "  ${RED}  this will EXPOSE your Ollama API proxy to the public${RESET}\n"
printf "  ${RED}  internet. Anyone who finds it can use your API key and${RESET}\n"
printf "  ${RED}  consume your Ollama Cloud quota.${RESET}\n\n"
printf "  ${RED}  - Bind to 127.0.0.1 unless you need remote access${RESET}\n"
printf "  ${RED}  - Use a firewall or auth proxy if you must expose it${RESET}\n"
printf "  ${RED}  - Never run this on a public-facing VPS without auth${RESET}\n\n"

prompt PORT "Which port should Guanaco use" "$DEFAULT_PORT"

# Detect Tailscale
TS_IP=""
if command -v tailscale >/dev/null 2>&1; then
    TS_IP=$(tailscale ip -4 2>/dev/null || true)
fi
if [ -z "$TS_IP" ] && [ -S /run/tailscale/tailscaled.sock ]; then
    TS_IP=$(tailscale ip -4 2>/dev/null || true)
fi
if [ -z "$TS_IP" ]; then
    TS_IP=$(ip -4 addr show | grep -oP 'inet 100\.\d+\.\d+\.\d+' | head -1 | awk '{print $2}' 2>/dev/null || true)
fi

if [ -n "$TS_IP" ]; then
    printf "  ${CYAN}Tailscale detected at ${TS_IP}${RESET}\n"
    prompt_yesno BIND_LOCAL "Bind to 0.0.0.0 (accessible via Tailscale)?" "y"
    if [ "$BIND_LOCAL" = "y" ]; then
        BIND_HOST="0.0.0.0"
        info "Bound to 0.0.0.0 — accessible at http://${TS_IP}:${PORT}/dashboard/"
    else
        BIND_HOST="127.0.0.1"
    fi
else
    prompt_yesno BIND_LOCAL "Bind to localhost only (127.0.0.1)?" "y"
    if [ "$BIND_LOCAL" = "y" ]; then
        BIND_HOST="127.0.0.1"
    else
        BIND_HOST="0.0.0.0"
        warn "Binding to 0.0.0.0 — accessible from ALL interfaces. Ensure you have auth/firewall."
    fi
fi

# ── Install ──
step "Installing"

# Clone or update
if [ -d "$INSTALL_DIR/repo" ]; then
    info "Updating existing installation..."
    cd "$INSTALL_DIR/repo"
    git pull --ff-only || { warn "Could not pull updates. Using existing version."; }
else
    info "Downloading Guanaco..."
    git clone "https://github.com/$REPO.git" "$INSTALL_DIR/repo"
    cd "$INSTALL_DIR/repo"
fi

# Create venv
info "Setting up virtual environment..."
python3 -m venv "$VENV_DIR"

# Source platform env if exists
if [ -f "$INSTALL_DIR/env" ]; then
    source "$INSTALL_DIR/env"
fi

# Install
info "Installing dependencies..."
"$VENV_DIR/bin/pip" install -e . --quiet 2>&1 | tail -1

# ── Write config ──
info "Writing configuration..."

mkdir -p "$INSTALL_DIR"

cat > "$INSTALL_DIR/config.yaml" << EOF
# Guanaco configuration
# Generated by install.sh on $(date -I)

router:
  host: "${BIND_HOST}"
  port: ${PORT}

llm:
  provider: ollama_cloud
  base_url: "https://api.ollama.com/v1"
  api_key_env: OLLAMA_API_KEY

fallback:
  enabled: false
  provider: ""
  api_key: ""
  model: ""
  primary_timeout: 30.0
  stream_chunk_timeout: 180.0
  max_tokens: 128000

cache:
  enabled: false
EOF

# Write API key env file
if [ -n "$OLLAMA_API_KEY" ]; then
    # Preserve existing env lines except old OLLAMA_API_KEY
    TMP_ENV=$(mktemp)
    grep -v "^export OLLAMA_API_KEY=" "$INSTALL_DIR/env" 2>/dev/null > "$TMP_ENV" || true
    echo "export OLLAMA_API_KEY=\"${OLLAMA_API_KEY}\"" >> "$TMP_ENV"
    mv "$TMP_ENV" "$INSTALL_DIR/env"
fi

# ── Create guanaco binary ──
mkdir -p "$BIN_DIR"

cat > "$BIN_DIR/guanaco" << SCRIPT
#!/usr/bin/env bash
GUANACO_DIR="$INSTALL_DIR"
if [ -f "\$GUANACO_DIR/env" ]; then
    source "\$GUANACO_DIR/env"
fi
exec "$VENV_DIR/bin/python" -m guanaco.cli "\$@"
SCRIPT
chmod +x "$BIN_DIR/guanaco"

# Also create 'oct' alias for backward compat
cat > "$BIN_DIR/oct" << SCRIPT
#!/usr/bin/env bash
GUANACO_DIR="$INSTALL_DIR"
if [ -f "\$GUANACO_DIR/env" ]; then
    source "\$GUANACO_DIR/env"
fi
exec "$VENV_DIR/bin/python" -m guanaco.cli "\$@"
SCRIPT
chmod +x "$BIN_DIR/oct"

# ── Add to PATH if needed ──
DETECT_SHELL="${SHELL##*/}"
case "$DETECT_SHELL" in
    zsh) PROFILE_FILE="$HOME/.zshrc" ;;
    bash) PROFILE_FILE="$HOME/.bashrc" ;;
    *) PROFILE_FILE="$HOME/.profile" ;;
esac

if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    echo ""
    warn "$BIN_DIR is not in your PATH."
    info "Adding to $PROFILE_FILE..."
    echo "" >> "$PROFILE_FILE"
    echo "# Added by Guanaco" >> "$PROFILE_FILE"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$PROFILE_FILE"
    export PATH="$BIN_DIR:$PATH"
    success "Added to $PROFILE_FILE"
fi

# ── Install complete ──
step "Install complete"

success "Guanaco installed successfully"
echo ""
printf "  ${BOLD}Start the server:${RESET}\n"
info "$BIN_DIR/guanaco start"
echo ""
printf "  ${DIM}Or reload your shell and use 'guanaco' directly:${RESET}\n"
info "source $PROFILE_FILE"
echo ""
printf "  ${BOLD}Dashboard:${RESET}\n"
if [ -n "$TS_IP" ]; then
    info "http://${TS_IP}:${PORT}/dashboard/"
    info "http://127.0.0.1:${PORT}/dashboard/  (local)"
else
    info "http://${BIND_HOST}:${PORT}/dashboard/"
fi
echo ""
printf "  ${BOLD}CLI commands:${RESET}\n"
printf "    %-24s %s\n" "guanaco status" "Show service & connection status"
printf "    %-24s %s\n" "guanaco models" "List available cloud models"
printf "    %-24s %s\n" "guanaco usage" "Check your Ollama Cloud usage/quota"
printf "    %-24s %s\n" "guanaco analytics" "View request analytics & stats"
printf "    %-24s %s\n" "guanaco key generate" "Generate an API key"
printf "    %-24s %s\n" "guanaco config --show" "Show current configuration"
printf "    %-24s %s\n" "guanaco setup" "Reconfigure (API key, ports, etc.)"

# ── Service setup ──
echo ""
printf "  ${BOLD}How should Guanaco run?${RESET}\n"
printf "    1) Foreground — press Ctrl+C to stop\n"
printf "    2) systemd service — auto-starts on boot, runs in background\n"
echo ""
prompt_yesno USE_SYSTEMD "Install as systemd service?" "y"

SERVICE_NAME="guanaco"
SERVICE_FILE="/etc/systemd/system/${SERVICE_NAME}.service"

if [ "$USE_SYSTEMD" = "y" ]; then
    printf "\n"
    info "Installing Guanaco as a systemd service..."

    CONFIG_DIR="$INSTALL_DIR"
    if [ ! -d "$CONFIG_DIR" ]; then
        CONFIG_DIR="$HOME/.guanaco"
    fi

    cat << SERVICEEOF | sudo tee "$SERVICE_FILE" > /dev/null
[Unit]
Description=Guanaco - LLM Proxy & Dashboard
Documentation=https://github.com/evangit2/guanaco
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=$(whoami)
Group=$(whoami)
Environment=PATH=${VENV_DIR}/bin:/usr/bin:/usr/local/bin
Environment=GUANACO_CONFIG_DIR=${CONFIG_DIR}
WorkingDirectory=${INSTALL_DIR}
ExecStart=${VENV_DIR}/bin/python -m uvicorn guanaco.app:create_app --factory --host ${BIND_HOST} --port ${PORT} --log-level info
Restart=on-failure
RestartSec=5

[Install]
WantedBy=multi-user.target
SERVICEEOF

    sudo systemctl daemon-reload
    sudo systemctl enable "$SERVICE_NAME" 2>/dev/null
    sudo systemctl restart "$SERVICE_NAME"
    sleep 2

    STATUS=$(sudo systemctl is-active "$SERVICE_NAME" 2>/dev/null)
    if [ "$STATUS" = "active" ]; then
        echo ""
        success "Guanaco service is running!"
        if [ -n "$TS_IP" ]; then
            printf "  ${BOLD}Dashboard:${RESET}  http://${TS_IP}:${PORT}/dashboard/\n"
        else
            printf "  ${BOLD}Dashboard:${RESET}  http://${BIND_HOST}:${PORT}/dashboard/\n"
        fi
        echo ""
        info "Manage with:"
        printf "    %-30s %s\n" "sudo systemctl status guanaco" "Check status"
        printf "    %-30s %s\n" "sudo systemctl stop guanaco" "Stop service"
        printf "    %-30s %s\n" "sudo systemctl restart guanaco" "Restart service"
        printf "    %-30s %s\n" "sudo journalctl -u guanaco -f" "View live logs"
    else
        echo ""
        warn "Service may not have started. Check logs:"
        info "sudo journalctl -u guanaco -n 50"
    fi
else
    echo ""
    info "Starting Guanaco in foreground..."
    info "Press Ctrl+C to stop. Run 'guanaco install' later for background service."
    echo ""
    "$BIN_DIR/guanaco" start
fi

# ── Platform tips ──
if [ "$PLATFORM" = "macos" ]; then
    printf "\n  ${DIM}macOS: To auto-start on login, see contrib/com.guanaco.start.plist${RESET}\n"
fi
if [ "$PLATFORM" = "wsl" ]; then
    WIN_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
    printf "  ${DIM}WSL: Access dashboard from Windows at http://${WIN_IP}:${PORT}/dashboard${RESET}\n"
fi

printf "\n  ${DIM}Docs: https://github.com/$REPO${RESET}\n"