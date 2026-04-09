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

# ── Colors (only if terminal) ──
if [ -t 1 ]; then
    RED='\033[0;31m'
    GREEN='\033[0;32m'
    YELLOW='\033[1;33m'
    CYAN='\033[0;36m'
    BOLD='\033[1m'
    DIM='\033[2m'
    RESET='\033[0m'
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

echo ""
echo "${BOLD}🦙 Guanaco Installer${RESET}"
echo "${DIM}   OpenAI-compatible LLM proxy for Ollama Cloud${RESET}"
echo ""
echo "Platform: $PLATFORM"
echo ""

# ── Auto-install prereqs ──
echo "${BOLD}━━━ Checking prerequisites ━━━${RESET}"

# git
if ! command -v git &>/dev/null; then
    echo "  ${YELLOW}⚠ git not found — installing...${RESET}"
    case "$PLATFORM" in
        linux|wsl)
            sudo apt update -qq 2>/dev/null && sudo apt install -y -qq git 2>/dev/null || \
            sudo dnf install -y -q git 2>/dev/null || \
            sudo pacman -S --noconfirm git 2>/dev/null || {
                echo "${RED}  ❌ Could not install git automatically. Please install it and re-run.${RESET}"
                exit 1
            }
            ;;
        macos)
            xcode-select --install 2>/dev/null || true
            ;;
    esac
    command -v git &>/dev/null && echo "  ✅ git installed" || { echo "${RED}  ❌ git still not found${RESET}"; exit 1; }
fi

# python3
if ! command -v python3 &>/dev/null; then
    echo "  ${YELLOW}⚠ Python 3.10+ not found — installing...${RESET}"
    case "$PLATFORM" in
        macos)
            if command -v brew &>/dev/null; then
                brew install python@3.12
            else
                echo "${RED}  ❌ No Homebrew found. Install Python from https://python.org or install Homebrew first.${RESET}"
                exit 1
            fi
            ;;
        linux|wsl)
            sudo apt update -qq 2>/dev/null && sudo apt install -y -qq python3 python3-venv python3-pip 2>/dev/null || \
            sudo dnf install -y -q python3 python3-pip 2>/dev/null || \
            sudo pacman -S --noconfirm python python-pip 2>/dev/null || {
                echo "${RED}  ❌ Could not install Python automatically. Please install it and re-run.${RESET}"
                exit 1
            }
            ;;
        *)
            echo "${RED}  ❌ Please install Python 3.10+ from https://python.org and re-run.${RESET}"
            exit 1
            ;;
    esac
fi

# python3-venv (test actual creation, not just import)
if command -v python3 &>/dev/null; then
    VENV_TEST_DIR=$(mktemp -d)
    if ! python3 -m venv "$VENV_TEST_DIR/test_venv" &>/dev/null; then
        echo "  ${YELLOW}⚠ python3-venv not working — installing...${RESET}"
        case "$PLATFORM" in
            linux|wsl)
                sudo apt install -y -qq python3-venv 2>/dev/null || \
                sudo apt install -y -qq python3."${PYTHON_VERSION}"-venv 2>/dev/null || \
                sudo dnf install -y -q python3-venv 2>/dev/null || {
                    echo "${RED}  ❌ Could not install python3-venv. Please install it and re-run.${RESET}"
                    exit 1
                }
                ;;
            macos)
                echo "  ${YELLOW}⚠ venv broken — try: brew reinstall python@3.12${RESET}"
                ;;
        esac
    fi
    rm -rf "$VENV_TEST_DIR"
fi

echo "  ✅ git       $(git --version 2>/dev/null | awk '{print $3}')"
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "?")
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "  ✅ python    $PYTHON_VERSION"
else
    echo "${RED}  ❌ Python 3.10+ required, found $PYTHON_VERSION${RESET}"
    exit 1
fi
if python3 -m venv /tmp/guanaco_venv_test &>/dev/null; then
    rm -rf /tmp/guanaco_venv_test
    echo "  ✅ venv      ok"
else
    rm -rf /tmp/guanaco_venv_test
    echo "  ${YELLOW}⚠ venv      missing (will attempt install)${RESET}"
fi
echo ""

# ── macOS SSL cert fix ──
if [ "$PLATFORM" = "macos" ]; then
    SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())" 2>/dev/null || echo "")
    if [ -n "$SSL_CERT_FILE" ]; then
        export SSL_CERT_FILE
        mkdir -p "$INSTALL_DIR"
        # Append to env file (don't overwrite — it may have OLLAMA_API_KEY later)
        grep -q "^export SSL_CERT_FILE=" "$INSTALL_DIR/env" 2>/dev/null && \
            sed -i '' "s|^export SSL_CERT_FILE=.*|export SSL_CERT_FILE=$SSL_CERT_FILE|" "$INSTALL_DIR/env" 2>/dev/null || \
            echo "export SSL_CERT_FILE=$SSL_CERT_FILE" >> "$INSTALL_DIR/env"
        echo "  ✅ ssl_certs  $SSL_CERT_FILE"
    fi
fi
echo ""

# ── Step 2: Configuration ──
echo "${BOLD}━━━ Step 2: Configuration ━━━${RESET}"
echo ""

echo "  You'll need an Ollama Cloud API key."
echo "  Get one at: ${CYAN}https://ollama.com${RESET}"
echo ""

prompt OLLAMA_API_KEY "Enter your Ollama API key" ""

if [ -z "$OLLAMA_API_KEY" ]; then
    echo ""
    echo "${YELLOW}⚠ No API key provided. You can set it later with:${RESET}"
    echo "  guanaco setup"
    echo ""
fi

# ── Step 3: Port configuration with security warning ──
echo ""
echo "${BOLD}━━━ Step 3: Network configuration ━━━${RESET}"
echo ""
echo -e "  ${YELLOW}⚠ Guanaco will start a server on port ${DEFAULT_PORT}.${RESET}"
echo ""
echo -e "  ${RED}╔══════════════════════════════════════════════════════════════╗${RESET}"
echo -e "  ${RED}║  ⚠  SECURITY WARNING                                       ║${RESET}"
echo -e "  ${RED}║                                                             ║${RESET}"
echo -e "  ${RED}║  If your machine has automatic port forwarding (some VPS   ║${RESET}"
echo -e "  ${RED}║  providers, routers with UPnP, Cloudflare tunnels, etc.),   ║${RESET}"
echo -e "  ${RED}║  this will EXPOSE your Ollama API proxy to the public      ║${RESET}"
echo -e "  ${RED}║  internet. Anyone who finds it can use your API key and    ║${RESET}"
echo -e "  ${RED}║  consume your Ollama Cloud quota.                          ║${RESET}"
echo -e "  ${RED}║                                                             ║${RESET}"
echo -e "  ${RED}║  • Bind to 127.0.0.1 unless you need remote access         ║${RESET}"
echo -e "  ${RED}║  • Use a firewall or auth proxy if you must expose it      ║${RESET}"
echo -e "  ${RED}║  • Never run this on a public-facing VPS without auth      ║${RESET}"
echo -e "  ${RED}╚══════════════════════════════════════════════════════════════╝${RESET}"
echo ""

prompt PORT "Which port should Guanaco use" "$DEFAULT_PORT"

prompt_yesno BIND_LOCAL "Bind to localhost only (127.0.0.1)?" "y"

if [ "$BIND_LOCAL" = "y" ]; then
    BIND_HOST="127.0.0.1"
else
    # Check for Tailscale — if found, suggest binding to 0.0.0.0 (Tailscale provides auth)
    TS_IP=$(tailscale ip -4 2>/dev/null || true)
    if [ -n "$TS_IP" ]; then
        echo ""
        echo -e "  ${CYAN}🌐 Tailscale detected at ${TS_IP}${RESET}"
        echo -e "  ${CYAN}   Binding to 0.0.0.0 — Tailscale provides authentication.${RESET}"
        echo ""
    else
        echo ""
        echo -e "  ${RED}⚠ Binding to 0.0.0.0 — Guanaco will be accessible from ALL network interfaces.${RESET}"
        echo -e "  ${RED}  Make sure you have a firewall or authentication layer in place.${RESET}"
        echo ""
    fi
    BIND_HOST="0.0.0.0"
fi

echo ""

# ── Step 4: Install ──
echo "${BOLD}━━━ Step 4: Installing ━━━${RESET}"
echo ""

# Clone or update
if [ -d "$INSTALL_DIR/repo" ]; then
    echo "  📦 Updating existing installation..."
    cd "$INSTALL_DIR/repo"
    git pull --ff-only || { echo "  ${YELLOW}⚠ Could not pull updates. Using existing version.${RESET}"; }
else
    echo "  📦 Downloading Guanaco..."
    git clone "https://github.com/$REPO.git" "$INSTALL_DIR/repo"
    cd "$INSTALL_DIR/repo"
fi

# Create venv
echo "  🐍 Setting up virtual environment..."
python3 -m venv "$VENV_DIR"

# Source platform env if exists
if [ -f "$INSTALL_DIR/env" ]; then
    source "$INSTALL_DIR/env"
fi

# Install
echo "  📥 Installing dependencies..."
"$VENV_DIR/bin/pip" install -e . --quiet 2>&1 | tail -1

# ── Write config ──
echo "  ⚙️  Writing configuration..."

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
    cat > "$INSTALL_DIR/env" << EOF
$(grep -v "^export OLLAMA_API_KEY=" "$INSTALL_DIR/env" 2>/dev/null || true)
export OLLAMA_API_KEY="${OLLAMA_API_KEY}"
EOF
fi

# ── Create guanaco binary ──
mkdir -p "$BIN_DIR"

case "$PLATFORM" in
    macos)
        cat > "$BIN_DIR/guanaco" << SCRIPT
#!/usr/bin/env bash
if [ -f "$INSTALL_DIR/env" ]; then
    source "$INSTALL_DIR/env"
fi
exec "$VENV_DIR/bin/python" -m guanaco.cli "\$@"
SCRIPT
        ;;
    *)
        cat > "$BIN_DIR/guanaco" << SCRIPT
#!/usr/bin/env bash
source "$INSTALL_DIR/env" 2>/dev/null || true
exec "$VENV_DIR/bin/python" -m guanaco.cli "\$@"
SCRIPT
        ;;
esac
chmod +x "$BIN_DIR/guanaco"

# Also create an oct alias for backward compat
ln -sf "$BIN_DIR/guanaco" "$BIN_DIR/oct" 2>/dev/null || true

# ── Add to PATH if needed ──
DETECT_SHELL="${SHELL##*/}"
case "$DETECT_SHELL" in
    zsh) PROFILE_FILE="$HOME/.zshrc" ;;
    bash) PROFILE_FILE="$HOME/.bashrc" ;;
    *) PROFILE_FILE="$HOME/.profile" ;;
esac

if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    echo ""
    echo "  ${YELLOW}⚠ $BIN_DIR is not in your PATH.${RESET}"
    echo "  Adding to $PROFILE_FILE..."
    echo "" >> "$PROFILE_FILE"
    echo "# Added by Guanaco" >> "$PROFILE_FILE"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$PROFILE_FILE"
    export PATH="$BIN_DIR:$PATH"
    echo "  ${GREEN}✅ Added. Reload your shell: source $PROFILE_FILE${RESET}"
fi

# ── Platform-specific tips ──
echo ""
echo "${BOLD}━━━ Installation complete! ━━━${RESET}"
echo ""
echo "  ${GREEN}✅ Guanaco installed successfully${RESET}"
echo ""
echo "  ${BOLD}Start the server:${RESET}"
echo "    guanaco start"
echo ""
echo "  ${BOLD}Dashboard:${RESET}"
echo "    http://${BIND_HOST}:${PORT}/dashboard"
echo ""
echo "  ${BOLD}CLI commands:${RESET}"
echo "    guanaco status         Show service & connection status"
echo "    guanaco models         List available cloud models"
echo "    guanaco usage          Check your Ollama Cloud usage/quota"
echo "    guanaco analytics      View request analytics & stats"
echo "    guanaco key generate   Generate an API key"
echo "    guanaco config --show  Show current configuration"
echo "    guanaco setup          Reconfigure (API key, ports, etc.)"
echo ""

# ── macOS auto-start tip ──
if [ "$PLATFORM" = "macos" ]; then
    echo "  ${DIM}🍎 macOS: To auto-start on login, see contrib/com.guanaco.start.plist${RESET}"
    echo ""
fi

# ── WSL tip ──
if [ "$PLATFORM" = "wsl" ]; then
    WIN_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
    echo "  ${DIM}🪟 WSL: Access dashboard from Windows at http://${WIN_IP}:${PORT}/dashboard${RESET}"
    echo ""
fi

echo "  ${DIM}Docs: https://github.com/$REPO${RESET}"
echo ""