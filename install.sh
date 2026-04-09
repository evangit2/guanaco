#!/usr/bin/env bash
# ollama-cloud-tools installer
# Usage: curl -sSL https://raw.githubusercontent.com/evanrice/ollama-cloud-tools/main/install.sh | bash
#
# Supports: Linux, macOS, WSL (Windows Subsystem for Linux)

set -euo pipefail

REPO="evanrice/ollama-cloud-tools"
INSTALL_DIR="$HOME/.oct"
VENV_DIR="$INSTALL_DIR/venv"
BIN_DIR="$HOME/.local/bin"

# ── Detect platform ──
detect_platform() {
    local os_name="$(uname -s)"
    case "$os_name" in
        Linux)
            # Check if WSL
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
echo "🦙 Ollama Cloud Tools Installer"
echo "================================"
echo "Platform: $PLATFORM"
echo ""

# ── Check Python ──
if ! command -v python3 &>/dev/null; then
    echo "❌ Python 3.10+ is required but not found."
    case "$PLATFORM" in
        macos)
            echo "   Install with: brew install python@3.11"
            echo "   Or download from: https://python.org"
            ;;
        linux|wsl)
            echo "   Install with:"
            echo "     Ubuntu/Debian: sudo apt install python3 python3-venv python3-pip"
            echo "     Fedora/RHEL:   sudo dnf install python3 python3-pip"
            echo "     Arch:           sudo pacman -S python python-pip"
            ;;
        *)
            echo "   Install from: https://python.org"
            ;;
    esac
    exit 1
fi

PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
if python3 -c "import sys; exit(0 if sys.version_info >= (3, 10) else 1)"; then
    echo "✅ Python $PYTHON_VERSION found"
else
    echo "❌ Python 3.10+ required, found $PYTHON_VERSION"
    exit 1
fi

# ── Platform-specific setup ──
case "$PLATFORM" in
    macos)
        # Check for Homebrew
        if ! command -v brew &>/dev/null; then
            echo "⚠️  Homebrew not found. Some dependencies may need manual install."
            echo "   Install Homebrew: /bin/bash -c \"\$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)\""
        fi

        # macOS needs certifi for SSL
        echo "🍎 Setting up macOS SSL certificates..."
        export SSL_CERT_FILE=$(python3 -c "import certifi; print(certifi.where())" 2>/dev/null || echo "")
        if [ -n "$SSL_CERT_FILE" ]; then
            echo "   SSL certs: $SSL_CERT_FILE"
            # Write to env file for persistence
            mkdir -p "$INSTALL_DIR"
            echo "export SSL_CERT_FILE=$SSL_CERT_FILE" > "$INSTALL_DIR/env"
        fi

        # Check for Xcode CLI tools (needed for some pip builds)
        if ! xcode-select -p &>/dev/null; then
            echo "⚠️  Xcode CLI tools not found. Installing..."
            xcode-select --install 2>/dev/null || true
            echo "   You may need to restart the installer after Xcode tools install."
        fi
        ;;

    wsl)
        echo "🐧 Windows Subsystem for Linux detected"
        # WSL should work like Linux but check for common issues
        if ! command -v git &>/dev/null; then
            echo "⚠️  git not found. Installing..."
            sudo apt update -qq && sudo apt install -y -qq git 2>/dev/null || {
                echo "   Could not install git. Please install manually:"
                echo "     sudo apt install git"
            }
        fi

        # WSL2 may need resolv.conf fix for DNS
        if ! ping -c1 -W2 ollama.com &>/dev/null; then
            echo "⚠️  Cannot reach ollama.com — DNS may need fixing for WSL2"
            echo "   Try: echo 'nameserver 8.8.8.8' | sudo tee /etc/resolv.conf"
        fi

        # Set up Windows browser access
        WIN_IP=$(hostname -I 2>/dev/null | awk '{print $1}')
        echo "   Access dashboard from Windows: http://$WIN_IP:8080/dashboard"
        ;;

    linux)
        # Standard Linux — check for venv support
        if ! python3 -c "import venv" &>/dev/null; then
            echo "⚠️  Python venv module not found. Installing..."
            sudo apt install -y python3-venv 2>/dev/null || \
            sudo dnf install -y python3-venv 2>/dev/null || {
                echo "   Could not install python3-venv automatically."
                echo "   Please install it manually and re-run."
                exit 1
            }
        fi
        ;;
esac

# ── Clone or update repo ──
if [ -d "$INSTALL_DIR/repo" ]; then
    echo "📦 Updating existing installation..."
    cd "$INSTALL_DIR/repo"
    git pull --ff-only || { echo "⚠️  Could not pull updates. Continuing with existing version."; }
else
    echo "📦 Cloning repository..."
    git clone "https://github.com/$REPO.git" "$INSTALL_DIR/repo"
    cd "$INSTALL_DIR/repo"
fi

# ── Create venv ──
echo "🐍 Creating virtual environment..."
python3 -m venv "$VENV_DIR"

# ── Source platform env if exists ──
if [ -f "$INSTALL_DIR/env" ]; then
    source "$INSTALL_DIR/env"
fi

# ── Install ──
echo "📥 Installing dependencies..."
"$VENV_DIR/bin/pip" install -e . --quiet

# ── Create oct binary ──
mkdir -p "$BIN_DIR"
case "$PLATFORM" in
    macos)
        # macOS wrapper that sets SSL certs
        cat > "$BIN_DIR/oct" << SCRIPT
#!/usr/bin/env bash
if [ -f "$INSTALL_DIR/env" ]; then
    source "$INSTALL_DIR/env"
fi
exec "$VENV_DIR/bin/python" -m oct.cli "\$@"
SCRIPT
        ;;
    *)
        # Standard wrapper
        cat > "$BIN_DIR/oct" << 'SCRIPT'
#!/usr/bin/env bash
exec "$HOME/.oct/venv/bin/python" -m oct.cli "$@"
SCRIPT
        ;;
esac
chmod +x "$BIN_DIR/oct"

# ── Shell profile detection ──
DETECT_SHELL="${SHELL##*/}"
case "$DETECT_SHELL" in
    zsh) PROFILE_FILE="$HOME/.zshrc" ;;
    bash) PROFILE_FILE="$HOME/.bashrc" ;;
    *) PROFILE_FILE="$HOME/.profile" ;;
esac

# ── Add to PATH if needed ──
if [[ ":$PATH:" != *":$BIN_DIR:"* ]]; then
    echo ""
    echo "⚠️  $BIN_DIR is not in your PATH."
    echo "   Adding to $PROFILE_FILE..."
    echo "" >> "$PROFILE_FILE"
    echo "# Added by ollama-cloud-tools" >> "$PROFILE_FILE"
    echo 'export PATH="$HOME/.local/bin:$PATH"' >> "$PROFILE_FILE"

    # If macOS with Homebrew, add to /etc/paths.d too
    if [ "$PLATFORM" = "macos" ] && [ -w /usr/local/bin ]; then
        ln -sf "$BIN_DIR/oct" /usr/local/bin/oct 2>/dev/null || true
    fi

    echo ""
    echo "   Reload your shell: source $PROFILE_FILE"
fi

# ── macOS-specific: create LaunchAgent for auto-start ──
if [ "$PLATFORM" = "macos" ]; then
    echo ""
    echo "🍎 macOS Tip: To auto-start oct on login, create a LaunchAgent:"
    echo "   cp $INSTALL_DIR/repo/contrib/com.oct.start.plist ~/Library/LaunchAgents/"
    echo "   launchctl load ~/Library/LaunchAgents/com.oct.start.plist"
fi

# ── WSL-specific: create startup script ──
if [ "$PLATFORM" = "wsl" ]; then
    cat > "$INSTALL_DIR/start.sh" << 'WSLSCRIPT'
#!/usr/bin/env bash
# WSL startup script for oct
source "$HOME/.oct/env" 2>/dev/null || true
export PATH="$HOME/.local/bin:$PATH"
exec oct start "$@"
WSLSCRIPT
    chmod +x "$INSTALL_DIR/start.sh"
    echo "🪟 WSL Tip: Run oct via: $INSTALL_DIR/start.sh"
fi

# ── Done ──
echo ""
echo "✅ Installation complete! (Platform: $PLATFORM)"
echo ""
echo "Next steps:"
echo "   1. Run 'oct setup' to configure your Ollama API key"
echo "   2. Run 'oct start' to launch the services"
echo "   3. Visit http://localhost:8080/dashboard for the web UI"
echo ""
echo "CLI commands:"
echo "   oct models           List available cloud models"
echo "   oct models --caps    Show model capabilities"
echo "   oct usage            Check your Ollama Cloud usage/quota"
echo "   oct status           Show service & connection status"
echo "   oct status -v        Verbose status with endpoint info"
echo "   oct analytics        View request analytics & stats"
echo "   oct analytics -m MODEL  History for a specific model"
echo "   oct key generate     Generate an API key"
echo "   oct config --show    Show current configuration"
echo ""
echo "Docs: https://github.com/$REPO"