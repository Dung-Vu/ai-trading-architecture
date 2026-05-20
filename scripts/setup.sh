#!/usr/bin/env bash
# =============================================================================
# setup.sh — AI Trading Architecture: One-Command Setup
# =============================================================================
# Usage:
#   ./scripts/setup.sh          # Full setup (venv + docker)
#   ./scripts/setup.sh --venv   # Python venv only
#   ./scripts/setup.sh --docker # Docker services only
#   ./scripts/setup.sh --check  # Health check only
# =============================================================================

set -euo pipefail

# ── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

info()  { echo -e "${BLUE}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ── Paths ───────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

ENV_FILE="$PROJECT_DIR/.env"
ENV_EXAMPLE="$PROJECT_DIR/.env.example"
VENV_DIR="$PROJECT_DIR/venv"
REQ_FILE="$PROJECT_DIR/requirements.txt"
COMPOSE_FILE="$PROJECT_DIR/docker-compose.yml"

# ── Helpers ─────────────────────────────────────────────────────────────────

check_command() {
    if ! command -v "$1" &>/dev/null; then
        error "$1 is not installed. Please install it first."
        exit 1
    fi
}

wait_for_service() {
    local container="$1"
    local max_wait="${2:-60}"
    local interval=3
    local elapsed=0

    info "Waiting for $container to be healthy..."
    while [ $elapsed -lt $max_wait ]; do
        local status
        status=$(docker inspect --format='{{.State.Health.Status}}' "$container" 2>/dev/null || echo "not_found")
        if [ "$status" = "healthy" ]; then
            ok "$container is healthy"
            return 0
        fi
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done
    warn "$container did not become healthy within ${max_wait}s (status: $status)"
    return 1
}

# ── Setup Functions ─────────────────────────────────────────────────────────

setup_env() {
    info "Checking .env file..."
    if [ ! -f "$ENV_FILE" ]; then
        if [ -f "$ENV_EXAMPLE" ]; then
            cp "$ENV_EXAMPLE" "$ENV_FILE"
            ok "Created .env from .env.example"
            warn "Please edit .env and fill in your API keys before running the bot"
        else
            error ".env.example not found. Create a .env file manually."
            exit 1
        fi
    else
        ok ".env already exists — skipping"
    fi
}

setup_venv() {
    info "Setting up Python virtual environment..."

    if [ ! -f "$VENV_DIR/bin/python" ]; then
        python3 -m venv "$VENV_DIR"
        ok "Virtual environment created at $VENV_DIR"
    else
        ok "Virtual environment already exists"
    fi

    # Activate
    # shellcheck disable=SC1091
    source "$VENV_DIR/bin/activate"
    ok "Virtual environment activated"

    # Upgrade pip
    info "Upgrading pip..."
    pip install --upgrade pip -q

    # Install requirements
    if [ -f "$REQ_FILE" ]; then
        info "Installing dependencies from $REQ_FILE..."
        pip install -r "$REQ_FILE" -q
        ok "Dependencies installed"
    else
        warn "requirements.txt not found — skipping pip install"
    fi
}

setup_docker() {
    info "Starting Docker Compose services..."

    if ! command -v docker &>/dev/null; then
        error "Docker is not installed. Please install Docker first."
        exit 1
    fi

    if ! command -v docker-compose &>/dev/null && ! docker compose version &>/dev/null; then
        error "Docker Compose is not installed."
        exit 1
    fi

    # Use `docker compose` (v2) or `docker-compose` (v1)
    COMPOSE_CMD="docker compose"
    if ! $COMPOSE_CMD version &>/dev/null; then
        COMPOSE_CMD="docker-compose"
    fi

    # Pull latest images
    info "Pulling latest images..."
    $COMPOSE_CMD pull --quiet 2>/dev/null || warn "Some images failed to pull — using cached versions"

    # Start services
    info "Starting services..."
    $COMPOSE_CMD up -d --remove-orphans

    ok "Services started"

    # Wait for health checks
    info "Waiting for services to become healthy..."
    sleep 5  # Initial grace period

    wait_for_service "trading-redis" 30 || true
    wait_for_service "trading-questdb" 60 || true
    wait_for_service "trading-postgres" 30 || true
    wait_for_service "trading-qdrant" 30 || true
    wait_for_service "trading-grafana" 30 || true

    ok "All services are up"
}

check_health() {
    info "Service health check:"
    echo ""

    local services=("trading-redis" "trading-questdb" "trading-postgres" "trading-qdrant" "trading-grafana")
    local ports=("6379" "9000" "5432" "6333" "3000")
    local labels=("Redis" "QuestDB" "PostgreSQL" "Qdrant" "Grafana")

    for i in "${!services[@]}"; do
        local status
        status=$(docker inspect --format='{{.State.Health.Status}}' "${services[$i]}" 2>/dev/null || echo "not_found")
        local icon="❌"
        [ "$status" = "healthy" ] && icon="✅"
        [ "$status" = "starting" ] && icon="⏳"
        printf "  %s %-15s port %-6s %s\n" "$icon" "${labels[$i]}" "${ports[$i]}:" "${status}"
    done

    echo ""
}

print_success() {
    echo ""
    echo -e "${GREEN}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${GREEN}║          ✅  Setup Complete — Next Steps  ✅            ║${NC}"
    echo -e "${GREEN}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""
    echo -e "${BLUE}1. Configure your API keys:${NC}"
    echo "   nano .env"
    echo ""
    echo -e "${BLUE}2. Activate the virtual environment:${NC}"
    echo "   source venv/bin/activate"
    echo ""
    echo -e "${BLUE}3. Run the bot (dry-run mode):${NC}"
    echo "   python src/main.py --mode dryrun --config config/settings.yaml"
    echo ""
    echo -e "${BLUE}4. Run backtests:${NC}"
    echo "   python -c \"from strategy.backtest import BacktestRunner; ...\""
    echo ""
    echo -e "${BLUE}5. Optimize strategy parameters:${NC}"
    echo "   python -c \"from strategy.optimizer import ParameterOptimizer; ...\""
    echo ""
    echo -e "${BLUE}6. Open dashboards:${NC}"
    echo "   Grafana:    http://localhost:3000  (admin/admin)"
    echo "   QuestDB:    http://localhost:9000"
    echo "   Qdrant:     http://localhost:6333/dashboard"
    echo ""
    echo -e "${BLUE}7. Run tests:${NC}"
    echo "   pytest tests/ -v"
    echo ""
    echo -e "${YELLOW}⚠  Reminder: Do NOT use real API keys in dry-run mode.${NC}"
    echo -e "${YELLOW}   Always test on testnet before going live.${NC}"
    echo ""
}

# ── Main ────────────────────────────────────────────────────────────────────

main() {
    echo ""
    echo -e "${BLUE}╔══════════════════════════════════════════════════════════╗${NC}"
    echo -e "${BLUE}║     🤖  AI Trading Architecture — Setup Script          ║${NC}"
    echo -e "${BLUE}╚══════════════════════════════════════════════════════════╝${NC}"
    echo ""

    local mode="full"
    for arg in "$@"; do
        case "$arg" in
            --venv)    mode="venv" ;;
            --docker)  mode="docker" ;;
            --check)   mode="check" ;;
            --help|-h)
                echo "Usage: $0 [--venv|--docker|--check|--help]"
                exit 0
                ;;
            *)
                warn "Unknown argument: $arg"
                ;;
        esac
    done

    case "$mode" in
        full)
            setup_env
            setup_venv
            setup_docker
            check_health
            print_success
            ;;
        venv)
            setup_env
            setup_venv
            ok "Python environment ready"
            ;;
        docker)
            setup_docker
            check_health
            ;;
        check)
            check_health
            ;;
    esac
}

main "$@"
