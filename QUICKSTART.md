# 🚀 Quickstart Guide — AI Trading Architecture

## 1-Minute Setup

```bash
# 1. Copy environment file, edit with your keys
cp .env.example .env
# Edit .env — add API keys if needed

# 2. Start infrastructure (Redis, QuestDB, PostgreSQL, Qdrant, Grafana)
docker compose up -d

# 3. Install dependencies
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Or use Makefile:
make setup && make docker-up
```

## Running Modes

### Phase 1 — Data & Execution

```bash
# Collect real-time market data from Binance
python -m src.main --data-pipeline

# Dry-run trading (simulated, no real money)
python -m src.main --mode dryrun

# Backtest last 90 days
python -m src.main --backtest

# Backtest with specific strategy
python -m src.main --backtest --strategy bbands --backtest-days 180

# Start Telegram monitoring bot
python -m src.main --monitor
```

### Phase 2 — AI Brain

```bash
# AI debate-powered trading
python -m src.main_ai --strategy ai_debate

# SMA strategy with AI confirmation
python -m src.main_ai --strategy sma_cross --debate-verify

# DSPy prompt optimization
python -m src.main_ai --optimize

# Weekly review report
python -m src.main_ai --weekly-review
```

### Phase 3 — Production

```bash
# Streamlit dashboard (dark theme, 4 pages)
python -m src.main --monitor
streamlit run src/dashboard.py
# Open http://localhost:8501

# Grafana dashboard (P&L, metrics, bot health)
# Open http://localhost:3000 (admin/admin)

# Generate backtest HTML report
python -c "
from src.reports import BacktestReport
report = BacktestReport(results, 'SMA Cross', 'BTC/USDT')
report.generate_html('backtest_report.html')
"
```

### Phase 4 — Self-Learning

```bash
# Full unified bot (all modules)
python -m src.main_full --mode dryrun --strategy ai_debate

# Auto-tune optimization cycle
python -c "
from src.autotune import AutoTuner
from src.memory.trade_memory import TradeMemory
memory = TradeMemory()
tuner = AutoTuner(trade_memory=memory, strategy_name='sma_cross')
tuner.weekly_optimization_cycle()
"

# News sentiment analysis
python -c "
from src.data.news_pipeline import NewsPipeline
pipeline = NewsPipeline(symbols=['BTC-USDT', 'ETH-USDT'])
sentiment = pipeline.get_market_sentiment()
print(sentiment)
"
```

## Docker Production

```bash
# Build and run production image
docker compose -f docker-compose.prod.yml up -d

# View logs
docker compose -f docker-compose.prod.yml logs -f app

# Check service health
docker compose -f docker-compose.prod.yml ps
```

## Makefile Targets

```bash
make help          # List all targets
make setup         # Create venv + install deps
make docker-up     # Start all services
make docker-down   # Stop all services
make run-data      # Run data pipeline
make run-dryrun    # Paper trading
make run-ai        # AI debate trading
make run-dashboard # Streamlit dashboard
make test          # Run tests
make lint          # Lint code
make format        # Format code
make clean         # Clean caches
```

## Architecture Overview

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Binance WS  │────▶│ Data Pipeline│────▶│ Redis/QuestDB│
└─────────────┘     └──────────────┘     └─────────────┘
                                                │
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│  Execution  │◀────│  AI Debate   │◀────│  Strategy   │
│  (CCXT)     │     │  (LangGraph) │     │  (SMA/BB)   │
└─────────────┘     └──────────────┘     └─────────────┘
       │                    │                    │
       ▼                    ▼                    ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│ Risk Engine │     │  Memory/     │     │  News/      │
│ + Kill Sw.  │     │  Learning    │     │  Sentiment  │
└─────────────┘     └──────────────┘     └─────────────┘
```

## Project Structure

```
ai-trading-architecture/
├── src/                    # Source code (40 files, ~16,000 lines)
│   ├── main.py             # Phase 1 CLI
│   ├── main_ai.py          # Phase 2 AI CLI
│   ├── main_full.py        # Full unified bot
│   ├── dashboard.py        # Streamlit dashboard
│   ├── autotune.py         # Auto-optimization
│   ├── reports.py          # Backtest report generator
│   ├── data/               # Data pipeline (6 files)
│   ├── strategy/           # Strategies (6 files)
│   ├── execution/          # Execution (4 files)
│   ├── risk/               # Risk management (2 files)
│   ├── monitoring/         # Telegram + logging (3 files)
│   ├── debate/             # AI debate engine (6 files)
│   └── memory/             # Self-learning (4 files)
├── tests/                  # Unit tests (7 files)
├── config/                 # Settings + Grafana (7 files)
├── scripts/                # Setup scripts
├── Dockerfile              # Multi-stage build
├── docker-compose.yml      # Dev infrastructure
├── docker-compose.prod.yml # Production infrastructure
├── Makefile                # Build automation
├── requirements.txt        # Python dependencies
└── pyproject.toml          # Project configuration
```
