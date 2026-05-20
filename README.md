# 🚀 AI Autonomous Trading Architecture

> **Dự án xây dựng hệ thống giao dịch crypto tự động sử dụng Multi-Agent LLM Debate.**
>
> ⚠️ **Cảnh báo:** Dự án này ở giai đoạn R&D. Không sử dụng vốn thật khi chưa hoàn thiện Phase 3.

## 🎯 Mục tiêu
Xây dựng hệ thống AI có khả năng:
1. Tự thu thập & phân tích dữ liệu thị trường real-time.
2. Tự tranh luận (Bull vs Bear vs Risk Manager) trước khi ra quyết định.
3. Tự khớp lệnh trên sàn Binance (hoặc CEX khác) qua API.
4. Tự học hỏi từ kết quả giao dịch để cải thiện qua thời gian.

## 📂 Cấu trúc Repo
```
ai-trading-architecture/
├── README.md               # Tài liệu này
├── AGENTS.md               # Hướng dẫn cho AI agents (Claude/Codex)
├── ARCHITECTURE.md         # Chi tiết kiến trúc hệ thống
├── TECH_STACK.md           # Công nghệ, version, setup guide
├── PHASE1_PLAN.md          # Kế hoạch chi tiết Phase 1 (Data + MVP)
├── src/                    # Source code (sẽ được tạo)
│   ├── data/               # Data pipeline (Cryptofeed, WebSocket)
│   ├── debate/             # AI Debate Engine (LangGraph, DSPy)
│   ├── execution/          # Execution Layer (CCXT, Lumibot)
│   ├── risk/               # Risk Management & Kill Switch
│   ├── memory/             # Mem0, Trade Logging, PostgreSQL
│   └── monitoring/         # Telegram Bot, Streamlit, Grafana
└── tests/                  # Unit & Integration tests
```

## 🚦 Trạng thái hiện tại
| Phase | Trạng thái | Files | Mô tả |
|---|---|---|---|
| **Phase 1: Data & MVP** | ✅ COMPLETE | ~25 files | Data pipeline, strategies, backtesting, execution, risk, monitoring |
| **Phase 2: AI Brain** | ✅ COMPLETE | ~15 files | Multi-agent debate (LangGraph), DSPy optimizer, memory |
| **Phase 3: Production** | ✅ COMPLETE | ~8 files | Streamlit dashboard, Docker, Grafana, reports, Makefile |
| **Phase 4: Self-Learning** | ✅ COMPLETE | ~6 files | Mem0, knowledge graph, news sentiment, auto-tuner |

## 📂 Source Code (Phase 1 — Đã triển khai)
```
ai-trading-architecture/
├── docker-compose.yml          # Redis, QuestDB, PostgreSQL, Qdrant
├── pyproject.toml              # Project config + dependencies
├── requirements.txt            # Pip requirements
├── config/settings.yaml        # Trading/strategy/risk settings
│
├── src/
│   ├── main.py                 # CLI entry point (Phase 1)
│   ├── main_ai.py              # AI-powered trading entry point (Phase 2) ⭐
│   ├── config.py               # Config loader (.env + yaml)
│   ├── data/                   # Data Pipeline Layer
│   │   ├── binance_connector.py    # Cryptofeed + Binance WS
│   │   ├── questdb_writer.py       # QuestDB ILP writer
│   │   ├── redis_cache.py          # Redis hot cache + pub/sub
│   │   ├── quality_gates.py        # Latency, spread, spike detection
│   │   └── config.py               # Data pipeline config
│   ├── strategy/               # Trading Strategy Layer
│   │   ├── base.py                 # Abstract base strategy
│   │   ├── sma_cross.py            # SMA crossover + RSI filter
│   │   ├── bbands.py               # Bollinger Bands + volume
│   │   ├── backtest.py             # Backtest runner (CCXT)
│   │   └── metrics.py              # Sharpe, Sortino, drawdown, etc.
│   ├── execution/              # Execution Layer
│   │   ├── exchange_client.py      # CCXT Binance wrapper
│   │   ├── order_manager.py        # Market/limit/SL/TP/bracket orders
│   │   ├── dry_run.py              # Simulated trading executor
│   │   └── position_sizer.py       # Half-Kelly + Van Tharp
│   ├── risk/                   # Risk Management Layer
│   │   ├── risk_engine.py          # Pre-trade checks, daily loss
│   │   └── kill_switch.py          # Kill switch state machine
│   ├── debate/                 # AI Debate Engine Layer ⭐
│   │   ├── __init__.py             # Package exports
│   │   ├── debate_engine.py        # LangGraph StateGraph workflow
│   │   ├── agents.py               # Bull, Bear, Devil, Judge, Risk agents
│   │   ├── llm_client.py           # LiteLLM wrapper with fallback
│   │   ├── models.py               # Pydantic v2 models
│   │   ├── prompts.py              # System prompts for each agent
│   │   └── optimizer.py            # DSPy prompt optimizer ⭐
│   ├── memory/                 # Memory & Learning Layer ⭐
│   │   ├── __init__.py             # Package exports
│   │   ├── trade_memory.py         # TradeMemory: PostgreSQL + Redis
│   │   └── weekly_review.py        # WeeklyReviewer: reports + insights
│   └── monitoring/             # Monitoring Layer
│       ├── telegram_bot.py         # PTB v21+ trading bot
│       ├── trading_logger.py       # Loguru + JSONL logging
│       └── alert_formatter.py      # HTML alert templates
│
├── tests/                    # Unit tests
│   ├── data/test_quality_gates.py
│   ├── strategy/test_metrics.py
│   ├── execution/test_position_sizer.py
│   ├── execution/test_dry_run.py
│   ├── execution/test_risk_engine.py
│   ├── execution/test_kill_switch.py
│   └── monitoring/test_alert_formatter.py
│
└── docs/                     # Research docs
    ├── ARCHITECTURE.md           # System architecture (Phase 1+2)
    ├── research-data-pipeline.md
    └── research-execution.md
```

## 🤖 AI Trading Usage (Phase 2)

### Quick Start
```bash
# Activate virtual environment
source venv/bin/activate

# Run AI debate trading (dry-run)
python -m src.main_ai --mode dryrun --strategy ai_debate

# Run SMA Cross with AI debate confirmation
python -m src.main_ai --mode dryrun --strategy sma_cross

# Run a single debate without executing trades
python -m src.main_ai --debate-only --debate-symbol BTC/USDT

# Run backtest
python -m src.main_ai --backtest --backtest-days 90

# Run with DSPy optimization enabled
python -m src.main_ai --mode dryrun --strategy ai_debate --optimize
```

### AI Debate Engine (Programmatic)
```python
from src.debate import DebateEngine, DebateConfig
from src.debate.llm_client import LLMClient

llm = LLMClient(model="anthropic/claude-sonnet-4")
config = DebateConfig(max_rounds=3, symbols=["BTC/USDT"])
engine = DebateEngine(config, llm)

result = engine.run_debate(
    market_data={"price": 67500, "rsi": 45, "volume": 1000},
    symbol="BTC/USDT",
)
print(f"Action: {result.action}, Confidence: {result.confidence}%")
```

### Memory & Learning
```python
from src.memory import TradeMemory, WeeklyReviewer

async with TradeMemory() as memory:
    # Log a trade
    await memory.log_trade({
        "symbol": "BTC/USDT", "side": "BUY", "quantity": 0.01,
        "price": 67500.0, "strategy": "ai_debate", "mode": "dryrun",
    })

    # Get performance
    summary = await memory.get_performance_summary()
    print(f"Win rate: {summary.win_rate:.1f}%, P&L: ${summary.total_pnl:+,.2f}")

    # Weekly review
    reviewer = WeeklyReviewer(memory)
    report = await reviewer.generate_report()
    reviewer.save_report(report)

    # Extract DSPy optimization insights
    insights = await reviewer.extract_insights()
    for insight in insights:
        print(f"  → {insight}")
```

### DSPy Prompt Optimization
```python
from src.memory import TradeMemory
from src.debate import DebateConfig
from src.debate.optimizer import DSPyOptimizer

async with TradeMemory() as memory:
    optimizer = DSPyOptimizer(
        trade_memory=memory,
        debate_config=DebateConfig(),
        llm_model="anthropic/claude-sonnet-4",
    )

    # Setup and optimize
    program = optimizer.setup_program()
    optimized = await optimizer.optimize(metric="sharpe_ratio")
    optimizer.save_optimized_prompts(optimized)

    # Weekly auto-optimization
    result = await optimizer.weekly_review_cycle(min_trades=20)
    print(f"Optimized: {result['optimized']}, Reason: {result['reason']}")
```

## 🔗 Tài liệu tham khảo
- [TradingAgents (UCLA/Tauric)](https://github.com/TauricResearch/TradingAgents)
- [LLM-TradeBot (Adversarial)](https://github.com/EthanAlgoX/LLM-TradeBot)
- [Lumibot (Execution)](https://github.com/Lumiwealth/lumibot)
- [Cryptofeed](https://github.com/bmoscon/cryptofeed)
- [DSPy](https://github.com/stanfordnlp/dspy)
- [Mem0](https://github.com/mem0ai/mem0)
- [CCXT](https://github.com/ccxt/ccxt)
