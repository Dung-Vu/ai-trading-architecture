# 🏗️ AI Trading Architecture — System Design

> Comprehensive architecture documentation for Phase 1 (Data & MVP) and Phase 2 (AI Brain).

---

## 📋 Table of Contents
1. [High-Level Architecture](#high-level-architecture)
2. [Phase 1: Data & MVP Architecture](#phase-1-data--mvp-architecture)
3. [Phase 2: AI Brain Architecture](#phase-2-ai-brain-architecture)
4. [LangGraph Debate Flow](#langgraph-debate-flow)
5. [Memory Layers](#memory-layers)
6. [Data Flow Diagram](#data-flow-diagram)
7. [Component Interactions](#component-interactions)

---

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                     AI TRADING ARCHITECTURE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐        │
│  │  Data    │──▶│ Strategy │──▶│  Debate  │──▶│  Risk    │        │
│  │ Pipeline │   │  Layer   │   │  Engine  │   │ Manager  │        │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘        │
│       │              │              │              │                │
│       ▼              ▼              ▼              ▼                │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐        │
│  │  Redis   │   │  Lumibot │   │ LangGraph│   │  Limits  │        │
│  │  QuestDB │   │  CCXT    │   │ + DSPy   │   │  Checks  │        │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘        │
│                                                      │              │
│       ┌──────────┐   ┌──────────┐                    │              │
│       │ Telegram │◀──│Execution │◀───────────────────┘              │
│       │ Grafana  │   │  Layer   │                                    │
│       └──────────┘   └──────────┘                                    │
│                     │                                                │
│                     ▼                                                │
│              ┌──────────────┐                                        │
│              │   Memory &   │                                        │
│              │   Learning   │◀─── Weekly Review + DSPy Optimization   │
│              │   Layer      │                                        │
│              └──────────────┘                                        │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Phase 1: Data & MVP Architecture

### 6-Layer Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Layer 1: Data Pipeline                                         │
│  ┌─────────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ Cryptofeed WS   │─▶│ Quality Gates│─▶│ Redis Cache       │  │
│  │ Binance Connect │  │ (latency,    │  │ price:latest:{sym}│  │
│  │                 │  │  spread,     │  │ ticker:{sym}      │  │
│  │                 │  │  z-score)    │  │ pub/sub channels  │  │
│  └─────────────────┘  └──────────────┘  └───────────────────┘  │
│                              │                                  │
│                              ▼                                  │
│                        ┌──────────────┐                         │
│                        │  QuestDB     │                         │
│                        │  (time-series│                         │
│                        │   storage)   │                         │
│                        └──────────────┘                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Layer 2: Strategy                                              │
│  ┌─────────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ BaseStrategy    │  │ SMACross     │  │ BBands            │  │
│  │ (Lumibot)       │  │ (SMA 20/50   │  │ (Bollinger + vol)│  │
│  │ + indicators    │  │  + RSI)      │  │                   │  │
│  └─────────────────┘  └──────────────┘  └───────────────────┘  │
│                                                                 │
│  ┌─────────────────┐  ┌──────────────┐                         │
│  │ BacktestRunner  │  │ MetricsCalc  │                         │
│  │ (historical     │  │ (Sharpe,     │                         │
│  │  simulation)    │  │  drawdown)   │                         │
│  └─────────────────┘  └──────────────┘                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Layer 3: Risk Management                                       │
│  ┌─────────────────┐  ┌──────────────┐                         │
│  │ RiskEngine      │  │ KillSwitch   │                         │
│  │ - Daily loss    │  │ - State      │                         │
│  │ - Drawdown      │  │   machine    │                         │
│  │ - Position size │  │ - Manual     │                         │
│  │ - Leverage      │  │   / Auto     │                         │
│  └─────────────────┘  └──────────────┘                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Layer 4: Execution                                             │
│  ┌─────────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ ExchangeClient  │  │ OrderManager │  │ DryRunExecutor    │  │
│  │ (CCXT Binance)  │  │ (market,     │  │ (simulated       │  │
│  │                 │  │  limit, SL/TP│  │  trading)        │  │
│  └─────────────────┘  └──────────────┘  └───────────────────┘  │
│                              │                                  │
│                        ┌──────────────┐                         │
│                        │PositionSizer │                         │
│                        │(Half-Kelly,  │                         │
│                        │ Van Tharp)   │                         │
│                        └──────────────┘                         │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Layer 5: Monitoring                                            │
│  ┌─────────────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │ TelegramBot     │  │ TradingLogger│  │ AlertFormatter    │  │
│  │ (PTB v21+,      │  │ (loguru,     │  │ (HTML templates   │  │
│  │  async)         │  │  JSONL)      │  │  for alerts)      │  │
│  └─────────────────┘  └──────────────┘  └───────────────────┘  │
│                                                                 │
│  Grafana dashboards (PostgreSQL + QuestDB datasources)          │
└─────────────────────────────────────────────────────────────────┘
```

---

## Phase 2: AI Brain Architecture

### AI Debate Engine Architecture

The Phase 2 AI Brain adds three new subsystems:

```
┌─────────────────────────────────────────────────────────────────┐
│  AI Debate Engine (LangGraph)                                   │
│                                                                  │
│  Market Data ──▶ Bull Agent ──▶ Bear Agent ──▶ Devil's Advocate │
│                    ▲               │               │            │
│                    └──── Round N ──┘               │            │
│                                                    │            │
│                                          ┌─────────▼──────┐    │
│                                          │    Judge       │    │
│                                          │  (Synthesis)   │    │
│                                          └─────────┬──────┘    │
│                                                    │            │
│                                          ┌─────────▼──────┐    │
│                                          │ Risk Manager   │    │
│                                          │  (Final check) │    │
│                                          └─────────┬──────┘    │
│                                                    │            │
│                                          ┌─────────▼──────┐    │
│                                          │ DebateResult   │    │
│                                          │ (action, conf, │    │
│                                          │  SL, TP)       │    │
│                                          └────────────────┘    │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  Memory & Learning Layer                                        │
│                                                                  │
│  ┌──────────────┐    ┌─────────────────┐    ┌───────────────┐  │
│  │ TradeMemory  │───▶│ PostgreSQL      │───▶│ TradeHistory  │  │
│  │ (logging)    │    │ (trades, debates│    │ (query, filter│  │
│  │              │    │  tables)        │    │  aggregate)   │  │
│  └──────┬───────┘    └─────────────────┘    └───────────────┘  │
│         │                                                       │
│         ▼                                                       │
│  ┌──────────────┐    ┌─────────────────┐    ┌───────────────┐  │
│  │ Redis Cache  │    │ WeeklyReviewer  │───▶│ Performance   │  │
│  │ (hot data,   │    │ (reports,       │    │ Reports       │  │
│  │  latest      │    │  insights)      │    │ (P&L, Sharpe, │  │
│  │  prices)     │    │                 │    │  win rate)    │  │
│  └──────────────┘    └─────────────────┘    └───────────────┘  │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│  DSPy Prompt Optimizer                                          │
│                                                                  │
│  Historical Trades ──▶ DSPy Demonstrations ──▶ MIPROv2          │
│                          (market_data →        Optimizer        │
│                           action, conf)         │               │
│                                                  ▼               │
│                                          ┌───────────────┐      │
│                                          │ Optimized     │      │
│                                          │ Prompts       │      │
│                                          │ (saved to     │      │
│                                          │  config/)     │      │
│                                          └───────────────┘      │
└─────────────────────────────────────────────────────────────────┘
```

---

## LangGraph Debate Flow

### StateGraph Architecture

The debate engine uses LangGraph's `StateGraph` for workflow orchestration:

```
                    ┌─────────────┐
                    │   START     │
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
              ┌────▶│  Bull Agent │────┐
              │     └──────┬──────┘    │
              │            │           │
              │            ▼           │
              │     ┌─────────────┐    │
              │     │  Bear Agent │    │
              │     └──────┬──────┘    │
              │            │           │
              │            ▼           │
              │     ┌─────────────┐    │
              │     │   Devil's   │    │
              │     │  Advocate   │    │
              │     └──────┬──────┘    │
              │            │           │
              │     ┌──────▼──────┐    │
              │     │ round < max?│───┘
              │     └──────┬──────┘
              │            │ No
              │            ▼
              │     ┌─────────────┐
              │     │   Judge     │
              │     │ (Synthesis) │
              │     └──────┬──────┘
              │            │
              │            ▼
              │     ┌─────────────┐
              └─────│ Risk Manager│
                    └──────┬──────┘
                           │
                           ▼
                    ┌─────────────┐
                    │     END     │
                    │(DebateResult│
                    └─────────────┘
```

### DebateState (TypedDict)

```python
class DebateState(TypedDict, total=False):
    # Input
    market_data: dict[str, Any]
    current_positions: dict[str, Any]
    portfolio: dict[str, Any]
    config: dict[str, Any]

    # Accumulated debate data
    current_round: int
    max_rounds: int
    debate_rounds: list[dict[str, Any]]

    # Per-round agent outputs
    bull_output: dict[str, Any]
    bear_output: dict[str, Any]
    devil_output: dict[str, Any]

    # Final outputs
    judge_output: dict[str, Any]
    risk_output: dict[str, Any]

    # Metadata
    symbol: str
    start_time: float
```

### Agent Roles & Anti-Sycophancy Rules

| Agent | Role | Temperature | Key Behavior |
|-------|------|-------------|--------------|
| **Bull** | Find BUY reasons | 0.7 | Optimistic, cites bullish indicators |
| **Bear** | Find SELL reasons | 0.7 | Pessimistic, cites bearish indicators |
| **Devil** | Challenge both | 0.77 | Skeptical, finds flaws in both arguments |
| **Judge** | Synthesize decision | 0.35 | Weighs all arguments, makes final call |
| **Risk Manager** | Approve/reject | 0.35 | Enforces risk limits on Judge's decision |

**Anti-Sycophancy Rules:**
1. Each round MUST provide NEW evidence.
2. MUST rebut the strongest opposing point.
3. Stance changes require explicit justification.
4. All claims MUST have specific numbers.
5. FORBIDDEN to agree unless genuinely convinced.

### DebateResult Output

```python
class DebateResult(BaseModel):
    action: str           # BUY, SELL, HOLD
    confidence: float     # 0-100
    reason: str           # Synthesized reasoning
    stop_loss: float      # SL price
    take_profit: float    # TP price
    bull_argument: str    # Best bull argument summary
    bear_argument: str    # Best bear argument summary
    devil_argument: str   # Devil's critique summary
    risk_decision: str    # APPROVE, REJECT, REDUCE, FLATTEN
    risk_reasoning: str   # Risk manager explanation
    rounds: list[DebateRound]  # Full debate transcript
    symbol: str           # Trading symbol
    metadata: dict        # Latency, tokens, etc.
```

---

## Memory Layers

### Layer 1: Hot Cache (Redis)

**Purpose:** Fast access to latest prices and recent trade state.

| Key Pattern | Type | TTL | Content |
|-------------|------|-----|---------|
| `price:latest:{symbol}` | Hash | 24h | Latest trade price, side, amount |
| `ticker:{symbol}` | Hash | 5min | Best bid/ask |
| `trade:latest:{symbol}` | Hash | 24h | Most recent trade details |
| `price:{symbol}` | Pub/Sub | N/A | Real-time price stream |

### Layer 2: Persistent Storage (PostgreSQL)

**Tables:**

```sql
-- trades table
CREATE TABLE trades (
    id              SERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol          VARCHAR(32) NOT NULL,
    side            VARCHAR(8) NOT NULL,    -- BUY or SELL
    quantity        DOUBLE PRECISION NOT NULL,
    price           DOUBLE PRECISION NOT NULL,
    pnl             DOUBLE PRECISION DEFAULT 0.0,
    pnl_pct         DOUBLE PRECISION DEFAULT 0.0,
    strategy        VARCHAR(64) DEFAULT 'unknown',
    mode            VARCHAR(16) DEFAULT 'dryrun',
    ai_confidence   DOUBLE PRECISION,
    debate_result   JSONB,                  -- Full debate context
    stop_loss       DOUBLE PRECISION,
    take_profit     DOUBLE PRECISION,
    order_id        VARCHAR(128),
    created_at      TIMESTAMPTZ DEFAULT NOW()
);

-- debates table
CREATE TABLE debates (
    id              SERIAL PRIMARY KEY,
    timestamp       TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    symbol          VARCHAR(32) NOT NULL,
    bull_arg        TEXT DEFAULT '',
    bear_arg        TEXT DEFAULT '',
    devil_arg       TEXT DEFAULT '',
    judge_action    VARCHAR(8) DEFAULT 'HOLD',
    judge_confidence DOUBLE PRECISION DEFAULT 50.0,
    risk_action     VARCHAR(16) DEFAULT 'APPROVE',
    risk_reasoning  TEXT DEFAULT '',
    rounds          INTEGER DEFAULT 0,
    latency_seconds DOUBLE PRECISION DEFAULT 0.0,
    created_at      TIMESTAMPTZ DEFAULT NOW()
);
```

### Layer 3: Analytics (TradeMemory)

**Query Methods:**
- `get_trade_history(symbol, start_date, end_date, limit)` — Filtered trade history
- `get_recent_trades(symbol, limit)` — Latest trades for a symbol
- `get_performance_summary()` — Aggregated metrics (win rate, Sharpe, drawdown)
- `get_strategy_performance()` — Performance by strategy
- `get_trade_patterns()` — Pattern detection (symbol+side, time-of-day, confidence correlation)

### Layer 4: Weekly Review (WeeklyReviewer)

**Report Sections:**
1. Performance Summary (P&L, win rate, Sharpe, drawdown)
2. Best & Worst Trades (with AI confidence analysis)
3. Strategy Comparison (ranked by total P&L)
4. Pattern Analysis (symbol+side, time-of-day, confidence correlation)
5. Reflection (What went well? What went poorly? What to change?)
6. Action Items (prioritized recommendations)

**Insight Extraction for DSPy:**
- Confidence calibration analysis
- Symbol-specific pattern recommendations
- Stop-loss/take-profit effectiveness
- Time-of-day optimization suggestions
- Overall performance directives

---

## Data Flow Diagram

### Full Trade Execution Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                    Trading Loop (every N seconds)                │
│                                                                  │
│  1. Redis Cache                                                  │
│     └── Get latest price for symbol                              │
│                                                                  │
│  2. Build Market Data                                            │
│     └── Combine price + indicators (RSI, MACD, BB, volume)       │
│                                                                  │
│  3. Run Strategy                                                 │
│     ├── ai_debate  → Signal from debate engine                   │
│     ├── sma_cross  → Traditional SMA crossover signal            │
│     └── bbands     → Bollinger Bands signal                      │
│                                                                  │
│  4. Run Debate Engine (if AI strategy or confirmation)           │
│     └── Bull → Bear → Devil → (repeat) → Judge → Risk Manager    │
│     └── Output: DebateResult (action, confidence, SL, TP)        │
│                                                                  │
│  5. Risk Engine Pre-Trade Check                                  │
│     ├── Daily loss limit                                         │
│     ├── Drawdown limit                                           │
│     ├── Position concentration                                   │
│     └── Leverage limit                                           │
│     └── Output: (approved, reason)                               │
│                                                                  │
│  6. Execute Order                                                │
│     ├── dryrun → DryRunExecutor (simulated)                      │
│     ├── testnet → CCXT Binance Testnet                           │
│     └── live → CCXT Binance (production)                         │
│                                                                  │
│  7. Log to Memory                                                │
│     ├── TradeMemory.log_trade() → PostgreSQL trades table        │
│     └── TradeMemory.log_debate() → PostgreSQL debates table      │
│                                                                  │
│  8. Send Telegram Alert                                          │
│     └── Formatted trade notification with AI confidence          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Weekly Review Cycle

```
Every 7 days (or on-demand):

┌──────────────────────────────────────────────────────┐
│  WeeklyReviewer.generate_report()                    │
│  ├── Query TradeMemory for last 7 days               │
│  ├── Compute performance metrics                     │
│  ├── Analyze patterns                                │
│  ├── Generate reflective sections                    │
│  └── Save report to docs/weekly_reviews/             │
│                                                      │
│  WeeklyReviewer.extract_insights()                   │
│  ├── Confidence calibration analysis                 │
│  ├── Symbol-specific recommendations                 │
│  ├── SL/TP effectiveness                             │
│  ├── Time-of-day patterns                            │
│  └── Output: List of actionable insight strings      │
│                                                      │
│  DSPyOptimizer.weekly_review_cycle()                 │
│  ├── Check if enough new trades (>= min_trades)      │
│  ├── Prepare demonstrations from trade history       │
│  ├── Run MIPROv2 optimization                        │
│  └── Save optimized prompts to config/optimized/     │
└──────────────────────────────────────────────────────┘
```

---

## Component Interactions

### Module Dependency Graph

```
main_ai.py
├── config.py
├── memory/
│   ├── trade_memory.py ──── asyncpg, redis.asyncio
│   └── weekly_review.py ─── trade_memory
├── debate/
│   ├── debate_engine.py ─── langgraph
│   ├── agents.py ──── llm_client, prompts
│   ├── llm_client.py ─── litellm
│   ├── models.py ─── pydantic v2
│   ├── prompts.py
│   └── optimizer.py ─── dspy-ai
├── risk/
│   ├── risk_engine.py
│   └── kill_switch.py
├── execution/
│   ├── dry_run.py
│   ├── exchange_client.py ─── ccxt
│   └── order_manager.py
├── data/
│   ├── redis_cache.py ─── redis.asyncio
│   └── binance_connector.py ─── cryptofeed
└── monitoring/
    ├── telegram_bot.py ─── python-telegram-bot
    └── alert_formatter.py
```

### Key Design Decisions

1. **Lazy Imports:** Modules use `__getattr__` for lazy imports to avoid requiring all dependencies at package level (e.g., `dspy-ai`, `langgraph`).

2. **Async/Sync Split:** Database operations use async (`asyncpg`, `redis.asyncio`), while strategy and risk logic is sync (Lumibot-based).

3. **Graceful Degradation:** If Redis/PostgreSQL are unavailable, the bot continues running with in-memory state.

4. **No Hardcoded Secrets:** All API keys and credentials come from `.env` files or environment variables.

5. **Type Safety:** All public APIs use type hints. Pydantic v2 models for validation.

6. **Structured Logging:** Loguru with daily rotation, separate error log files, JSON-compatible format.
