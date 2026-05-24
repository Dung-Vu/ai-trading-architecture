# 🛠️ TECH STACK & SETUP GUIDE

## Công nghệ chính
| Thành phần | Công nghệ | Version | Ghi chú |
|---|---|---|---|
| **Language** | Python | 3.11+ | Async-first |
| **Data Pipeline** | Cryptofeed | Latest | 50+ exchanges |
| **Time-series DB** | QuestDB | Latest | Hoặc TimescaleDB |
| **Cache** | Redis | 7+ | Hot data, pub/sub |
| **Agent Framework** | LangGraph | Latest | State machine |
| **LLM Router** | LiteLLM | Latest | Multi-provider |
| **Prompt Optimization** | DSPy | Latest | MIPROv2 |
| **Memory** | Mem0 | Latest | + Qdrant |
| **Execution** | CCXT | Latest | 100+ exchanges |
| **Trading Framework** | Lumibot | Latest | Backtest + Live |
| **Monitoring** | Telegram Bot API | Latest | Alerts |
| **Dashboard** | Grafana + Streamlit | Latest | Visualization |
| **ML Tracking** | Weights & Biases | Latest | Hoặc MLflow |
| **Database** | PostgreSQL | 15+ | Audit + Analytics |

## Cài đặt môi trường
```bash
# 1. Clone repo
git clone <repo-url>
cd ai-trading-architecture

# 2. Setup Python env
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Setup Docker (cho Redis, QuestDB, PostgreSQL, Qdrant)
docker compose up -d

# 5. Copy env template
cp .env.example .env
# Edit .env với API keys thật
```

## Cấu trúc `.env`
```env
# Exchange
BINANCE_API_KEY=
BINANCE_API_SECRET=

# LLM Providers
BAILIAN_API_KEY=
BAILIAN_BASE_URL=https://coding-intl.dashscope.aliyuncs.com/apps/anthropic
OPENCODE_API_KEY=
OPENCODE_GO_BASE_URL=https://opencode.ai/zen/go/v1

# Database
DATABASE_URL=postgresql://user:pass@localhost:5432/trading_db
REDIS_URL=redis://localhost:6379
QDRANT_URL=http://localhost:6333

# Monitoring
TELEGRAM_BOT_TOKEN=
TELEGRAM_CHAT_ID=
WANDB_PROJECT=ai-trading-architecture

# Trading Mode
TRADING_MODE=dryrun  # dryrun hoặc live
```

## Docker Compose (services cơ bản)
```yaml
services:
  redis:
    image: redis:7-alpine
    ports: ["6379:6379"]
    volumes: [redis-data:/data]

  questdb:
    image: questdb/questdb:latest
    ports: ["9000:9000", "8812:8812", "9009:9009"]
    volumes: [questdb-data:/root/.questdb]

  postgres:
    image: postgres:15-alpine
    environment:
      POSTGRES_DB: trading_db
      POSTGRES_USER: trading_user
      POSTGRES_PASSWORD: trading_pass
    ports: ["5432:5432"]
    volumes: [pg-data:/var/lib/postgresql/data]

  qdrant:
    image: qdrant/qdrant:latest
    ports: ["6333:6333"]
    volumes: [qdrant-data:/qdrant/storage]

volumes:
  redis-data:
  questdb-data:
  pg-data:
  qdrant-data:
```

## Requirements cơ bản
```txt
# Core
asyncio
aiohttp
websockets

# Data
cryptofeed>=2.0
ccxt>=4.0

# AI/LLM
langgraph
langchain
litellm
dspy-ai
mem0ai

# Trading
lumibot>=3.0

# Database
redis
asyncpg
qdrant-client

# Monitoring
python-telegram-bot>=20.0
streamlit
wandb

# Utils
pydantic>=2.0
loguru
python-dotenv
pytest
```
