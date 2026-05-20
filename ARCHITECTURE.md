# 🏗️ KIẾN TRÚC HỆ THỐNG — CHI TIẾT KỸ THUẬT

## 1. DATA PIPELINE LAYER
### Mục tiêu
Thu thập dữ liệu thị trường real-time, đảm bảo độ trễ <100ms, không mất dữ liệu.

### Thành phần
- **Cryptofeed**: Thư viện chuẩn hóa data từ 50+ sàn crypto.
- **Binance WebSocket**: Stream trực tiếp giá, order book, trade.
- **News API + Crawl4AI**: Thu thập tin tức, sentiment từ Twitter/Reddit.
- **Redis**: Hot cache cho data real-time.
- **QuestDB/TimescaleDB**: Time-series database lưu OHLCV, order book history.

### Flow
```
Binance WS → Cryptofeed → Redis (hot) → QuestDB (cold) → AI Engine
```

### Quality Gates
- Reject stale data (>5s latency).
- Reject abnormal spreads.
- Statistical price spike detection (Z-score > 3σ).

## 2. AI DEBATE ENGINE
### Mục tiêu
Mô phỏng quy trình ra quyết định của đội ngũ chuyên gia tài chính.

### Kiến trúc Debate (Triangular Adversarial)
```
┌─────────────────────────────────────────────────────────────┐
│                    INPUT: Market Data                        │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  BULL AGENT: Tìm lý do MUA (Technical, News, Sentiment)     │
│  BEAR AGENT: Tìm lý do BÁN (Rủi ro, Kháng cự, Macro)        │
│  DEVIL'S ADVOCATE: Phản biện cả 2, tìm điểm mù             │
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  JUDGE / SYNTHESIS: Tổng hợp, ra quyết định cuối cùng       │
│  Output: {action: BUY/SELL/HOLD, confidence, reason, SL, TP}│
└──────────────────────────┬──────────────────────────────────┘
                           ▼
┌─────────────────────────────────────────────────────────────┐
│  RISK MANAGER: Kiểm tra giới hạn vốn, drawdown, exposure    │
│  Output: APPROVE / REJECT / REDUCE / FLATTEN                │
└─────────────────────────────────────────────────────────────┘
```

### Anti-Sycophancy Rules (BẮT BUỘC)
1. Mỗi round debate PHẢI đưa ra bằng chứng MỚI.
2. PHẢI phản biện điểm mạnh nhất của phe đối lập.
3. Nếu đổi立场 → giải thích CHÍNH XÁC bằng chứng nào đã thuyết phục.
4. Mọi khẳng định PHẢI có số liệu cụ thể.
5. CẤM đồng ý trừ khi thực sự bị thuyết phục.

### Công nghệ
- **LangGraph**: Quản lý state machine của debate loop.
- **DSPy MIPROv2**: Tự động tối ưu prompt hàng tuần dựa trên kết quả trade.
- **LiteLLM**: Routing multi-provider (Claude, GPT, v.v.) với fallback.

## 3. RISK MANAGEMENT LAYER
### Mục tiêu
Bảo vệ vốn là ưu tiên SỐ 1. AI có thể sai, Risk Manager không được sai.

### Hardcoded Limits (KHÔNG ĐƯỢC THAY ĐỔI)
| Rule | Giá trị | Hành động |
|---|---|---|
| Max loss / ngày | 3% vốn | Dừng trade 24h |
| Max drawdown | 10% vốn | Dừng toàn bộ hệ thống |
| Max concentration | 20% vốn / coin | Từ chối lệnh mới |
| Max leverage | 3x | Cấm mở vị thế >3x |
| Kill switch | Manual + Auto | Đóng tất cả lệnh |

### Position Sizing
```python
# Half-Kelly (ceiling) + Van Tharp (floor)
f_kelly = (b * p - q) / b / 2  # Half-Kelly
f_van_tharp = (equity * risk_pct) / stop_loss_distance
final_size = min(f_kelly, f_van_tharp, max_position_pct * equity)
```

## 4. EXECUTION LAYER
### Mục tiêu
Khớp lệnh chính xác, nhanh chóng, có cơ chế retry & fail-safe.

### Thành phần
- **CCXT**: Unified API cho 100+ sàn.
- **Lumibot**: Framework quản lý chiến lược, backtest, live trading.
- **Order Types**: Limit, Market, Stop-Loss, Take-Profit, Trailing Stop.

### Retry & Fail-Safe
- Retry 3 lần với exponential backoff (1s, 2s, 4s).
- Nếu API lỗi >5 lần → Dừng bot, gửi alert Telegram.
- Luôn kiểm tra balance trước khi đặt lệnh.

## 5. MEMORY & LEARNING
### Mục tiêu
AI phải học từ sai lầm, không lặp lại lỗi cũ.

### Kiến trúc Memory
| Layer | Mục đích | TTL | Công nghệ |
|---|---|---|---|
| Short-term | Context phiên debate | Session | In-memory |
| Working | Active analysis | Giờ-Ngày | LangGraph state |
| Long-term | Lịch sử trade, pattern | Vĩnh viễn | Mem0 + Qdrant |
| Meta-memory | Performance prompt | Vĩnh viễn | DSPy optimizers |

### Weekly Review Flow
1. Tổng hợp toàn bộ trade trong tuần.
2. Phân tích: Win rate, Sharpe ratio, Max drawdown, Lỗi lớn nhất.
3. AI tự viết báo cáo: "Điều gì làm tốt? Điều gì làm tệ? Cần thay đổi gì?"
4. Cập nhật prompt DSPy dựa trên insights.

## 6. MONITORING & ALERTING
### Telegram Bot
- Alert khi có lệnh mới, hit SL/TP, kill switch kích hoạt.
- Commands: `/status`, `/profit`, `/balance`, `/stop`, `/start`.

### Grafana Dashboard
- Real-time P&L, Win rate, Sharpe ratio.
- Bot health, API latency, Data quality metrics.

### Streamlit
- Backtest analysis, ML experiment review.
- Interactive charting với dữ liệu từ QuestDB.
