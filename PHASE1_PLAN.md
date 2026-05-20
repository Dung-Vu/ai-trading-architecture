# 📋 PHASE 1: DATA PIPELINE & MVP — KẾ HOẠCH CHI TIẾT

## Mục tiêu Phase 1
- [ ] Kết nối Binance WebSocket, nhận giá real-time (BTC, ETH, 5 altcoins).
- [ ] Lưu data vào QuestDB (OHLCV, trades).
- [ ] Xây dựng framework strategy cơ bản (MA crossover, RSI).
- [ ] Chạy backtest trên data lịch sử 3 tháng.
- [ ] Dry-run trading với Lumibot + CCXT.
- [ ] Telegram bot gửi alert khi có tín hiệu.
- [ ] Logging & monitoring cơ bản.

## Thời gian: 3 tuần

## Task breakdown

### Tuần 1: Data Infrastructure
| Task | Mô tả | Priority | Status |
|---|---|:---:|:---:|
| **1.1** | Setup Docker compose (Redis, QuestDB, PostgreSQL, Qdrant) | 🔴 High | ✅ DONE |
| **1.2** | Viết Cryptofeed connector cho Binance | 🔴 High | ✅ DONE |
| **1.3** | Implement WebSocket reconnection với exponential backoff | 🔴 High | ✅ DONE |
| **1.4** | Quality gates: stale data, abnormal spread, spike detection | 🟡 Medium | ✅ DONE |
| **1.5** | Lưu OHLCV vào QuestDB với batching | 🔴 High | ✅ DONE |
| **1.6** | Viết unit tests cho data pipeline | 🟡 Medium | ✅ DONE |

### Tuần 2: Strategy & Backtesting
| Task | Mô tả | Priority | Status |
|---|---|:---:|:---:|
| **2.1** | Setup Lumibot framework, config Binance sandbox | 🔴 High | ✅ DONE |
| **2.2** | Viết strategy: SMA Crossover + RSI filter | 🔴 High | ✅ DONE |
| **2.3** | Viết strategy: Bollinger Bands + Volume confirmation | 🟡 Medium | ✅ DONE |
| **2.4** | Backtest trên data 3 tháng, tính metrics | 🔴 High | ✅ DONE |
| **2.5** | Visualization: P&L, drawdown, win rate chart | 🟡 Medium | ✅ DONE |
| **2.6** | Optimize parameters (grid search hoặc optuna) | 🟡 Medium | ⚪ TODO |

### Tuần 3: Execution & Monitoring
| Task | Mô tả | Priority | Status |
|---|---|:---:|:---:|
| **3.1** | Setup CCXT với Binance testnet | 🔴 High | ✅ DONE |
| **3.2** | Implement dry-run trading với Lumibot | 🔴 High | ✅ DONE |
| **3.3** | Telegram bot: alert khi có entry/exit signal | 🔴 High | ✅ DONE |
| **3.4** | Logging system: log mọi decision, trade, error | 🔴 High | ✅ DONE |
| **3.5** | Grafana dashboard: real-time P&L, bot health | 🟡 Medium | ⚪ TODO |
| **3.6** | Documentation: hướng dẫn setup, chạy, monitor | 🟢 Low | ⚪ TODO |

## Acceptance Criteria
- [ ] Bot kết nối Binance stable >24h không disconnect.
- [ ] Data lưu vào QuestDB đầy đủ, không mất candle.
- [ ] Backtest chạy thành công, có báo cáo metrics.
- [ ] Dry-run trade được ít nhất 10 lệnh ảo.
- [ ] Telegram nhận được alert đúng thời điểm.
- [ ] Log đầy đủ, có thể debug khi có lỗi.

## Rủi ro & Mitigation
| Rủi ro | Impact | Mitigation |
|---|---|---|
| Binance API rate limit | Cao | Dùng CCXT rate limiter, batching requests |
| WebSocket disconnect | Cao | Auto-reconnect với backoff, buffer data |
| Data chất lượng thấp | Trung bình | Quality gates, reject outliers |
| Lumibot bugs | Trung bình | Test kỹ với sandbox trước khi live |
| Telegram bot bị block | Thấp | Dùng polling thay vì webhook, retry logic |
