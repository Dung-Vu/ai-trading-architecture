# 🤖 AGENTS.md — Hướng dẫn cho AI Developers

> Tài liệu này dành cho các AI assistant (Claude Code, Codex, OpenCode...) khi làm việc trong repo này.
> Hãy đọc kỹ trước khi thực hiện bất kỳ thay đổi nào.

## ⚠️ Quy tắc BẮT BUỘC
1. **KHÔNG BAO GIỜ dùng API key thật** trong code. Luôn dùng biến môi trường (`.env`).
2. **KHÔNG BAO GIỜ bypass Risk Engine**. Mọi lệnh phải qua lớp Risk checks trước khi gửi đi.
3. **Ưu tiên Dry-run** trước khi chuyển sang Live trading.
4. **Viết test** cho mọi module quan trọng (đặc biệt là Risk & Execution).
5. **Log mọi quyết định AI** vào database để audit sau này.

## 🏗️ Kiến trúc tổng quan
Hệ thống gồm 6 lớp:
1. **Data Pipeline** (Cryptofeed + WebSocket) → Thu thập giá, tin tức, sentiment.
2. **AI Debate Engine** (LangGraph + LLM) → Tranh luận Bull/Bear/Risk.
3. **Risk Management** (Hardcoded limits) → Kill switch, position sizing.
4. **Execution Layer** (CCXT + Lumibot) → Khớp lệnh trên sàn.
5. **Memory & Learning** (Mem0 + PostgreSQL) → Lưu log, rút kinh nghiệm.
6. **Monitoring** (Telegram + Grafana) → Giám sát & cảnh báo.

## 🚀 Workflow phát triển
1. Đọc `PHASE1_PLAN.md` để biết task cụ thể.
2. Tạo branch feature/ tương ứng.
3. Code + Viết test + Chạy test.
4. Commit với message rõ ràng (conventional commits).
5. Tạo PR để review.

## 🔧 Môi trường phát triển
```bash
# Setup
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Chạy test
pytest tests/ -v

# Chạy bot (Dry-run)
python src/main.py --mode dryrun --config config/dryrun.yaml
```

## 📋 Checklist trước khi commit
- [ ] Code đã được lint (ruff/black)
- [ ] Test pass 100%
- [ ] Không có hardcoded secrets
- [ ] Risk checks được gọi trước khi execute
- [ ] Log đầy đủ (input, decision, output)
- [ ] Documentation được cập nhật

## 🆘 Khi gặp vấn đề
1. Check `ARCHITECTURE.md` để hiểu thiết kế gốc.
2. Check logs trong `logs/` folder.
3. Nếu bug liên quan đến Risk → DỪNG BOT NGAY.
4. Hỏi user trước khi thay đổi cấu trúc quan trọng.
