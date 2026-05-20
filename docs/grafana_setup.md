# Grafana & QuestDB Monitoring Setup Guide

This document outlines the step-by-step instructions to connect **QuestDB** to **Grafana** for rich, premium real-time visualization of the AI Trading Bot's performance, including P&L, win rates, Sharpe ratio, and AI debate analytics.

---

## 1. Prerequisites

Ensure your infrastructure has the following running:
- **QuestDB** (PostgreSQL Wire Protocol on port `8812`)
- **Grafana** (Default port `3000`)
- **AI Trading Bot** actively logging data into QuestDB tables

---

## 2. QuestDB Connection Parameters in Grafana

QuestDB exposes a PostgreSQL-compatible wire protocol. To add QuestDB as a data source in Grafana:

1. Open your Grafana Dashboard (typically `http://localhost:3000`).
2. Navigate to **Connections** > **Data Sources** > **Add new data source**.
3. Select **PostgreSQL**.
4. Configure the settings as follows:

| Field | Configuration Value | Note |
| :--- | :--- | :--- |
| **Host** | `localhost:8812` (or Docker service container IP) | QuestDB PostgreSQL wire port is `8812` |
| **Database** | `qdb` | Default database name |
| **User** | `admin` | Default QuestDB user |
| **Password** | `quest` | Default QuestDB password |
| **SSL Mode** | `disable` | Standard local connection (disable SSL verification) |
| **PostgreSQL Version** | `9.6 or newer` | Choose the latest version compatible |

5. Click **Save & test**. You should see the message: `Database Connection OK`.

---

## 3. Recommended Grafana Dashboards & SQL Queries

Use these precise, high-performance SQL queries to configure beautiful visualization panels.

### Panel 1: Cumulative P&L over Time (TimeSeries Graph)
*Visualizing the growth of bot balance over time.*
```sql
SELECT
  timestamp AS "time",
  sum(pnl) OVER (ORDER BY timestamp) AS "Cumulative P&L ($)"
FROM trades
WHERE $__timeFilter(timestamp)
ORDER BY 1;
```
* **Panel Type:** Time Series
* **Line Style:** Smooth Gradient, Filled area with 15% opacity.
* **Colors:** Sleek Emerald Green (`#00C853`) for positive gains or Harmonic Red (`#FF1744`) for drawdowns.

### Panel 2: Strategy Metrics Summary (Stat Callouts)
*Instant visual summaries of key trading performance metrics.*

#### Win Rate (%)
```sql
SELECT
  (count(case when pnl > 0 then 1 end) * 100.0) / count(*) AS "Win Rate"
FROM trades
WHERE timestamp >= now() - INTERVAL '30 days';
```

#### Total Profit Factor
```sql
SELECT
  sum(case when pnl > 0 then pnl else 0 end) / 
  nullif(abs(sum(case when pnl < 0 then pnl else 0 end)), 0) AS "Profit Factor"
FROM trades;
```

* **Panel Type:** Stat
* **Color Mode:** Value, using HSL-tailored thresholds (e.g., Red below 45%, Orange 45-55%, Green above 55% for Win Rate).

### Panel 3: Live Position State & SL/TP Safety Boundaries (Table)
*Checking currently active symbol trades with target stop losses and take profits.*
```sql
SELECT
  symbol,
  side,
  entry_price AS "Entry Price",
  current_price AS "Current Price",
  stop_loss AS "SL Price",
  take_profit AS "TP Price",
  pnl_pct AS "P&L (%)"
FROM active_positions
ORDER BY timestamp DESC;
```
* **Panel Type:** Table
* **Formatting:** Enable color columns for `P&L (%)` to highlight active gains and losses.

### Panel 4: AI Debate Analytics - Confidence vs. Win Rate (Scatter Plot / Bar Chart)
*Measuring how AI debate confidence maps to profitable trades.*
```sql
SELECT
  debate_confidence AS "Confidence",
  avg(case when pnl > 0 then 1.0 else 0.0 end) * 100.0 AS "Win Rate (%)"
FROM trades
GROUP BY debate_confidence
ORDER BY debate_confidence ASC;
```
* **Panel Type:** Bar Chart or XY Chart
* **Purpose:** Audit AI Debate engine effectiveness. High confidence scores should strongly correlate with higher win rates.

---

## 4. Premium Aesthetic Customization Tips

Make your Grafana Dashboard look incredibly premium and modern:
1. **Dark Mode Only:** Ensure Grafana is forced to Dark Theme for deep indigo and slate aesthetics.
2. **Glassmorphism Border Styling:** Enable **Panel Borders** with thin, light borders against deep dark backgrounds.
3. **Smooth Micro-Animations:** Under **Standard Options**, enable smooth hover tooltip interactions with multi-series tracking.
4. **Use Inter or Outfit Fonts:** If hosted, customize custom stylesheet headers for a premium typography finish.
5. **No empty states:** Always configure default fallback values (`coalesce(..., 0)`) so the panels never display empty or broken database grids.

---

*For detailed trade logging code details, refer to [ARCHITECTURE.md](file:///d:/ai-trading-architecture/docs/ARCHITECTURE.md).*
