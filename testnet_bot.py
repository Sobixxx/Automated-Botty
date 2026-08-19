"""
testnet_bot.py

Automated LONG-only trading bot for Binance SPOT TESTNET.

SCOPE:
  - Binance Spot cannot short. SHORT signals are detected and sent to
    Telegram as an FYI only, never executed. Full long+short automation
    would need Binance Futures Testnet instead (separate build).
  - TESTNET ONLY -- https://testnet.binance.vision, virtual funds.

Each run (intended hourly via GitHub Actions):
  1. Fetch fresh 1H candles from Binance Testnet's own market data.
  2. Re-run the validated backtest engine on that data to determine the
     current signal state (deterministic given price history + params,
     so no external state file is needed for the strategy logic itself).
  3. Check Binance directly (open orders) for actual current position
     state -- source of truth, more robust than a local file.
  4. Long signal + flat: market buy sized by 1% risk / stop distance,
     then an OCO order (stop-loss + take-profit) brackets it.
  5. Position open past the time-stop (160 bars): cancel OCO, close at
     market.
"""

import os
import sys
import math
import requests
import pandas as pd
from binance.spot import Spot as BinanceClient

sys.path.insert(0, os.path.dirname(__file__))
from smc_backtest import run_backtest, FINAL_CONFIG

SYMBOL = os.environ.get("SMC_SYMBOL", "ETHUSDT")
INTERVAL = "1h"
LOOKBACK_BARS = 3000
BAR_HOURS = 1
MAX_HOLD_BARS = FINAL_CONFIG["max_hold_bars"]

TESTNET_BASE_URL = "https://testnet.binance.vision"
API_KEY = os.environ.get("BINANCE_TESTNET_API_KEY")
API_SECRET = os.environ.get("BINANCE_TESTNET_API_SECRET")

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")


def send_telegram(message):
    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Telegram not configured -- printing instead:\n", message)
        return
    url = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"
    try:
        requests.post(url, json={"chat_id": TELEGRAM_CHAT_ID, "text": message,
                                  "parse_mode": "Markdown"}, timeout=10)
    except Exception as e:
        print("Telegram send failed:", e)


def get_client():
    if not API_KEY or not API_SECRET:
        raise RuntimeError("BINANCE_TESTNET_API_KEY / BINANCE_TESTNET_API_SECRET not set")
    return BinanceClient(api_key=API_KEY, api_secret=API_SECRET, base_url=TESTNET_BASE_URL)


def fetch_testnet_klines(client, symbol, interval, limit):
    raw = client.klines(symbol, interval, limit=limit)
    cols = ["open_time", "open", "high", "low", "close", "volume", "close_time",
            "quote_volume", "trades", "taker_buy_base", "taker_buy_quote", "ignore"]
    df = pd.DataFrame(raw, columns=cols)
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = df[c].astype(float)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def get_symbol_filters(client, symbol):
    info = client.exchange_info(symbol=symbol)
    filters = {f["filterType"]: f for f in info["symbols"][0]["filters"]}
    lot = filters.get("LOT_SIZE", {})
    price_f = filters.get("PRICE_FILTER", {})
    return {
        "step_size": float(lot.get("stepSize", 0.00001)),
        "min_qty": float(lot.get("minQty", 0.0)),
        "tick_size": float(price_f.get("tickSize", 0.01)),
    }


def round_step(value, step):
    if step <= 0:
        return value, 8
    precision = max(0, -int(round(math.log10(step))))
    return math.floor(value / step) * step, precision


def main():
    client = get_client()
    df = fetch_testnet_klines(client, SYMBOL, INTERVAL, LOOKBACK_BARS)
    df_closed = df.iloc[:-1].reset_index(drop=True)
    last_idx = len(df_closed) - 1
    last_time = df_closed["timestamp"].iloc[-1]
    current_price = df_closed["close"].iloc[-1]

    trades, equity, enriched = run_backtest(df_closed, **FINAL_CONFIG)

    open_orders = client.get_open_orders(symbol=SYMBOL)
    in_position = len(open_orders) > 0

    if in_position:
        oldest_order_time = min(o["time"] for o in open_orders)
        hours_open = (pd.Timestamp.utcnow().tz_localize("UTC") -
                      pd.to_datetime(oldest_order_time, unit="ms", utc=True)).total_seconds() / 3600
        if hours_open >= MAX_HOLD_BARS * BAR_HOURS:
            for o in open_orders:
                client.cancel_order(symbol=SYMBOL, orderId=o["orderId"])
            balances = client.account()["balances"]
            base_asset = SYMBOL.replace("USDT", "")
            base_qty = float(next((b["free"] for b in balances if b["asset"] == base_asset), 0))
            if base_qty > 0:
                filters = get_symbol_filters(client, SYMBOL)
                qty, prec = round_step(base_qty, filters["step_size"])
                if qty > 0:
                    client.new_order(symbol=SYMBOL, side="SELL", type="MARKET", quantity=round(qty, prec))
            send_telegram(f"*TIME STOP \u2014 {SYMBOL}*\nHeld {hours_open:.0f}h, closed at market (~{current_price:.2f}).")
            print("Time stop triggered, position closed.")
            return

    if in_position:
        print(f"Currently in position on {SYMBOL} ({len(open_orders)} open order(s)). No new entry this run.")
        return

    if trades.empty:
        print(f"No signal on latest closed bar ({last_time}).")
        return

    latest_entries = trades[trades["entry_idx"] == last_idx]
    if latest_entries.empty:
        print(f"No new entry on latest closed bar ({last_time}).")
        return

    row = latest_entries.iloc[0]

    if row["direction"] == "short":
        send_telegram(
            f"*SHORT signal \u2014 {SYMBOL} (NOT EXECUTED)*\n"
            f"Spot testnet is long-only. Entry would be {row['entry_price']:.2f}, "
            f"stop {row['stop']:.2f}, target {row['target']:.2f}."
        )
        print("Short signal detected, not executed (spot is long-only).")
        return

    account = client.account()
    usdt_balance = float(next(b["free"] for b in account["balances"] if b["asset"] == "USDT"))
    risk_amount = usdt_balance * FINAL_CONFIG["risk_pct"]
    stop_price = row["stop"]
    risk_per_unit = abs(current_price - stop_price)
    if risk_per_unit <= 0:
        print("Invalid risk per unit, skipping.")
        return

    raw_qty = risk_amount / risk_per_unit
    filters = get_symbol_filters(client, SYMBOL)
    qty, qty_prec = round_step(raw_qty, filters["step_size"])
    if qty < filters["min_qty"] or qty <= 0:
        send_telegram(f"*Entry skipped \u2014 {SYMBOL}*\nComputed size {raw_qty:.6f} below exchange minimum.")
        print(f"Quantity {raw_qty} below min_qty {filters['min_qty']}, skipping.")
        return

    order = client.new_order(symbol=SYMBOL, side="BUY", type="MARKET", quantity=round(qty, qty_prec))
    filled_qty = float(order.get("executedQty", qty))

    target_price = row["target"]
    tick = filters["tick_size"]
    stop_limit_price, sp_prec = round_step(stop_price * 0.999, tick)
    target_price_r, tp_prec = round_step(target_price, tick)

    oco = client.new_oco_order(
        symbol=SYMBOL, side="SELL", quantity=round(filled_qty, qty_prec),
        price=round(target_price_r, tp_prec),
        stopPrice=round(stop_price, sp_prec),
        stopLimitPrice=round(stop_limit_price, sp_prec),
        stopLimitTimeInForce="GTC",
    )

    send_telegram(
        f"*LONG ENTRY EXECUTED \u2014 {SYMBOL} (TESTNET)*\n"
        f"Qty: `{filled_qty}`\nEntry: `{current_price:.2f}`\n"
        f"Stop: `{stop_price:.2f}` | Target: `{target_price:.2f}`\n"
        f"Bar time: {last_time}"
    )
    print("Long entry executed and OCO placed:", order, oco)


if __name__ == "__main__":
    main()
