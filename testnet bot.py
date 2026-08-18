"""
testnet_bot.py

Automated LONG-only trading bot for Binance SPOT TESTNET, driven by the
same validated run_backtest() engine used throughout this project.

IMPORTANT SCOPE:
  - Binance Spot (testnet or live) cannot short. This bot only executes
    LONG signals. SHORT signals are detected and logged/notified, but
    not traded. To trade both directions, Binance Futures Testnet would
    be needed instead (different API, different mechanics -- a separate
    build).
  - This is TESTNET ONLY. Uses https://testnet.binance.vision with virtual
    funds. Do not point this at production Binance without deliberately
    changing the base_url AND re-reading every risk consideration first.

How it works each run (designed for hourly execution via GitHub Actions):
  1. Fetch fresh 1H klines directly from Binance Testnet's own market data
     (not mainnet) -- keeps signal generation and execution on the same
     price feed, avoiding mismatch between "what triggered the signal"
     and "what price you'd actually get filled at".
  2. Re-run the full validated backtest engine on that data. Because the
     strategy is deterministic, this reconstructs current state (in a
     trade or not, cooldown active or not) with no separate state file
     needed for the STRATEGY LOGIC. Actual position state, however, is
     read from Binance itself (open orders) as the source of truth --
     more robust than trusting a local file if a run ever fails partway.
  3. If flat and a LONG signal just fired: place a MARKET buy sized by
     1% risk / stop distance, then immediately place an OCO (stop-loss +
     take-profit) order to bracket it -- Binance's own matching engine
     then handles the exit, no need to poll intra-hour.
  4. If currently holding a position and the time-stop (160 bars) has
     elapsed: cancel the OCO and market-sell to flatten.
  5. SHORT signals: notified via Telegram only, not executed.

Free requirements:
  - Binance Testnet account + API key (testnet.binance.vision, login via
    GitHub, generate HMAC_SHA256 key -- SEPARATE from any live account)
  - Telegram bot (as set up earlier)
  - GitHub Actions free tier
"""

import os
import sys
import time
import math
import requests
import pandas as pd
from binance.spot import Spot as BinanceClient

sys.path.insert(0, os.path.dirname(__file__))
from smc_backtest import run_backtest

SYMBOL = os.environ.get("SMC_SYMBOL", "ETHUSDT")
INTERVAL = "1h"
LOOKBACK_BARS = 3000
BAR_HOURS = 1
MAX_HOLD_BARS = 160

TESTNET_BASE_URL = "https://testnet.binance.vision"
API_KEY = os.environ.get("BINANCE_TESTNET_API_KEY")
API_SECRET = os.environ.get("BINANCE_TESTNET_API_SECRET")

TELEGRAM_TOKEN = os.environ.get("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID = os.environ.get("TELEGRAM_CHAT_ID")

LIVE_CONFIG = dict(
    swing_lookback=12, ob_lookback=20, atr_mult=0.75,
    min_reward_risk=0.9, max_reward_risk=5.0, structure_target_lookback=200,
    zone_filter=False, use_atr_stop=True, fee_pct=0.0004, bos_only=True,
    use_structure_target=True, reward_risk=2.0,
    use_macd_filter=True, use_bb_filter=True, bb_mode="contained",
    risk_pct=0.01, max_hold_bars=MAX_HOLD_BARS,
    use_htf_filter=True, htf_rule="4h", htf_bias_mode="structure",
    max_concurrent_positions=1,
    use_loss_cooldown=True, cooldown_after_losses=2, cooldown_bars=300,
)


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
        return value
    precision = max(0, -int(round(math.log10(step))))
    return math.floor(value / step) * step, precision


def get_open_position(client, symbol):
    """Source of truth for whether we're currently in a trade: any open
    orders on this symbol (the OCO bracket) means yes."""
    open_orders = client.get_open_orders(symbol=symbol)
    return open_orders  # list, empty if none


def main():
    client = get_client()
    df = fetch_testnet_klines(client, SYMBOL, INTERVAL, LOOKBACK_BARS)
    df_closed = df.iloc[:-1].reset_index(drop=True)
    last_idx = len(df_closed) - 1
    last_time = df_closed["timestamp"].iloc[-1]
    current_price = df_closed["close"].iloc[-1]

    trades, equity, enriched = run_backtest(df_closed, **LIVE_CONFIG)

    open_orders = get_open_position(client, SYMBOL)
    in_position = len(open_orders) > 0

    # --- time-stop check on any existing open position ---
    if in_position:
        oldest_order_time = min(o["time"] for o in open_orders)
        hours_open = (pd.Timestamp.utcnow().tz_localize("UTC") -
                      pd.to_datetime(oldest_order_time, unit="ms", utc=True)).total_seconds() / 3600
        if hours_open >= MAX_HOLD_BARS * BAR_HOURS:
            for o in open_orders:
                client.cancel_order(symbol=SYMBOL, orderId=o["orderId"])
            base_asset_qty = float(client.account()["balances"]
                                    and next((b["free"] for b in client.account()["balances"]
                                              if b["asset"] == SYMBOL.replace("USDT", "")), 0))
            if base_asset_qty > 0:
                filters = get_symbol_filters(client, SYMBOL)
                qty, prec = round_step(base_asset_qty, filters["step_size"])
                if qty > 0:
                    client.new_order(symbol=SYMBOL, side="SELL", type="MARKET",
                                      quantity=round(qty, prec))
            send_telegram(f"*TIME STOP \u2014 {SYMBOL}*\nPosition held {hours_open:.0f}h, closed at market (~{current_price:.2f}).")
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
            f"stop {row['stop']:.2f}, target {row['target']:.2f}.\n"
            f"Use Futures Testnet to trade shorts."
        )
        print("Short signal detected, not executed (spot is long-only).")
        return

    # --- execute the LONG entry ---
    account = client.account()
    usdt_balance = float(next(b["free"] for b in account["balances"] if b["asset"] == "USDT"))
    risk_amount = usdt_balance * LIVE_CONFIG["risk_pct"]
    stop_price = row["stop"]
    risk_per_unit = abs(current_price - stop_price)
    if risk_per_unit <= 0:
        print("Invalid risk per unit, skipping.")
        return

    raw_qty = risk_amount / risk_per_unit
    filters = get_symbol_filters(client, SYMBOL)
    qty, qty_prec = round_step(raw_qty, filters["step_size"])
    if qty < filters["min_qty"] or qty <= 0:
        send_telegram(f"*Entry skipped \u2014 {SYMBOL}*\nComputed size {raw_qty:.6f} below exchange minimum. "
                       f"Testnet balance may be too small for 1% risk sizing at current price.")
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
        f"Qty: `{filled_qty}`\n"
        f"Entry: `{current_price:.2f}`\n"
        f"Stop: `{stop_price:.2f}` | Target: `{target_price:.2f}`\n"
        f"OCO order placed. Bar time: {last_time}"
    )
    print("Long entry executed and OCO placed:", order, oco)


if __name__ == "__main__":
    main()
