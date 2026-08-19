"""
smc_backtest.py -- backtest engine for the validated MTF SMC strategy.
Entries: 4H structural bias + 1H BOS + fresh order block + MACD + BB
contained filter. Exits: ATR stop, adaptive structure target (fallback
2R), time stop, loss-cooldown circuit breaker.
"""
import numpy as np
import pandas as pd

from smc_core import find_swings, detect_structure, detect_order_blocks, atr, \
    higher_timeframe_trend, add_classical_indicators

# Validated configuration: ETH/USDT 1H, 4H structural bias, single position,
# 2-loss/300-bar cooldown. See project history for full validation results.
FINAL_CONFIG = dict(
    swing_lookback=12, ob_lookback=20, atr_mult=0.75,
    min_reward_risk=0.9, max_reward_risk=5.0, structure_target_lookback=200,
    use_atr_stop=True, fee_pct=0.0004, bos_only=True,
    use_structure_target=True, reward_risk=2.0,
    use_macd_filter=True, use_bb_filter=True,
    risk_pct=0.01, max_hold_bars=160,
    use_htf_filter=True, htf_rule="4h",
    max_concurrent_positions=1,
    use_loss_cooldown=True, cooldown_after_losses=2, cooldown_bars=300,
)


def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.rename(columns={ts_col: "timestamp"})
    df = df.sort_values("timestamp").reset_index(drop=True)
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def run_backtest(
    df,
    swing_lookback=12, ob_lookback=20, atr_mult=0.75,
    min_reward_risk=0.9, max_reward_risk=5.0, structure_target_lookback=200,
    use_atr_stop=True, fee_pct=0.0004, bos_only=True,
    use_structure_target=True, reward_risk=2.0,
    use_macd_filter=True, use_bb_filter=True,
    risk_pct=0.01, max_hold_bars=160,
    use_htf_filter=True, htf_rule="4h",
    max_concurrent_positions=1,
    use_loss_cooldown=True, cooldown_after_losses=2, cooldown_bars=300,
):
    df = find_swings(df, lookback=swing_lookback)
    df = detect_structure(df)
    df["atr14"] = atr(df, period=14)
    df = add_classical_indicators(df)
    if use_htf_filter:
        df = higher_timeframe_trend(df, rule=htf_rule, swing_lookback=swing_lookback)
    else:
        df["htf_trend"] = 0

    order_blocks = detect_order_blocks(df, max_lookback=ob_lookback)

    n = len(df)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    trend_arr = df["trend"].values
    htf_trend = df["htf_trend"].values
    macd_hist = df["macd_hist"].values
    bb_upper = df["bb_upper"].values
    bb_lower = df["bb_lower"].values
    atr_vals = df["atr14"].values

    swing_high_flags = df["swing_high"].values
    swing_low_flags = df["swing_low"].values

    trades = []
    equity = 10000.0
    equity_curve = np.full(n, np.nan)
    equity_curve[0] = equity

    open_positions = []
    used_ob_ids = set()
    consecutive_losses = 0
    cooldown_until_idx = -1

    for i in range(n):
        equity_curve[i] = equity

        still_open = []
        for pos in open_positions:
            hit_stop = (low[i] <= pos["stop"]) if pos["dir"] == "long" else (high[i] >= pos["stop"])
            hit_target = (high[i] >= pos["target"]) if pos["dir"] == "long" else (low[i] <= pos["target"])
            bars_held = i - pos["entry_idx"]
            timed_out = bars_held >= max_hold_bars

            if hit_stop or hit_target or timed_out or i == n - 1:
                if hit_target:
                    exit_price = pos["target"]
                elif hit_stop:
                    exit_price = pos["stop"]
                else:
                    exit_price = close[i]
                risk_per_unit = abs(pos["entry"] - pos["stop"])
                raw = (exit_price - pos["entry"]) if pos["dir"] == "long" else (pos["entry"] - exit_price)
                r_multiple = raw / risk_per_unit if risk_per_unit > 0 else 0.0

                risk_amount = equity * risk_pct
                pnl = risk_amount * r_multiple - equity * fee_pct
                equity += pnl

                trades.append({
                    "entry_idx": pos["entry_idx"], "exit_idx": i,
                    "entry_time": df["timestamp"].iloc[pos["entry_idx"]],
                    "exit_time": df["timestamp"].iloc[i],
                    "direction": pos["dir"], "entry_price": pos["entry"],
                    "stop": pos["stop"], "target": pos["target"], "exit_price": exit_price,
                    "r_multiple": r_multiple, "pnl": pnl, "equity_after": equity,
                    "bars_held": bars_held,
                })

                if use_loss_cooldown:
                    consecutive_losses = consecutive_losses + 1 if pnl <= 0 else 0
                    if consecutive_losses >= cooldown_after_losses:
                        cooldown_until_idx = i + cooldown_bars
                        consecutive_losses = 0
            else:
                still_open.append(pos)
        open_positions = still_open

        cooldown_active = use_loss_cooldown and i < cooldown_until_idx
        if len(open_positions) < max_concurrent_positions and i > swing_lookback * 10 and not cooldown_active:
            recent_bull = [ob for ob in order_blocks
                           if ob.direction == "bull" and ob.confirmed_idx <= i
                           and (i - ob.confirmed_idx) <= ob_lookback * 3
                           and id(ob) not in used_ob_ids
                           and (not bos_only or ob.event_type == "BOS_up")]
            recent_bear = [ob for ob in order_blocks
                           if ob.direction == "bear" and ob.confirmed_idx <= i
                           and (i - ob.confirmed_idx) <= ob_lookback * 3
                           and id(ob) not in used_ob_ids
                           and (not bos_only or ob.event_type == "BOS_down")]

            price = close[i]
            trend = trend_arr[i]
            macd_ok_long = (not use_macd_filter) or macd_hist[i] > 0
            macd_ok_short = (not use_macd_filter) or macd_hist[i] < 0
            bb_ok_long = (not use_bb_filter) or np.isnan(bb_upper[i]) or price <= bb_upper[i]
            bb_ok_short = (not use_bb_filter) or np.isnan(bb_lower[i]) or price >= bb_lower[i]
            htf_ok_long = (not use_htf_filter) or htf_trend[i] == 1
            htf_ok_short = (not use_htf_filter) or htf_trend[i] == -1

            if trend == 1 and recent_bull and htf_ok_long and macd_ok_long and bb_ok_long:
                ob = max(recent_bull, key=lambda o: o.confirmed_idx)
                if low[i] <= ob.top and low[i] >= ob.bottom * 0.98:
                    used_ob_ids.add(id(ob))
                    entry = price
                    stop = min(ob.bottom, entry) - atr_mult * atr_vals[i] if use_atr_stop else ob.bottom
                    risk = entry - stop
                    if risk > 0:
                        target = entry + reward_risk * risk
                        if use_structure_target:
                            lookback_start = max(0, i - structure_target_lookback)
                            candidates = [high[k] for k in range(lookback_start, i)
                                          if swing_high_flags[k] and high[k] > entry]
                            if candidates:
                                struct_target = min(candidates)
                                implied_rr = (struct_target - entry) / risk
                                if min_reward_risk <= implied_rr <= max_reward_risk:
                                    target = struct_target
                        open_positions.append({"dir": "long", "entry": entry, "stop": stop,
                                                "target": target, "entry_idx": i})

            elif trend == -1 and recent_bear and htf_ok_short and macd_ok_short and bb_ok_short:
                ob = max(recent_bear, key=lambda o: o.confirmed_idx)
                if high[i] >= ob.bottom and high[i] <= ob.top * 1.02:
                    used_ob_ids.add(id(ob))
                    entry = price
                    stop = max(ob.top, entry) + atr_mult * atr_vals[i] if use_atr_stop else ob.top
                    risk = stop - entry
                    if risk > 0:
                        target = entry - reward_risk * risk
                        if use_structure_target:
                            lookback_start = max(0, i - structure_target_lookback)
                            candidates = [low[k] for k in range(lookback_start, i)
                                          if swing_low_flags[k] and low[k] < entry]
                            if candidates:
                                struct_target = max(candidates)
                                implied_rr = (entry - struct_target) / risk
                                if min_reward_risk <= implied_rr <= max_reward_risk:
                                    target = struct_target
                        open_positions.append({"dir": "short", "entry": entry, "stop": stop,
                                                "target": target, "entry_idx": i})

    equity_curve = pd.Series(equity_curve).ffill().values
    trades_df = pd.DataFrame(trades)
    return trades_df, equity_curve, df


def summarize(trades_df, equity_curve, starting_equity=10000.0):
    if trades_df.empty:
        return {"total_trades": 0}
    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    win_rate = len(wins) / len(trades_df)
    profit_factor = (wins["pnl"].sum() / abs(losses["pnl"].sum())) if len(losses) and losses["pnl"].sum() != 0 else None
    return {
        "total_trades": len(trades_df),
        "win_rate_pct": round(win_rate * 100, 1),
        "avg_r_multiple": round(trades_df["r_multiple"].mean(), 2),
        "profit_factor": round(profit_factor, 2) if profit_factor else None,
        "total_return_pct": round((equity_curve[-1] / starting_equity - 1) * 100, 1),
        "final_equity": round(equity_curve[-1], 2),
    }
