import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from smc_core import (find_swings, detect_structure, detect_order_blocks, detect_fvgs,
                       premium_discount, atr, higher_timeframe_trend, fvg_fill_indices,
                       add_classical_indicators, adx_indicator, higher_timeframe_trend_ema)

TIMEFRAME_CONFIGS = {
    "4h": dict(
        swing_lookback=6, ob_lookback=15, max_hold_bars=40, atr_mult=0.65,
        min_reward_risk=0.7, structure_target_lookback=100,
        zone_filter=False, use_atr_stop=True, fee_pct=0.0004, bos_only=True,
        use_structure_target=True, max_reward_risk=5.0, reward_risk=2.0,
        use_macd_filter=True, use_bb_filter=True, bb_mode="contained", risk_pct=0.01,
    ),
    "1h": dict(
        swing_lookback=12, ob_lookback=20, max_hold_bars=160, atr_mult=0.8,
        min_reward_risk=0.9, structure_target_lookback=200,
        zone_filter=False, use_atr_stop=True, fee_pct=0.0004, bos_only=True,
        use_structure_target=True, max_reward_risk=5.0, reward_risk=2.0,
        use_macd_filter=True, use_bb_filter=True, bb_mode="contained", risk_pct=0.01,
    ),
    "1h_mtf": dict(
        swing_lookback=12, ob_lookback=20, max_hold_bars=160, atr_mult=0.75,
        min_reward_risk=0.9, structure_target_lookback=200,
        zone_filter=False, use_atr_stop=True, fee_pct=0.0004, bos_only=True,
        use_structure_target=True, max_reward_risk=5.0, reward_risk=2.0,
        use_macd_filter=True, use_bb_filter=True, bb_mode="contained", risk_pct=0.01,
        use_htf_filter=True, htf_rule="4h", max_concurrent_positions=2,
        use_loss_cooldown=True, cooldown_after_losses=2, cooldown_bars=300,
    ),
}

# BEST-VALIDATED STRATEGY: 1H entries, gated by 4H structural bias (MTF).
# Grid-searched on TRAIN (Jan 2025-Apr 2026), confirmed on TEST
# (May-Jul 2026, never seen during optimization):
#   Train: 32 trades, WR 59.4%, PF 2.69, expectancy +0.734R
#   Test:   8 trades, WR 62.5%, PF 2.50, expectancy +0.648R  (held up OOS)
#   Full 19mo: 43 trades, WR 60.5%, PF 2.72, expectancy +0.737R, Sharpe 2.64, return +34.2%
#   Long/short balanced: 22 long (WR 63.6%) / 21 short (WR 57.1%)
# This is the current default. Requires BOTH 1H and 4H OHLCV data covering
# the same period (4H bias is resampled live from the 1H data, so a
# separate 4H file is not strictly required -- see higher_timeframe_trend()).
FINAL_CONFIG = dict(TIMEFRAME_CONFIGS["1h_mtf"])


def load_csv(path):
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    ts_col = "timestamp" if "timestamp" in df.columns else df.columns[0]
    df[ts_col] = pd.to_datetime(df[ts_col])
    df = df.rename(columns={ts_col: "timestamp"})
    df = df.sort_values("timestamp").reset_index(drop=True)
    required = {"open", "high", "low", "close"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")
    if "volume" not in df.columns:
        df["volume"] = 0.0
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def run_backtest(
    df,
    swing_lookback=5,
    ob_lookback=15,
    max_hold_bars=40,
    reward_risk=2.0,
    zone_filter=True,
    risk_pct=0.01,
    starting_equity=10000.0,
    fee_pct=0.0004,
    use_htf_filter=False,
    htf_rule="1D",
    htf_bias_mode="structure",  # "structure" (BOS-based, original) or "ema" (matches Pine v4)
    use_atr_stop=False,
    atr_mult=0.75,
    use_volume_filter=False,
    volume_mult=1.2,
    bos_only=False,
    choch_only=False,
    require_close_in_zone=False,
    require_fvg_confluence=False,
    fvg_lookback=30,
    use_structure_target=False,
    structure_target_lookback=60,
    min_reward_risk=1.0,
    max_reward_risk=5.0,
    use_ema_filter=False,
    ema_fast=50,
    ema_slow=200,
    use_macd_filter=False,
    use_rsi_filter=False,
    rsi_overbought=70,
    rsi_oversold=30,
    use_sar_filter=False,
    use_bb_filter=False,
    bb_mode="breakout",
    indicator_time_scale=1,  # multiply all classical indicator periods by this (for lower timeframes)
    use_vol_regime_filter=False,
    vol_regime_lookback=100,
    vol_regime_min_percentile=25,  # skip entries when ATR is in the bottom X% of its recent range
    use_ob_strength_filter=False,
    min_ob_strength_atr_mult=0.8,  # OB source candle range must be >= this many ATRs
    max_concurrent_positions=1,
    use_advanced_exit=False,
    tp1_r=1.0, tp1_fraction=0.3,
    tp2_r=2.0, tp2_fraction=0.3,
    breakeven_r=1.25,
    trail_atr_mult=0.75,
    time_stop_check_bars=48,
    time_stop_check_min_r=0.5,
    time_stop_max_bars=96,
    use_ema200_filter=False,
    use_adx_filter=False,
    adx_threshold=20,
    use_displacement_filter=False,
    displacement_atr_mult=1.0,
    use_loss_cooldown=False,
    cooldown_after_losses=3,
    cooldown_bars=48,
):
    df = find_swings(df, lookback=swing_lookback)
    df = detect_structure(df)
    df = premium_discount(df, lookback=swing_lookback * 10)
    df["atr14"] = atr(df, period=14 * indicator_time_scale)
    df["atr_pct_rank"] = df["atr14"].rolling(vol_regime_lookback, min_periods=vol_regime_lookback).apply(
        lambda x: (x < x.iloc[-1]).mean() * 100, raw=False
    ) if use_vol_regime_filter else np.nan
    df["vol_avg20"] = df["volume"].rolling(20 * indicator_time_scale, min_periods=20 * indicator_time_scale).mean()
    df = add_classical_indicators(
        df, ema_fast=ema_fast, ema_slow=ema_slow,
        rsi_period=14 * indicator_time_scale,
        macd_fast=12 * indicator_time_scale,
        macd_slow=26 * indicator_time_scale,
        macd_signal=9 * indicator_time_scale,
        bb_period=20 * indicator_time_scale,
    )
    df = adx_indicator(df, period=14 * indicator_time_scale)
    if use_htf_filter:
        if htf_bias_mode == "ema":
            df = higher_timeframe_trend_ema(df, rule=htf_rule, ema_period=50)
        else:
            df = higher_timeframe_trend(df, rule=htf_rule, swing_lookback=swing_lookback)
    else:
        df["htf_trend"] = 0  # neutral = no filter applied
    order_blocks = detect_order_blocks(df, max_lookback=ob_lookback)
    fvgs = detect_fvgs(df)
    fvg_filled_map = fvg_fill_indices(df, fvgs) if require_fvg_confluence else {}

    n = len(df)
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    events = df["structure_event"].values
    range_pos = df["range_position"].values
    atr_vals = df["atr14"].values
    atr_pct_rank = df["atr_pct_rank"].values
    vol_avg = df["vol_avg20"].values
    volume = df["volume"].values
    htf_trend = df["htf_trend"].values
    swing_high_flags = df["swing_high"].values
    swing_low_flags = df["swing_low"].values
    ema_fast_vals = df["ema_fast"].values
    ema_slow_vals = df["ema_slow"].values
    macd_hist = df["macd_hist"].values
    rsi_vals = df["rsi"].values
    sar_trend = df["sar_trend"].values
    bb_upper = df["bb_upper"].values
    bb_lower = df["bb_lower"].values
    adx_vals = df["adx"].values
    ema_slow_arr = df["ema_slow"].values  # reused for EMA200 filter when ema_slow=200

    # index order blocks / fvgs by the bar they were confirmed at (the structure event bar)
    obs_by_event_bar = {}
    for ob in order_blocks:
        obs_by_event_bar.setdefault(ob.idx, []).append(ob)

    trades = []
    equity = starting_equity
    equity_curve = np.full(n, np.nan)
    equity_curve[0] = equity

    open_positions = []   # list of dicts -- supports concurrent positions
    used_ob_ids = set()   # order blocks already traded or invalidated -> never reused
    consecutive_losses = 0
    cooldown_until_idx = -1

    for i in range(n):
        equity_curve[i] = equity

        # register new zones (order blocks) whenever a structure event just confirmed one
        ev = events[i]
        if ev in ("BOS_up", "CHoCH_up", "BOS_down", "CHoCH_down"):
            for j in range(max(0, i - ob_lookback), i):
                pass  # zones already generated globally; filter below by proximity

        # --- manage open positions ---
        still_open = []
        for pos in open_positions:

            if use_advanced_exit:
                exited = False
                risk_per_unit = pos["initial_risk"]
                bars_held = i - pos["entry_idx"]

                # 1. Invalidation: opposite CHoCH + displacement candle
                displacement = (high[i] - low[i]) >= atr_vals[i] if not np.isnan(atr_vals[i]) else False
                invalidated = ((pos["dir"] == "long" and events[i] == "CHoCH_down" and displacement) or
                               (pos["dir"] == "short" and events[i] == "CHoCH_up" and displacement))
                if invalidated and pos["remaining_fraction"] > 0:
                    exit_price = close[i]
                    raw = (exit_price - pos["entry"]) if pos["dir"] == "long" else (pos["entry"] - exit_price)
                    r_multiple = raw / risk_per_unit if risk_per_unit > 0 else 0.0
                    risk_amount = equity * risk_pct * pos["remaining_fraction"]
                    pnl = risk_amount * r_multiple - equity * fee_pct * pos["remaining_fraction"]
                    equity += pnl
                    trades.append({"entry_idx": pos["entry_idx"], "exit_idx": i,
                        "entry_time": df["timestamp"].iloc[pos["entry_idx"]], "exit_time": df["timestamp"].iloc[i],
                        "direction": pos["dir"], "entry_price": pos["entry"], "stop": pos["stop"],
                        "target": exit_price, "exit_price": exit_price, "r_multiple": r_multiple,
                        "pnl": pnl, "equity_after": equity, "bars_held": bars_held, "leg": "invalidated"})
                    exited = True

                # 2. Stop hit (initial, breakeven, or trailed)
                if not exited:
                    hit_stop = (low[i] <= pos["stop"]) if pos["dir"] == "long" else (high[i] >= pos["stop"])
                    if hit_stop and pos["remaining_fraction"] > 0:
                        exit_price = pos["stop"]
                        raw = (exit_price - pos["entry"]) if pos["dir"] == "long" else (pos["entry"] - exit_price)
                        r_multiple = raw / risk_per_unit if risk_per_unit > 0 else 0.0
                        risk_amount = equity * risk_pct * pos["remaining_fraction"]
                        pnl = risk_amount * r_multiple - equity * fee_pct * pos["remaining_fraction"]
                        equity += pnl
                        trades.append({"entry_idx": pos["entry_idx"], "exit_idx": i,
                            "entry_time": df["timestamp"].iloc[pos["entry_idx"]], "exit_time": df["timestamp"].iloc[i],
                            "direction": pos["dir"], "entry_price": pos["entry"], "stop": pos["stop"],
                            "target": exit_price, "exit_price": exit_price, "r_multiple": r_multiple,
                            "pnl": pnl, "equity_after": equity, "bars_held": bars_held, "leg": "stop"})
                        exited = True

                # 3. TP1 / TP2 partials
                if not exited:
                    if not pos["tp1_done"]:
                        hit_tp1 = (high[i] >= pos["tp1_price"]) if pos["dir"] == "long" else (low[i] <= pos["tp1_price"])
                        if hit_tp1:
                            risk_amount = equity * risk_pct * tp1_fraction
                            pnl = risk_amount * tp1_r - equity * fee_pct * tp1_fraction
                            equity += pnl
                            trades.append({"entry_idx": pos["entry_idx"], "exit_idx": i,
                                "entry_time": df["timestamp"].iloc[pos["entry_idx"]], "exit_time": df["timestamp"].iloc[i],
                                "direction": pos["dir"], "entry_price": pos["entry"], "stop": pos["stop"],
                                "target": pos["tp1_price"], "exit_price": pos["tp1_price"], "r_multiple": tp1_r,
                                "pnl": pnl, "equity_after": equity, "bars_held": bars_held, "leg": "tp1"})
                            pos["tp1_done"] = True
                            pos["remaining_fraction"] -= tp1_fraction
                    elif not pos["tp2_done"]:
                        hit_tp2 = (high[i] >= pos["tp2_price"]) if pos["dir"] == "long" else (low[i] <= pos["tp2_price"])
                        if hit_tp2:
                            risk_amount = equity * risk_pct * tp2_fraction
                            pnl = risk_amount * tp2_r - equity * fee_pct * tp2_fraction
                            equity += pnl
                            trades.append({"entry_idx": pos["entry_idx"], "exit_idx": i,
                                "entry_time": df["timestamp"].iloc[pos["entry_idx"]], "exit_time": df["timestamp"].iloc[i],
                                "direction": pos["dir"], "entry_price": pos["entry"], "stop": pos["stop"],
                                "target": pos["tp2_price"], "exit_price": pos["tp2_price"], "r_multiple": tp2_r,
                                "pnl": pnl, "equity_after": equity, "bars_held": bars_held, "leg": "tp2"})
                            pos["tp2_done"] = True
                            pos["remaining_fraction"] -= tp2_fraction
                            pos["runner_active"] = True

                    # 4. Breakeven move (only after TP1 confirms continued structure)
                    if not exited and not pos["breakeven_done"] and pos["tp1_done"]:
                        extreme = high[i] if pos["dir"] == "long" else low[i]
                        moved_r = ((extreme - pos["entry"]) if pos["dir"] == "long" else (pos["entry"] - extreme)) / risk_per_unit
                        if moved_r >= breakeven_r:
                            new_stop = pos["entry"]
                            if (pos["dir"] == "long" and new_stop > pos["stop"]) or (pos["dir"] == "short" and new_stop < pos["stop"]):
                                pos["stop"] = new_stop
                            pos["breakeven_done"] = True

                    # 5. Structural trailing stop (active once runner phase begins)
                    if not exited and pos["runner_active"]:
                        lookback_start = max(pos["entry_idx"], i - 50)
                        if pos["dir"] == "long":
                            recent_lows = [low[k] for k in range(lookback_start, i) if swing_low_flags[k]]
                            if recent_lows and not np.isnan(atr_vals[i]):
                                candidate = max(recent_lows) - trail_atr_mult * atr_vals[i]
                                if candidate > pos["stop"]:
                                    pos["stop"] = candidate
                        else:
                            recent_highs = [high[k] for k in range(lookback_start, i) if swing_high_flags[k]]
                            if recent_highs and not np.isnan(atr_vals[i]):
                                candidate = min(recent_highs) + trail_atr_mult * atr_vals[i]
                                if candidate < pos["stop"]:
                                    pos["stop"] = candidate

                    # 6. Time stop -- conditional early exit, or hard cap (skipped once runner active)
                    if not exited and pos["remaining_fraction"] > 0 and not pos["runner_active"]:
                        if bars_held >= time_stop_check_bars and not pos["tp1_done"]:
                            unrealized_r = ((close[i] - pos["entry"]) if pos["dir"] == "long"
                                            else (pos["entry"] - close[i])) / risk_per_unit
                            favorable_bos = "BOS_up" if pos["dir"] == "long" else "BOS_down"
                            had_bos = np.any(events[pos["entry_idx"]+1:i+1] == favorable_bos)
                            if unrealized_r < time_stop_check_min_r and not had_bos:
                                exit_price = close[i]
                                raw = (exit_price - pos["entry"]) if pos["dir"] == "long" else (pos["entry"] - exit_price)
                                r_multiple = raw / risk_per_unit if risk_per_unit > 0 else 0.0
                                risk_amount = equity * risk_pct * pos["remaining_fraction"]
                                pnl = risk_amount * r_multiple - equity * fee_pct * pos["remaining_fraction"]
                                equity += pnl
                                trades.append({"entry_idx": pos["entry_idx"], "exit_idx": i,
                                    "entry_time": df["timestamp"].iloc[pos["entry_idx"]], "exit_time": df["timestamp"].iloc[i],
                                    "direction": pos["dir"], "entry_price": pos["entry"], "stop": pos["stop"],
                                    "target": exit_price, "exit_price": exit_price, "r_multiple": r_multiple,
                                    "pnl": pnl, "equity_after": equity, "bars_held": bars_held, "leg": "time_stop_early"})
                                exited = True
                        if not exited and bars_held >= time_stop_max_bars:
                            exit_price = close[i]
                            raw = (exit_price - pos["entry"]) if pos["dir"] == "long" else (pos["entry"] - exit_price)
                            r_multiple = raw / risk_per_unit if risk_per_unit > 0 else 0.0
                            risk_amount = equity * risk_pct * pos["remaining_fraction"]
                            pnl = risk_amount * r_multiple - equity * fee_pct * pos["remaining_fraction"]
                            equity += pnl
                            trades.append({"entry_idx": pos["entry_idx"], "exit_idx": i,
                                "entry_time": df["timestamp"].iloc[pos["entry_idx"]], "exit_time": df["timestamp"].iloc[i],
                                "direction": pos["dir"], "entry_price": pos["entry"], "stop": pos["stop"],
                                "target": exit_price, "exit_price": exit_price, "r_multiple": r_multiple,
                                "pnl": pnl, "equity_after": equity, "bars_held": bars_held, "leg": "time_stop_max"})
                            exited = True

                if not exited and (pos["remaining_fraction"] <= 1e-9 == False) and i == n - 1 and pos["remaining_fraction"] > 0:
                    exit_price = close[i]
                    raw = (exit_price - pos["entry"]) if pos["dir"] == "long" else (pos["entry"] - exit_price)
                    r_multiple = raw / risk_per_unit if risk_per_unit > 0 else 0.0
                    risk_amount = equity * risk_pct * pos["remaining_fraction"]
                    pnl = risk_amount * r_multiple - equity * fee_pct * pos["remaining_fraction"]
                    equity += pnl
                    trades.append({"entry_idx": pos["entry_idx"], "exit_idx": i,
                        "entry_time": df["timestamp"].iloc[pos["entry_idx"]], "exit_time": df["timestamp"].iloc[i],
                        "direction": pos["dir"], "entry_price": pos["entry"], "stop": pos["stop"],
                        "target": exit_price, "exit_price": exit_price, "r_multiple": r_multiple,
                        "pnl": pnl, "equity_after": equity, "bars_held": bars_held, "leg": "end_of_data"})
                    exited = True

                if not exited and pos["remaining_fraction"] > 1e-9:
                    still_open.append(pos)
                continue

            hit_stop = False
            hit_target = False
            if pos["dir"] == "long":
                if low[i] <= pos["stop"]:
                    hit_stop = True
                elif high[i] >= pos["target"]:
                    hit_target = True
            else:
                if high[i] >= pos["stop"]:
                    hit_stop = True
                elif low[i] <= pos["target"]:
                    hit_target = True

            bars_held = i - pos["entry_idx"]
            timed_out = bars_held >= max_hold_bars

            if hit_stop or hit_target or timed_out or i == n - 1:
                if hit_target:
                    exit_price = pos["target"]
                    risk_per_unit = abs(pos["entry"] - pos["stop"])
                    raw = (exit_price - pos["entry"]) if pos["dir"] == "long" else (pos["entry"] - exit_price)
                    r_multiple = raw / risk_per_unit if risk_per_unit > 0 else 0.0
                elif hit_stop:
                    exit_price = pos["stop"]
                    r_multiple = -1.0
                else:
                    exit_price = close[i]
                    risk_per_unit = abs(pos["entry"] - pos["stop"])
                    raw = (exit_price - pos["entry"]) if pos["dir"] == "long" else (pos["entry"] - exit_price)
                    r_multiple = raw / risk_per_unit if risk_per_unit > 0 else 0.0

                risk_amount = equity * risk_pct
                pnl = risk_amount * r_multiple
                pnl -= equity * fee_pct  # rough round-trip fee drag
                equity += pnl

                trades.append({
                    "entry_idx": pos["entry_idx"],
                    "exit_idx": i,
                    "entry_time": df["timestamp"].iloc[pos["entry_idx"]],
                    "exit_time": df["timestamp"].iloc[i],
                    "direction": pos["dir"],
                    "entry_price": pos["entry"],
                    "stop": pos["stop"],
                    "target": pos["target"],
                    "exit_price": exit_price,
                    "r_multiple": r_multiple,
                    "pnl": pnl,
                    "equity_after": equity,
                    "bars_held": bars_held,
                })
                consecutive_losses = consecutive_losses + 1 if pnl <= 0 else 0
                if use_loss_cooldown and consecutive_losses >= cooldown_after_losses:
                    cooldown_until_idx = i + cooldown_bars
                    consecutive_losses = 0  # reset so cooldown expires cleanly, doesn't re-trigger instantly
            else:
                still_open.append(pos)
        open_positions = still_open

        # --- look for new entries if there's room for another concurrent slot ---
        cooldown_active = use_loss_cooldown and i < cooldown_until_idx
        if len(open_positions) < max_concurrent_positions and i > swing_lookback * 10 and not cooldown_active:
            # gather recent, not-yet-used bullish/bearish OBs formed within ob_lookback*3 bars
            recent_bull_obs = [ob for ob in order_blocks
                                if ob.direction == "bull" and ob.confirmed_idx <= i and (i - ob.confirmed_idx) <= ob_lookback * 3
                                and id(ob) not in used_ob_ids
                                and (not bos_only or ob.event_type == "BOS_up")
                                and (not choch_only or ob.event_type == "CHoCH_up")
                                and (not use_ob_strength_filter or np.isnan(atr_vals[ob.idx]) or
                                     (high[ob.idx] - low[ob.idx]) >= min_ob_strength_atr_mult * atr_vals[ob.idx])]
            recent_bear_obs = [ob for ob in order_blocks
                                if ob.direction == "bear" and ob.confirmed_idx <= i and (i - ob.confirmed_idx) <= ob_lookback * 3
                                and id(ob) not in used_ob_ids
                                and (not bos_only or ob.event_type == "BOS_down")
                                and (not choch_only or ob.event_type == "CHoCH_down")
                                and (not use_ob_strength_filter or np.isnan(atr_vals[ob.idx]) or
                                     (high[ob.idx] - low[ob.idx]) >= min_ob_strength_atr_mult * atr_vals[ob.idx])]

            price = close[i]
            trend = df["trend"].iloc[i]
            pos_in_range = range_pos[i]
            bar_atr = atr_vals[i]
            volume_ok = (not use_volume_filter) or (
                not np.isnan(vol_avg[i]) and volume[i] > volume_mult * vol_avg[i]
            )
            htf_ok_long = (not use_htf_filter) or (htf_trend[i] == 1)
            htf_ok_short = (not use_htf_filter) or (htf_trend[i] == -1)

            ema_ok_long = (not use_ema_filter) or (not np.isnan(ema_fast_vals[i]) and not np.isnan(ema_slow_vals[i])
                                                     and price > ema_slow_vals[i] and ema_fast_vals[i] > ema_slow_vals[i])
            ema_ok_short = (not use_ema_filter) or (not np.isnan(ema_fast_vals[i]) and not np.isnan(ema_slow_vals[i])
                                                      and price < ema_slow_vals[i] and ema_fast_vals[i] < ema_slow_vals[i])
            macd_ok_long = (not use_macd_filter) or (not np.isnan(macd_hist[i]) and macd_hist[i] > 0)
            macd_ok_short = (not use_macd_filter) or (not np.isnan(macd_hist[i]) and macd_hist[i] < 0)
            rsi_ok_long = (not use_rsi_filter) or (not np.isnan(rsi_vals[i]) and rsi_vals[i] < rsi_overbought)
            rsi_ok_short = (not use_rsi_filter) or (not np.isnan(rsi_vals[i]) and rsi_vals[i] > rsi_oversold)

            sar_ok_long = (not use_sar_filter) or (sar_trend[i] == 1)
            sar_ok_short = (not use_sar_filter) or (sar_trend[i] == -1)

            if not use_bb_filter or np.isnan(bb_upper[i]) or np.isnan(bb_lower[i]):
                bb_ok_long = True
                bb_ok_short = True
            elif bb_mode == "breakout":
                bb_ok_long = price >= bb_upper[i]
                bb_ok_short = price <= bb_lower[i]
            else:  # "contained" -- avoid entries already stretched beyond the bands
                bb_ok_long = price <= bb_upper[i]
                bb_ok_short = price >= bb_lower[i]

            vol_regime_ok = (not use_vol_regime_filter) or (
                not np.isnan(atr_pct_rank[i]) and atr_pct_rank[i] >= vol_regime_min_percentile
            )

            ema200_ok_long = (not use_ema200_filter) or (not np.isnan(ema_slow_arr[i]) and price > ema_slow_arr[i])
            ema200_ok_short = (not use_ema200_filter) or (not np.isnan(ema_slow_arr[i]) and price < ema_slow_arr[i])

            adx_ok = (not use_adx_filter) or (not np.isnan(adx_vals[i]) and adx_vals[i] >= adx_threshold)

            displacement_ok = (not use_displacement_filter) or (
                not np.isnan(bar_atr) and (high[i] - low[i]) >= displacement_atr_mult * bar_atr
            )

            # LONG setup: uptrend, price tags a bullish OB, zone in discount (or filter off)
            if trend == 1 and recent_bull_obs and htf_ok_long and volume_ok and ema_ok_long and macd_ok_long and rsi_ok_long and sar_ok_long and bb_ok_long and vol_regime_ok and ema200_ok_long and adx_ok and displacement_ok:
                ob = max(recent_bull_obs, key=lambda o: o.idx)  # most recent
                if low[i] <= ob.top and low[i] >= ob.bottom * 0.98:  # price entered the block
                    close_ok = (not require_close_in_zone) or (close[i] <= ob.top and close[i] >= ob.bottom)
                    zone_ok = (not zone_filter) or (not np.isnan(pos_in_range) and pos_in_range < 0.5)
                    fvg_ok = True
                    if require_fvg_confluence:
                        fvg_ok = any(
                            f.direction == "bull" and f.idx < i and (i - f.idx) <= fvg_lookback
                            and (fvg_filled_map.get(id(f)) is None or fvg_filled_map[id(f)] > i)
                            and f.bottom <= ob.top and f.top >= ob.bottom  # zone overlap = confluence
                            for f in fvgs
                        )
                    used_ob_ids.add(id(ob))  # block is consumed whether or not we take the trade
                    if zone_ok and fvg_ok and close_ok:
                        entry = price
                        if use_atr_stop and not np.isnan(bar_atr):
                            stop = min(ob.bottom, entry) - atr_mult * bar_atr
                        else:
                            stop = ob.bottom * 0.999
                        risk = entry - stop
                        if risk > 0:
                            target = entry + reward_risk * risk
                            if use_structure_target:
                                # nearest already-confirmed swing high above entry, within lookback
                                lookback_start = max(0, i - structure_target_lookback)
                                candidates = [high[k] for k in range(lookback_start, i)
                                              if swing_high_flags[k] and high[k] > entry]
                                if candidates:
                                    struct_target = min(candidates)  # nearest one above
                                    implied_rr = (struct_target - entry) / risk
                                    if min_reward_risk <= implied_rr <= max_reward_risk:
                                        target = struct_target
                            open_positions.append({"dir": "long", "entry": entry, "stop": stop,
                                              "target": target, "entry_idx": i,
                                              "initial_risk": risk, "tp1_price": entry + tp1_r * risk,
                                              "tp2_price": entry + tp2_r * risk, "tp1_done": False,
                                              "tp2_done": False, "breakeven_done": False,
                                              "runner_active": False, "remaining_fraction": 1.0})

            # SHORT setup: downtrend, price tags a bearish OB, zone in premium
            elif trend == -1 and recent_bear_obs and htf_ok_short and volume_ok and ema_ok_short and macd_ok_short and rsi_ok_short and sar_ok_short and bb_ok_short and vol_regime_ok and ema200_ok_short and adx_ok and displacement_ok:
                ob = max(recent_bear_obs, key=lambda o: o.idx)
                if high[i] >= ob.bottom and high[i] <= ob.top * 1.02:
                    close_ok = (not require_close_in_zone) or (close[i] <= ob.top and close[i] >= ob.bottom)
                    zone_ok = (not zone_filter) or (not np.isnan(pos_in_range) and pos_in_range > 0.5)
                    fvg_ok = True
                    if require_fvg_confluence:
                        fvg_ok = any(
                            f.direction == "bear" and f.idx < i and (i - f.idx) <= fvg_lookback
                            and (fvg_filled_map.get(id(f)) is None or fvg_filled_map[id(f)] > i)
                            and f.bottom <= ob.top and f.top >= ob.bottom
                            for f in fvgs
                        )
                    used_ob_ids.add(id(ob))
                    if zone_ok and fvg_ok and close_ok:
                        entry = price
                        if use_atr_stop and not np.isnan(bar_atr):
                            stop = max(ob.top, entry) + atr_mult * bar_atr
                        else:
                            stop = ob.top * 1.001
                        risk = stop - entry
                        if risk > 0:
                            target = entry - reward_risk * risk
                            if use_structure_target:
                                lookback_start = max(0, i - structure_target_lookback)
                                candidates = [low[k] for k in range(lookback_start, i)
                                              if swing_low_flags[k] and low[k] < entry]
                                if candidates:
                                    struct_target = max(candidates)  # nearest one below
                                    implied_rr = (entry - struct_target) / risk
                                    if min_reward_risk <= implied_rr <= max_reward_risk:
                                        target = struct_target
                            open_positions.append({"dir": "short", "entry": entry, "stop": stop,
                                              "target": target, "entry_idx": i,
                                              "initial_risk": risk, "tp1_price": entry - tp1_r * risk,
                                              "tp2_price": entry - tp2_r * risk, "tp1_done": False,
                                              "tp2_done": False, "breakeven_done": False,
                                              "runner_active": False, "remaining_fraction": 1.0})

    equity_curve = pd.Series(equity_curve).ffill().values
    trades_df = pd.DataFrame(trades)
    return trades_df, equity_curve, df


def summarize(trades_df, equity_curve, starting_equity):
    if trades_df.empty:
        print("No trades were generated — try loosening zone_filter or ob_lookback.")
        return {}

    wins = trades_df[trades_df["pnl"] > 0]
    losses = trades_df[trades_df["pnl"] <= 0]
    win_rate = len(wins) / len(trades_df)
    avg_r = trades_df["r_multiple"].mean()
    profit_factor = (wins["pnl"].sum() / abs(losses["pnl"].sum())) if len(losses) and losses["pnl"].sum() != 0 else np.nan
    total_return_pct = (equity_curve[-1] / starting_equity - 1) * 100

    running_max = np.maximum.accumulate(equity_curve)
    drawdown = (equity_curve - running_max) / running_max
    max_dd_pct = drawdown.min() * 100

    # --- Sharpe ratio, computed on the per-trade return series ---
    # Each trade's return is treated as one "period" (trade-based Sharpe,
    # not calendar-based, since holding periods vary). Annualization uses
    # the average trades-per-year implied by the backtest's date range.
    trade_returns_pct = trades_df["pnl"] / starting_equity  # simple return per trade on starting capital
    if trade_returns_pct.std(ddof=1) > 0 and len(trades_df) > 1:
        span_days = (trades_df["exit_time"].max() - trades_df["entry_time"].min()).total_seconds() / 86400
        trades_per_year = len(trades_df) / span_days * 365 if span_days > 0 else len(trades_df)
        sharpe = (trade_returns_pct.mean() / trade_returns_pct.std(ddof=1)) * np.sqrt(trades_per_year)
    else:
        sharpe = np.nan

    # --- Return per trade per unit of risk taken (avg R multiple is this,
    # but expressed here explicitly as $ return per $ risked, i.e. expectancy) ---
    return_per_trade_per_risk = trades_df["r_multiple"].mean()  # this IS $ return / $ risked, per trade

    stats = {
        "total_trades": len(trades_df),
        "win_rate_pct": round(win_rate * 100, 1),
        "avg_r_multiple": round(avg_r, 2),
        "profit_factor": round(profit_factor, 2) if not np.isnan(profit_factor) else None,
        "total_return_pct": round(total_return_pct, 1),
        "max_drawdown_pct": round(max_dd_pct, 1),
        "sharpe_ratio_annualized": round(sharpe, 2) if not np.isnan(sharpe) else None,
        "expectancy_R_per_trade": round(return_per_trade_per_risk, 3),
        "final_equity": round(equity_curve[-1], 2),
        "long_trades": int((trades_df["direction"] == "long").sum()),
        "short_trades": int((trades_df["direction"] == "short").sum()),
    }
    return stats


def plot_results(df, trades_df, equity_curve, out_path):
    fig, axes = plt.subplots(2, 1, figsize=(11, 8), gridspec_kw={"height_ratios": [2, 1]})

    ax0 = axes[0]
    ax0.plot(df["timestamp"], df["close"], color="#333333", linewidth=0.8, label="Close")
    if not trades_df.empty:
        longs = trades_df[trades_df["direction"] == "long"]
        shorts = trades_df[trades_df["direction"] == "short"]
        ax0.scatter(df["timestamp"].iloc[longs["entry_idx"]], longs["entry_price"],
                    marker="^", color="green", s=35, label="Long entry", zorder=5)
        ax0.scatter(df["timestamp"].iloc[shorts["entry_idx"]], shorts["entry_price"],
                    marker="v", color="red", s=35, label="Short entry", zorder=5)
    ax0.set_title("Price with SMC trade entries")
    ax0.legend(loc="upper left", fontsize=8)
    ax0.tick_params(axis='x', labelrotation=20)

    ax1 = axes[1]
    ax1.plot(df["timestamp"], equity_curve, color="#1f77b4", linewidth=1.2)
    ax1.set_title("Equity curve")
    ax1.tick_params(axis='x', labelrotation=20)

    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Simplified SMC backtester")
    parser.add_argument("--csv", type=str, default="demo_btc_4h.csv")
    parser.add_argument("--reward_risk", type=float, default=2.0)
    parser.add_argument("--risk_pct", type=float, default=0.01)
    parser.add_argument("--no_zone_filter", action="store_true")
    parser.add_argument("--preset", type=str, choices=["final"], default=None,
                         help="Use the validated 'final' config (BOS-only + adaptive structure target)")
    parser.add_argument("--timeframe", type=str, choices=["4h", "1h", "1h_mtf"], default="1h_mtf",
                         help="Which validated config to use with --preset (default: 1h_mtf, best-validated)")
    parser.add_argument("--out_prefix", type=str, default="results")
    args = parser.parse_args()

    df = load_csv(args.csv)

    if args.preset == "final":
        params = dict(TIMEFRAME_CONFIGS[args.timeframe])
    else:
        params = dict(
            reward_risk=args.reward_risk,
            risk_pct=args.risk_pct,
            zone_filter=not args.no_zone_filter,
        )

    trades_df, equity_curve, enriched_df = run_backtest(df, **params)
    stats = summarize(trades_df, equity_curve, starting_equity=10000.0)

    print("\n=== SMC Backtest Results ===")
    for k, v in stats.items():
        print(f"{k}: {v}")

    if not trades_df.empty:
        trades_df.to_csv(f"{args.out_prefix}_trades.csv", index=False)
    plot_results(enriched_df, trades_df, equity_curve, f"{args.out_prefix}_chart.png")
    print(f"\nSaved: {args.out_prefix}_trades.csv, {args.out_prefix}_chart.png")


if __name__ == "__main__":
    main()
