"""
Simplified, rule-based Smart Money Concepts detection.
This approximates the logic described publicly for LuxAlgo-style SMC
indicators (swing pivots, BOS/CHoCH, order blocks, FVGs, premium/discount
zones). It is an independent re-implementation for backtesting purposes,
not a copy of any proprietary script.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass, field


@dataclass
class OrderBlock:
    idx: int             # the source candle (down/up-close candle itself)
    confirmed_idx: int    # the bar where the structure event CONFIRMED this OB exists
    event_type: str       # "BOS_up","CHoCH_up","BOS_down","CHoCH_down" -- what confirmed it
    direction: str      # "bull" or "bear"
    top: float
    bottom: float
    mitigated: bool = False
    mitigated_idx: int = None


@dataclass
class FVG:
    idx: int
    direction: str
    top: float
    bottom: float
    filled: bool = False


def find_swings(df: pd.DataFrame, lookback: int = 5):
    """Pivot-based swing high/low detection."""
    highs = df["high"].values
    lows = df["low"].values
    n = len(df)
    swing_high = np.full(n, False)
    swing_low = np.full(n, False)

    for i in range(lookback, n - lookback):
        window_h = highs[i - lookback:i + lookback + 1]
        window_l = lows[i - lookback:i + lookback + 1]
        if highs[i] == window_h.max() and np.argmax(window_h) == lookback:
            swing_high[i] = True
        if lows[i] == window_l.min() and np.argmin(window_l) == lookback:
            swing_low[i] = True

    df = df.copy()
    df["swing_high"] = swing_high
    df["swing_low"] = swing_low
    return df


def detect_structure(df: pd.DataFrame):
    """
    Walk the series tracking the last confirmed swing high/low and flag
    BOS (continuation) vs CHoCH (reversal) events on close-through breaks.
    """
    n = len(df)
    close = df["close"].values
    swing_high_flags = df["swing_high"].values
    swing_low_flags = df["swing_low"].values
    highs = df["high"].values
    lows = df["low"].values

    event = np.array([None] * n, dtype=object)   # "BOS_up","CHoCH_up","BOS_down","CHoCH_down"
    trend = np.zeros(n, dtype=int)                # 1 = up, -1 = down, 0 = undefined

    last_swing_high = None
    last_swing_low = None
    current_trend = 0

    for i in range(n):
        # update the last confirmed swing points as they appear
        if swing_high_flags[i]:
            last_swing_high = highs[i]
        if swing_low_flags[i]:
            last_swing_low = lows[i]

        if last_swing_high is not None and close[i] > last_swing_high:
            if current_trend <= 0:
                event[i] = "CHoCH_up"
                current_trend = 1
            else:
                event[i] = "BOS_up"
            last_swing_high = None  # consumed, wait for a new one

        elif last_swing_low is not None and close[i] < last_swing_low:
            if current_trend >= 0:
                event[i] = "CHoCH_down"
                current_trend = -1
            else:
                event[i] = "BOS_down"
            last_swing_low = None

        trend[i] = current_trend

    df = df.copy()
    df["structure_event"] = event
    df["trend"] = trend
    return df


def detect_order_blocks(df: pd.DataFrame, max_lookback: int = 15):
    """
    For each bullish structure event, the order block is the last down-close
    candle before the impulsive move. Mirror for bearish events.
    """
    n = len(df)
    open_ = df["open"].values
    close = df["close"].values
    high = df["high"].values
    low = df["low"].values
    events = df["structure_event"].values

    blocks = []
    for i in range(n):
        ev = events[i]
        if ev in ("BOS_up", "CHoCH_up"):
            for j in range(i - 1, max(0, i - max_lookback) - 1, -1):
                if close[j] < open_[j]:  # down-close candle
                    blocks.append(OrderBlock(idx=j, confirmed_idx=i, event_type=ev, direction="bull", top=high[j], bottom=low[j]))
                    break
        elif ev in ("BOS_down", "CHoCH_down"):
            for j in range(i - 1, max(0, i - max_lookback) - 1, -1):
                if close[j] > open_[j]:  # up-close candle
                    blocks.append(OrderBlock(idx=j, confirmed_idx=i, event_type=ev, direction="bear", top=high[j], bottom=low[j]))
                    break
    return blocks


def fvg_fill_indices(df: pd.DataFrame, fvgs):
    """For each FVG, find the first later bar where price fully closes the gap."""
    low = df["low"].values
    high = df["high"].values
    n = len(df)
    filled_idx_map = {}
    for fvg in fvgs:
        filled = None
        for j in range(fvg.idx + 1, n):
            if fvg.direction == "bull" and low[j] <= fvg.bottom:
                filled = j
                break
            if fvg.direction == "bear" and high[j] >= fvg.top:
                filled = j
                break
        filled_idx_map[id(fvg)] = filled
    return filled_idx_map


def detect_fvgs(df: pd.DataFrame):
    """3-candle imbalance detection."""
    n = len(df)
    high = df["high"].values
    low = df["low"].values
    fvgs = []
    for i in range(2, n):
        if low[i] > high[i - 2]:
            fvgs.append(FVG(idx=i, direction="bull", top=low[i], bottom=high[i - 2]))
        elif high[i] < low[i - 2]:
            fvgs.append(FVG(idx=i, direction="bear", top=low[i - 2], bottom=high[i]))
    return fvgs


def atr(df: pd.DataFrame, period: int = 14):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def higher_timeframe_trend_ema(df: pd.DataFrame, rule: str = "4h", ema_period: int = 50):
    """
    EMA-crossover HTF bias -- matches the Pine Script v4 implementation:
    htfClose = prior HTF bar's close, htfEma = EMA(50) of HTF close (also
    offset by one bar). Bias up if htfClose > htfEma, down if less.
    This is a SIMPLER proxy than the BOS/CHoCH-based higher_timeframe_trend()
    above -- deliberately replicated here to match what the Pine script
    actually does, not what the original Python backtest used.
    """
    htf = df.set_index("timestamp").resample(rule).agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum",
    }).dropna().reset_index()

    htf["ema50"] = htf["close"].ewm(span=ema_period, adjust=False).mean()
    # Pine uses close[1]/ema[1] (prior closed HTF bar) via request.security lookahead_off
    htf["htf_close_prev"] = htf["close"].shift(1)
    htf["htf_ema_prev"] = htf["ema50"].shift(1)
    htf["bias"] = np.where(htf["htf_close_prev"] > htf["htf_ema_prev"], 1,
                    np.where(htf["htf_close_prev"] < htf["htf_ema_prev"], -1, 0))

    bias_series = htf.set_index("timestamp")["bias"]
    mapped = df["timestamp"].apply(
        lambda t: bias_series[bias_series.index <= t].iloc[-1]
        if (bias_series.index <= t).any() else 0
    )
    df = df.copy()
    df["htf_trend"] = mapped.values
    return df


def higher_timeframe_trend(df: pd.DataFrame, rule: str = "1D", swing_lookback: int = 5):
    """
    Resample to a higher timeframe (e.g. Daily), run the same swing/BOS/CHoCH
    logic, then map each HTF trend value back down onto every original bar.
    """
    htf = df.set_index("timestamp").resample(rule).agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum",
    }).dropna().reset_index()

    htf = find_swings(htf, lookback=swing_lookback)
    htf = detect_structure(htf)

    # forward-map: for each original bar, use the most recently completed HTF trend
    htf_trend_series = htf.set_index("timestamp")["trend"]
    mapped = df["timestamp"].apply(
        lambda t: htf_trend_series[htf_trend_series.index <= t].iloc[-1]
        if (htf_trend_series.index <= t).any() else 0
    )
    df = df.copy()
    df["htf_trend"] = mapped.values
    return df


def adx_indicator(df: pd.DataFrame, period=14):
    """Standard Wilder ADX (trend-strength), plus +DI/-DI."""
    high = df["high"]
    low = df["low"]
    close = df["close"]
    prev_close = close.shift(1)
    prev_high = high.shift(1)
    prev_low = low.shift(1)

    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(axis=1)
    atr_smooth = tr.ewm(alpha=1/period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr_smooth
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1/period, adjust=False).mean() / atr_smooth
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(alpha=1/period, adjust=False).mean()

    df = df.copy()
    df["adx"] = adx
    df["plus_di"] = plus_di
    df["minus_di"] = minus_di
    return df


def parabolic_sar(df: pd.DataFrame, af_start=0.02, af_step=0.02, af_max=0.2):
    """Standard Wilder Parabolic SAR."""
    high = df["high"].values
    low = df["low"].values
    n = len(df)
    sar = np.zeros(n)
    trend = np.zeros(n, dtype=int)  # 1 = up, -1 = down

    # init: assume uptrend starting, SAR = first low
    trend[0] = 1
    sar[0] = low[0]
    ep = high[0]   # extreme point
    af = af_start

    for i in range(1, n):
        prev_sar = sar[i - 1]
        if trend[i - 1] == 1:
            new_sar = prev_sar + af * (ep - prev_sar)
            new_sar = min(new_sar, low[i - 1], low[i - 2] if i >= 2 else low[i - 1])
            if low[i] < new_sar:
                trend[i] = -1
                sar[i] = ep
                ep = low[i]
                af = af_start
            else:
                trend[i] = 1
                sar[i] = new_sar
                if high[i] > ep:
                    ep = high[i]
                    af = min(af + af_step, af_max)
        else:
            new_sar = prev_sar + af * (ep - prev_sar)
            new_sar = max(new_sar, high[i - 1], high[i - 2] if i >= 2 else high[i - 1])
            if high[i] > new_sar:
                trend[i] = 1
                sar[i] = ep
                ep = high[i]
                af = af_start
            else:
                trend[i] = -1
                sar[i] = new_sar
                if low[i] < ep:
                    ep = low[i]
                    af = min(af + af_step, af_max)

    df = df.copy()
    df["sar"] = sar
    df["sar_trend"] = trend
    return df


def add_classical_indicators(df: pd.DataFrame, ema_fast=50, ema_slow=200, rsi_period=14,
                               macd_fast=12, macd_slow=26, macd_signal=9, bb_period=20):
    """EMA trend filter, MACD, RSI, Bollinger Bands -- classical TA layer."""
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=ema_fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=ema_slow, adjust=False).mean()

    ema_f = df["close"].ewm(span=macd_fast, adjust=False).mean()
    ema_s = df["close"].ewm(span=macd_slow, adjust=False).mean()
    df["macd"] = ema_f - ema_s
    df["macd_signal"] = df["macd"].ewm(span=macd_signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    delta = df["close"].diff()
    gain = delta.clip(lower=0).rolling(rsi_period).mean()
    loss = (-delta.clip(upper=0)).rolling(rsi_period).mean()
    rs = gain / loss.replace(0, np.nan)
    df["rsi"] = 100 - (100 / (1 + rs))

    mid = df["close"].rolling(bb_period).mean()
    std = df["close"].rolling(bb_period).std()
    df["bb_mid"] = mid
    df["bb_upper"] = mid + 2 * std
    df["bb_lower"] = mid - 2 * std
    df["bb_width"] = (df["bb_upper"] - df["bb_lower"]) / mid

    df = parabolic_sar(df)

    return df


def premium_discount(df: pd.DataFrame, lookback: int = 50):
    """Rolling range-based premium(>0.5)/discount(<0.5) position, 0-1 scale."""
    roll_high = df["high"].rolling(lookback, min_periods=lookback).max()
    roll_low = df["low"].rolling(lookback, min_periods=lookback).min()
    rng = (roll_high - roll_low).replace(0, np.nan)
    pos = (df["close"] - roll_low) / rng
    df = df.copy()
    df["range_position"] = pos  # <0.5 discount, >0.5 premium
    return df
