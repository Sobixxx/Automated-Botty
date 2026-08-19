"""
smc_core.py -- Smart Money Concepts detection primitives.
Rule-based, independent implementation (not a copy of any proprietary
indicator), used by both the backtester and the live testnet bot.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass


@dataclass
class OrderBlock:
    idx: int              # source candle (down/up-close candle itself)
    confirmed_idx: int    # bar where the structure event confirmed this OB
    event_type: str       # "BOS_up", "CHoCH_up", "BOS_down", "CHoCH_down"
    direction: str        # "bull" or "bear"
    top: float
    bottom: float


def find_swings(df: pd.DataFrame, lookback: int = 12):
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
    """Walk the series, flagging BOS (continuation) vs CHoCH (reversal)."""
    n = len(df)
    close = df["close"].values
    swing_high_flags = df["swing_high"].values
    swing_low_flags = df["swing_low"].values
    highs = df["high"].values
    lows = df["low"].values

    event = np.array([None] * n, dtype=object)
    trend = np.zeros(n, dtype=int)

    last_swing_high = None
    last_swing_low = None
    current_trend = 0

    for i in range(n):
        if swing_high_flags[i]:
            last_swing_high = highs[i]
        if swing_low_flags[i]:
            last_swing_low = lows[i]

        if last_swing_high is not None and close[i] > last_swing_high:
            event[i] = "CHoCH_up" if current_trend <= 0 else "BOS_up"
            current_trend = 1
            last_swing_high = None
        elif last_swing_low is not None and close[i] < last_swing_low:
            event[i] = "CHoCH_down" if current_trend >= 0 else "BOS_down"
            current_trend = -1
            last_swing_low = None

        trend[i] = current_trend

    df = df.copy()
    df["structure_event"] = event
    df["trend"] = trend
    return df


def detect_order_blocks(df: pd.DataFrame, max_lookback: int = 20):
    """Last opposite-close candle before an impulsive structural break."""
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
                if close[j] < open_[j]:
                    blocks.append(OrderBlock(idx=j, confirmed_idx=i, event_type=ev,
                                              direction="bull", top=high[j], bottom=low[j]))
                    break
        elif ev in ("BOS_down", "CHoCH_down"):
            for j in range(i - 1, max(0, i - max_lookback) - 1, -1):
                if close[j] > open_[j]:
                    blocks.append(OrderBlock(idx=j, confirmed_idx=i, event_type=ev,
                                              direction="bear", top=high[j], bottom=low[j]))
                    break
    return blocks


def atr(df: pd.DataFrame, period: int = 14):
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low, (high - prev_close).abs(), (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period, min_periods=period).mean()


def higher_timeframe_trend(df: pd.DataFrame, rule: str = "4h", swing_lookback: int = 12):
    """
    True BOS/CHoCH-based higher-timeframe bias: resample to the HTF rule,
    run the same swing/structure logic there, then map each HTF trend
    value back onto every original (LTF) bar using only fully-closed
    HTF bars (no lookahead).
    """
    htf = df.set_index("timestamp").resample(rule).agg({
        "open": "first", "high": "max", "low": "min", "close": "last",
        "volume": "sum",
    }).dropna().reset_index()

    htf = find_swings(htf, lookback=swing_lookback)
    htf = detect_structure(htf)

    htf_trend_series = htf.set_index("timestamp")["trend"]
    mapped = df["timestamp"].apply(
        lambda t: htf_trend_series[htf_trend_series.index <= t].iloc[-1]
        if (htf_trend_series.index <= t).any() else 0
    )
    df = df.copy()
    df["htf_trend"] = mapped.values
    return df


def add_classical_indicators(df: pd.DataFrame, macd_fast=12, macd_slow=26, macd_signal=9,
                              bb_period=20, bb_mult=2.0):
    """MACD and Bollinger Bands."""
    df = df.copy()
    ema_fast = df["close"].ewm(span=macd_fast, adjust=False).mean()
    ema_slow = df["close"].ewm(span=macd_slow, adjust=False).mean()
    df["macd"] = ema_fast - ema_slow
    df["macd_signal"] = df["macd"].ewm(span=macd_signal, adjust=False).mean()
    df["macd_hist"] = df["macd"] - df["macd_signal"]

    mid = df["close"].rolling(bb_period).mean()
    std = df["close"].rolling(bb_period).std()
    df["bb_mid"] = mid
    df["bb_upper"] = mid + bb_mult * std
    df["bb_lower"] = mid - bb_mult * std
    return df
