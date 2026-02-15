#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SMART GUARDIAN V39.7 - STREAMLIT DASHBOARD
Modern visual interface for EUR/USD Trading Analysis
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import sys

# Import toate funcțiile din codul original
# Presupunem că ai salvat codul original ca 'guardian_core.py'
# Dacă nu, includem tot aici
import os
import csv
import json
import time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, List, Tuple, Any

# Third-party imports
try:
    import requests
    import numpy as np
    import yfinance as yf
    DEPENDENCIES_OK = True
except ImportError as e:
    st.error(f"Missing dependency: {e}")
    st.error("Install with: pip install requests pandas numpy yfinance MetaTrader5 plotly streamlit")
    DEPENDENCIES_OK = False
    st.stop()

# MT5 import (optional)
try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
except ImportError:
    MT5_AVAILABLE = False

# ============================================================================
# COPIAZĂ TOT DIN CODUL ORIGINAL (dataclasses, config, functions)
# ============================================================================

CONFIG = {
    'csv_file': 'guardian_v39_7_tracking.csv',
    'cache_file': 'guardian_v39_7_cache.json',
    'cache_duration': 3600,
    'fred_api_key': '4fa2b1e89ac686cc006839fe22973292',
    'mt5_timeout': 60000,
    'mt5_symbol': 'EURUSD',
    'rsi_period': 14,
    'atr_period': 14,
}

@dataclass
class MeanReversionAnalysis:
    active: bool
    phase: str
    direction: str
    magnitude: float
    momentum_weight: float
    explanation: str

@dataclass
class US2YAnalysis:
    us2y: float
    de2y: float
    us10y: float
    differential: float
    change_7d: float
    yield_curve: float
    carry_signal: str
    policy_bias: str
    alert_level: str
    verdict: str
    weight_override: float
    interpretation: str
    recommendation: str

@dataclass
class SpikeAnalysis:
    spike_value: float
    level: str
    volatility_risk: str
    confluence_active: bool
    confluence_level: str
    override_weight: float
    explanation: str
    recommendation: str

@dataclass
class PriceZScoreAnalysis:
    price_z_value: float
    level: str
    volatility_risk: str
    direction: str
    confidence_adjustment: float
    explanation: str
    recommendation: str

@dataclass
class DivergenceAnalysis:
    detected: bool
    divergence_type: str
    priority_override: bool
    override_weight: float
    explanation: str
    recommendation: str

@dataclass
class VolumeAnalysis:
    current_volume: float
    avg_volume_20: float
    volume_ratio: float
    obv_current: float
    obv_slope: str
    obv_divergence: bool
    obv_divergence_type: str
    delta_5d: float
    delta_signal: str
    pv_relationship: str
    smart_money_signal: str
    phase_detected: str
    consolidation_quality: str
    wick_pressure: str
    range_duration: int
    verdict: str
    confidence_boost: float
    explanation: str
    recommendation: str

@dataclass
class GeopoliticalAlert:
    level: str
    trigger_type: str
    severity: float
    override_weight: float
    explanation: str

@dataclass
class LayerVerdict:
    layer_name: str
    verdict: str
    confidence: float
    weight: float
    key_metrics: Dict[str, Any]
    explanation: str

@dataclass
class GuardianReport:
    timestamp: str
    z_score: float
    z_score_spike: float
    z_score_price: float
    spread: float
    ma_value: float
    vix: float
    correlation_h4: float
    us2y: float
    de2y: float
    us10y: float
    rsi_daily: float
    z_score_prev: float
    z_score_2days: float
    rsi_prev: float
    rsi_2days: float
    us2y_diff_7d_ago: float
    macro_verdict: LayerVerdict
    us2y_analysis: US2YAnalysis
    institutional_verdict: LayerVerdict
    correlation_verdict: LayerVerdict
    spike_analysis: SpikeAnalysis
    price_z_analysis: PriceZScoreAnalysis
    divergence_analysis: DivergenceAnalysis
    volume_analysis: VolumeAnalysis
    geopolitical_alert: GeopoliticalAlert
    technical_verdict: LayerVerdict
    mr_z_analysis: MeanReversionAnalysis
    mr_rsi_analysis: MeanReversionAnalysis
    risk_verdict: LayerVerdict
    v38_integrated_score: float
    v38_phase: str
    v38_action: str
    v38_confidence: float
    priority_evaluation: str
    final_action: str
    final_confidence: float
    final_explanation: str
    execution_plan: str
    double_mr_confirmation: bool
    active_overrides: List[str]

# ============================================================================
# UTILITY FUNCTIONS (copy from original)
# ============================================================================

def load_cache() -> Dict:
    if not Path(CONFIG['cache_file']).exists():
        return {}
    try:
        with open(CONFIG['cache_file'], 'r') as f:
            cache = json.load(f)
        if time.time() - cache.get('timestamp', 0) > CONFIG['cache_duration']:
            return {}
        return cache
    except:
        return {}

def save_cache(data: Dict):
    data['timestamp'] = time.time()
    with open(CONFIG['cache_file'], 'w') as f:
        json.dump(data, f, indent=2)

def calculate_rsi(prices: pd.Series, period: int = 14) -> float:
    if len(prices) < period + 1:
        return 50.0
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return float(rsi.iloc[-1])

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return float(atr.iloc[-1])

def init_mt5() -> bool:
    if not MT5_AVAILABLE:
        return False
    if not mt5.initialize():
        return False
    return True

def get_mt5_data(symbol: str = 'EURUSD', timeframe=None, bars: int = 100) -> Optional[pd.DataFrame]:
    if not MT5_AVAILABLE or not init_mt5():
        return None
    if timeframe is None:
        timeframe = mt5.TIMEFRAME_D1
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None or len(rates) == 0:
            return None
        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s')
        return df
    except Exception as e:
        return None
    finally:
        mt5.shutdown()

def get_eurusd_data_with_fallback(bars: int = 100) -> pd.DataFrame:
    df = get_mt5_data('EURUSD', bars=bars)
    if df is not None and len(df) > 0:
        return df
    try:
        ticker = yf.Ticker('EURUSD=X')
        df = ticker.history(period='6mo')
        if df.empty:
            raise Exception("YFinance returned empty data")
        df = df.rename(columns={
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'tick_volume'
        })
        df = df.reset_index()
        df = df.rename(columns={'Date': 'time'})
        return df
    except Exception as e:
        raise Exception("Could not fetch EUR/USD data from any source")

def fetch_fred_rate(series_id: str, cache: Dict) -> Optional[float]:
    cache_key = f'fred_{series_id}'
    if cache_key in cache:
        return cache[cache_key]
    if not CONFIG['fred_api_key']:
        return None
    url = f"https://api.stlouisfed.org/fred/series/observations"
    params = {
        'series_id': series_id,
        'api_key': CONFIG['fred_api_key'],
        'file_type': 'json',
        'sort_order': 'desc',
        'limit': 1
    }
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if 'observations' in data and len(data['observations']) > 0:
            value = float(data['observations'][0]['value'])
            cache[cache_key] = value
            return value
    except Exception as e:
        pass
    return None

def get_macro_rates(cache: Dict) -> Tuple[Optional[float], Optional[float]]:
    us_real = fetch_fred_rate('REAINTRATREARAT10Y', cache)
    eu_real = fetch_fred_rate('ECBDFR', cache)
    return us_real, eu_real

def get_historical_z_scores() -> Tuple[float, float, float]:
    if not Path(CONFIG['csv_file']).exists():
        return 0.0, 0.0, 0.0
    try:
        df = pd.read_csv(CONFIG['csv_file'])
        if len(df) == 0:
            return 0.0, 0.0, 0.0
        recent = df.tail(3)
        if len(recent) >= 3:
            return (
                float(recent.iloc[-1]['Z_Score']),
                float(recent.iloc[-2]['Z_Score']),
                float(recent.iloc[-3]['Z_Score'])
            )
        elif len(recent) == 2:
            return float(recent.iloc[-1]['Z_Score']), float(recent.iloc[-2]['Z_Score']), 0.0
        elif len(recent) == 1:
            return float(recent.iloc[-1]['Z_Score']), 0.0, 0.0
        else:
            return 0.0, 0.0, 0.0
    except Exception as e:
        return 0.0, 0.0, 0.0

def get_historical_rsi() -> Tuple[float, float, float]:
    if not Path(CONFIG['csv_file']).exists():
        return 50.0, 50.0, 50.0
    try:
        df = pd.read_csv(CONFIG['csv_file'])
        if len(df) == 0:
            return 50.0, 50.0, 50.0
        recent = df.tail(3)
        if len(recent) >= 3:
            return (
                float(recent.iloc[-1]['RSI_Daily']),
                float(recent.iloc[-2]['RSI_Daily']),
                float(recent.iloc[-3]['RSI_Daily'])
            )
        elif len(recent) == 2:
            return float(recent.iloc[-1]['RSI_Daily']), float(recent.iloc[-2]['RSI_Daily']), 50.0
        elif len(recent) == 1:
            return float(recent.iloc[-1]['RSI_Daily']), 50.0, 50.0
        else:
            return 50.0, 50.0, 50.0
    except Exception as e:
        return 50.0, 50.0, 50.0

def get_historical_us2y_differential() -> float:
    if not Path(CONFIG['csv_file']).exists():
        return 0.0
    try:
        df = pd.read_csv(CONFIG['csv_file'])
        if len(df) < 7:
            return 0.0
        row_7d = df.iloc[-7]
        return float(row_7d.get('US2Y_Diff', 0.0))
    except Exception as e:
        return 0.0

def detect_mean_reversion_z(z: float, z_prev: float, z_2days: float) -> MeanReversionAnalysis:
    if -0.5 <= z <= 0.5:
        return MeanReversionAnalysis(
            active=False,
            phase='EQUILIBRIUM',
            direction='NEUTRAL',
            magnitude=0.0,
            momentum_weight=1.0,
            explanation='Z-Score in echilibru (-0.5 to 0.5). Nicio presiune Mean Reversion.'
        )
    from_extreme_high = z_2days > 2.0 or z_prev > 2.0
    from_extreme_low = z_2days < -2.0 or z_prev < -2.0
    in_extreme_high = z > 2.0
    in_extreme_low = z < -2.0
    crossed_zero_down = z_2days > 0.5 and z < -0.5
    crossed_zero_up = z_2days < -0.5 and z > 0.5
    if crossed_zero_down:
        return MeanReversionAnalysis(
            active=True,
            phase='TREND_REVERSAL',
            direction='BEARISH',
            magnitude=abs(z),
            momentum_weight=0.5,
            explanation=f'TREND REVERSAL detectat: Z-Score a trecut de la zona pozitiva ({z_2days:.2f}) la negativ ({z:.2f}). Posibila inversare trend BEARISH.'
        )
    if crossed_zero_up:
        return MeanReversionAnalysis(
            active=True,
            phase='TREND_REVERSAL',
            direction='BULLISH',
            magnitude=abs(z),
            momentum_weight=0.5,
            explanation=f'TREND REVERSAL detectat: Z-Score a trecut de la zona negativa ({z_2days:.2f}) la pozitiv ({z:.2f}). Posibila inversare trend BULLISH.'
        )
    if from_extreme_high and -0.5 < z < 0.5:
        return MeanReversionAnalysis(
            active=True,
            phase='RETEST',
            direction='BEARISH',
            magnitude=abs(z_2days),
            momentum_weight=0.4,
            explanation=f'RETEST dupa reversal: Z-Score testeaza directia bearish dupa ce a fost extrem de pozitiv ({z_2days:.2f}). Posibila continuare Mean Reversion.'
        )
    if from_extreme_low and -0.5 < z < 0.5:
        return MeanReversionAnalysis(
            active=True,
            phase='RETEST',
            direction='BULLISH',
            magnitude=abs(z_2days),
            momentum_weight=0.4,
            explanation=f'RETEST dupa reversal: Z-Score testeaza directia bullish dupa ce a fost extrem de negativ ({z_2days:.2f}). Posibila continuare Mean Reversion.'
        )
    if in_extreme_high:
        return MeanReversionAnalysis(
            active=True,
            phase='ACTIVE',
            direction='BEARISH',
            magnitude=abs(z),
            momentum_weight=0.3,
            explanation=f'Mean Reversion ACTIV: Z-Score extrem de pozitiv ({z:.2f}). Spread prea larg, expect revenire. Directie BEARISH.'
        )
    if in_extreme_low:
        return MeanReversionAnalysis(
            active=True,
            phase='ACTIVE',
            direction='BULLISH',
            magnitude=abs(z),
            momentum_weight=0.3,
            explanation=f'Mean Reversion ACTIV: Z-Score extrem de negativ ({z:.2f}). Spread prea ingust, expect revenire. Directie BULLISH.'
        )
    return MeanReversionAnalysis(
        active=False,
        phase='NONE',
        direction='NEUTRAL',
        magnitude=0.0,
        momentum_weight=1.0,
        explanation=f'Z-Score la {z:.2f}. Nicio presiune Mean Reversion detectata.'
    )

def detect_mean_reversion_rsi(rsi: float, rsi_prev: float, rsi_2days: float) -> MeanReversionAnalysis:
    if 45 <= rsi <= 55:
        return MeanReversionAnalysis(
            active=False,
            phase='EQUILIBRIUM',
            direction='NEUTRAL',
            magnitude=0.0,
            momentum_weight=1.0,
            explanation='RSI in echilibru (45-55). Nicio presiune Mean Reversion.'
        )
    from_overbought = rsi_2days > 70 or rsi_prev > 70
    from_oversold = rsi_2days < 30 or rsi_prev < 30
    in_overbought = rsi > 70
    in_oversold = rsi < 30
    crossed_50_down = rsi_2days > 55 and rsi < 45
    crossed_50_up = rsi_2days < 45 and rsi > 55
    if crossed_50_down:
        return MeanReversionAnalysis(
            active=True,
            phase='TREND_REVERSAL',
            direction='BEARISH',
            magnitude=abs(rsi - 50),
            momentum_weight=0.5,
            explanation=f'TREND REVERSAL RSI detectat: RSI a trecut de la zona superioara ({rsi_2days:.1f}) sub 50 ({rsi:.1f}). Posibila inversare trend BEARISH.'
        )
    if crossed_50_up:
        return MeanReversionAnalysis(
            active=True,
            phase='TREND_REVERSAL',
            direction='BULLISH',
            magnitude=abs(rsi - 50),
            momentum_weight=0.5,
            explanation=f'TREND REVERSAL RSI detectat: RSI a trecut de la zona inferioara ({rsi_2days:.1f}) peste 50 ({rsi:.1f}). Posibila inversare trend BULLISH.'
        )
    if from_overbought and 45 < rsi < 55:
        return MeanReversionAnalysis(
            active=True,
            phase='RETEST',
            direction='BEARISH',
            magnitude=abs(rsi_2days - 50),
            momentum_weight=0.4,
            explanation=f'RETEST RSI dupa reversal: RSI testeaza directia bearish dupa ce a fost overbought ({rsi_2days:.1f}). Posibila continuare Mean Reversion.'
        )
    if from_oversold and 45 < rsi < 55:
        return MeanReversionAnalysis(
            active=True,
            phase='RETEST',
            direction='BULLISH',
            magnitude=abs(rsi_2days - 50),
            momentum_weight=0.4,
            explanation=f'RETEST RSI dupa reversal: RSI testeaza directia bullish dupa ce a fost oversold ({rsi_2days:.1f}). Posibila continuare Mean Reversion.'
        )
    if in_overbought:
        return MeanReversionAnalysis(
            active=True,
            phase='ACTIVE',
            direction='BEARISH',
            magnitude=rsi - 50,
            momentum_weight=0.3,
            explanation=f'Mean Reversion RSI ACTIV: RSI overbought ({rsi:.1f}). EUR/USD supracumparat, expect corecție. Directie BEARISH.'
        )
    if in_oversold:
        return MeanReversionAnalysis(
            active=True,
            phase='ACTIVE',
            direction='BULLISH',
            magnitude=50 - rsi,
            momentum_weight=0.3,
            explanation=f'Mean Reversion RSI ACTIV: RSI oversold ({rsi:.1f}). EUR/USD supravandut, expect rebound. Directie BULLISH.'
        )
    return MeanReversionAnalysis(
        active=False,
        phase='NONE',
        direction='NEUTRAL',
        magnitude=0.0,
        momentum_weight=1.0,
        explanation=f'RSI la {rsi:.1f}. Nicio presiune Mean Reversion detectata.'
    )

def analyze_spike_volatility(z_spike: float, mr_z: MeanReversionAnalysis) -> SpikeAnalysis:
    spike_abs = abs(z_spike)
    if spike_abs > 3.0:
        level = 'EXTREME'
        volatility_risk = 'CRITICAL'
    elif spike_abs > 2.5:
        level = 'SPIKE'
        volatility_risk = 'ELEVATED'
    else:
        level = 'NORMAL'
        volatility_risk = 'Normal'
    mr_active = mr_z.active and mr_z.phase in ['ACTIVE', 'TREND_REVERSAL']
    confluence_critical = mr_active and spike_abs > 3.0
    confluence_warning = mr_active and spike_abs > 2.5
    if confluence_critical:
        confluence_level = 'CRITICAL'
        override_weight = 0.25
        confluence_active = True
        explanation = f"CONFLUENCE CRITICAL: Strategic Z-Score Mean Reversion {mr_z.direction} ({mr_z.phase}) + Spike EXTREME ({z_spike:.2f}). Probabilitate MARE de exhaustion. Volatilitate critica!"
        recommendation = "DO NOT CHASE the move! Wait pentru confirmarea reversal. Spike extreme + MR activ = posibil turning point."
    elif confluence_warning:
        confluence_level = 'WARNING'
        override_weight = 0.15
        confluence_active = True
        explanation = f"CONFLUENCE WARNING: Strategic Z-Score Mean Reversion {mr_z.direction} + Spike detectat ({z_spike:.2f}). Volatilitate ridicata. High risk environment."
        recommendation = "Trade cu cautiune. Tighten stops. Spike + MR = risc crescut de inversare brusca."
    elif spike_abs > 3.0:
        confluence_level = 'NONE'
        override_weight = 0.10
        confluence_active = False
        explanation = f"SPIKE EXTREME detectat ({z_spike:.2f}) fara MR activ. Volatilitate critica, dar fara confluenta strategica."
        recommendation = "Volatilitate extrema. Reduce pozitii, tighten stops. Asteapta stabilizare."
    elif spike_abs > 2.5:
        confluence_level = 'NONE'
        override_weight = 0.05
        confluence_active = False
        explanation = f"SPIKE moderat detectat ({z_spike:.2f}). Volatilitate crescuta."
        recommendation = "Monitorizare atenta. Posibile miscari brusce."
    else:
        confluence_level = 'NONE'
        override_weight = 0.0
        confluence_active = False
        explanation = f"Z-Score Spike normal ({z_spike:.2f}). Volatilitate in parametri normali."
        recommendation = "Nicio alerta spike. Proceed normal."
    return SpikeAnalysis(
        spike_value=z_spike,
        level=level,
        volatility_risk=volatility_risk,
        confluence_active=confluence_active,
        confluence_level=confluence_level,
        override_weight=override_weight,
        explanation=explanation,
        recommendation=recommendation
    )

def fetch_volume_data(bars: int = 30) -> Optional[pd.DataFrame]:
    if MT5_AVAILABLE and init_mt5():
        try:
            rates = mt5.copy_rates_from_pos('EURUSD', mt5.TIMEFRAME_D1, 0, bars)
            mt5.shutdown()
            if rates is not None and len(rates) > 5:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                return df
        except Exception as e:
            pass
    try:
        ticker = yf.Ticker('EURUSD=X')
        df = ticker.history(period='3mo')
        if df.empty:
            return None
        df = df.rename(columns={'Open':'open','High':'high','Low':'low',
                                 'Close':'close','Volume':'tick_volume'})
        df = df.reset_index().rename(columns={'Date':'time'})
        return df
    except Exception as e:
        return None

def calculate_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    obv = [0.0]
    for i in range(1, len(close)):
        if close.iloc[i] > close.iloc[i-1]:
            obv.append(obv[-1] + volume.iloc[i])
        elif close.iloc[i] < close.iloc[i-1]:
            obv.append(obv[-1] - volume.iloc[i])
        else:
            obv.append(obv[-1])
    return pd.Series(obv, index=close.index)

def calculate_volume_delta(open_: pd.Series, close: pd.Series, volume: pd.Series) -> pd.Series:
    delta = pd.Series(0.0, index=close.index)
    bullish = close > open_
    bearish = close < open_
    delta[bullish] = volume[bullish]
    delta[bearish] = -volume[bearish]
    return delta

def analyze_volume(df: Optional[pd.DataFrame]) -> VolumeAnalysis:
    if df is None or len(df) < 15:
        return VolumeAnalysis(
            current_volume=0.0, avg_volume_20=0.0, volume_ratio=1.0,
            obv_current=0.0, obv_slope='FLAT', obv_divergence=False,
            obv_divergence_type='NONE', delta_5d=0.0,
            delta_signal='NEUTRAL', pv_relationship='NEUTRAL',
            smart_money_signal='ABSENT', phase_detected='NONE',
            consolidation_quality='NONE', wick_pressure='NEUTRAL',
            range_duration=0, verdict='NEUTRAL', confidence_boost=0.0,
            explanation='Date volum indisponibile. Volume layer inactiv.',
            recommendation='Nicio analiza volum posibila.'
        )
    for col in ['open', 'high', 'low', 'close', 'tick_volume']:
        if col not in df.columns:
            return VolumeAnalysis(
                current_volume=0.0, avg_volume_20=0.0, volume_ratio=1.0,
                obv_current=0.0, obv_slope='FLAT', obv_divergence=False,
                obv_divergence_type='NONE', delta_5d=0.0,
                delta_signal='NEUTRAL', pv_relationship='NEUTRAL',
                smart_money_signal='ABSENT', phase_detected='NONE',
                consolidation_quality='NONE', wick_pressure='NEUTRAL',
                range_duration=0, verdict='NEUTRAL', confidence_boost=0.0,
                explanation=f'Coloana {col} lipseste. Volume layer inactiv.',
                recommendation='Verifica sursa de date.'
            )
    close   = df['close']
    open_   = df['open']
    high    = df['high']
    low     = df['low']
    volume  = df['tick_volume'].astype(float)
    current_volume = float(volume.iloc[-1])
    avg_volume_20  = float(volume.tail(20).mean()) if len(volume) >= 20 else float(volume.mean())
    avg_volume_20  = max(avg_volume_20, 1.0)
    volume_ratio   = current_volume / avg_volume_20
    if volume_ratio > 1.5:
        vol_label = 'HIGH'
    elif volume_ratio < 0.5:
        vol_label = 'LOW'
    else:
        vol_label = 'NORMAL'
    obv = calculate_obv(close, volume)
    obv_current = float(obv.iloc[-1])
    obv_5d_ago = float(obv.iloc[-6]) if len(obv) >= 6 else float(obv.iloc[0])
    obv_change  = obv_current - obv_5d_ago
    obv_range   = float(obv.tail(20).max() - obv.tail(20).min()) if len(obv) >= 20 else 1.0
    obv_range   = max(obv_range, 1.0)
    if obv_change / obv_range > 0.10:
        obv_slope = 'RISING'
    elif obv_change / obv_range < -0.10:
        obv_slope = 'FALLING'
    else:
        obv_slope = 'FLAT'
    price_5d_change = float(close.iloc[-1]) - float(close.iloc[-6]) if len(close) >= 6 else 0.0
    price_rising    = price_5d_change > 0.0005
    price_falling   = price_5d_change < -0.0005
    obv_divergence       = False
    obv_divergence_type = 'NONE'
    if price_rising and obv_slope == 'FALLING':
        obv_divergence      = True
        obv_divergence_type = 'BEARISH_DIV'
    elif price_falling and obv_slope == 'RISING':
        obv_divergence      = True
        obv_divergence_type = 'BULLISH_DIV'
    delta    = calculate_volume_delta(open_, close, volume)
    delta_5d = float(delta.tail(5).sum())
    avg_vol_5d   = float(volume.tail(5).mean()) * 5
    avg_vol_5d   = max(avg_vol_5d, 1.0)
    delta_ratio  = delta_5d / avg_vol_5d
    if delta_ratio > 0.20:
        delta_signal = 'ACCUMULATION'
    elif delta_ratio < -0.20:
        delta_signal = 'DISTRIBUTION'
    else:
        delta_signal = 'NEUTRAL'
    current_bar_bullish = float(close.iloc[-1]) > float(open_.iloc[-1])
    if vol_label == 'HIGH' and current_bar_bullish:
        pv_relationship = 'CONFIRMED_BULL'
    elif vol_label == 'HIGH' and not current_bar_bullish:
        pv_relationship = 'CONFIRMED_BEAR'
    elif vol_label == 'LOW' and current_bar_bullish:
        pv_relationship = 'FAKE_BULL'
    elif vol_label == 'LOW' and not current_bar_bullish:
        pv_relationship = 'FAKE_BEAR'
    else:
        pv_relationship = 'NEUTRAL'
    window_days = 10
    recent_window = df.tail(window_days)
    window_high = recent_window['high'].max()
    window_low  = recent_window['low'].min()
    range_height = window_high - window_low
    avg_candle_size = (recent_window['high'] - recent_window['low']).mean()
    if range_height < (avg_candle_size * 3.5):
        consolidation_quality = 'TIGHT_RANGE'
    elif range_height < (avg_candle_size * 6.0):
        consolidation_quality = 'WIDE_RANGE'
    else:
        consolidation_quality = 'TRENDING'
    selling_wicks_count = 0
    buying_wicks_count = 0
    for i in range(len(recent_window)):
        row = recent_window.iloc[i]
        body_top = max(row['open'], row['close'])
        body_bottom = min(row['open'], row['close'])
        body_size = body_top - body_bottom
        upper_wick = row['high'] - body_top
        lower_wick = body_bottom - row['low']
        total_range = row['high'] - row['low']
        if total_range == 0: continue
        if upper_wick > (total_range * 0.35) and upper_wick > lower_wick:
            selling_wicks_count += 1
        elif lower_wick > (total_range * 0.35) and lower_wick > upper_wick:
            buying_wicks_count += 1
    if selling_wicks_count >= 3:
        wick_pressure = 'SELLING_WICKS'
    elif buying_wicks_count >= 3:
        wick_pressure = 'BUYING_WICKS'
    else:
        wick_pressure = 'NEUTRAL'
    phase_detected = 'NONE'
    obv_window_start = float(obv.iloc[-window_days])
    obv_window_end = obv_current
    obv_window_trend = 'FLAT'
    if obv_window_end < obv_window_start:
        obv_window_trend = 'FALLING'
    elif obv_window_end > obv_window_start:
        obv_window_trend = 'RISING'
    is_consolidation = consolidation_quality in ['TIGHT_RANGE', 'WIDE_RANGE']
    if is_consolidation:
        dist_signals = 0
        if obv_window_trend == 'FALLING': dist_signals += 1
        if wick_pressure == 'SELLING_WICKS': dist_signals += 1
        if obv_divergence_type == 'BEARISH_DIV': dist_signals += 1
        if delta_signal == 'DISTRIBUTION': dist_signals += 1
        if dist_signals >= 2:
            phase_detected = 'DISTRIBUTION_PHASE'
        accum_signals = 0
        if obv_window_trend == 'RISING': accum_signals += 1
        if wick_pressure == 'BUYING_WICKS': accum_signals += 1
        if obv_divergence_type == 'BULLISH_DIV': accum_signals += 1
        if delta_signal == 'ACCUMULATION': accum_signals += 1
        if accum_signals >= 2:
            phase_detected = 'ACCUMULATION_PHASE'
    price_range_pct = abs(float(close.iloc[-1]) - float(open_.iloc[-1])) / float(open_.iloc[-1]) * 100
    absorption = vol_label == 'HIGH' and price_range_pct < 0.15
    if absorption and current_bar_bullish:
        smart_money_signal = 'ACCUMULATING'
    elif absorption and not current_bar_bullish:
        smart_money_signal = 'DISTRIBUTING'
    elif phase_detected == 'ACCUMULATION_PHASE':
        smart_money_signal = 'ACCUMULATING_RANGE'
    elif phase_detected == 'DISTRIBUTION_PHASE':
        smart_money_signal = 'DISTRIBUTING_RANGE'
    elif delta_signal == 'ACCUMULATION' and obv_slope == 'RISING':
        smart_money_signal = 'ACCUMULATING'
    elif delta_signal == 'DISTRIBUTION' and obv_slope == 'FALLING':
        smart_money_signal = 'DISTRIBUTING'
    elif vol_label == 'LOW':
        smart_money_signal = 'ABSENT'
    else:
        smart_money_signal = 'NEUTRAL'
    bull_signals = sum([
        obv_slope == 'RISING',
        obv_divergence_type == 'BULLISH_DIV',
        delta_signal == 'ACCUMULATION',
        pv_relationship == 'CONFIRMED_BULL',
        smart_money_signal in ['ACCUMULATING', 'ACCUMULATING_RANGE'],
        phase_detected == 'ACCUMULATION_PHASE'
    ])
    bear_signals = sum([
        obv_slope == 'FALLING',
        obv_divergence_type == 'BEARISH_DIV',
        delta_signal == 'DISTRIBUTION',
        pv_relationship == 'CONFIRMED_BEAR',
        smart_money_signal in ['DISTRIBUTING', 'DISTRIBUTING_RANGE'],
        phase_detected == 'DISTRIBUTION_PHASE'
    ])
    fake_signals = pv_relationship in ['FAKE_BULL', 'FAKE_BEAR']
    confidence_boost = 0.0
    if bull_signals >= 3:
        verdict = 'BULLISH'
        confidence_boost = +0.15
    elif bear_signals >= 3:
        verdict = 'BEARISH'
        confidence_boost = -0.15
    elif bull_signals >= 2:
        verdict = 'BULLISH'
        confidence_boost = +0.08
    elif bear_signals >= 2:
        verdict = 'BEARISH'
        confidence_boost = -0.08
    else:
        verdict = 'NEUTRAL'
        confidence_boost = 0.0
    if phase_detected == 'DISTRIBUTION_PHASE':
        verdict = 'BEARISH'
        confidence_boost = min(confidence_boost, -0.20)
    elif phase_detected == 'ACCUMULATION_PHASE':
        verdict = 'BULLISH'
        confidence_boost = max(confidence_boost, +0.20)
    if fake_signals:
        confidence_boost *= 0.5
    parts = []
    parts.append(f"Volume ratio: {volume_ratio:.2f}x ({vol_label}).")
    if is_consolidation:
        parts.append(f"Range Analysis ({window_days}d): {consolidation_quality}.")
        parts.append(f"Wick Pressure: {wick_pressure} ({selling_wicks_count} up / {buying_wicks_count} down).")
    if phase_detected != 'NONE':
        parts.append(f"STRUCTURA WYCKOFF: {phase_detected} CONFIRMATA.")
    if obv_divergence:
        parts.append(f"OBV DIVERGENTA {obv_divergence_type}.")
    if smart_money_signal not in ['NEUTRAL', 'ABSENT']:
        parts.append(f"SMART MONEY: {smart_money_signal}.")
    explanation = ' '.join(parts)
    if verdict == 'BULLISH':
        recommendation = (f"Volume & Structure confirm BULLISH. "
                          f"Boost confidence +{abs(confidence_boost):.0%}.")
    elif verdict == 'BEARISH':
        recommendation = (f"Volume & Structure confirm BEARISH. "
                          f"Boost confidence +{abs(confidence_boost):.0%}.")
    else:
        recommendation = "Volume neutru. Nicio ajustare majora."
    return VolumeAnalysis(
        current_volume=current_volume,
        avg_volume_20=avg_volume_20,
        volume_ratio=volume_ratio,
        obv_current=obv_current,
        obv_slope=obv_slope,
        obv_divergence=obv_divergence,
        obv_divergence_type=obv_divergence_type,
        delta_5d=delta_5d,
        delta_signal=delta_signal,
        pv_relationship=pv_relationship,
        smart_money_signal=smart_money_signal,
        phase_detected=phase_detected,
        consolidation_quality=consolidation_quality,
        wick_pressure=wick_pressure,
        range_duration=window_days,
        verdict=verdict,
        confidence_boost=confidence_boost,
        explanation=explanation,
        recommendation=recommendation
    )

def analyze_price_zscore(z_price: float, z_spike_spread: float) -> PriceZScoreAnalysis:
    price_abs = abs(z_price)
    if price_abs > 3.0:
        level = 'EXTREME'
        volatility_risk = 'CRITICAL'
    elif price_abs > 2.5:
        level = 'SPIKE'
        volatility_risk = 'ELEVATED'
    else:
        level = 'NORMAL'
        volatility_risk = 'Normal'
    if z_price > 2.0:
        direction = 'OVERBOUGHT'
    elif z_price < -2.0:
        direction = 'OVERSOLD'
    else:
        direction = 'NEUTRAL'
    if level == 'EXTREME':
        confidence_adjustment = -0.20
    elif level == 'SPIKE':
        confidence_adjustment = -0.12
    else:
        confidence_adjustment = 0.0
    if direction == 'OVERBOUGHT':
        explanation = (f"EUR/USD overbought pe Price Z-Score: {z_price:.2f} ({level}). "
                       f"Pretul EUR/USD extins fata de media 20 bars. Risc corecție.")
        recommendation = ("Reduce confidence pentru BUY signals. "
                          "Price Z overbought = potential reversal zone.")
    elif direction == 'OVERSOLD':
        explanation = (f"EUR/USD oversold pe Price Z-Score: {z_price:.2f} ({level}). "
                       f"Pretul EUR/USD comprimat fata de media 20 bars. Risc rebound.")
        recommendation = ("Reduce confidence pentru SELL signals. "
                          "Price Z oversold = potential bounce zone.")
    else:
        explanation = (f"EUR/USD Price Z-Score normal: {z_price:.2f}. "
                       f"Pretul in parametri normali fata de media 20 bars.")
        recommendation = "Nicio ajustare necesara. Price Z neutral."
    return PriceZScoreAnalysis(
        price_z_value=z_price,
        level=level,
        volatility_risk=volatility_risk,
        direction=direction,
        confidence_adjustment=confidence_adjustment,
        explanation=explanation,
        recommendation=recommendation
    )

def analyze_divergence(z_spike_spread: float, z_spike_price: float) -> DivergenceAnalysis:
    spread_abs = abs(z_spike_spread)
    price_abs = abs(z_spike_price)
    spread_extreme = spread_abs > 3.0
    spread_spike   = spread_abs > 2.5
    spread_normal  = spread_abs <= 2.5
    price_extreme  = price_abs > 3.0
    price_spike    = price_abs > 2.5
    price_normal   = price_abs <= 2.5
    if price_extreme and not spread_extreme:
        return DivergenceAnalysis(
            detected=True,
            divergence_type='PRICE_NO_YIELDS',
            priority_override=True,
            override_weight=0.30,
            explanation=(f"DIVERGENTA CRITICA: Price Z Spike EXTREME ({z_spike_price:.2f}) "
                         f"fara suport yields (Spread Z: {z_spike_spread:.2f}). "
                         f"EUR/USD s-a miscat fara confirmare din bond spreads. "
                         f"Posibil fake breakout, short squeeze sau flow valutar temporar."),
            recommendation="DO NOT CHASE momentum! Wait pentru confirmarea yields sau pullback catre MA. High risk zone."
        )
    elif spread_extreme and not price_extreme:
        return DivergenceAnalysis(
            detected=True,
            divergence_type='YIELDS_NO_PRICE',
            priority_override=False,
            override_weight=0.10,
            explanation=(f"DIVERGENTA OPORTUNITATE: Yields Z Spike EXTREME ({z_spike_spread:.2f}) "
                         f"fara miscare price inca (Price Z: {z_spike_price:.2f}). "
                         f"Bond spreads au explodat dar EUR/USD nu a urmat inca. "
                         f"Historic: price urmareaza yields cu 1-3 zile lag."),
            recommendation="Monitorizeaza aproape. EUR/USD poate exploda in 1-2 zile. Pregateste entry in directia yields."
        )
    elif price_spike and spread_normal:
        return DivergenceAnalysis(
            detected=True,
            divergence_type='PRICE_WEAK',
            priority_override=False,
            override_weight=0.10,
            explanation=(f"Divergenta moderata: Price Z SPIKE ({z_spike_price:.2f}) "
                         f"fara confirmare yields ({z_spike_spread:.2f}). "
                         f"Miscare pret fara suport fundamental din spread-uri."),
            recommendation="Cautiune moderata. Reduce position size. Mismatch intre price si yields."
        )
    elif spread_spike and price_normal:
        return DivergenceAnalysis(
            detected=True,
            divergence_type='YIELDS_WEAK',
            priority_override=False,
            override_weight=0.05,
            explanation=(f"Divergenta minora: Yields Z SPIKE ({z_spike_spread:.2f}) "
                         f"fara miscare semnificativa in price ({z_spike_price:.2f}). "
                         f"Yields se misca dar EUR/USD rezista inca."),
            recommendation="Monitorizare. Yields pot duce pretul in curand sau pot normaliza."
        )
    else:
        return DivergenceAnalysis(
            detected=False,
            divergence_type='NONE',
            priority_override=False,
            override_weight=0.0,
            explanation=(f"Nicio divergenta detectata. "
                         f"Spread Z: {z_spike_spread:.2f}, Price Z: {z_spike_price:.2f}. "
                         f"Yields si Price se misca sincronizat."),
            recommendation="Nicio alerta divergenta. Proceed normal."
        )

def analyze_us2y_rates(us2y: float, de2y: float, us10y: float, diff_7d_ago: float) -> US2YAnalysis:
    differential = us2y - de2y
    change_7d = differential - diff_7d_ago
    yield_curve = us10y - us2y
    if differential > 1.5:
        carry_signal = 'STRONG_USD'
    elif differential > 1.0:
        carry_signal = 'MODERATE_USD'
    elif differential > -1.0:
        carry_signal = 'NEUTRAL'
    else:
        carry_signal = 'MODERATE_EUR'
    if us2y > 2.0 and change_7d > 0.10:
        policy_bias = 'HAWKISH_USD'
    elif us2y < 1.5 and change_7d < -0.10:
        policy_bias = 'DOVISH_USD'
    else:
        policy_bias = 'NEUTRAL'
    if abs(change_7d) > 0.20:
        alert_level = 'ALERT'
        weight_override = 0.35
    elif abs(change_7d) > 0.12:
        alert_level = 'WATCH'
        weight_override = 0.15
    else:
        alert_level = 'NONE'
        weight_override = 0.15
    if differential > 1.0 and change_7d > 0.10:
        verdict = 'BEARISH'
    elif differential < 0.5 and change_7d < -0.10:
        verdict = 'BULLISH'
    else:
        verdict = 'NEUTRAL'
    interpretation = f"Diferentialul US2Y-DE2Y este {differential:.2f}% "
    if change_7d > 0:
        interpretation += f"(+{change_7d:.2f}% in 7 zile). "
    else:
        interpretation += f"({change_7d:.2f}% in 7 zile). "
    interpretation += f"Yield curve 2s10s: {yield_curve:.2f}%. "
    if yield_curve < -0.10:
        interpretation += "INVERSIUNE detectata - risc recessiune. "
    interpretation += f"Carry trade: {carry_signal}. Policy bias: {policy_bias}."
    if alert_level == 'ALERT':
        recommendation = f"ALERT: Miscare semnificativa >20 bps in diferential ({change_7d*100:.0f} bps). Prioritate MARE in decizie finala."
    elif alert_level == 'WATCH':
        recommendation = f"WATCH: Miscare moderata 12-20 bps ({change_7d*100:.0f} bps). Monitorizare atenta."
    else:
        recommendation = "Diferentialul stabil. Influenta normala in analiza."
    return US2YAnalysis(
        us2y=us2y,
        de2y=de2y,
        us10y=us10y,
        differential=differential,
        change_7d=change_7d,
        yield_curve=yield_curve,
        carry_signal=carry_signal,
        policy_bias=policy_bias,
        alert_level=alert_level,
        verdict=verdict,
        weight_override=weight_override,
        interpretation=interpretation,
        recommendation=recommendation
    )

def detect_geopolitical_pressure(z_score: float, correlation: float, rsi: float, spike: SpikeAnalysis) -> GeopoliticalAlert:
    correlation_divergence = correlation < -0.3
    z_extreme = abs(z_score) > 3.0
    rsi_extreme = rsi > 80 or rsi < 20
    spike_extreme = spike.level == 'EXTREME'
    combined_pressure = correlation_divergence and (z_extreme or rsi_extreme or spike_extreme)
    if combined_pressure and (abs(z_score) > 3.5 or spike_extreme):
        level = 'CRITICAL'
        severity = 1.0
        override_weight = 0.90
        trigger_type = 'combined_with_spike' if spike_extreme else 'combined'
        explanation = f"PRESIUNE GEOPOLITICA CRITICA: Corelatie {correlation:.2f} (divergenta severa) + Z-Score extrem {z_score:.2f}"
        if spike_extreme:
            explanation += f" + Spike EXTREME ({spike.spike_value:.2f})"
        explanation += ". Posibil conflict major sau criza financiara. OVERRIDE 90% in decizie finala."
    elif combined_pressure or (correlation_divergence and abs(z_score) > 2.5):
        level = 'WARNING'
        severity = 0.6
        override_weight = 0.20
        trigger_type = 'combined_with_spike' if spike_extreme else 'combined' if combined_pressure else 'correlation_divergence'
        explanation = f"WARNING: Corelatie {correlation:.2f} cu Z-Score {z_score:.2f}. Posibile tensiuni geopolitice. Influence 20% in decizie."
    elif correlation < 0.0:
        level = 'PRE_ALERT'
        severity = 0.3
        override_weight = 0.0
        trigger_type = 'correlation_divergence'
        explanation = f"PRE-ALERT: Corelatie negativa {correlation:.2f}. Monitorizare atenta, dar fara override."
    else:
        level = 'NONE'
        severity = 0.0
        override_weight = 0.0
        trigger_type = 'none'
        explanation = f"Corelatie normala {correlation:.2f}. Nicio presiune geopolitica detectata."
    return GeopoliticalAlert(
        level=level,
        trigger_type=trigger_type,
        severity=severity,
        override_weight=override_weight,
        explanation=explanation
    )

def analyze_macro_layer(us_real: Optional[float], eu_real: Optional[float]) -> LayerVerdict:
    if us_real is None or eu_real is None:
        return LayerVerdict(
            layer_name='MACRO',
            verdict='NEUTRAL',
            confidence=0.0,
            weight=0.30,
            key_metrics={'us_real': None, 'eu_real': None},
            explanation='Date macro indisponibile. Verdict NEUTRAL.'
        )
    differential = us_real - eu_real
    if differential > 0.5:
        verdict = 'BEARISH'
        confidence = min(0.9, 0.5 + abs(differential) * 0.2)
        explanation = f"USD mai atractiv: US real rate {us_real:.2f}% vs EU {eu_real:.2f}% (diff: +{differential:.2f}%). Presiune BEARISH pe EUR/USD."
    elif differential < -0.5:
        verdict = 'BULLISH'
        confidence = min(0.9, 0.5 + abs(differential) * 0.2)
        explanation = f"EUR mai atractiv: EU real rate {eu_real:.2f}% vs US {us_real:.2f}% (diff: {differential:.2f}%). Presiune BULLISH pe EUR/USD."
    else:
        verdict = 'NEUTRAL'
        confidence = 0.5
        explanation = f"Echilibru macro: Differential minim ({differential:.2f}%). Nicio presiune clara."
    return LayerVerdict(
        layer_name='MACRO',
        verdict=verdict,
        confidence=confidence,
        weight=0.30,
        key_metrics={'us_real': us_real, 'eu_real': eu_real, 'differential': differential},
        explanation=explanation
    )

def analyze_institutional_layer(z_score: float, spread: float, ma_value: float, mr_z: MeanReversionAnalysis) -> LayerVerdict:
    spread_vs_ma = spread - ma_value
    if z_score > 2.0:
        z_signal = 'BEARISH'
        z_conf = min(0.9, 0.5 + abs(z_score - 2.0) * 0.2)
    elif z_score < -2.0:
        z_signal = 'BULLISH'
        z_conf = min(0.9, 0.5 + abs(z_score + 2.0) * 0.2)
    else:
        z_signal = 'NEUTRAL'
        z_conf = 0.5
    if spread_vs_ma > 0.1:
        spread_signal = 'BEARISH'
    elif spread_vs_ma < -0.1:
        spread_signal = 'BULLISH'
    else:
        spread_signal = 'NEUTRAL'
    if mr_z.active and mr_z.phase == 'TREND_REVERSAL':
        verdict = mr_z.direction
        confidence = 0.8
        explanation = f"TREND REVERSAL Z-Score domina: {mr_z.explanation} Z-Score actual: {z_score:.2f}, Spread vs MA: {spread_vs_ma:+.2f}."
    elif mr_z.active:
        verdict = mr_z.direction
        confidence = 0.7
        explanation = f"Mean Reversion Z activ: {mr_z.explanation} Z-Score: {z_score:.2f}."
    else:
        if z_signal == spread_signal and z_signal != 'NEUTRAL':
            verdict = z_signal
            confidence = min(0.85, (z_conf + 0.6) / 2)
            explanation = f"Z-Score ({z_score:.2f}) si Spread structure ({spread_vs_ma:+.2f}) align {verdict}. Confidence ridicata."
        elif z_signal != 'NEUTRAL':
            verdict = z_signal
            confidence = z_conf * 0.8
            explanation = f"Z-Score {z_score:.2f} sugereaza {verdict}, dar Spread structure mixt. Confidence moderata."
        else:
            verdict = 'NEUTRAL'
            confidence = 0.5
            explanation = f"Z-Score neutru ({z_score:.2f}), Spread vs MA: {spread_vs_ma:+.2f}. Nicio directie clara."
    return LayerVerdict(
        layer_name='INSTITUTIONAL',
        verdict=verdict,
        confidence=confidence,
        weight=0.40,
        key_metrics={
            'z_score': z_score,
            'spread': spread,
            'ma_value': ma_value,
            'spread_vs_ma': spread_vs_ma,
            'mr_active': mr_z.active,
            'mr_phase': mr_z.phase
        },
        explanation=explanation
    )

def analyze_technical_layer(spot: float, ema_daily: float, ema_weekly: float,
                            mr_rsi: MeanReversionAnalysis,
                            price_z: PriceZScoreAnalysis) -> LayerVerdict:
    spot_vs_daily = ((spot - ema_daily) / ema_daily) * 100
    spot_vs_weekly = ((spot - ema_weekly) / ema_weekly) * 100
    mtf_aligned = (spot_vs_daily > 0 and spot_vs_weekly > 0) or (spot_vs_daily < 0 and spot_vs_weekly < 0)
    if spot_vs_daily > 0.3 and spot_vs_weekly > 0.3:
        tech_signal = 'BULLISH'
        tech_conf = 0.8
    elif spot_vs_daily < -0.3 and spot_vs_weekly < -0.3:
        tech_signal = 'BEARISH'
        tech_conf = 0.8
    elif spot_vs_daily > 0:
        tech_signal = 'BULLISH'
        tech_conf = 0.6
    elif spot_vs_daily < 0:
        tech_signal = 'BEARISH'
        tech_conf = 0.6
    else:
        tech_signal = 'NEUTRAL'
        tech_conf = 0.5
    if mr_rsi.active and mr_rsi.phase == 'TREND_REVERSAL':
        verdict = mr_rsi.direction
        confidence = 0.85
        explanation = f"TREND REVERSAL RSI domina: {mr_rsi.explanation} Spot: {spot:.5f} vs EMA Daily: {ema_daily:.5f} ({spot_vs_daily:+.2f}%)."
    elif mr_rsi.active:
        verdict = mr_rsi.direction
        confidence = 0.75
        explanation = f"Mean Reversion RSI activ: {mr_rsi.explanation} Technical: Spot {spot_vs_daily:+.2f}% vs Daily EMA."
    else:
        verdict = tech_signal
        confidence = tech_conf
        if mtf_aligned:
            explanation = f"MTF alignment {verdict}: Spot {spot_vs_daily:+.2f}% vs Daily, {spot_vs_weekly:+.2f}% vs Weekly."
        else:
            explanation = f"MTF mixt: Daily {spot_vs_daily:+.2f}%, Weekly {spot_vs_weekly:+.2f}%. Verdict: {verdict}."
    price_z_note = ""
    if price_z.confidence_adjustment != 0.0:
        contradicts = (
            (verdict == 'BULLISH' and price_z.direction == 'OVERBOUGHT') or
            (verdict == 'BEARISH' and price_z.direction == 'OVERSOLD')
        )
        confirms = (
            (verdict == 'BULLISH' and price_z.direction == 'OVERSOLD') or
            (verdict == 'BEARISH' and price_z.direction == 'OVERBOUGHT')
        )
        if contradicts:
            confidence = max(0.1, confidence + price_z.confidence_adjustment)
            price_z_note = (f" PRICE Z ({price_z.price_z_value:.2f}) {price_z.direction}: "
                           f"contradicts {verdict} signal. Confidence ajustat {price_z.confidence_adjustment:+.0%}.")
        elif confirms:
            confidence = min(0.95, confidence + 0.08)
            price_z_note = (f" PRICE Z ({price_z.price_z_value:.2f}) confirma {verdict} signal. "
                           f"Confidence boost +8%.")
        else:
            price_z_note = (f" PRICE Z ({price_z.price_z_value:.2f}) {price_z.level}: "
                           f"volatilitate ridicata, cautiune.")
    explanation += price_z_note
    return LayerVerdict(
        layer_name='TECHNICAL',
        verdict=verdict,
        confidence=confidence,
        weight=0.20,
        key_metrics={
            'spot': spot,
            'ema_daily': ema_daily,
            'ema_weekly': ema_weekly,
            'spot_vs_daily_pct': spot_vs_daily,
            'spot_vs_weekly_pct': spot_vs_weekly,
            'mtf_aligned': mtf_aligned,
            'mr_rsi_active': mr_rsi.active,
            'price_z_value': price_z.price_z_value,
            'price_z_direction': price_z.direction,
            'price_z_adjustment': price_z.confidence_adjustment
        },
        explanation=explanation
    )

def analyze_correlation_layer(correlation: float) -> LayerVerdict:
    if correlation > 0.7:
        verdict = 'NEUTRAL'
        confidence = 0.8
        modifier = 1.0
        explanation = f"Corelatie normala {correlation:.2f}. Relatia EUR/USD-Spread functioneaza corect. Nicio penalizare."
    elif 0.3 <= correlation <= 0.7:
        verdict = 'NEUTRAL'
        confidence = 0.6
        modifier = 0.9
        explanation = f"Corelatie moderata {correlation:.2f}. Relatia slabita, dar acceptabila. Penalizare minima 10%."
    elif 0.0 <= correlation < 0.3:
        verdict = 'WAIT'
        confidence = 0.7
        modifier = 0.7
        explanation = f"Corelatie slaba {correlation:.2f}. Relatia EUR/USD-Spread perturbata. Penalizare 30%. Cautiune recomandata."
    elif -0.3 <= correlation < 0.0:
        verdict = 'WAIT'
        confidence = 0.8
        modifier = 0.5
        explanation = f"Corelatie negativa {correlation:.2f}. DIVERGENTA detectata. Penalizare 50%. PRE-ALERT geopolitic."
    else:
        verdict = 'WAIT'
        confidence = 0.9
        modifier = 0.5
        explanation = f"Corelatie negativa severa {correlation:.2f}. DIVERGENTA MAJORA. Penalizare 50%. Posibil WARNING geopolitic."
    return LayerVerdict(
        layer_name='CORRELATION',
        verdict=verdict,
        confidence=confidence,
        weight=modifier,
        key_metrics={'correlation': correlation, 'modifier': modifier},
        explanation=explanation
    )

def analyze_risk_layer(vix: float, atr: float) -> LayerVerdict:
    if vix < 15:
        vix_signal = 'LOW_VOL'
        risk_env = 'FAVORABIL'
    elif vix < 25:
        vix_signal = 'MODERATE_VOL'
        risk_env = 'NORMAL'
    elif vix < 35:
        vix_signal = 'HIGH_VOL'
        risk_env = 'RIDICAT'
    else:
        vix_signal = 'EXTREME_VOL'
        risk_env = 'EXTREM'
    sl_distance = atr * 1.5
    tp_distance = atr * 2.5
    explanation = f"VIX: {vix:.1f} ({vix_signal}, risc {risk_env}). ATR: {atr:.5f}. SL sugerat: {sl_distance:.5f}, TP: {tp_distance:.5f}."
    return LayerVerdict(
        layer_name='RISK',
        verdict='NEUTRAL',
        confidence=0.8,
        weight=0.10,
        key_metrics={
            'vix': vix,
            'vix_signal': vix_signal,
            'risk_environment': risk_env,
            'atr': atr,
            'sl_distance': sl_distance,
            'tp_distance': tp_distance
        },
        explanation=explanation
    )

def calculate_v38_core(macro: LayerVerdict, institutional: LayerVerdict,
                       technical: LayerVerdict, risk: LayerVerdict,
                       correlation_modifier: float,
                       volume: VolumeAnalysis) -> Tuple[float, str, str, float]:
    def verdict_to_score(verdict: str) -> float:
        return {'BULLISH': 1.0, 'BEARISH': -1.0, 'NEUTRAL': 0.0, 'WAIT': 0.0}.get(verdict, 0.0)
    macro_score = verdict_to_score(macro.verdict) * macro.confidence
    inst_score  = verdict_to_score(institutional.verdict) * institutional.confidence
    tech_score  = verdict_to_score(technical.verdict) * technical.confidence
    risk_score  = 0.0
    integrated = (
        macro_score * 0.30 +
        inst_score  * 0.40 +
        tech_score  * 0.20 +
        risk_score  * 0.10
    )
    integrated *= correlation_modifier
    if volume.verdict != 'NEUTRAL' and volume.confidence_boost != 0.0:
        same_direction = (
            (integrated > 0 and volume.verdict == 'BULLISH') or
            (integrated < 0 and volume.verdict == 'BEARISH')
        )
        opposite_direction = (
            (integrated > 0 and volume.verdict == 'BEARISH') or
            (integrated < 0 and volume.verdict == 'BULLISH')
        )
        if same_direction:
            integrated *= (1.0 + abs(volume.confidence_boost))
        elif opposite_direction:
            integrated *= (1.0 - abs(volume.confidence_boost))
    if integrated > 0.5:
        phase = 'ACCUMULATION'
    elif integrated > 0.2:
        phase = 'BULLISH_BIAS'
    elif integrated < -0.5:
        phase = 'DISTRIBUTION'
    elif integrated < -0.2:
        phase = 'BEARISH_BIAS'
    else:
        phase = 'CONSOLIDATION'
    if integrated > 0.6:
        action = 'BUY'
    elif integrated < -0.6:
        action = 'SELL'
    else:
        action = 'WAIT'
    confidence = min(0.95, abs(integrated))
    return integrated, phase, action, confidence

def integrate_priority_hierarchy(
    macro: LayerVerdict,
    us2y: US2YAnalysis,
    institutional: LayerVerdict,
    correlation: LayerVerdict,
    spike: SpikeAnalysis,
    divergence: DivergenceAnalysis,
    volume: VolumeAnalysis,
    geopolitical: GeopoliticalAlert,
    technical: LayerVerdict,
    mr_z: MeanReversionAnalysis,
    mr_rsi: MeanReversionAnalysis,
    risk: LayerVerdict,
    v38_score: float,
    v38_action: str
) -> Tuple[str, float, str, str]:
    active_overrides = []
    weights_applied = {}
    double_mr = (mr_z.active and mr_rsi.active and
                 mr_z.direction == mr_rsi.direction and
                 mr_z.phase in ['TREND_REVERSAL', 'ACTIVE'] and
                 mr_rsi.phase in ['TREND_REVERSAL', 'ACTIVE'])
    if divergence.detected and divergence.divergence_type == 'PRICE_NO_YIELDS' and divergence.priority_override:
        active_overrides.append('DIVERGENCE_PRICE_NO_YIELDS')
        final_action = 'WAIT'
        final_confidence = 0.88
        priority_eval = "Priority 0.5: DIVERGENCE PRICE_NO_YIELDS - fake breakout risk"
        explanation = (f"DIVERGENTA CRITICA: {divergence.explanation} "
                       f"{divergence.recommendation} "
                       f"Nicio pozitie recomandata pana la confirmarea yields.")
        return final_action, final_confidence, priority_eval, explanation
    if geopolitical.level == 'CRITICAL':
        active_overrides.append('GEOPOLITICAL_CRITICAL')
        final_action = 'WAIT'
        final_confidence = 0.95
        priority_eval = "Priority 1: GEOPOLITICAL CRITICAL override 90%"
        explanation = f"DECIZIE DOMINATA DE CRIZA GEOPOLITICA: {geopolitical.explanation} Toate celelalte semnale sunt ignorate. WAIT obligatoriu."
        return final_action, final_confidence, priority_eval, explanation
    if spike.confluence_level == 'CRITICAL':
        active_overrides.append('SPIKE_CONFLUENCE_CRITICAL')
        final_action = 'WAIT'
        final_confidence = 0.90
        priority_eval = "Priority 1.5: SPIKE CONFLUENCE CRITICAL override 25%"
        explanation = f"CONFLUENCE CRITICAL detectat: {spike.explanation} {spike.recommendation} Probabilitate MARE de exhaustion. WAIT pentru confirmarea directiei."
        return final_action, final_confidence, priority_eval, explanation
    if us2y.alert_level == 'ALERT':
        active_overrides.append('US2Y_ALERT')
        weights_applied['US2Y'] = 0.35
        us2y_weight = 0.35
        if us2y.verdict == 'BEARISH':
            us2y_score = -0.7
        elif us2y.verdict == 'BULLISH':
            us2y_score = 0.7
        else:
            us2y_score = 0.0
        v38_weighted = v38_score * 0.65
        final_score = us2y_score * 0.35 + v38_weighted
        if final_score > 0.5:
            final_action = 'BUY'
        elif final_score < -0.5:
            final_action = 'SELL'
        else:
            final_action = 'WAIT'
        if spike.confluence_level == 'WARNING':
            active_overrides.append('SPIKE_CONFLUENCE_WARNING')
            final_confidence_boost = 0.10
        else:
            final_confidence_boost = 0.0
        final_confidence = min(0.90, abs(final_score) + 0.3 + final_confidence_boost)
        priority_eval = f"Priority 2: US2Y ALERT override 35%. US2Y {us2y.verdict}, V38 score adjusted."
        explanation = f"US2Y ALERT domina decizia: {us2y.recommendation} {us2y.interpretation} V38 core ({v38_action}) este moderat de US2Y signal."
        if spike.confluence_level == 'WARNING':
            explanation += f" SPIKE WARNING: {spike.explanation}"
        if double_mr:
            active_overrides.append('DOUBLE_MR_CONFIRMATION')
            final_confidence = min(0.95, final_confidence + 0.15)
            explanation += f" BONUS: Double MR confirmation (Z-Score + RSI align {mr_z.direction}). Confidence +15%."
        return final_action, final_confidence, priority_eval, explanation
    if spike.confluence_level == 'WARNING':
        active_overrides.append('SPIKE_CONFLUENCE_WARNING')
        weights_applied['SPIKE'] = 0.15
        v38_weighted = v38_score * 0.85
        final_score = v38_weighted
        if final_score > 0.5:
            final_action = 'WATCH'
        elif final_score < -0.5:
            final_action = 'WATCH'
        else:
            final_action = 'WAIT'
        final_confidence = min(0.75, abs(final_score) + 0.20)
        priority_eval = "Priority 2.5: SPIKE CONFLUENCE WARNING 15% influence. Action downgraded to WATCH."
        explanation = f"SPIKE CONFLUENCE WARNING: {spike.explanation} {spike.recommendation} V38 sugera {v38_action}, dar spike + MR impune CAUTIUNE."
        if double_mr:
            active_overrides.append('DOUBLE_MR_CONFIRMATION')
            final_confidence = min(0.85, final_confidence + 0.10)
            explanation += f" BONUS partial: Double MR confirmation (+10%)."
        return final_action, final_confidence, priority_eval, explanation
    mr_trend_reversal = False
    mr_direction = None
    if mr_z.phase == 'TREND_REVERSAL':
        mr_trend_reversal = True
        mr_direction = mr_z.direction
        mr_source = 'Z-Score'
        mr_explanation = mr_z.explanation
    elif mr_rsi.phase == 'TREND_REVERSAL':
        mr_trend_reversal = True
        mr_direction = mr_rsi.direction
        mr_source = 'RSI'
        mr_explanation = mr_rsi.explanation
    if mr_trend_reversal:
        active_overrides.append(f'MR_TREND_REVERSAL_{mr_source}')
        weights_applied['MR_REVERSAL'] = 0.25
        if mr_direction == 'BULLISH':
            mr_score = 0.8
        else:
            mr_score = -0.8
        v38_weighted = v38_score * 0.75
        final_score = mr_score * 0.25 + v38_weighted
        if final_score > 0.4:
            final_action = 'BUY'
        elif final_score < -0.4:
            final_action = 'SELL'
        else:
            final_action = 'WAIT'
        final_confidence = min(0.85, abs(final_score) + 0.25)
        priority_eval = f"Priority 3: {mr_source} TREND REVERSAL 25% weight. Direction: {mr_direction}."
        explanation = f"TREND REVERSAL {mr_source} influențeaza decizia: {mr_explanation} V38 core ({v38_action}) este moderat de signal Mean Reversion."
        if double_mr:
            active_overrides.append('DOUBLE_MR_CONFIRMATION')
            final_confidence = min(0.95, final_confidence + 0.15)
            explanation += f" BONUS: Double MR confirmation (ambele MR active si align). Confidence +15%."
        return final_action, final_confidence, priority_eval, explanation
    if volume.phase_detected == 'DISTRIBUTION_PHASE':
        active_overrides.append('WYCKOFF_DISTRIBUTION')
        if v38_action == 'BUY':
            final_action = 'WAIT'
            final_confidence = 0.75
            priority_eval = "Priority 3.2: WYCKOFF DISTRIBUTION override. BUY -> WAIT."
            explanation = (f"DISTRIBUTIE DETECTATA: Consolidare in range cu presiune de vanzare (Wicks/OBV). "
                           f"Desi tehnic este Bullish, structura volumului indica vanzare institutionala. "
                           f"WAIT obligatoriu pentru breakdown.")
            return final_action, final_confidence, priority_eval, explanation
        elif v38_action == 'SELL':
            final_action = 'SELL'
            final_confidence = min(0.95, abs(v38_score) + 0.20)
            priority_eval = "Priority 3.2: WYCKOFF DISTRIBUTION confirma SELL. Strong signal."
            explanation = f"DISTRIBUTIE CONFIRMATA: {volume.explanation}. Aliniere perfecta cu semnalul V38 SELL."
            return final_action, final_confidence, priority_eval, explanation
    if volume.obv_divergence and volume.obv_divergence_type != 'NONE':
        active_overrides.append(f'OBV_DIVERGENCE_{volume.obv_divergence_type}')
        if volume.obv_divergence_type == 'BEARISH_DIV':
            if v38_action == 'BUY':
                final_action = 'WATCH'
                final_confidence = max(0.55, v38_score * 0.70)
                priority_eval = "Priority 3.5: OBV BEARISH DIVERGENCE - distributie silentioasa. BUY downgradat la WATCH."
                explanation = (f"OBV BEARISH DIVERGENCE: Pretul urca dar OBV scade. "
                               f"Banii mari DISTRIBUIE in timp ce retailul cumpara. "
                               f"{volume.explanation} V38 sugera BUY dar volume contrazice. WATCH.")
                return final_action, final_confidence, priority_eval, explanation
            elif v38_action == 'SELL':
                final_action = 'SELL'
                final_confidence = min(0.92, abs(v38_score) + abs(volume.confidence_boost))
                priority_eval = "Priority 3.5: OBV BEARISH DIV confirma SELL. Confidence boostat."
                explanation = (f"OBV BEARISH DIVERGENCE confirma SELL: {volume.explanation} "
                               f"Distributie detectata. Confidence boost +{abs(volume.confidence_boost):.0%}.")
                return final_action, final_confidence, priority_eval, explanation
        elif volume.obv_divergence_type == 'BULLISH_DIV':
            if v38_action == 'SELL':
                final_action = 'WATCH'
                final_confidence = max(0.55, abs(v38_score) * 0.70)
                priority_eval = "Priority 3.5: OBV BULLISH DIVERGENCE - acumulare silentioasa. SELL downgradat la WATCH."
                explanation = (f"OBV BULLISH DIVERGENCE: Pretul scade dar OBV urca. "
                               f"Banii mari ACUMULEAZA in timp ce retailul vinde. "
                               f"{volume.explanation} V38 sugera SELL dar volume contrazice. WATCH.")
                return final_action, final_confidence, priority_eval, explanation
            elif v38_action == 'BUY':
                final_action = 'BUY'
                final_confidence = min(0.92, abs(v38_score) + abs(volume.confidence_boost))
                priority_eval = "Priority 3.5: OBV BULLISH DIV confirma BUY. Confidence boostat."
                explanation = (f"OBV BULLISH DIVERGENCE confirma BUY: {volume.explanation} "
                               f"Acumulare detectata. Confidence boost +{abs(volume.confidence_boost):.0%}.")
                return final_action, final_confidence, priority_eval, explanation
    if geopolitical.level == 'WARNING':
        active_overrides.append('GEOPOLITICAL_WARNING')
        weights_applied['GEOPOLITICAL'] = 0.20
        if v38_action in ['BUY', 'SELL']:
            final_action = 'WATCH'
            final_confidence = 0.60
            priority_eval = "Priority 4: GEOPOLITICAL WARNING 20% influence. Action downgraded to WATCH."
            explanation = f"GEOPOLITICAL WARNING reduce confidence: {geopolitical.explanation} V38 sugera {v38_action}, dar presiunea geopolitica impune CAUTIUNE. Recomandare: WATCH."
        else:
            final_action = 'WAIT'
            final_confidence = 0.65
            priority_eval = "Priority 4: GEOPOLITICAL WARNING reinforces WAIT."
            explanation = f"GEOPOLITICAL WARNING: {geopolitical.explanation} V38 already suggests WAIT. Confirmare cautiune."
        return final_action, final_confidence, priority_eval, explanation
    mr_active = mr_z.active or mr_rsi.active
    if mr_active and not mr_trend_reversal:
        active_overrides.append('MR_ACTIVE')
        if mr_z.active:
            momentum_adj = mr_z.momentum_weight
            mr_source = 'Z-Score'
        else:
            momentum_adj = mr_rsi.momentum_weight
            mr_source = 'RSI'
        final_score = v38_score * momentum_adj
        if final_score > 0.5:
            final_action = 'BUY'
        elif final_score < -0.5:
            final_action = 'SELL'
        else:
            final_action = 'WAIT'
        final_confidence = min(0.80, abs(final_score) + 0.2)
        priority_eval = f"Priority 5: {mr_source} Mean Reversion ACTIVE adjusts momentum (weight {momentum_adj})."
        explanation = f"Mean Reversion {mr_source} activ ajusteaza momentum: V38 score {v38_score:.2f} → adjusted {final_score:.2f}. Final action: {final_action}."
        if double_mr:
            active_overrides.append('DOUBLE_MR_CONFIRMATION')
            final_confidence = min(0.90, final_confidence + 0.15)
            explanation += f" BONUS: Double MR confirmation (Z + RSI align {mr_z.direction}). Confidence +15%."
        return final_action, final_confidence, priority_eval, explanation
    if double_mr:
        active_overrides.append('DOUBLE_MR_CONFIRMATION')
        if mr_z.direction == 'BULLISH':
            mr_boost = 0.3
        else:
            mr_boost = -0.3
        final_score = v38_score + mr_boost
        if final_score > 0.5:
            final_action = 'BUY'
        elif final_score < -0.5:
            final_action = 'SELL'
        else:
            final_action = 'WAIT'
        final_confidence = min(0.90, abs(v38_score) + 0.30)
        priority_eval = "Priority 6: DOUBLE MR CONFIRMATION (Z-Score + RSI align). Boost +15% confidence."
        explanation = f"Double Mean Reversion confirmation: Z-Score si RSI ambele {mr_z.direction}. V38 score {v38_score:.2f} boosted by MR alignment. Final: {final_action}."
        return final_action, final_confidence, priority_eval, explanation
    final_action = v38_action
    final_confidence = min(0.80, abs(v38_score) + 0.2)
    priority_eval = "Priority 7: V38 Core decision (no overrides active)."
    explanation = (f"Decizie bazata pe V38 Core: Integrated score {v38_score:.2f}. "
                   f"Macro {macro.verdict}, Institutional {institutional.verdict}, "
                   f"Technical {technical.verdict}. Action: {final_action}.")
    if volume.verdict != 'NEUTRAL' and volume.confidence_boost != 0.0:
        vol_confirms = (
            (final_action == 'BUY'  and volume.verdict == 'BULLISH') or
            (final_action == 'SELL' and volume.verdict == 'BEARISH')
        )
        vol_contradicts = (
            (final_action == 'BUY'  and volume.verdict == 'BEARISH') or
            (final_action == 'SELL' and volume.verdict == 'BULLISH')
        )
        if vol_confirms:
            boost = abs(volume.confidence_boost)
            final_confidence = min(0.92, final_confidence + boost)
            explanation += (f" VOLUME CONFIRMA {final_action}: {volume.smart_money_signal}. "
                            f"Confidence +{boost:.0%}.")
            priority_eval += f" Volume confirmation +{boost:.0%}."
        elif vol_contradicts:
            penalty = abs(volume.confidence_boost)
            final_confidence = max(0.30, final_confidence - penalty)
            explanation += (f" VOLUME CONTRAZICE {final_action}: {volume.smart_money_signal}. "
                            f"Confidence -{penalty:.0%}. Cautiune!")
            priority_eval += f" Volume contradiction -{penalty:.0%}."
    if us2y.alert_level == 'WATCH':
        active_overrides.append('US2Y_WATCH')
        explanation += f" US2Y WATCH: {us2y.recommendation} Monitorizare atenta, dar fara override."
    return final_action, final_confidence, priority_eval, explanation

def save_to_csv(report: GuardianReport):
    file_exists = Path(CONFIG['csv_file']).exists()
    fieldnames = [
        'Timestamp', 'Z_Score', 'Z_Score_Spike', 'Z_Score_Price', 'Z_Score_Prev', 'Z_Score_2Days',
        'Spread', 'MA_Value', 'VIX', 'Correlation_H4',
        'US2Y', 'DE2Y', 'US10Y', 'US2Y_Diff', 'US2Y_Change_7D',
        'Yield_Curve_2s10s', 'Carry_Signal', 'Policy_Bias', 'US2Y_Alert',
        'RSI_Daily', 'RSI_Prev', 'RSI_2Days',
        'Spike_Level', 'Spike_Volatility_Risk', 'Spike_Confluence', 'Spike_Confluence_Level',
        'Price_Z_Value', 'Price_Z_Level', 'Price_Z_Direction', 'Price_Z_Adjustment',
        'Divergence_Detected', 'Divergence_Type', 'Divergence_Override',
        'Vol_Ratio', 'OBV_Slope', 'OBV_Divergence', 'OBV_Div_Type',
        'Delta_5D_Signal', 'PV_Relationship', 'Smart_Money',
        'Wyckoff_Phase', 'Consolidation_Quality', 'Wick_Pressure',
        'Vol_Verdict', 'Vol_Confidence_Boost',
        'MR_Z_Active', 'MR_Z_Phase', 'MR_Z_Direction', 'MR_Z_Magnitude',
        'MR_RSI_Active', 'MR_RSI_Phase', 'MR_RSI_Direction', 'MR_RSI_Magnitude',
        'MR_Double_Confirmation',
        'Macro_Verdict', 'Macro_Confidence',
        'Institutional_Verdict', 'Institutional_Confidence',
        'Correlation_Verdict', 'Correlation_Modifier',
        'Geopolitical_Level', 'Geopolitical_Severity',
        'Technical_Verdict', 'Technical_Confidence',
        'V38_Score', 'V38_Phase', 'V38_Action', 'V38_Confidence',
        'Final_Action', 'Final_Confidence',
        'Active_Overrides', 'Priority_Evaluation'
    ]
    with open(CONFIG['csv_file'], 'a', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow({
            'Timestamp': report.timestamp,
            'Z_Score': f"{report.z_score:.4f}",
            'Z_Score_Spike': f"{report.z_score_spike:.4f}",
            'Z_Score_Price': f"{report.z_score_price:.4f}",
            'Z_Score_Prev': f"{report.z_score_prev:.4f}",
            'Z_Score_2Days': f"{report.z_score_2days:.4f}",
            'Spread': f"{report.spread:.4f}",
            'MA_Value': f"{report.ma_value:.4f}",
            'VIX': f"{report.vix:.2f}",
            'Correlation_H4': f"{report.correlation_h4:.4f}",
            'US2Y': f"{report.us2y_analysis.us2y:.4f}",
            'DE2Y': f"{report.us2y_analysis.de2y:.4f}",
            'US10Y': f"{report.us2y_analysis.us10y:.4f}",
            'US2Y_Diff': f"{report.us2y_analysis.differential:.4f}",
            'US2Y_Change_7D': f"{report.us2y_analysis.change_7d:.4f}",
            'Yield_Curve_2s10s': f"{report.us2y_analysis.yield_curve:.4f}",
            'Carry_Signal': report.us2y_analysis.carry_signal,
            'Policy_Bias': report.us2y_analysis.policy_bias,
            'US2Y_Alert': report.us2y_analysis.alert_level,
            'RSI_Daily': f"{report.rsi_daily:.2f}",
            'RSI_Prev': f"{report.rsi_prev:.2f}",
            'RSI_2Days': f"{report.rsi_2days:.2f}",
            'Spike_Level': report.spike_analysis.level,
            'Spike_Volatility_Risk': report.spike_analysis.volatility_risk,
            'Spike_Confluence': report.spike_analysis.confluence_active,
            'Spike_Confluence_Level': report.spike_analysis.confluence_level,
            'Price_Z_Value': f"{report.price_z_analysis.price_z_value:.4f}",
            'Price_Z_Level': report.price_z_analysis.level,
            'Price_Z_Direction': report.price_z_analysis.direction,
            'Price_Z_Adjustment': f"{report.price_z_analysis.confidence_adjustment:+.2f}",
            'Divergence_Detected': report.divergence_analysis.detected,
            'Divergence_Type': report.divergence_analysis.divergence_type,
            'Divergence_Override': report.divergence_analysis.priority_override,
            'Vol_Ratio': f"{report.volume_analysis.volume_ratio:.3f}",
            'OBV_Slope': report.volume_analysis.obv_slope,
            'OBV_Divergence': report.volume_analysis.obv_divergence,
            'OBV_Div_Type': report.volume_analysis.obv_divergence_type,
            'Delta_5D_Signal': report.volume_analysis.delta_signal,
            'PV_Relationship': report.volume_analysis.pv_relationship,
            'Smart_Money': report.volume_analysis.smart_money_signal,
            'Wyckoff_Phase': report.volume_analysis.phase_detected,
            'Consolidation_Quality': report.volume_analysis.consolidation_quality,
            'Wick_Pressure': report.volume_analysis.wick_pressure,
            'Vol_Verdict': report.volume_analysis.verdict,
            'Vol_Confidence_Boost': f"{report.volume_analysis.confidence_boost:+.3f}",
            'MR_Z_Active': report.mr_z_analysis.active,
            'MR_Z_Phase': report.mr_z_analysis.phase,
            'MR_Z_Direction': report.mr_z_analysis.direction,
            'MR_Z_Magnitude': f"{report.mr_z_analysis.magnitude:.4f}",
            'MR_RSI_Active': report.mr_rsi_analysis.active,
            'MR_RSI_Phase': report.mr_rsi_analysis.phase,
            'MR_RSI_Direction': report.mr_rsi_analysis.direction,
            'MR_RSI_Magnitude': f"{report.mr_rsi_analysis.magnitude:.2f}",
            'MR_Double_Confirmation': report.double_mr_confirmation,
            'Macro_Verdict': report.macro_verdict.verdict,
            'Macro_Confidence': f"{report.macro_verdict.confidence:.2f}",
            'Institutional_Verdict': report.institutional_verdict.verdict,
            'Institutional_Confidence': f"{report.institutional_verdict.confidence:.2f}",
            'Correlation_Verdict': report.correlation_verdict.verdict,
            'Correlation_Modifier': f"{report.correlation_verdict.key_metrics['modifier']:.2f}",
            'Geopolitical_Level': report.geopolitical_alert.level,
            'Geopolitical_Severity': f"{report.geopolitical_alert.severity:.2f}",
            'Technical_Verdict': report.technical_verdict.verdict,
            'Technical_Confidence': f"{report.technical_verdict.confidence:.2f}",
            'V38_Score': f"{report.v38_integrated_score:.4f}",
            'V38_Phase': report.v38_phase,
            'V38_Action': report.v38_action,
            'V38_Confidence': f"{report.v38_confidence:.2f}",
            'Final_Action': report.final_action,
            'Final_Confidence': f"{report.final_confidence:.2f}",
            'Active_Overrides': '|'.join(report.active_overrides),
            'Priority_Evaluation': report.priority_evaluation
        })

# ============================================================================
# STREAMLIT UI & CHARTS
# ============================================================================

def get_action_color(action: str) -> str:
    """Return color for action"""
    if action == 'BUY':
        return '#00ff00'
    elif action == 'SELL':
        return '#ff0000'
    elif action == 'WATCH':
        return '#ffa500'
    else:
        return '#808080'

def get_verdict_color(verdict: str) -> str:
    """Return color for verdict"""
    if verdict == 'BULLISH':
        return '#00ff00'
    elif verdict == 'BEARISH':
        return '#ff0000'
    else:
        return '#808080'

def create_confidence_gauge(confidence: float, title: str) -> go.Figure:
    """Create gauge chart for confidence"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': title},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 30], 'color': "lightgray"},
                {'range': [30, 60], 'color': "gray"},
                {'range': [60, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig

def create_layer_comparison(report: GuardianReport) -> go.Figure:
    """Create bar chart comparing layer verdicts"""
    layers = ['Macro', 'US2Y', 'Institutional', 'Correlation', 'Technical', 'Volume']
    verdicts = [
        report.macro_verdict.verdict,
        report.us2y_analysis.verdict,
        report.institutional_verdict.verdict,
        report.correlation_verdict.verdict,
        report.technical_verdict.verdict,
        report.volume_analysis.verdict
    ]
    confidences = [
        report.macro_verdict.confidence,
        1.0 if report.us2y_analysis.alert_level == 'ALERT' else 0.5,
        report.institutional_verdict.confidence,
        report.correlation_verdict.confidence,
        report.technical_verdict.confidence,
        abs(report.volume_analysis.confidence_boost) if report.volume_analysis.verdict != 'NEUTRAL' else 0.0
    ]
    
    colors = [get_verdict_color(v) for v in verdicts]
    
    fig = go.Figure(data=[
        go.Bar(
            x=layers,
            y=confidences,
            marker_color=colors,
            text=[f"{v}<br>{c:.0%}" for v, c in zip(verdicts, confidences)],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title="Layer Verdicts Comparison",
        yaxis_title="Confidence",
        height=400,
        showlegend=False
    )
    
    return fig

def create_historical_chart() -> Optional[go.Figure]:
    """Create historical Z-Score and RSI chart"""
    if not Path(CONFIG['csv_file']).exists():
        return None
    
    try:
        df = pd.read_csv(CONFIG['csv_file'])
        if len(df) == 0:
            return None
        
        df = df.tail(30)  # Last 30 days
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Z-Score History', 'RSI History'),
            vertical_spacing=0.12
        )
        
        # Z-Score
        fig.add_trace(
            go.Scatter(x=df['Timestamp'], y=df['Z_Score'], 
                      mode='lines+markers', name='Z-Score',
                      line=dict(color='blue', width=2)),
            row=1, col=1
        )
        
        # Extreme zones
        fig.add_hline(y=2.0, line_dash="dash", line_color="red", row=1, col=1)
        fig.add_hline(y=-2.0, line_dash="dash", line_color="green", row=1, col=1)
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=1, col=1)
        
        # RSI
        fig.add_trace(
            go.Scatter(x=df['Timestamp'], y=df['RSI_Daily'],
                      mode='lines+markers', name='RSI',
                      line=dict(color='purple', width=2)),
            row=2, col=1
        )
        
        # RSI zones
        fig.add_hline(y=70, line_dash="dash", line_color="red", row=2, col=1)
        fig.add_hline(y=30, line_dash="dash", line_color="green", row=2, col=1)
        fig.add_hline(y=50, line_dash="dot", line_color="gray", row=2, col=1)
        
        fig.update_layout(height=600, showlegend=False)
        
        return fig
    
    except Exception as e:
        st.error(f"Error loading historical data: {e}")
        return None

def create_wyckoff_visualization(volume: VolumeAnalysis) -> go.Figure:
    """Create Wyckoff phase visualization"""
    
    phases = ['ACCUMULATION', 'DISTRIBUTION', 'NONE']
    values = [0, 0, 0]
    
    if volume.phase_detected == 'ACCUMULATION_PHASE':
        values[0] = 1
        color = 'green'
    elif volume.phase_detected == 'DISTRIBUTION_PHASE':
        values[1] = 1
        color = 'red'
    else:
        values[2] = 1
        color = 'gray'
    
    fig = go.Figure(data=[
        go.Bar(
            x=phases,
            y=values,
            marker_color=[color if v > 0 else 'lightgray' for v in values],
            text=[volume.phase_detected if v > 0 else '' for v in values],
            textposition='auto'
        )
    ])
    
    fig.update_layout(
        title=f"Wyckoff Phase: {volume.phase_detected}",
        yaxis_title="Active",
        height=300,
        showlegend=False
    )
    
    return fig

# ============================================================================
# MAIN STREAMLIT APP
# ============================================================================

def main_streamlit():
    st.set_page_config(
        page_title="Smart Guardian V39.7",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .big-font {
        font-size:30px !important;
        font-weight: bold;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<p class="big-font">📊 SMART GUARDIAN V39.7</p>', unsafe_allow_html=True)
    st.markdown("**EUR/USD Trading Analysis System** | 11 Layers + Wyckoff + Volume Analysis")
    st.markdown("---")
    
    # Sidebar - Manual Inputs
    st.sidebar.header("📝 Manual Inputs (TradingView)")
    
    with st.sidebar.expander("Z-Score & Spread Data", expanded=True):
        z_score = st.number_input("1. Z-Score Strategic (126 bars)", value=0.0, format="%.4f")
        z_score_spike = st.number_input("2. Z-Score Spike Spread (20 bars)", value=0.0, format="%.4f")
        z_score_price = st.number_input("3. Z-Score Spike Price (20 bars)", value=0.0, format="%.4f")
        spread = st.number_input("4. Spread (US10Y-DE10Y) %", value=0.0, format="%.4f")
        ma_value = st.number_input("5. MA Value %", value=0.0, format="%.4f")
    
    with st.sidebar.expander("Market Indicators", expanded=True):
        vix = st.number_input("6. VIX", value=15.0, format="%.2f")
        correlation_h4 = st.number_input("7. Correlation H4", value=0.7, format="%.4f")
    
    with st.sidebar.expander("Yield Data", expanded=True):
        us2y = st.number_input("8. US 2-Year Yield %", value=4.0, format="%.4f")
        de2y = st.number_input("9. DE 2-Year Yield %", value=2.0, format="%.4f")
        use_spread_for_us10y = st.checkbox("Use Spread for US10Y?", value=True)
        if use_spread_for_us10y:
            us10y = spread
            st.info(f"US10Y = {us10y:.4f}%")
        else:
            us10y = st.number_input("10. US 10-Year Yield %", value=4.0, format="%.4f")
    
    # Auto-fetch data
    st.sidebar.header("🤖 Auto Data")
    
    with st.sidebar:
        if st.button("🔄 Fetch Auto Data", type="primary"):
            with st.spinner("Fetching data..."):
                cache = load_cache()
                
                # EUR/USD data
                try:
                    eurusd_df = get_eurusd_data_with_fallback(bars=100)
                    rsi_auto = calculate_rsi(eurusd_df['close'], CONFIG['rsi_period'])
                    st.success(f"✓ EUR/USD: RSI {rsi_auto:.1f}")
                    st.session_state['rsi_auto'] = rsi_auto
                    st.session_state['eurusd_df'] = eurusd_df
                except Exception as e:
                    st.error(f"EUR/USD fetch failed: {e}")
                    st.session_state['rsi_auto'] = None
                
                # Macro rates
                us_real, eu_real = get_macro_rates(cache)
                if us_real:
                    st.success(f"✓ US Real: {us_real:.2f}%")
                if eu_real:
                    st.success(f"✓ EU Real: {eu_real:.2f}%")
                
                st.session_state['us_real'] = us_real
                st.session_state['eu_real'] = eu_real
                
                save_cache(cache)
    
    # RSI Input
    rsi_auto = st.session_state.get('rsi_auto', None)
    if rsi_auto:
        use_auto_rsi = st.sidebar.checkbox(f"Use auto RSI ({rsi_auto:.1f})?", value=True)
        if use_auto_rsi:
            rsi_daily = rsi_auto
        else:
            rsi_daily = st.sidebar.number_input("11. RSI Daily (manual)", value=50.0, format="%.2f")
    else:
        rsi_daily = st.sidebar.number_input("11. RSI Daily", value=50.0, format="%.2f")
    
    st.sidebar.markdown("---")
    
    # Run Analysis Button
    if st.sidebar.button("🚀 RUN ANALYSIS", type="primary", use_container_width=True):
        
        with st.spinner("Running Smart Guardian V39.7 Analysis..."):
            
            # Load cache & historical data
            cache = load_cache()
            us_real = st.session_state.get('us_real', None)
            eu_real = st.session_state.get('eu_real', None)
            eurusd_df = st.session_state.get('eurusd_df', None)
            
            if eurusd_df is not None:
                atr_calculated = calculate_atr(eurusd_df['high'], eurusd_df['low'], eurusd_df['close'], CONFIG['atr_period'])
                spot_price = float(eurusd_df['close'].iloc[-1])
                ema_daily = float(eurusd_df['close'].rolling(window=20).mean().iloc[-1])
                ema_weekly = float(eurusd_df['close'].rolling(window=100).mean().iloc[-1])
            else:
                atr_calculated = 0.0050
                spot_price = 1.08000
                ema_daily = 1.07500
                ema_weekly = 1.07000
            
            z_score_today, z_score_prev, z_score_2days = get_historical_z_scores()
            rsi_today, rsi_prev, rsi_2days = get_historical_rsi()
            us2y_diff_7d_ago = get_historical_us2y_differential()
            
            z_score_today = z_score
            rsi_today = rsi_daily
            
            volume_df = fetch_volume_data(bars=30)
            
            # Analyze layers
            macro_verdict = analyze_macro_layer(us_real, eu_real)
            us2y_analysis = analyze_us2y_rates(us2y, de2y, us10y, us2y_diff_7d_ago)
            mr_z_analysis = detect_mean_reversion_z(z_score, z_score_prev, z_score_2days)
            mr_rsi_analysis = detect_mean_reversion_rsi(rsi_daily, rsi_prev, rsi_2days)
            double_mr = (mr_z_analysis.active and mr_rsi_analysis.active and mr_z_analysis.direction == mr_rsi_analysis.direction)
            spike_analysis = analyze_spike_volatility(z_score_spike, mr_z_analysis)
            price_z_analysis = analyze_price_zscore(z_score_price, z_score_spike)
            divergence_analysis = analyze_divergence(z_score_spike, z_score_price)
            volume_analysis = analyze_volume(volume_df)
            institutional_verdict = analyze_institutional_layer(z_score, spread, ma_value, mr_z_analysis)
            correlation_verdict = analyze_correlation_layer(correlation_h4)
            correlation_modifier = correlation_verdict.key_metrics['modifier']
            geopolitical_alert = detect_geopolitical_pressure(z_score, correlation_h4, rsi_daily, spike_analysis)
            technical_verdict = analyze_technical_layer(spot_price, ema_daily, ema_weekly, mr_rsi_analysis, price_z_analysis)
            risk_verdict = analyze_risk_layer(vix, atr_calculated)
            
            v38_score, v38_phase, v38_action, v38_confidence = calculate_v38_core(
                macro_verdict, institutional_verdict, technical_verdict, risk_verdict,
                correlation_modifier, volume_analysis
            )
            
            final_action, final_confidence, priority_eval, final_explanation = integrate_priority_hierarchy(
                macro_verdict, us2y_analysis, institutional_verdict, correlation_verdict,
                spike_analysis, divergence_analysis, volume_analysis,
                geopolitical_alert, technical_verdict, mr_z_analysis, mr_rsi_analysis,
                risk_verdict, v38_score, v38_action
            )
            
            # Execution plan
            if final_action in ['BUY', 'SELL']:
                direction = 'LONG' if final_action == 'BUY' else 'SHORT'
                sl = risk_verdict.key_metrics['sl_distance']
                tp = risk_verdict.key_metrics['tp_distance']
                if final_action == 'BUY':
                    sl_price = spot_price - sl
                    tp_price = spot_price + tp
                else:
                    sl_price = spot_price + sl
                    tp_price = spot_price - tp
                execution_plan = f"{direction} EUR/USD at {spot_price:.5f}. SL: {sl_price:.5f}, TP: {tp_price:.5f}. Risk: {risk_verdict.key_metrics['risk_environment']}."
            else:
                execution_plan = f"No position recommended. Action: {final_action}. Monitor pentru entry opportunity."
            
            # Active overrides
            active_overrides = []
            if geopolitical_alert.level == 'CRITICAL':
                active_overrides.append('GEOPOLITICAL_CRITICAL')
            elif geopolitical_alert.level == 'WARNING':
                active_overrides.append('GEOPOLITICAL_WARNING')
            if spike_analysis.confluence_level == 'CRITICAL':
                active_overrides.append('SPIKE_CONFLUENCE_CRITICAL')
            elif spike_analysis.confluence_level == 'WARNING':
                active_overrides.append('SPIKE_CONFLUENCE_WARNING')
            if divergence_analysis.detected:
                active_overrides.append(f'DIVERGENCE_{divergence_analysis.divergence_type}')
            if volume_analysis.phase_detected != 'NONE':
                active_overrides.append(f'WYCKOFF_{volume_analysis.phase_detected}')
            if volume_analysis.obv_divergence:
                active_overrides.append(f'OBV_DIV_{volume_analysis.obv_divergence_type}')
            if us2y_analysis.alert_level == 'ALERT':
                active_overrides.append('US2Y_ALERT')
            if mr_z_analysis.phase == 'TREND_REVERSAL':
                active_overrides.append('MR_TREND_REVERSAL_Z')
            if mr_rsi_analysis.phase == 'TREND_REVERSAL':
                active_overrides.append('MR_TREND_REVERSAL_RSI')
            if double_mr:
                active_overrides.append('DOUBLE_MR_CONFIRMATION')
            
            # Create report
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            report = GuardianReport(
                timestamp=timestamp,
                z_score=z_score,
                z_score_spike=z_score_spike,
                z_score_price=z_score_price,
                spread=spread,
                ma_value=ma_value,
                vix=vix,
                correlation_h4=correlation_h4,
                us2y=us2y,
                de2y=de2y,
                us10y=us10y,
                rsi_daily=rsi_daily,
                z_score_prev=z_score_prev,
                z_score_2days=z_score_2days,
                rsi_prev=rsi_prev,
                rsi_2days=rsi_2days,
                us2y_diff_7d_ago=us2y_diff_7d_ago,
                macro_verdict=macro_verdict,
                us2y_analysis=us2y_analysis,
                institutional_verdict=institutional_verdict,
                correlation_verdict=correlation_verdict,
                spike_analysis=spike_analysis,
                price_z_analysis=price_z_analysis,
                divergence_analysis=divergence_analysis,
                volume_analysis=volume_analysis,
                geopolitical_alert=geopolitical_alert,
                technical_verdict=technical_verdict,
                mr_z_analysis=mr_z_analysis,
                mr_rsi_analysis=mr_rsi_analysis,
                risk_verdict=risk_verdict,
                v38_integrated_score=v38_score,
                v38_phase=v38_phase,
                v38_action=v38_action,
                v38_confidence=v38_confidence,
                priority_evaluation=priority_eval,
                final_action=final_action,
                final_confidence=final_confidence,
                final_explanation=final_explanation,
                execution_plan=execution_plan,
                double_mr_confirmation=double_mr,
                active_overrides=active_overrides
            )
            
            # Save to CSV
            save_to_csv(report)
            save_cache(cache)
            
            st.session_state['report'] = report
            st.success("✅ Analysis Complete!")
    
    # Display Results
    if 'report' in st.session_state and st.session_state['report'] is not None:
        report = st.session_state['report']
        
        # Final Decision - Hero Section
        st.markdown("## 🎯 FINAL DECISION")
        
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col1:
            action_color = get_action_color(report.final_action)
            st.markdown(f"<div class='metric-card'><h3 style='color:{action_color};'>{report.final_action}</h3><p>Action</p></div>", unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"<div class='metric-card'><h3>{report.final_confidence:.0%}</h3><p>Confidence</p></div>", unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"<div class='metric-card'><h3>{report.v38_phase}</h3><p>Market Phase</p></div>", unsafe_allow_html=True)
        
        # Gauges
        col1, col2 = st.columns(2)
        
        with col1:
            st.plotly_chart(create_confidence_gauge(report.final_confidence, "Final Confidence"), use_container_width=True)
        
        with col2:
            st.plotly_chart(create_confidence_gauge(report.v38_confidence, "V38 Core Confidence"), use_container_width=True)
        
        # Execution Plan
        st.markdown("### 📋 Execution Plan")
        st.info(report.execution_plan)
        
        # Final Explanation
        st.markdown("### 💡 Explanation")
        st.write(report.final_explanation)
        
        # Priority Evaluation
        st.markdown("### 🔍 Priority Evaluation")
        st.write(report.priority_evaluation)
        
        # Active Overrides
        if report.active_overrides:
            st.markdown("### ⚠️ Active Overrides")
            for override in report.active_overrides:
                st.warning(f"🔴 {override}")
        
        st.markdown("---")
        
        # Layer Analysis
        st.markdown("## 📊 Layer Analysis")
        
        tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
            "📈 Layers Comparison",
            "🏛️ Institutional",
            "📉 Technical",
            "📊 Volume & Wyckoff",
            "🌍 Geopolitical",
            "📜 Historical"
        ])
        
        with tab1:
            st.plotly_chart(create_layer_comparison(report), use_container_width=True)
            
            # Summary table
            st.markdown("### Summary Table")
            
            summary_data = {
                'Layer': ['Macro', 'US2Y', 'Institutional', 'Correlation', 'Volume', 'Technical'],
                'Verdict': [
                    report.macro_verdict.verdict,
                    report.us2y_analysis.verdict,
                    report.institutional_verdict.verdict,
                    report.correlation_verdict.verdict,
                    report.volume_analysis.verdict,
                    report.technical_verdict.verdict
                ],
                'Confidence': [
                    f"{report.macro_verdict.confidence:.0%}",
                    f"{report.us2y_analysis.weight_override:.0%}",
                    f"{report.institutional_verdict.confidence:.0%}",
                    f"{report.correlation_verdict.confidence:.0%}",
                    f"{abs(report.volume_analysis.confidence_boost):.0%}" if report.volume_analysis.verdict != 'NEUTRAL' else "N/A",
                    f"{report.technical_verdict.confidence:.0%}"
                ],
                'Weight': [
                    f"{report.macro_verdict.weight:.0%}",
                    f"{report.us2y_analysis.weight_override:.0%}",
                    f"{report.institutional_verdict.weight:.0%}",
                    f"{report.correlation_verdict.key_metrics['modifier']:.1f}x",
                    f"{report.volume_analysis.confidence_boost:+.0%}",
                    f"{report.technical_verdict.weight:.0%}"
                ]
            }
            
            st.dataframe(pd.DataFrame(summary_data), use_container_width=True)
        
        with tab2:
            st.markdown("### Institutional Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Z-Score", f"{report.z_score:.2f}")
                st.metric("Spread", f"{report.spread:.2f}%")
                st.metric("MA Value", f"{report.ma_value:.2f}%")
            
            with col2:
                st.metric("Verdict", report.institutional_verdict.verdict)
                st.metric("Confidence", f"{report.institutional_verdict.confidence:.0%}")
                st.metric("MR Z-Score Phase", report.mr_z_analysis.phase)
            
            st.info(report.institutional_verdict.explanation)
            
            # Spike Analysis
            st.markdown("#### Spike Analysis")
            st.write(f"**Level:** {report.spike_analysis.level} | **Confluence:** {report.spike_analysis.confluence_level}")
            st.write(report.spike_analysis.explanation)
            
            # Divergence
            if report.divergence_analysis.detected:
                st.warning(f"**Divergence Detected:** {report.divergence_analysis.divergence_type}")
                st.write(report.divergence_analysis.explanation)
        
        with tab3:
            st.markdown("### Technical Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Spot Price", f"{report.technical_verdict.key_metrics['spot']:.5f}")
                st.metric("EMA Daily", f"{report.technical_verdict.key_metrics['ema_daily']:.5f}")
                st.metric("EMA Weekly", f"{report.technical_verdict.key_metrics['ema_weekly']:.5f}")
            
            with col2:
                st.metric("RSI", f"{report.rsi_daily:.1f}")
                st.metric("Verdict", report.technical_verdict.verdict)
                st.metric("Confidence", f"{report.technical_verdict.confidence:.0%}")
            
            st.info(report.technical_verdict.explanation)
            
            # RSI MR
            st.markdown("#### RSI Mean Reversion")
            st.write(f"**Phase:** {report.mr_rsi_analysis.phase}")
            st.write(report.mr_rsi_analysis.explanation)
        
        with tab4:
            st.markdown("### Volume & Wyckoff Analysis")
            
            st.plotly_chart(create_wyckoff_visualization(report.volume_analysis), use_container_width=True)
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Volume Ratio", f"{report.volume_analysis.volume_ratio:.2f}x")
                st.metric("OBV Slope", report.volume_analysis.obv_slope)
            
            with col2:
                st.metric("Smart Money", report.volume_analysis.smart_money_signal)
                st.metric("Delta Signal", report.volume_analysis.delta_signal)
            
            with col3:
                st.metric("Wick Pressure", report.volume_analysis.wick_pressure)
                st.metric("Consolidation", report.volume_analysis.consolidation_quality)
            
            st.info(report.volume_analysis.explanation)
            
            if report.volume_analysis.obv_divergence:
                st.warning(f"**OBV Divergence:** {report.volume_analysis.obv_divergence_type}")
        
        with tab5:
            st.markdown("### Geopolitical & Risk")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Geopolitical Level", report.geopolitical_alert.level)
                st.metric("VIX", f"{report.vix:.1f}")
                st.metric("Risk Environment", report.risk_verdict.key_metrics['risk_environment'])
            
            with col2:
                st.metric("Correlation H4", f"{report.correlation_h4:.2f}")
                st.metric("Correlation Modifier", f"{report.correlation_verdict.key_metrics['modifier']:.1f}x")
                st.metric("ATR", f"{report.risk_verdict.key_metrics['atr']:.5f}")
            
            st.write(report.geopolitical_alert.explanation)
        
        with tab6:
            st.markdown("### Historical Charts")
            
            hist_chart = create_historical_chart()
            if hist_chart:
                st.plotly_chart(hist_chart, use_container_width=True)
            else:
                st.info("No historical data available yet. Run analysis a few times to build history.")

if __name__ == "__main__":
    if 'report' not in st.session_state:
        st.session_state['report'] = None
    
    main_streamlit()
            
    