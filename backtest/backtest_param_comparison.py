"""
Backtest: Old vs New Parameter Comparison
==========================================
Fetches ~2 months of Bybit 5m OHLCV for all 8 regime assets,
reconstructs the Regime Filter bar-by-bar under OLD and NEW parameters,
counts signal frequency, regime distribution, and simulates trade outcomes.

Directory:
  backtest/
  ├── backtest_param_comparison.py   <- this script
  └── output/

Usage:
  cd backtest
  python backtest_param_comparison.py
"""

import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
from datetime import datetime, timezone, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR  = SCRIPT_DIR / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

try:
    from dotenv import load_dotenv
    for env in [SCRIPT_DIR / '.env', SCRIPT_DIR.parent / '.env']:
        if env.exists():
            load_dotenv(env); break
except ImportError:
    pass

# -----------------------------------------
# Regime assets (identical to Bot)
# -----------------------------------------
REGIME_ASSETS = [
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'BNB/USDT:USDT',
    'XRP/USDT:USDT', 'AVAX/USDT:USDT', 'ADA/USDT:USDT', 'DOGE/USDT:USDT'
]
LABELS    = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'AVAX', 'ADA', 'DOGE']
TIMEFRAME = '5m'
ADX_WIN   = 14
BACKTEST_DAYS = 60   # ~2 months

# -----------------------------------------
# Parameter sets: OLD vs NEW
# -----------------------------------------
PARAM_SETS = {
    'OLD': dict(
        MR_SCORE_THR  = 0.55,
        TR_SCORE_THR  = 0.40,
        Z_LONG_PCT    = 20,
        Z_SHORT_PCT   = 80,
        EMA_SLOPE_BARS= 3,
        TR_BB_PCT     = 60,
        HVOL_ATR_PCT  = 85,
        ADX_THR       = 20,
        EMA_CONSENSUS = 0.60,   # 60% = 5/8
        NDIPDI_THR    = 5.0,
        SLOPE_LOOKBACK= 3,      # 15 min
        BEAR_NEEDS_ADX= False,
        DROP_TR_SCORE = False,  # OLD 仍保留 score <= tr_thr 關卡
        DI_SLOPE_EPS  = 0.0,    # >0 表 ndi_slope > 0
    ),
    'NEW': dict(
        MR_SCORE_THR  = 0.52,
        TR_SCORE_THR  = 0.42,
        Z_LONG_PCT    = 25,
        Z_SHORT_PCT   = 75,
        EMA_SLOPE_BARS= 2,
        TR_BB_PCT     = 50,
        HVOL_ATR_PCT  = 90,
        ADX_THR       = 18,
        EMA_CONSENSUS = 0.50,   # 50% = 4/8
        NDIPDI_THR    = 5.0,
        SLOPE_LOOKBACK= 3,
        BEAR_NEEDS_ADX= False,
        DROP_TR_SCORE = False,
        DI_SLOPE_EPS  = 0.0,
    ),
    # [v4] 對應 prototype 七個改動 + [v5] Combo A+B（與 live 對齊）
    'V4': dict(
        MR_SCORE_THR  = 0.52,
        TR_SCORE_THR  = 0.42,   # （V4 不再使用，留作參考）
        Z_LONG_PCT    = 25,
        Z_SHORT_PCT   = 75,
        EMA_SLOPE_BARS= 2,
        TR_BB_PCT     = 30,     # [v5 Combo A] 50→30
        HVOL_ATR_PCT  = 90,
        ADX_THR       = 18,
        EMA_CONSENSUS = 0.375,  # 改動7
        NDIPDI_THR    = 3.0,    # 改動3
        SLOPE_LOOKBACK= 1,      # 改動4：3 → 1 bar
        BEAR_NEEDS_ADX= True,   # 改動5：is_bear 須配 mean_adx > 30
        DROP_TR_SCORE = True,   # 改動2：移除 score <= tr_thr 雙重關卡
        DI_SLOPE_EPS  = -0.5,   # [v5 Combo B] 容忍 1-bar DI 輕微負斜率
    ),
}

PARAM_LABELS = {
    'MR_SCORE_THR':   'MR threshold',
    'TR_SCORE_THR':   'TR threshold',
    'Z_LONG_PCT':     'Z long pct',
    'Z_SHORT_PCT':    'Z short pct',
    'EMA_SLOPE_BARS': 'EMA slope bars',
    'TR_BB_PCT':      'BB width pct',
    'HVOL_ATR_PCT':   'HVol ATR pct',
    'ADX_THR':        'ADX threshold',
    'EMA_CONSENSUS':  'EMA consensus',
}

# -----------------------------------------
# ADX / DI (identical to Bot)
# -----------------------------------------
def rolling_adx_wilder(highs, lows, closes, win=ADX_WIN):
    n = len(closes)
    h, l, c = np.array(highs), np.array(lows), np.array(closes)
    tr  = np.maximum(h[1:]-l[1:],
          np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    dmp = np.where((h[1:]-h[:-1]) > (l[:-1]-l[1:]),
                   np.maximum(h[1:]-h[:-1], 0), 0)
    dmn = np.where((l[:-1]-l[1:]) > (h[1:]-h[:-1]),
                   np.maximum(l[:-1]-l[1:], 0), 0)
    atr_w = np.zeros(n-1); pdi_w = np.zeros(n-1); ndi_w = np.zeros(n-1)
    if win <= len(tr):
        atr_w[win-1] = tr[:win].sum()
        pdi_w[win-1] = dmp[:win].sum()
        ndi_w[win-1] = dmn[:win].sum()
    for i in range(win, n-1):
        atr_w[i] = atr_w[i-1] - atr_w[i-1]/win + tr[i]
        pdi_w[i] = pdi_w[i-1] - pdi_w[i-1]/win + dmp[i]
        ndi_w[i] = ndi_w[i-1] - ndi_w[i-1]/win + dmn[i]
    with np.errstate(divide='ignore', invalid='ignore'):
        pdi = np.where(atr_w > 0, 100*pdi_w/atr_w, 0)
        ndi = np.where(atr_w > 0, 100*ndi_w/atr_w, 0)
        dx  = np.where((pdi+ndi) > 0, 100*np.abs(pdi-ndi)/(pdi+ndi), 0)
    adx_arr = np.full(n-1, 25.0)
    if 2*win < len(dx):
        adx_arr[2*win] = dx[win:2*win].mean()
    for i in range(2*win+1, n-1):
        adx_arr[i] = (adx_arr[i-1]*(win-1) + dx[i]) / win
    adx_arr[:2*win] = adx_arr[2*win] if 2*win < len(adx_arr) else 25.0
    full_adx = np.full(n, 25.0); full_adx[1:] = adx_arr
    full_pdi = np.zeros(n);      full_pdi[1:] = pdi
    full_ndi = np.zeros(n);      full_ndi[1:] = ndi
    return full_adx, full_pdi, full_ndi

def ema(arr, period):
    result = np.full_like(arr, np.nan, dtype=float)
    k = 2.0 / (period + 1)
    start = next((i for i, v in enumerate(arr) if not np.isnan(v)), None)
    if start is None:
        return result
    result[start] = arr[start]
    for i in range(start + 1, len(arr)):
        result[i] = arr[i] * k + result[i-1] * (1 - k)
    return result

def bbw(closes, period=20):
    c = np.array(closes, dtype=float)
    upper = np.full_like(c, np.nan)
    lower = np.full_like(c, np.nan)
    mid   = np.full_like(c, np.nan)
    for i in range(period-1, len(c)):
        sl = c[i-period+1:i+1]
        m  = sl.mean(); s = sl.std(ddof=0)
        mid[i] = m; upper[i] = m + 2*s; lower[i] = m - 2*s
    with np.errstate(divide='ignore', invalid='ignore'):
        bw = np.where(mid > 0, (upper - lower) / mid, np.nan)
    return bw

def zscore(closes, period=20):
    c = np.array(closes, dtype=float)
    z = np.full_like(c, np.nan)
    for i in range(period-1, len(c)):
        sl = c[i-period+1:i+1]
        m  = sl.mean(); s = sl.std(ddof=0)
        z[i] = (c[i] - m) / s if s > 0 else 0
    return z

def atr_pct(highs, lows, closes):
    h, l, c = np.array(highs), np.array(lows), np.array(closes)
    tr  = np.maximum(h[1:]-l[1:],
          np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    atr_vals = np.full(len(c), np.nan)
    for i in range(14, len(tr)):
        atr_vals[i+1] = tr[i-14:i].mean()
    with np.errstate(divide='ignore', invalid='ignore'):
        return np.where(c > 0, atr_vals / c, np.nan)

def safe_pct(arr, pct):
    vals = [v for v in arr if v is not None and not np.isnan(v)]
    return float(np.percentile(vals, pct)) if vals else 0.0

# -----------------------------------------
# Fetch 2 months of OHLCV
# -----------------------------------------
def fetch_data():
    print(f"Connecting to Bybit, fetching {BACKTEST_DAYS} days of 5m data...")
    exchange = ccxt.bybit({'enableRateLimit': True})

    end_dt   = datetime.now(timezone.utc)
    start_dt = end_dt - timedelta(days=BACKTEST_DAYS)
    since_ms = int(start_dt.timestamp() * 1000)

    # 5m bars per day = 288; 60 days = 17280 bars
    # Bybit returns max 1000 per call; need pagination
    all_data = {}
    for sym, lbl in zip(REGIME_ASSETS, LABELS):
        print(f"  -> {lbl:6s}", end='', flush=True)
        frames = []
        cur_ms = since_ms
        while True:
            try:
                raw = exchange.fetch_ohlcv(
                    sym, timeframe=TIMEFRAME, since=cur_ms, limit=1000)
            except Exception as e:
                print(f" ERROR: {e}"); break
            if not raw:
                break
            frames.extend(raw)
            last_ts = raw[-1][0]
            if last_ts >= int(end_dt.timestamp() * 1000) - 300_000:
                break
            cur_ms = last_ts + 300_000   # +5 min

        df = pd.DataFrame(frames, columns=['ts','open','high','low','close','vol'])
        df['time'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
        df = df[(df['time'] >= pd.Timestamp(start_dt)) &
                (df['time'] <= pd.Timestamp(end_dt))].drop_duplicates('ts').reset_index(drop=True)

        # Pre-compute indicators
        df['adx'], df['pdi'], df['ndi'] = rolling_adx_wilder(
            df['high'].values, df['low'].values, df['close'].values)
        df['ndipdi'] = df['ndi'] - df['pdi']
        df['bbw']    = bbw(df['close'].values)
        df['zscore'] = zscore(df['close'].values)
        df['atr_p']  = atr_pct(df['high'].values, df['low'].values, df['close'].values)
        df['ema21']  = ema(df['close'].values, 21)

        all_data[lbl] = df
        print(f" {len(df)} bars OK")

    return all_data

# -----------------------------------------
# Align all assets to BTC timeline
# -----------------------------------------
def align_to_btc(all_data):
    btc = all_data['BTC'].copy()
    aligned = {'BTC': btc}
    for lbl, df in all_data.items():
        if lbl == 'BTC':
            continue
        merged = btc[['time']].merge(
            df[['time','adx','pdi','ndi','ndipdi','bbw','zscore','atr_p','ema21','close']],
            on='time', how='left')
        aligned[lbl] = merged
    return aligned

# -----------------------------------------
# Reconstruct regime signal bar-by-bar for one parameter set
# -----------------------------------------
def simulate_regime(aligned, params, slope_bars=None):
    if slope_bars is None:
        slope_bars = params['EMA_SLOPE_BARS']

    di_slope_lookback = params.get('SLOPE_LOOKBACK', 3)  # [v4] 改動4
    ndipdi_thr        = params.get('NDIPDI_THR', 5.0)    # [v4] 改動3
    bear_needs_adx    = params.get('BEAR_NEEDS_ADX', False)  # [v4] 改動5
    drop_tr_score     = params.get('DROP_TR_SCORE', False)   # [v4] 改動2

    btc   = aligned['BTC']
    n     = len(btc)
    times = btc['time'].values

    results = []
    warmup  = max(ADX_WIN * 2 + 5, 30)

    for i in range(warmup, n):
        # Collect cross-asset metrics at bar i
        adx_list=[]; pdi_list=[]; ndi_list=[]; ndipdi_list=[]
        bbw_list=[]; z_list=[]; atr_list=[]
        ndi_prev_list=[]; pdi_prev_list=[]
        ret_list = []
        ema_slopes = []
        n_assets = 0

        for lbl, df in aligned.items():
            row = df.iloc[i]
            if pd.isna(row['adx']) or pd.isna(row['bbw']):
                continue
            adx_list.append(row['adx'])
            pdi_list.append(row['pdi'])
            ndi_list.append(row['ndi'])
            ndipdi_list.append(row['ndipdi'])
            bbw_list.append(row['bbw'])
            z_list.append(row['zscore'] if not pd.isna(row['zscore']) else 0)
            atr_list.append(row['atr_p'] if not pd.isna(row['atr_p']) else 0)
            # [v4] 改動4：DI slope 用 SLOPE_LOOKBACK bars
            prev_i = max(0, i - di_slope_lookback)
            ndi_prev_list.append(df.iloc[prev_i]['ndi'])
            pdi_prev_list.append(df.iloc[prev_i]['pdi'])
            # 24h return（288 bars × 5min）
            ret_i = max(0, i - 288)
            close_now  = df.iloc[i]['close']
            close_prev = df.iloc[ret_i]['close']
            if close_prev and close_prev > 0:
                ret_list.append((close_now - close_prev) / close_prev)
            n_assets += 1

            # EMA slope (slope_bars)
            if i >= slope_bars:
                ema_now  = df.iloc[i]['ema21']
                ema_prev = df.iloc[i - slope_bars]['ema21']
                if not (pd.isna(ema_now) or pd.isna(ema_prev)):
                    ema_slopes.append(1 if ema_now > ema_prev else -1)

        if n_assets < 4:
            results.append({'time': times[i], 'regime': 0, 'is_highvol': False,
                           'mean_adx': 0, 'score': 0.5, 'ema_dir': 0})
            continue

        # Compute mean metrics
        mean_adx    = float(np.mean(adx_list))
        mean_ndi    = float(np.mean(ndi_list))
        mean_pdi    = float(np.mean(pdi_list))
        mean_ndipdi = float(np.mean(ndipdi_list))
        ndi_slope   = mean_ndi - float(np.mean(ndi_prev_list))
        pdi_slope   = mean_pdi - float(np.mean(pdi_prev_list))
        di_slope_eps = params.get('DI_SLOPE_EPS', 0.0)
        if di_slope_eps < 0:
            ndi_rising = ndi_slope >= di_slope_eps
            pdi_rising = pdi_slope >= di_slope_eps
        else:
            ndi_rising = ndi_slope > di_slope_eps  # 0 → strict > 0
            pdi_rising = pdi_slope > di_slope_eps

        # Thresholds from rolling percentile
        mr_thr  = params['MR_SCORE_THR']
        tr_thr  = params['TR_SCORE_THR']
        zl_thr  = safe_pct(z_list, params['Z_LONG_PCT'])
        zs_thr  = safe_pct(z_list, params['Z_SHORT_PCT'])
        bb_thr  = safe_pct(bbw_list, params['TR_BB_PCT'])
        atr_hi  = safe_pct(atr_list, params['HVOL_ATR_PCT'])
        adx_thr = params['ADX_THR']

        # HighVol gate
        mean_atr    = float(np.nanmean(atr_list))
        is_highvol  = mean_atr >= atr_hi

        # Composite score (simplified: ADX/bbw/zscore blend)
        mean_bbw = float(np.nanmean(bbw_list))
        mean_z   = float(np.nanmean(z_list))
        # Normalize each component 0~1
        norm_adx = min(mean_adx / 60.0, 1.0)
        norm_bbw = min(mean_bbw / 0.15, 1.0)
        norm_z   = min(abs(mean_z) / 3.0, 1.0)
        # score: high=MR (low volatility/trend), low=TR (trending)
        score = (1 - norm_adx) * 0.5 + (1 - norm_bbw) * 0.3 + norm_z * 0.2

        # EMA direction (majority vote)
        threshold = max(1, int(n_assets * params['EMA_CONSENSUS']))
        up_votes  = sum(1 for s in ema_slopes if s > 0)
        dn_votes  = sum(1 for s in ema_slopes if s < 0)
        if up_votes >= threshold:
            ema_dir = +1
        elif dn_votes >= threshold:
            ema_dir = -1
        else:
            ema_dir = 0

        # [v4] is_bear 計算（改動5）
        bear_votes = sum(1 for r in ret_list if r < -0.03)
        bull_votes = sum(1 for r in ret_list if r > +0.02)
        is_bear_raw = bear_votes > n_assets // 2
        if bear_needs_adx:
            is_bear = is_bear_raw and mean_adx > 30
        else:
            is_bear = is_bear_raw
        if bull_votes > n_assets // 2:
            is_bear = False

        # Regime signal
        regime = 0
        if is_highvol:
            regime = 0
        elif score >= mr_thr:
            # MR zone
            if mean_z < zl_thr:
                regime = 0 if is_bear else +1
            elif mean_z > zs_thr:
                regime = -1
        else:
            # [v4] 改動2：DROP_TR_SCORE=True 移除 score <= tr_thr 關卡
            if drop_tr_score:
                trend_ok = mean_adx >= adx_thr and mean_bbw >= bb_thr
            else:
                trend_ok = score <= tr_thr and mean_adx >= adx_thr and mean_bbw >= bb_thr
            if trend_ok:
                # [v4] 改動3：NDIPDI_THR 由 5 → 3
                if mean_ndipdi < -ndipdi_thr and ema_dir == +1 and pdi_rising:
                    regime = 0 if is_bear else +2
                elif mean_ndipdi > +ndipdi_thr and ema_dir == -1 and ndi_rising:
                    regime = -2

        results.append({
            'time':       times[i],
            'regime':     regime,
            'is_highvol': is_highvol,
            'mean_adx':   round(mean_adx, 2),
            'score':      round(score, 4),
            'ema_dir':    ema_dir,
            'mean_ndipdi':round(mean_ndipdi, 2),
            'ndi_rising': int(ndi_rising),
            'pdi_rising': int(pdi_rising),
        })

    return pd.DataFrame(results)

# -----------------------------------------
# Compute summary statistics
# -----------------------------------------
def summarise(df, label):
    total   = len(df)
    counts  = df['regime'].value_counts().to_dict()
    regimes = {-2: 'SHORT TR', -1: 'SHORT MR', 0: 'No signal', 1: 'LONG MR', 2: 'LONG TR'}

    print(f"\n{'='*55}")
    print(f"  {label} Results  ({total} bars = {total*5/60/24:.0f} days)")
    print(f"{'='*55}")

    rows = []
    for code, name in sorted(regimes.items()):
        cnt = counts.get(code, 0)
        pct = cnt / total * 100
        print(f"  {name:12s}  {cnt:6d} bars  {pct:5.1f}%")
        rows.append({'regime': name, 'bars': cnt, 'pct': round(pct, 1)})

    active = sum(counts.get(c, 0) for c in [-2,-1,1,2])
    hvol   = df['is_highvol'].sum()
    print(f"  {'---':12s}  {'---':>6s}  {'---':>6}")
    print(f"  {'Active signal':12s}  {active:6d} bars  {active/total*100:5.1f}%")
    print(f"  {'High vol':12s}  {int(hvol):6d} bars  {hvol/total*100:5.1f}%")

    # Signal transitions (proxy for trade opportunities)
    df2 = df.copy()
    df2['prev_regime'] = df2['regime'].shift(1)
    entries = df2[(df2['regime'] != df2['prev_regime']) & (df2['regime'] != 0)]
    long_entries  = entries[entries['regime'] > 0]
    short_entries = entries[entries['regime'] < 0]
    print(f"\n  Regime transitions (entry signals):")
    print(f"  Long  regime triggers : {len(long_entries)}")
    print(f"  Short regime triggers : {len(short_entries)}")

    return pd.DataFrame(rows), len(long_entries), len(short_entries)

# -----------------------------------------
# Plot comparison
# -----------------------------------------
def plot_comparison(old_df, new_df, start_dt, end_dt):
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle(
        f'Parameter Comparison: OLD vs NEW  |  {start_dt.date()} to {end_dt.date()}  |  5m bars\n'
        f'Regime signal distribution and ADX timeline',
        color='#e6edf3', fontsize=12, y=0.99
    )

    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.3,
                           top=0.94, bottom=0.05)

    REGIME_COLORS = {
        -2: '#ff7b72', -1: '#ffa657', 0: '#30363d', 1: '#79c0ff', 2: '#3fb950'
    }
    REGIME_NAMES = {
        -2: 'SHORT TR', -1: 'SHORT MR', 0: 'No signal', 1: 'LONG MR', 2: 'LONG TR'
    }

    for col, (label, df) in enumerate([('OLD', old_df), ('NEW', new_df)]):
        color_main = '#58a6ff' if col == 0 else '#3fb950'

        # Row 0: Regime heatmap
        ax0 = fig.add_subplot(gs[0, col])
        ax0.set_facecolor('#161b22')
        ax0.spines[:].set_color('#30363d')
        ax0.tick_params(colors='#8b949e', labelsize=7)
        times = pd.to_datetime(df['time'], utc=True)
        bar_colors = [REGIME_COLORS.get(r, '#30363d') for r in df['regime']]
        ax0.bar(times, [1]*len(df), width=0.003, color=bar_colors, alpha=0.85)
        ax0.set_yticks([])
        ax0.set_title(f'{label} — Regime signal timeline',
                      color=color_main, fontsize=9, loc='left')
        ax0.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax0.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
        ax0.grid(axis='x', color='#30363d', lw=0.4, alpha=0.5)

        from matplotlib.patches import Patch
        legend_patches = [Patch(color=v, label=REGIME_NAMES[k])
                          for k, v in REGIME_COLORS.items()]
        ax0.legend(handles=legend_patches, loc='upper right', fontsize=6,
                   facecolor='#21262d', labelcolor='#8b949e', edgecolor='#30363d', ncol=3)

        # Row 1: ADX + score timeline
        ax1 = fig.add_subplot(gs[1, col])
        ax1.set_facecolor('#161b22')
        ax1.spines[:].set_color('#30363d')
        ax1.tick_params(colors='#8b949e', labelsize=7)
        ax1.plot(times, df['mean_adx'], color=color_main, lw=0.8, alpha=0.8, label='mean ADX')
        adx_thr = 20 if col == 0 else 18
        ax1.axhline(adx_thr, color='#f0883e', lw=0.7, ls='--',
                    label=f'ADX threshold ({adx_thr})')
        ax1.axhline(40, color='#ff7b72', lw=0.5, ls=':', alpha=0.5)
        ax1.set_ylabel('mean ADX', color='#8b949e', fontsize=8)
        ax1.set_title(f'{label} — ADX timeline', color=color_main, fontsize=9, loc='left')
        ax1.legend(loc='upper right', fontsize=6, facecolor='#21262d',
                   labelcolor='#8b949e', edgecolor='#30363d')
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax1.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
        ax1.grid(axis='x', color='#30363d', lw=0.4, alpha=0.5)
        ax1.set_ylim(0, 65)

        # Row 2: Score timeline with MR/TR bands
        ax2 = fig.add_subplot(gs[2, col])
        ax2.set_facecolor('#161b22')
        ax2.spines[:].set_color('#30363d')
        ax2.tick_params(colors='#8b949e', labelsize=7)
        ax2.plot(times, df['score'], color=color_main, lw=0.7, alpha=0.7, label='score')

        mr_thr = 0.55 if col == 0 else 0.52
        tr_thr = 0.40 if col == 0 else 0.42
        ax2.axhline(mr_thr, color='#79c0ff', lw=0.8, ls='--',
                    label=f'MR thr ({mr_thr})')
        ax2.axhline(tr_thr, color='#ff7b72', lw=0.8, ls='--',
                    label=f'TR thr ({tr_thr})')
        ax2.fill_between(times, mr_thr, 1.0, alpha=0.05, color='#79c0ff', label='MR zone')
        ax2.fill_between(times, 0, tr_thr, alpha=0.05, color='#ff7b72', label='TR zone')
        ax2.set_ylabel('Composite score', color='#8b949e', fontsize=8)
        ax2.set_title(f'{label} — Score with MR/TR bands', color=color_main, fontsize=9, loc='left')
        ax2.legend(loc='upper right', fontsize=6, facecolor='#21262d',
                   labelcolor='#8b949e', edgecolor='#30363d', ncol=2)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        ax2.xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=0, interval=2))
        ax2.grid(axis='x', color='#30363d', lw=0.4, alpha=0.5)
        ax2.set_ylim(0, 1)

    out = str(OUTPUT_DIR / f'param_comparison_{start_dt.date()}_{end_dt.date()}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"\nChart saved: {out}")

# -----------------------------------------
# Export comparison CSV
# -----------------------------------------
def export_csv(old_df, new_df, v4_df, old_entries, new_entries, v4_entries,
               start_dt, end_dt):
    REGIME_NAMES = {
        -2: 'SHORT_TR', -1: 'SHORT_MR', 0: 'No_signal', 1: 'LONG_MR', 2: 'LONG_TR'
    }
    rows = []
    total = len(old_df)
    for code in [-2, -1, 0, 1, 2]:
        name = REGIME_NAMES[code]
        old_cnt = (old_df['regime'] == code).sum()
        new_cnt = (new_df['regime'] == code).sum()
        v4_cnt  = (v4_df['regime']  == code).sum()
        rows.append({
            'regime':     name,
            'old_bars':   int(old_cnt), 'old_pct': round(old_cnt / total * 100, 2),
            'new_bars':   int(new_cnt), 'new_pct': round(new_cnt / total * 100, 2),
            'v4_bars':    int(v4_cnt),  'v4_pct':  round(v4_cnt  / total * 100, 2),
            'v4_vs_new':  int(v4_cnt - new_cnt),
            'v4_x_new':   round(v4_cnt / max(new_cnt, 1), 2),
        })
    rows.append({
        'regime':     'LONG_triggers',
        'old_bars':   old_entries[0], 'old_pct': '',
        'new_bars':   new_entries[0], 'new_pct': '',
        'v4_bars':    v4_entries[0],  'v4_pct':  '',
        'v4_vs_new':  v4_entries[0] - new_entries[0],
        'v4_x_new':   round(v4_entries[0] / max(new_entries[0], 1), 2),
    })
    rows.append({
        'regime':     'SHORT_triggers',
        'old_bars':   old_entries[1], 'old_pct': '',
        'new_bars':   new_entries[1], 'new_pct': '',
        'v4_bars':    v4_entries[1],  'v4_pct':  '',
        'v4_vs_new':  v4_entries[1] - new_entries[1],
        'v4_x_new':   round(v4_entries[1] / max(new_entries[1], 1), 2),
    })
    out_df = pd.DataFrame(rows)
    out = str(OUTPUT_DIR / f'param_comparison_{start_dt.date()}_{end_dt.date()}.csv')
    out_df.to_csv(out, index=False)
    print(f"CSV saved: {out}")
    print(f"\n{out_df.to_string(index=False)}")
    return out_df

# -----------------------------------------
# Print parameter diff table
# -----------------------------------------
def print_param_diff():
    print(f"\n{'='*65}")
    print(f"  Parameter Changes: OLD vs NEW")
    print(f"{'='*65}")
    print(f"  {'Parameter':20s}  {'OLD':>8s}  {'NEW':>8s}  {'Direction':>20s}")
    print(f"  {'-'*62}")
    directions = {
        'MR_SCORE_THR':   'Dead zone narrows (more MR)',
        'TR_SCORE_THR':   'Dead zone narrows (more TR)',
        'Z_LONG_PCT':     'Long Z condition looser',
        'Z_SHORT_PCT':    'Short Z condition looser',
        'EMA_SLOPE_BARS': 'Trend triggers earlier',
        'TR_BB_PCT':      'BBW threshold lower',
        'HVOL_ATR_PCT':   'Harder to trigger HVol block',
        'ADX_THR':        'Lower ADX market enters TR',
        'EMA_CONSENSUS':  'Fewer assets needed for EMA vote',
    }
    for k, label in PARAM_LABELS.items():
        old_v = PARAM_SETS['OLD'][k]
        new_v = PARAM_SETS['NEW'][k]
        print(f"  {label:20s}  {str(old_v):>8s}  {str(new_v):>8s}  {directions[k]}")
    print(f"{'='*65}")

# -----------------------------------------
# MAIN
# -----------------------------------------
if __name__ == '__main__':
    print("=" * 65)
    print("  Old vs New Parameter Backtest (~2 months)")
    print("  Using full 8-asset Regime Filter reconstruction")
    print("=" * 65)

    print_param_diff()

    all_data = fetch_data()
    if not all_data or 'BTC' not in all_data:
        print("ERROR: Could not fetch data."); exit(1)

    aligned = align_to_btc(all_data)
    btc_df  = aligned['BTC']
    start_dt = btc_df['time'].iloc[0].to_pydatetime()
    end_dt   = btc_df['time'].iloc[-1].to_pydatetime()

    print(f"\nRunning OLD parameter simulation...")
    old_df = simulate_regime(aligned, PARAM_SETS['OLD'])
    old_summary, old_long, old_short = summarise(old_df, "OLD parameters")

    print(f"\nRunning NEW parameter simulation...")
    new_df = simulate_regime(aligned, PARAM_SETS['NEW'])
    new_summary, new_long, new_short = summarise(new_df, "NEW parameters")

    print(f"\nRunning V4 parameter simulation (改動 2/3/4/5/7)...")
    v4_df = simulate_regime(aligned, PARAM_SETS['V4'])
    v4_summary, v4_long, v4_short = summarise(v4_df, "V4 parameters")

    print(f"\nRendering charts (OLD vs NEW)...")
    plot_comparison(old_df, new_df, start_dt, end_dt)

    print(f"\nExporting 3-way comparison CSV...")
    export_csv(old_df, new_df, v4_df,
               (old_long, old_short),
               (new_long, new_short),
               (v4_long, v4_short),
               start_dt, end_dt)

    total = len(old_df)
    print(f"\n{'='*65}")
    print(f"  Final Comparison: OLD → NEW → V4")
    print(f"{'='*65}")
    for code, name in [(-2,'SHORT TR'),(2,'LONG TR'),(-1,'SHORT MR'),(1,'LONG MR')]:
        old_c = (old_df['regime'] == code).sum()
        new_c = (new_df['regime'] == code).sum()
        v4_c  = (v4_df['regime']  == code).sum()
        print(f"  {name:10s}  OLD={old_c/total*100:5.2f}%  NEW={new_c/total*100:5.2f}%  "
              f"V4={v4_c/total*100:5.2f}%  (V4 vs NEW: {v4_c - new_c:+d} bars, "
              f"x{v4_c / max(new_c, 1):.1f})")
    print(f"\n  Long  triggers  OLD={old_long}  NEW={new_long}  V4={v4_long}  "
          f"(V4 vs NEW: {v4_long - new_long:+d}, x{v4_long / max(new_long, 1):.1f})")
    print(f"  Short triggers  OLD={old_short}  NEW={new_short}  V4={v4_short}  "
          f"(V4 vs NEW: {v4_short - new_short:+d}, x{v4_short / max(new_short, 1):.1f})")
    print(f"\nDone. Check output/ folder.")
