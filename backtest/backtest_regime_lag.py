"""
Backtest: Regime Filter Signal Lag Verification
================================================
Fetches yesterday's Bybit 5-minute OHLCV, reconstructs the ADX timeline
offline, and quantifies the delay between momentum peak and Bot entry.

Directory structure:
  your_project/
  ├── prototype_long_short_v3.py
  ├── .env
  └── backtest/
      ├── backtest_regime_lag.py   <- this script
      └── output/                  <- auto-created

Usage:
  pip install ccxt pandas numpy matplotlib python-dotenv
  cd backtest
  python backtest_regime_lag.py

.env (optional): BYBIT_API_KEY / BYBIT_SECRET  (public data works without keys)
"""

import ccxt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from datetime import datetime, timezone, timedelta
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# -----------------------------------------
# Path setup: script lives in backtest/, output goes to backtest/output/
# -----------------------------------------
SCRIPT_DIR = Path(__file__).parent
OUTPUT_DIR  = SCRIPT_DIR / 'output'
OUTPUT_DIR.mkdir(exist_ok=True)

# Load .env from local dir first, then parent project dir
try:
    from dotenv import load_dotenv
    for env in [SCRIPT_DIR / '.env', SCRIPT_DIR.parent / '.env']:
        if env.exists():
            load_dotenv(env); break
except ImportError:
    pass

# -----------------------------------------
# Parameters -- identical to prototype_long_short_v3.py
# -----------------------------------------
REGIME_SYMBOLS   = [
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT',
    'ADA/USDT:USDT', 'DOGE/USDT:USDT'
]
TIMEFRAME        = '5m'
ADX_WIN          = 14
EMA_SLOPE_BARS   = 3
OHLCV_LIMIT      = 300
REGIME_CACHE_TTL = 60   # seconds

# Yesterday's actual trade log
ACTUAL_TRADES = [
    dict(time='2026-04-27 01:30', symbol='UNI', action='LONG_ENTRY',  price=3.296),
    dict(time='2026-04-27 01:31', symbol='SUI', action='LONG_ENTRY',  price=0.9525),
    dict(time='2026-04-27 01:31', symbol='OP',  action='LONG_ENTRY',  price=0.12658),
    dict(time='2026-04-27 02:07', symbol='OP',  action='NATIVE_EXIT', price=0.12683),
    dict(time='2026-04-27 02:08', symbol='SUI', action='NATIVE_EXIT', price=0.956),
    dict(time='2026-04-27 02:11', symbol='UNI', action='NATIVE_EXIT', price=3.308),
    dict(time='2026-04-27 16:08', symbol='ADA',  action='SHORT_ENTRY', price=0.2443),
    dict(time='2026-04-27 16:09', symbol='SOL',  action='SHORT_ENTRY', price=84.13),
    dict(time='2026-04-27 16:10', symbol='OP',   action='SHORT_ENTRY', price=0.11958),
    dict(time='2026-04-27 16:11', symbol='DOGE', action='SHORT_ENTRY', price=0.09731),
    dict(time='2026-04-27 16:12', symbol='ARB',  action='SHORT_ENTRY', price=0.12407),
    dict(time='2026-04-27 16:55', symbol='OP',   action='SHORT_EXIT',  price=0.12006),
    dict(time='2026-04-27 16:58', symbol='ARB',  action='SHORT_EXIT',  price=0.12446),
    dict(time='2026-04-27 17:00', symbol='ADA',  action='SHORT_EXIT',  price=0.2452),
    dict(time='2026-04-27 17:01', symbol='DOGE', action='SHORT_EXIT',  price=0.09763),
    dict(time='2026-04-27 17:24', symbol='SOL',  action='SHORT_EXIT',  price=84.4),
]

# -----------------------------------------
# Wilder smoothed ADX -- identical to Bot implementation
# -----------------------------------------
def rolling_adx_simple(highs, lows, closes, win=ADX_WIN):
    n = len(closes)
    adx = np.full(n, 25.0)
    if n < win * 2 + 1:
        return adx, np.zeros(n), np.zeros(n)

    h, l, c = np.array(highs), np.array(lows), np.array(closes)
    tr  = np.maximum(h[1:]-l[1:],
          np.maximum(np.abs(h[1:]-c[:-1]), np.abs(l[1:]-c[:-1])))
    dmp = np.where((h[1:]-h[:-1]) > (l[:-1]-l[1:]),
                   np.maximum(h[1:]-h[:-1], 0), 0)
    dmn = np.where((l[:-1]-l[1:]) > (h[1:]-h[:-1]),
                   np.maximum(l[:-1]-l[1:], 0), 0)

    atr_w = np.zeros(n-1); pdi_w = np.zeros(n-1); ndi_w = np.zeros(n-1)
    atr_w[win-1] = tr[:win].sum()
    pdi_w[win-1] = dmp[:win].sum()
    ndi_w[win-1] = dmn[:win].sum()
    for i in range(win, n-1):
        atr_w[i] = atr_w[i-1] - atr_w[i-1]/win + tr[i]
        pdi_w[i] = pdi_w[i-1] - pdi_w[i-1]/win + dmp[i]
        ndi_w[i] = ndi_w[i-1] - ndi_w[i-1]/win + dmn[i]

    with np.errstate(divide='ignore', invalid='ignore'):
        pdi = np.where(atr_w>0, 100*pdi_w/atr_w, 0)
        ndi = np.where(atr_w>0, 100*ndi_w/atr_w, 0)
        dx  = np.where((pdi+ndi)>0, 100*np.abs(pdi-ndi)/(pdi+ndi), 0)

    adx_arr = np.full(n-1, 25.0)
    adx_arr[2*win] = dx[win:2*win].mean() if 2*win <= len(dx) else 25.0
    for i in range(2*win+1, n-1):
        adx_arr[i] = (adx_arr[i-1]*(win-1) + dx[i]) / win
    adx_arr[:2*win] = adx_arr[2*win]

    full_adx = np.full(n, adx_arr[-1]); full_adx[1:] = adx_arr
    full_pdi = np.zeros(n);             full_pdi[1:] = pdi
    full_ndi = np.zeros(n);             full_ndi[1:] = ndi
    return full_adx, full_pdi, full_ndi

# -----------------------------------------
# Fetch yesterday's 5m OHLCV from Bybit
# -----------------------------------------
def fetch_yesterday_ohlcv():
    print("Connecting to Bybit, fetching yesterday's 5m OHLCV...")
    exchange = ccxt.bybit({'enableRateLimit': True})

    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    since_ms  = int(datetime(yesterday.year, yesterday.month, yesterday.day,
                              tzinfo=timezone.utc).timestamp() * 1000)
    all_data = {}
    for sym in REGIME_SYMBOLS:
        print(f"  -> {sym}")
        try:
            raw = exchange.fetch_ohlcv(sym, timeframe=TIMEFRAME,
                                       since=since_ms, limit=290)
            df  = pd.DataFrame(raw, columns=['ts','open','high','low','close','vol'])
            df['time'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
            df = df[df['time'].dt.date == yesterday].reset_index(drop=True)
            all_data[sym] = df
            print(f"     {len(df)} candles OK")
        except Exception as e:
            print(f"     ERROR: {e}")

    return all_data, yesterday

# -----------------------------------------
# Simulate Bot's Regime timeline bar-by-bar
# -----------------------------------------
def simulate_regime_timeline(all_data):
    btc_df = all_data.get('BTC/USDT:USDT')
    if btc_df is None or btc_df.empty:
        raise ValueError("Missing BTC data")

    records = []
    for i, ts in enumerate(btc_df['time'].tolist()):
        adx_vals = []; pdi_vals = []; ndi_vals = []
        for sym, df in all_data.items():
            if df.empty or i >= len(df):
                continue
            slice_df = df.iloc[:i+1]
            if len(slice_df) < ADX_WIN * 2 + 2:
                continue
            adx, pdi, ndi = rolling_adx_simple(
                slice_df['high'].values, slice_df['low'].values, slice_df['close'].values)
            adx_vals.append(adx[-1]); pdi_vals.append(pdi[-1]); ndi_vals.append(ndi[-1])

        if not adx_vals:
            continue
        mean_adx = float(np.mean(adx_vals))
        mean_pdi = float(np.mean(pdi_vals))
        mean_ndi = float(np.mean(ndi_vals))
        records.append({
            'time': ts,
            'mean_adx': round(mean_adx, 2),
            'mean_pdi': round(mean_pdi, 2),
            'mean_ndi': round(mean_ndi, 2),
            'regime':   -2 if (mean_ndi > mean_pdi and mean_adx >= 20) else +2,
        })
    return pd.DataFrame(records)

# -----------------------------------------
# Quantify momentum peak vs Bot entry lag
# -----------------------------------------
def analyse_lag(regime_df):
    sw = regime_df[
        (regime_df['time'].dt.hour >= 14) &
        (regime_df['time'].dt.hour <= 18)
    ].copy()
    if sw.empty:
        print("WARNING: No data in short session window"); return None, None

    peak_idx  = sw['mean_adx'].idxmax()
    peak_time = sw.loc[peak_idx, 'time']
    peak_adx  = sw.loc[peak_idx, 'mean_adx']
    bot_entry = pd.Timestamp('2026-04-27 16:08', tz='UTC')
    lag_min   = (bot_entry - peak_time).total_seconds() / 60

    print(f"\n{'='*55}")
    print(f"  Short Session Momentum Analysis")
    print(f"{'='*55}")
    print(f"  ADX peak time  : {peak_time.strftime('%H:%M UTC')}")
    print(f"  ADX peak value : {peak_adx:.2f}")
    print(f"  Bot first short: 16:08 UTC")
    print(f"  Signal lag     : {lag_min:+.0f} min")
    verdict = f"CONFIRMED: Bot entered {lag_min:.0f} min after ADX peak" \
              if lag_min > 0 else f"Bot entered {abs(lag_min):.0f} min before ADX peak"
    print(f"  {verdict}")
    print(f"{'='*55}\n")
    return peak_time, lag_min

# -----------------------------------------
# Plot regime analysis (3-panel)
# -----------------------------------------
def plot_regime_analysis(regime_df, peak_time, yesterday):
    fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
    fig.patch.set_facecolor('#0d1117')
    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.tick_params(colors='#8b949e', labelsize=9)
        ax.spines[:].set_color('#30363d')

    times = regime_df['time'].dt.tz_convert('UTC')

    # Panel 1: mean ADX
    ax1 = axes[0]
    ax1.plot(times, regime_df['mean_adx'], color='#58a6ff', lw=1.5,
             label='mean ADX (5-asset avg)')
    ax1.axhline(20, color='#f0883e', lw=0.8, ls='--', label='ADX=20 entry threshold')
    ax1.axhline(40, color='#ff7b72', lw=0.8, ls='--', label='ADX=40 overheated')
    if peak_time is not None:
        pv = regime_df.loc[regime_df['time'] == peak_time, 'mean_adx'].values
        if len(pv):
            ax1.axvline(peak_time, color='#ffa657', lw=1.2, ls=':', alpha=0.9)
            ax1.annotate(f"Momentum Peak\n{peak_time.strftime('%H:%M')}\nADX={pv[0]:.1f}",
                         xy=(peak_time, pv[0]), xytext=(10, -30),
                         textcoords='offset points', color='#ffa657', fontsize=8,
                         arrowprops=dict(arrowstyle='->', color='#ffa657', lw=0.8))
    for label, t, col in [('LONG 01:30',  pd.Timestamp('2026-04-27 01:30', tz='UTC'), '#3fb950'),
                           ('SHORT 16:08', pd.Timestamp('2026-04-27 16:08', tz='UTC'), '#ff7b72')]:
        ax1.axvline(t, color=col, lw=1.2, ls='-', alpha=0.7)
        ax1.text(t, 2, label, color=col, fontsize=7, va='bottom')
    ax1.set_ylabel('mean ADX', color='#8b949e', fontsize=9)
    ax1.set_title(f'Regime Filter Signal Lag Analysis — {yesterday} (UTC) | 5m candles',
                  color='#e6edf3', fontsize=11, pad=10)
    ax1.legend(loc='upper left', fontsize=8, facecolor='#21262d',
               labelcolor='#8b949e', edgecolor='#30363d')
    ax1.set_ylim(0, max(regime_df['mean_adx'].max() * 1.2, 50))

    # Panel 2: PDI vs NDI
    ax2 = axes[1]
    ax2.plot(times, regime_df['mean_pdi'], color='#3fb950', lw=1.2, label='+DI (bull)')
    ax2.plot(times, regime_df['mean_ndi'], color='#ff7b72', lw=1.2, label='-DI (bear)')
    ax2.fill_between(times, regime_df['mean_pdi'], regime_df['mean_ndi'],
                     where=regime_df['mean_ndi'] > regime_df['mean_pdi'],
                     alpha=0.15, color='#ff7b72', label='Bears dominant')
    ax2.fill_between(times, regime_df['mean_pdi'], regime_df['mean_ndi'],
                     where=regime_df['mean_pdi'] >= regime_df['mean_ndi'],
                     alpha=0.15, color='#3fb950', label='Bulls dominant')
    if peak_time is not None:
        ax2.axvline(peak_time, color='#ffa657', lw=1.2, ls=':', alpha=0.9)
    ax2.set_ylabel('+DI / -DI', color='#8b949e', fontsize=9)
    ax2.legend(loc='upper left', fontsize=8, facecolor='#21262d',
               labelcolor='#8b949e', edgecolor='#30363d')

    # Panel 3: Regime heatmap
    ax3 = axes[2]
    colors = ['#ff7b72' if r < 0 else '#3fb950' for r in regime_df['regime']]
    ax3.bar(times, [1]*len(regime_df), width=0.003, color=colors, alpha=0.85)
    trade_colors = {
        'LONG_ENTRY':  ('#3fb950', 'Long entry'),
        'SHORT_ENTRY': ('#ff7b72', 'Short entry'),
        'NATIVE_EXIT': ('#58a6ff', 'TP/SL exit'),
        'SHORT_EXIT':  ('#ffa657', 'Timeout exit'),
    }
    plotted = set()
    for t in ACTUAL_TRADES:
        ts  = pd.Timestamp(t['time'], tz='UTC')
        col, label = trade_colors.get(t['action'], ('#8b949e', t['action']))
        ax3.axvline(ts, color=col, lw=1.0, alpha=0.8,
                    label=label if label not in plotted else '_nolegend_')
        plotted.add(label)
    if peak_time is not None:
        ax3.axvline(peak_time, color='#ffa657', lw=1.5, ls=':', alpha=0.9,
                    label='ADX momentum peak')
    patches = [mpatches.Patch(color='#3fb950', alpha=0.6, label='Bull regime (+2)'),
               mpatches.Patch(color='#ff7b72', alpha=0.6, label='Bear regime (-2)')]
    patches += [mpatches.Patch(color=v[0], label=v[1]) for v in trade_colors.values()]
    ax3.legend(handles=patches, loc='upper left', fontsize=7, facecolor='#21262d',
               labelcolor='#8b949e', edgecolor='#30363d', ncol=3)
    ax3.set_ylabel('Regime (-2/+2)', color='#8b949e', fontsize=9)
    ax3.set_yticks([])

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
        ax.grid(axis='x', color='#30363d', lw=0.5, alpha=0.5)
    axes[-1].set_xlabel(f'Time (UTC)  {yesterday}', color='#8b949e', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = str(OUTPUT_DIR / f'regime_lag_analysis_{yesterday}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"Chart saved: {out}")

# -----------------------------------------
# BTC price chart
# -----------------------------------------
def plot_btc_price(all_data, yesterday):
    btc_df = all_data.get('BTC/USDT:USDT')
    if btc_df is None or btc_df.empty:
        return
    fig, ax = plt.subplots(figsize=(16, 4))
    fig.patch.set_facecolor('#0d1117')
    ax.set_facecolor('#161b22')
    ax.tick_params(colors='#8b949e', labelsize=9)
    ax.spines[:].set_color('#30363d')
    times = btc_df['time']
    ax.plot(times, btc_df['close'], color='#58a6ff', lw=1.2)
    ax.fill_between(times, btc_df['close'].min(), btc_df['close'],
                    alpha=0.08, color='#58a6ff')
    ax.axvspan(pd.Timestamp('2026-04-27 16:08', tz='UTC'),
               pd.Timestamp('2026-04-27 16:12', tz='UTC'),
               alpha=0.2, color='#ff7b72', label='Short entry window')
    ax.axvline(pd.Timestamp('2026-04-27 16:55', tz='UTC'),
               color='#ffa657', lw=0.8, ls='--', label='Timeout exits begin')
    ax.set_title(f'BTC/USDT 5m Close Price — {yesterday} (UTC)',
                 color='#e6edf3', fontsize=10)
    ax.set_ylabel('Price (USDT)', color='#8b949e', fontsize=9)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax.grid(color='#30363d', lw=0.5, alpha=0.5)
    ax.legend(fontsize=8, facecolor='#21262d', labelcolor='#8b949e', edgecolor='#30363d')
    plt.tight_layout()
    out = str(OUTPUT_DIR / f'btc_price_{yesterday}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"Chart saved: {out}")

# -----------------------------------------
# MAIN
# -----------------------------------------
if __name__ == '__main__':
    print("=" * 55)
    print("  Regime Filter Signal Lag Backtest")
    print("=" * 55)

    all_data, yesterday = fetch_yesterday_ohlcv()
    if not all_data:
        print("ERROR: Could not fetch data.")
        exit(1)

    print("\nReconstructing ADX / Regime timeline bar-by-bar...")
    regime_df = simulate_regime_timeline(all_data)
    print(f"Done -- {len(regime_df)} time points")

    peak_time, lag_min = analyse_lag(regime_df)

    print("Rendering charts...")
    plot_regime_analysis(regime_df, peak_time, yesterday)
    plot_btc_price(all_data, yesterday)

    out = str(OUTPUT_DIR / f'regime_timeline_{yesterday}.csv')
    regime_df.to_csv(out, index=False)
    print(f"CSV saved: {out}")

    sw = regime_df[(regime_df['time'].dt.hour >= 14) & (regime_df['time'].dt.hour <= 18)]
    if not sw.empty:
        print(f"\nShort session ADX stats (14:00-18:00):")
        print(f"  max={sw['mean_adx'].max():.2f}  min={sw['mean_adx'].min():.2f}  avg={sw['mean_adx'].mean():.2f}")
        peak_idx  = sw['mean_adx'].idxmax()
        post_peak = sw.loc[peak_idx:]
        declining = post_peak[post_peak['mean_adx'] < sw['mean_adx'].max() * 0.95]
        if not declining.empty:
            ds = declining.iloc[0]['time']
            mins = (pd.Timestamp('2026-04-27 16:08', tz='UTC') - ds).total_seconds() / 60
            if mins > 0:
                print(f"  WARNING: ADX started declining at {ds.strftime('%H:%M')},"
                      f" {mins:.0f} min before Bot entry")

    print("\nDone. Check output/ folder.")
