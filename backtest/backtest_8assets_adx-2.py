"""
Backtest: 8-Asset Individual ADX Timeline Analysis
===================================================
Reconstructs yesterday's full Regime Filter using all 8 assets defined in
prototype_long_short_v3.py (BTC/ETH/SOL/BNB/XRP/AVAX/ADA/DOGE).
Plots each asset's ADX, +DI, -DI individually to pinpoint momentum peak
times and compare them against Bot entry at 16:08 UTC.

Directory structure:
  your_project/
  ├── prototype_long_short_v3.py
  ├── .env
  └── backtest/
      ├── backtest_8assets_adx.py  <- this script
      └── output/                  <- auto-created

Usage:
  cd backtest
  python backtest_8assets_adx.py
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

# -----------------------------------------
# Path setup
# -----------------------------------------
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
# Parameters -- identical to prototype_long_short_v3.py
# -----------------------------------------
REGIME_ASSETS = [
    'BTC/USDT:USDT', 'ETH/USDT:USDT', 'SOL/USDT:USDT', 'BNB/USDT:USDT',
    'XRP/USDT:USDT', 'AVAX/USDT:USDT', 'ADA/USDT:USDT', 'DOGE/USDT:USDT'
]
LABELS    = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'AVAX', 'ADA', 'DOGE']
TIMEFRAME = '5m'
ADX_WIN   = 14

ASSET_COLORS = {
    'BTC':  '#F7931A', 'ETH':  '#627EEA', 'SOL':  '#9945FF',
    'BNB':  '#F3BA2F', 'XRP':  '#00AAE4', 'AVAX': '#E84142',
    'ADA':  '#0033AD', 'DOGE': '#C2A633',
}

BOT_SHORT_ENTRY = pd.Timestamp('2026-04-27 16:08', tz='UTC')
BOT_LONG_ENTRY  = pd.Timestamp('2026-04-27 01:30', tz='UTC')
BOT_TIMEOUT_START = pd.Timestamp('2026-04-27 16:55', tz='UTC')

# -----------------------------------------
# Wilder smoothed ADX -- identical to Bot
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

# -----------------------------------------
# Fetch yesterday's 5m OHLCV for all 8 assets
# -----------------------------------------
def fetch_all_assets():
    print("Connecting to Bybit, fetching 8 assets 5m OHLCV...")
    exchange = ccxt.bybit({'enableRateLimit': True})

    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)
    since_ms  = int(datetime(yesterday.year, yesterday.month, yesterday.day,
                             tzinfo=timezone.utc).timestamp() * 1000)
    all_data = {}
    for sym, lbl in zip(REGIME_ASSETS, LABELS):
        print(f"  -> {lbl:6s} ({sym})")
        try:
            raw = exchange.fetch_ohlcv(sym, timeframe=TIMEFRAME,
                                       since=since_ms, limit=300)
            df  = pd.DataFrame(raw, columns=['ts','open','high','low','close','vol'])
            df['time'] = pd.to_datetime(df['ts'], unit='ms', utc=True)
            df = df[df['time'].dt.date == yesterday].reset_index(drop=True)

            adx, pdi, ndi = rolling_adx_wilder(
                df['high'].values, df['low'].values, df['close'].values)
            df['adx']     = adx
            df['pdi']     = pdi
            df['ndi']     = ndi
            df['ndipdi']  = ndi - pdi

            all_data[lbl] = df
            sw = df[(df['time'].dt.hour >= 14) & (df['time'].dt.hour <= 18)]
            ndi_peak_t = sw.loc[sw['ndi'].idxmax(), 'time'].strftime('%H:%M') if not sw.empty else 'N/A'
            adx_peak_t = sw.loc[sw['adx'].idxmax(), 'time'].strftime('%H:%M') if not sw.empty else 'N/A'
            print(f"     {len(df)} candles | NDI peak: {ndi_peak_t} | ADX peak: {adx_peak_t}")
        except Exception as e:
            print(f"     ERROR: {e}")

    return all_data, yesterday

# -----------------------------------------
# Compute 8-asset mean series (aligned to BTC timeline)
# -----------------------------------------
def compute_mean_series(all_data):
    btc = all_data.get('BTC')
    if btc is None:
        return None
    adx_mat = []; ndipdi_mat = []
    for lbl, df in all_data.items():
        merged = btc[['time']].merge(df[['time','adx','ndipdi']], on='time', how='left')
        adx_mat.append(merged['adx'].values)
        ndipdi_mat.append(merged['ndipdi'].values)
    return pd.DataFrame({
        'time':        pd.to_datetime(btc['time'].values, utc=True),
        'mean_adx':    np.nanmean(adx_mat, axis=0),
        'mean_ndipdi': np.nanmean(ndipdi_mat, axis=0),
    })

# -----------------------------------------
# Print peak analysis table to terminal
# -----------------------------------------
def print_peak_analysis(all_data):
    print(f"\n{'='*70}")
    print(f"  Short Session Peak Analysis  (14:30 - 17:30 UTC)")
    print(f"  Bot short entry: 16:08 UTC")
    print(f"{'='*70}")
    print(f"  {'Asset':6s}  {'NDI peak':>9s}  {'NDI time':>8s}  "
          f"{'ADX peak':>9s}  {'ADX time':>8s}  {'NDI at entry':>12s}  {'Decay':>7s}  {'NDI lag':>8s}")
    print(f"  {'-'*68}")

    for lbl, df in all_data.items():
        w = df[(df['time'].dt.hour >= 14) & (df['time'].dt.hour <= 18)].copy()
        if w.empty:
            continue
        ndi_peak_row = w.loc[w['ndi'].idxmax()]
        adx_peak_row = w.loc[w['adx'].idxmax()]
        ndi_at_entry = df.loc[
            (df['time'] >= BOT_SHORT_ENTRY) &
            (df['time'] <= BOT_SHORT_ENTRY + pd.Timedelta('10min')), 'ndi'
        ].mean()
        decay_pct = (ndi_peak_row['ndi'] - ndi_at_entry) / ndi_peak_row['ndi'] * 100 \
                    if ndi_peak_row['ndi'] > 0 else 0
        ndi_lag   = (BOT_SHORT_ENTRY - ndi_peak_row['time']).total_seconds() / 60
        flag = f"DECAYED {decay_pct:.0f}%" if decay_pct > 5 else "still strong"

        print(f"  {lbl:6s}  {ndi_peak_row['ndi']:>9.1f}  "
              f"{ndi_peak_row['time'].strftime('%H:%M'):>8s}  "
              f"{adx_peak_row['adx']:>9.1f}  "
              f"{adx_peak_row['time'].strftime('%H:%M'):>8s}  "
              f"{ndi_at_entry:>12.1f}  {flag:>10s}  {ndi_lag:>+7.0f}m")
    print(f"{'='*70}\n")

# -----------------------------------------
# Chart 1: Full-day overview + 8 individual asset panels
# -----------------------------------------
def plot_8assets(all_data, mean_df, yesterday):
    fig = plt.figure(figsize=(20, 22))
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle(
        f'8-Asset ADX / NDI Individual Timeline  |  {yesterday} (UTC)  |  5m candles\n'
        f'Reconstructing prototype_long_short_v3.py Regime Filter',
        color='#e6edf3', fontsize=12, y=0.98
    )

    gs = gridspec.GridSpec(5, 2, figure=fig, hspace=0.55, wspace=0.3,
                           top=0.93, bottom=0.04)

    # --- Overview row: 8-asset mean ADX + mean NDI-PDI ---
    ax_mean = fig.add_subplot(gs[0, :])
    ax_mean.set_facecolor('#161b22')
    ax_mean.spines[:].set_color('#30363d')
    ax_mean.tick_params(colors='#8b949e', labelsize=8)

    if mean_df is not None:
        t = mean_df['time']
        ax_mean.plot(t, mean_df['mean_adx'],
                     color='#58a6ff', lw=1.8, label='mean ADX (8-asset avg)')
        ax2m = ax_mean.twinx()
        ax2m.plot(t, mean_df['mean_ndipdi'],
                  color='#ff7b72', lw=1.2, ls='--',
                  label='mean NDI-PDI (bear momentum)', alpha=0.8)
        ax2m.axhline(5, color='#ff7b72', lw=0.6, ls=':', alpha=0.5)
        ax2m.set_ylabel('NDI-PDI', color='#ff7b72', fontsize=8)
        ax2m.tick_params(colors='#ff7b72', labelsize=8)
        ax2m.spines[:].set_color('#30363d')
        lines1, lbl1 = ax_mean.get_legend_handles_labels()
        lines2, lbl2 = ax2m.get_legend_handles_labels()
        ax_mean.legend(lines1+lines2, lbl1+lbl2, loc='upper left', fontsize=7,
                       facecolor='#21262d', labelcolor='#8b949e', edgecolor='#30363d')

    ax_mean.axvline(BOT_SHORT_ENTRY, color='#ff7b72', lw=1.5, ls='-', alpha=0.8)
    ax_mean.text(BOT_SHORT_ENTRY, 5, ' SHORT\n 16:08', color='#ff7b72', fontsize=7, va='bottom')
    ax_mean.axvline(BOT_LONG_ENTRY, color='#3fb950', lw=1.0, ls='-', alpha=0.6)
    ax_mean.axhline(20, color='#f0883e', lw=0.7, ls='--', alpha=0.6)
    ax_mean.axhline(40, color='#ff7b72', lw=0.7, ls='--', alpha=0.4)
    ax_mean.set_ylabel('mean ADX', color='#8b949e', fontsize=9)
    ax_mean.set_title('Overview: 8-asset composite (values used by Regime Filter)',
                      color='#8b949e', fontsize=9, loc='left', pad=4)
    ax_mean.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    ax_mean.xaxis.set_major_locator(mdates.HourLocator(interval=2))
    ax_mean.grid(axis='x', color='#30363d', lw=0.4, alpha=0.5)

    # --- 8 individual asset panels ---
    for i, (lbl, df) in enumerate(all_data.items()):
        row = (i // 2) + 1
        col = i % 2
        ax  = fig.add_subplot(gs[row, col])
        ax.set_facecolor('#161b22')
        ax.spines[:].set_color('#30363d')
        ax.tick_params(colors='#8b949e', labelsize=7)

        color = ASSET_COLORS.get(lbl, '#58a6ff')
        t = df['time']

        ax.plot(t, df['adx'], color=color, lw=1.5, label='ADX')
        ax.axhline(20, color='#f0883e', lw=0.6, ls='--', alpha=0.5)
        ax.axhline(40, color='#ff7b72', lw=0.6, ls='--', alpha=0.4)

        ax2 = ax.twinx()
        ax2.plot(t, df['ndi'], color='#ff7b72', lw=0.9, alpha=0.7, label='-DI')
        ax2.plot(t, df['pdi'], color='#3fb950', lw=0.9, alpha=0.7, label='+DI')
        ax2.fill_between(t, df['ndi'], df['pdi'],
                         where=df['ndi'] >= df['pdi'],
                         alpha=0.08, color='#ff7b72')
        ax2.set_ylabel('-DI/+DI', color='#8b949e', fontsize=7)
        ax2.tick_params(colors='#8b949e', labelsize=7)
        ax2.spines[:].set_color('#30363d')

        # Annotate NDI and ADX peaks in short session window
        sw = df[(df['time'].dt.hour >= 14) & (df['time'].dt.hour <= 18)]
        if not sw.empty:
            ndi_pr = sw.loc[sw['ndi'].idxmax()]
            adx_pr = sw.loc[sw['adx'].idxmax()]
            ax2.annotate(
                f"-DI peak\n{ndi_pr['time'].strftime('%H:%M')}\n{ndi_pr['ndi']:.1f}",
                xy=(ndi_pr['time'], ndi_pr['ndi']),
                xytext=(8, 4), textcoords='offset points',
                color='#ff7b72', fontsize=6,
                arrowprops=dict(arrowstyle='->', color='#ff7b72', lw=0.6)
            )
            ax.annotate(
                f"ADX peak\n{adx_pr['time'].strftime('%H:%M')}\n{adx_pr['adx']:.1f}",
                xy=(adx_pr['time'], adx_pr['adx']),
                xytext=(8, -18), textcoords='offset points',
                color=color, fontsize=6,
                arrowprops=dict(arrowstyle='->', color=color, lw=0.6)
            )

        ax.axvline(BOT_SHORT_ENTRY, color='#ff7b72', lw=1.0, ls='-', alpha=0.7)
        ax.axvline(BOT_LONG_ENTRY,  color='#3fb950', lw=0.8, ls='-', alpha=0.5)
        ax.set_title(lbl, color=color, fontsize=9, loc='left', pad=3, fontweight='bold')
        ax.set_ylabel('ADX', color='#8b949e', fontsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=4))
        ax.grid(axis='x', color='#30363d', lw=0.4, alpha=0.4)
        ax.set_ylim(0, max(df['adx'].max() * 1.25, 50))

    out = str(OUTPUT_DIR / f'8assets_adx_{yesterday}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"Chart saved: {out}")

# -----------------------------------------
# Chart 2: Short session zoom (14:30-18:00) -- NDI aligned overlay
# -----------------------------------------
def plot_short_window_focus(all_data, yesterday):
    fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
    fig.patch.set_facecolor('#0d1117')
    fig.suptitle(
        f'Short Session Zoom (14:30 - 18:00 UTC) | -DI and ADX Aligned Overlay\n'
        f'{yesterday} | Dots mark individual peaks',
        color='#e6edf3', fontsize=11, y=0.99
    )

    start = pd.Timestamp('2026-04-27 14:30', tz='UTC')
    end   = pd.Timestamp('2026-04-27 18:00', tz='UTC')

    for ax in axes:
        ax.set_facecolor('#161b22')
        ax.spines[:].set_color('#30363d')
        ax.tick_params(colors='#8b949e', labelsize=8)
        ax.axvline(BOT_SHORT_ENTRY, color='#ff7b72', lw=1.5, ls='-', alpha=0.9,
                   label='Bot short entry 16:08')
        ax.axvline(BOT_TIMEOUT_START, color='#ffa657', lw=1.0, ls='--', alpha=0.7,
                   label='Timeout exits begin 16:55')

    ax_ndi, ax_adx = axes

    for lbl, df in all_data.items():
        w = df[(df['time'] >= start) & (df['time'] <= end)]
        if w.empty:
            continue
        color = ASSET_COLORS.get(lbl, '#58a6ff')

        # NDI panel
        ax_ndi.plot(w['time'], w['ndi'], color=color, lw=1.3, label=lbl, alpha=0.85)
        ndi_peak_idx = w['ndi'].idxmax()
        ndi_pt = w.loc[ndi_peak_idx, 'time']
        ndi_pv = w.loc[ndi_peak_idx, 'ndi']
        ax_ndi.scatter([ndi_pt], [ndi_pv], color=color, s=50, zorder=5)
        ax_ndi.text(ndi_pt, ndi_pv + 0.5,
                    f"{lbl}\n{ndi_pt.strftime('%H:%M')}",
                    color=color, fontsize=6.5, ha='center', va='bottom')

        # ADX panel
        ax_adx.plot(w['time'], w['adx'], color=color, lw=1.3, label=lbl, alpha=0.85)
        adx_peak_idx = w['adx'].idxmax()
        adx_pt = w.loc[adx_peak_idx, 'time']
        adx_pv = w.loc[adx_peak_idx, 'adx']
        ax_adx.scatter([adx_pt], [adx_pv], color=color, s=50, zorder=5)
        ax_adx.text(adx_pt, adx_pv + 0.3,
                    f"{lbl}\n{adx_pt.strftime('%H:%M')}",
                    color=color, fontsize=6.5, ha='center', va='bottom')

    ax_ndi.axhline(5,  color='#8b949e', lw=0.7, ls=':', alpha=0.5, label='NDI=5 threshold')
    ax_adx.axhline(20, color='#f0883e', lw=0.7, ls='--', alpha=0.6, label='ADX=20')
    ax_adx.axhline(40, color='#ff7b72', lw=0.7, ls='--', alpha=0.4, label='ADX=40')

    ax_ndi.set_ylabel('-DI (bearish momentum)', color='#8b949e', fontsize=9)
    ax_ndi.set_title('-DI overlay  |  Dots = individual asset peaks  |  '
                     'Note: peaks scattered across different times',
                     color='#8b949e', fontsize=9, loc='left')
    ax_ndi.legend(loc='upper right', fontsize=7, facecolor='#21262d',
                  labelcolor='#8b949e', edgecolor='#30363d', ncol=4)
    ax_ndi.grid(axis='x', color='#30363d', lw=0.4, alpha=0.5)

    ax_adx.set_ylabel('ADX (trend strength)', color='#8b949e', fontsize=9)
    ax_adx.set_title('ADX overlay  |  ADX peaks lag -DI peaks significantly',
                     color='#8b949e', fontsize=9, loc='left')
    ax_adx.legend(loc='upper right', fontsize=7, facecolor='#21262d',
                  labelcolor='#8b949e', edgecolor='#30363d', ncol=4)
    ax_adx.grid(axis='x', color='#30363d', lw=0.4, alpha=0.5)

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
        ax.xaxis.set_major_locator(mdates.MinuteLocator(byminute=[0, 30]))
        ax.set_xlim(start, end)
    axes[-1].set_xlabel(f'Time (UTC)  {yesterday}', color='#8b949e', fontsize=9)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    out = str(OUTPUT_DIR / f'short_window_focus_{yesterday}.png')
    plt.savefig(out, dpi=150, bbox_inches='tight', facecolor='#0d1117')
    print(f"Chart saved: {out}")

# -----------------------------------------
# Export peak comparison CSV
# -----------------------------------------
def export_peak_csv(all_data, yesterday):
    rows = []
    for lbl, df in all_data.items():
        sw = df[(df['time'].dt.hour >= 14) & (df['time'].dt.hour <= 18)]
        if sw.empty:
            continue
        ndi_pr = sw.loc[sw['ndi'].idxmax()]
        adx_pr = sw.loc[sw['adx'].idxmax()]
        ndi_at_entry = df.loc[
            (df['time'] >= BOT_SHORT_ENTRY) &
            (df['time'] <= BOT_SHORT_ENTRY + pd.Timedelta('10min')), 'ndi'
        ].mean()
        decay_pct = (ndi_pr['ndi'] - ndi_at_entry) / ndi_pr['ndi'] * 100 \
                    if ndi_pr['ndi'] > 0 else 0
        rows.append({
            'asset':          lbl,
            'ndi_peak_time':  ndi_pr['time'].strftime('%H:%M'),
            'ndi_peak_value': round(ndi_pr['ndi'], 2),
            'ndi_at_entry':   round(ndi_at_entry, 2),
            'ndi_decay_pct':  round(decay_pct, 1),
            'ndi_lag_min':    round((BOT_SHORT_ENTRY - ndi_pr['time']).total_seconds()/60, 0),
            'adx_peak_time':  adx_pr['time'].strftime('%H:%M'),
            'adx_peak_value': round(adx_pr['adx'], 2),
            'adx_lag_min':    round((BOT_SHORT_ENTRY - adx_pr['time']).total_seconds()/60, 0),
        })
    out_df = pd.DataFrame(rows)
    out = str(OUTPUT_DIR / f'peak_analysis_{yesterday}.csv')
    out_df.to_csv(out, index=False)
    print(f"CSV saved: {out}")
    return out_df

# -----------------------------------------
# MAIN
# -----------------------------------------
if __name__ == '__main__':
    print("=" * 65)
    print("  8-Asset Regime Filter ADX Timeline Backtest")
    print("  Reconstructing prototype_long_short_v3.py")
    print("=" * 65)

    all_data, yesterday = fetch_all_assets()
    if not all_data:
        print("ERROR: Could not fetch data.")
        exit(1)

    print(f"\nFetched {len(all_data)} assets successfully.")
    print_peak_analysis(all_data)

    mean_df = compute_mean_series(all_data)

    print("Rendering charts...")
    plot_8assets(all_data, mean_df, yesterday)
    plot_short_window_focus(all_data, yesterday)

    print("\nExporting peak analysis CSV...")
    out_df = export_peak_csv(all_data, yesterday)
    print(out_df.to_string(index=False))

    print("\nDone. Output files:")
    print(f"  output/8assets_adx_{yesterday}.png")
    print(f"  output/short_window_focus_{yesterday}.png")
    print(f"  output/peak_analysis_{yesterday}.csv")
