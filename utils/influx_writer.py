import os
import logging
from influxdb_client import InfluxDBClient, Point
from influxdb_client.client.write_api import SYNCHRONOUS

logger = logging.getLogger('AlgoTrade_LongShort_V3')

INFLUX_URL    = os.getenv('INFLUX_URL',    'http://localhost:8086')
INFLUX_TOKEN  = os.getenv('INFLUX_TOKEN',  'mytoken123456')
INFLUX_ORG    = os.getenv('INFLUX_ORG',    'algo')
INFLUX_BUCKET = os.getenv('INFLUX_BUCKET', 'trading')
INFLUX_ENABLED = os.getenv('INFLUX_ENABLED', 'true').lower() == 'true'

_client = None
_write_api = None

if INFLUX_ENABLED:
    try:
        _client = InfluxDBClient(url=INFLUX_URL, token=INFLUX_TOKEN, org=INFLUX_ORG)
        _write_api = _client.write_api(write_options=SYNCHRONOUS)
        logger.info(f"✅ InfluxDB 已連線: {INFLUX_URL} bucket={INFLUX_BUCKET}")
    except Exception as e:
        INFLUX_ENABLED = False
        logger.warning(f"⚠️ InfluxDB 初始化失敗，停用: {e}")


def _write(point: Point):
    """安全寫入 InfluxDB，失敗時靜默記錄 warning。"""
    if not INFLUX_ENABLED or _write_api is None:
        return
    try:
        _write_api.write(bucket=INFLUX_BUCKET, record=point)
    except Exception as e:
        logger.warning(f"⚠️ InfluxDB write 失敗: {e}")


# ──────────────────────────────────────────────────────────
# Measurement: market_regime
#   Grafana 用途：顯示 BTC Regime 信號、ADX、Score 趨勢
# ──────────────────────────────────────────────────────────
def write_regime(regime_signal: int, mean_adx: float, market_score: float,
                 adx_mei: float = 0.0, is_ranging: bool = False,
                 adx_dynamic_floor: float = 20.0,
                 brake: bool = False, soft_brake: bool = False,
                 sim_mode: bool = False):
    point = (
        Point("market_regime")
        .tag("mode", "SIM" if sim_mode else "LIVE")
        .field("regime_signal",    int(regime_signal))
        .field("mean_adx",         float(mean_adx))
        .field("market_score",     float(market_score))
        .field("adx_mei",          float(adx_mei))
        .field("is_ranging",       int(is_ranging))
        .field("adx_dynamic_floor", float(adx_dynamic_floor))
        .field("brake",            int(brake))
        .field("soft_brake",       int(soft_brake))
    )
    _write(point)


# ──────────────────────────────────────────────────────────
# Measurement: trade_event
#   Grafana 用途：標記每筆開平倉事件，含 PnL / TP / SL
# ──────────────────────────────────────────────────────────
def write_trade(symbol: str, action: str, price: float, amount: float,
                atr: float = 0.0, net_flow: float = 0.0,
                tp_price: float = 0.0, sl_price: float = 0.0,
                realized_pnl: float = 0.0,
                regime_signal: int = 0, mean_adx: float = 0.0,
                market_score: float = 0.0, sim_mode: bool = False):
    point = (
        Point("trade_event")
        .tag("symbol",  symbol)
        .tag("action",  action)                        # LONG_ENTRY / SHORT_ENTRY / LONG_EXIT / SHORT_EXIT / NATIVE_EXIT
        .tag("mode",    "SIM" if sim_mode else "LIVE")
        .field("price",          float(price))
        .field("amount",         float(amount))
        .field("trade_value",    float(price * amount))
        .field("atr",            float(atr))
        .field("net_flow",       float(net_flow))
        .field("tp_price",       float(tp_price))
        .field("sl_price",       float(sl_price))
        .field("realized_pnl",   float(realized_pnl))
        .field("regime_signal",  int(regime_signal))
        .field("mean_adx",       float(mean_adx))
        .field("market_score",   float(market_score))
    )
    _write(point)


# ──────────────────────────────────────────────────────────
# Measurement: account_balance
#   Grafana 用途：資金曲線、淨值追蹤
# ──────────────────────────────────────────────────────────
def write_balance(balance: float, equity: float = 0.0,
                  total_pnl: float = 0.0, sim_mode: bool = False):
    point = (
        Point("account_balance")
        .tag("mode", "SIM" if sim_mode else "LIVE")
        .field("balance",   float(balance))
        .field("equity",    float(equity) if equity else float(balance))
        .field("total_pnl", float(total_pnl))
    )
    _write(point)


# ──────────────────────────────────────────────────────────
# Measurement: position_status
#   Grafana 用途：持倉未實現盈虧 / Max Drawdown 監控
# ──────────────────────────────────────────────────────────
def write_position(symbol: str, side: str, entry_price: float,
                   current_price: float, pnl_pct: float,
                   unrealized_pnl: float = 0.0, time_held_secs: float = 0.0,
                   sim_mode: bool = False):
    point = (
        Point("position_status")
        .tag("symbol",  symbol)
        .tag("side",    side)          # "long" / "short"
        .tag("mode",    "SIM" if sim_mode else "LIVE")
        .field("entry_price",     float(entry_price))
        .field("current_price",   float(current_price))
        .field("pnl_pct",         float(pnl_pct))
        .field("unrealized_pnl",  float(unrealized_pnl))
        .field("time_held_mins",  float(time_held_secs / 60.0))
    )
    _write(point)


# ──────────────────────────────────────────────────────────
# Legacy – kept for backward compatibility
# ──────────────────────────────────────────────────────────
def write_signal(pair: str, z_score: float, adx: float,
                 signal: str, decision: str,
                 bear_votes: int = 0, bull_votes: int = 0):
    point = (
        Point("trading_signal")
        .tag("pair", pair)
        .field("z_score",    float(z_score))
        .field("adx",        float(adx))
        .field("bear_votes", int(bear_votes))
        .field("bull_votes", int(bull_votes))
        .field("signal",     signal)
        .field("decision",   decision)
    )
    _write(point)
