"""輕量、可擴充的台股資料與回測套件。"""

from .backtest import BacktestConfig, BacktestResult, SwingConfig, run_dca, run_smart_dip, run_swing
from .providers import PriceRequest, ProviderRouter, build_default_router

__all__ = [
    "BacktestConfig", "BacktestResult", "SwingConfig", "PriceRequest", "ProviderRouter",
    "build_default_router", "run_dca", "run_smart_dip", "run_swing",
]
