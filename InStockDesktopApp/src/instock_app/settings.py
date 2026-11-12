from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from .contracts import StrategySpec


@dataclass(frozen=True)
class AppPaths:
    install_root: Path
    app_dir: Path
    workers_dir: Path
    engine_dir: Path
    data_dir: Path
    logs_dir: Path
    reports_dir: Path
    instock_root: Path
    python_exe: Path
    mariadb_exe: Path
    mariadb_data_dir: Path

    @classmethod
    def default(cls) -> "AppPaths":
        root = Path(r"C:\InStockLocal\InStockDesktopApp")
        return cls(
            install_root=root,
            app_dir=root / "app",
            workers_dir=root / "workers",
            engine_dir=root / "engine",
            data_dir=root / "data",
            logs_dir=Path(r"C:\InStockLocal\logs\desktop-app"),
            reports_dir=Path(r"C:\InStockLocal\reports"),
            instock_root=Path(r"C:\InStockLocal\InStock"),
            python_exe=Path(r"C:\InStockLocal\venv\Scripts\python.exe"),
            mariadb_exe=Path(r"C:\Program Files\MariaDB 12.3\bin\mariadbd.exe"),
            mariadb_data_dir=Path(r"C:\InStockLocal\db"),
        )


BASE_STRATEGIES: dict[str, StrategySpec] = {
    "all_market": StrategySpec(
        "all_market", "全市场精选", "all_market_config.json", "all_market.json", "全市场Top"
    ),
    "tech_attack": StrategySpec(
        "tech_attack", "科技进攻", "tech_attack_config.json", "tech_attack.json", "科技进攻Top"
    ),
    "rebound_trend": StrategySpec(
        "rebound_trend", "反弹趋势", "rebound_trend_config.json", "rebound_trend.json", "反弹趋势Top"
    ),
    "zhengxi_prosperity": StrategySpec(
        "zhengxi_prosperity",
        "郑希景气精选",
        "zhengxi_prosperity_config.json",
        "zhengxi_prosperity.json",
        "郑希景气Top",
    ),
}

HIGH_PRECISION_STRATEGY = StrategySpec(
    "high_precision",
    "高胜率精选",
    "high_precision_config.json",
    "high_precision.json",
    "高胜率精选",
)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def build_strategy_registry(
    engine_dir: Path, data_dir: Path
) -> tuple[dict[str, StrategySpec], str]:
    del engine_dir, data_dir
    return dict(BASE_STRATEGIES), "all_market"


STRATEGIES, DEFAULT_STRATEGY_KEY = build_strategy_registry(
    Path(r"C:\InStockLocal\InStockDesktopApp\engine"),
    Path(r"C:\InStockLocal\InStockDesktopApp\data"),
)

SYSTEM_ACTIONS = ("start", "stop", "update")
