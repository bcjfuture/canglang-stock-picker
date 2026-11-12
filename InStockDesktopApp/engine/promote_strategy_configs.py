from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
from datetime import datetime
from pathlib import Path


PROFILE_FILES = {
    "all_market": "all_market_config.json",
    "tech_attack": "tech_attack_config.json",
    "rebound_trend": "rebound_trend_config.json",
}
TRADE_GRID = {
    "holding_days": {5, 7, 10, 15, 20},
    "atr_multiplier": {1.0, 1.25, 1.5, 1.75, 2.0},
    "reward_risk": {1.5, 2.0, 2.5},
}
MARKET_GRID = {
    "breadth_floor": {0.0, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55},
    "momentum_floor": {-0.20, -0.05, -0.03, -0.01, 0.0, 0.01, 0.03},
    "drawdown_floor": {-0.60, -0.35, -0.30, -0.25, -0.20, -0.15},
    "strong_pick_count": {12, 15},
    "normal_pick_count": {5, 8, 10},
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validate_profile(name: str, item: dict) -> None:
    gate = item.get("gate", {})
    checks = gate.get("checks", [])
    policy_checks = gate.get("policy_checks", [])
    if not gate.get("passed") or len(checks) != 6 or not all(check.get("passed") for check in checks):
        raise ValueError(f"{name}: original risk gate failed")
    if len(policy_checks) < 5 or not all(check.get("passed") for check in policy_checks):
        raise ValueError(f"{name}: policy gate failed")
    if not item.get("neighbor_stability", {}).get("passed"):
        raise ValueError(f"{name}: neighbor stability failed")
    policy = item.get("policy") or {}
    market = item.get("market_policy") or {}
    for key, allowed in TRADE_GRID.items():
        if policy.get(key) not in allowed:
            raise ValueError(f"{name}: {key} outside approved grid")
    for key, allowed in MARKET_GRID.items():
        if market.get(key) not in allowed:
            raise ValueError(f"{name}: {key} outside approved grid")
    weights = item.get("chosen_weights")
    if not isinstance(weights, dict) or not weights:
        raise ValueError(f"{name}: missing chosen weights")


def promote_strategy_configs(evidence: dict, engine_dir: Path, manifest_path: Path) -> dict:
    profiles = evidence.get("profiles", {})
    if set(profiles) != set(PROFILE_FILES):
        raise ValueError("evidence must contain exactly all three strategies")
    for name in PROFILE_FILES:
        _validate_profile(name, profiles[name])

    engine_dir = Path(engine_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    backup_dir = engine_dir / "backups" / timestamp
    prepared: dict[str, tuple[Path, Path, Path, dict]] = {}
    for name, filename in PROFILE_FILES.items():
        target = engine_dir / filename
        if not target.exists():
            raise FileNotFoundError(target)
        current = json.loads(target.read_text(encoding="utf-8"))
        item = profiles[name]
        policy = dict(item["policy"])
        current.update(
            {
                "policy_version": policy["policy_version"],
                "strategy_gate_passed": True,
                "trade_policy": policy,
                "market_policy": dict(item["market_policy"]),
                "historical_factor_weights": dict(item["chosen_weights"]),
                "require_historical_factors": True,
                "promoted_at": datetime.now().isoformat(timespec="seconds"),
            }
        )
        temporary = target.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(current, ensure_ascii=False, indent=2), encoding="utf-8")
        json.loads(temporary.read_text(encoding="utf-8"))
        prepared[name] = (target, temporary, backup_dir / filename, current)

    backup_dir.mkdir(parents=True, exist_ok=False)
    for target, _, backup, _ in prepared.values():
        shutil.copy2(target, backup)
    replaced = []
    try:
        for name in PROFILE_FILES:
            target, temporary, _, _ = prepared[name]
            os.replace(temporary, target)
            replaced.append(name)
    except Exception:
        for name in replaced:
            target, _, backup, _ = prepared[name]
            shutil.copy2(backup, target)
        raise

    manifest = {
        "promoted_at": datetime.now().isoformat(timespec="seconds"),
        "backup_dir": str(backup_dir),
        "profiles": {
            name: {
                "policy_version": profiles[name]["policy"]["policy_version"],
                "config": str(prepared[name][0]),
                "sha256": _sha256(prepared[name][0]),
            }
            for name in PROFILE_FILES
        },
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary_manifest, manifest_path)
    return manifest


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Atomically promote three validated InStock strategies")
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--engine-dir", type=Path, default=Path(__file__).parent)
    parser.add_argument("--manifest", type=Path, required=True)
    args = parser.parse_args(argv)
    evidence = json.loads(args.evidence.read_text(encoding="utf-8"))
    manifest = promote_strategy_configs(evidence, args.engine_dir, args.manifest)
    print(f"PROMOTION_PASS={len(manifest['profiles'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
