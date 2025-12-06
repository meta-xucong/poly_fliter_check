"""
工具脚本：校验 markets_history.csv 与 markets_history_prices.csv 的一致性，
用来核实每个盘口是否包含两个 outcome 的价格序列。

使用方式：
    python validate_polymarket_data.py

输出内容：
- 市场数量统计
- 价格历史中 outcome 数量不是 2 的盘口列表
- 价格历史与元数据 outcomes 不匹配的盘口列表
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Set

DATA_DIR = Path(__file__).resolve().parent
MARKETS_FILE = DATA_DIR / "markets_history.csv"
PRICES_FILE = DATA_DIR / "markets_history_prices.csv"


def load_market_outcomes(csv_path: Path) -> Dict[str, Set[str]]:
    """读取市场元数据，返回 market_slug -> outcome 名称集合。"""
    outcomes: Dict[str, Set[str]] = {}
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            slug = row.get("market_slug") or ""
            outcome_str = row.get("outcomes") or ""
            outcome_set = {part.strip() for part in outcome_str.split("|") if part.strip()}
            if slug:
                outcomes[slug] = outcome_set
    return outcomes


def load_price_outcomes(csv_path: Path) -> Dict[str, Set[str]]:
    """读取价格历史，返回 market_slug -> 价格中出现的 outcome 名称集合。"""
    price_outcomes: Dict[str, Set[str]] = defaultdict(set)
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            slug = row.get("market_slug") or ""
            outcome = row.get("outcome") or ""
            if slug and outcome:
                price_outcomes[slug].add(outcome)
    return price_outcomes


def validate():
    market_outcomes = load_market_outcomes(MARKETS_FILE)
    price_outcomes = load_price_outcomes(PRICES_FILE)

    print(f"市场数量（元数据）：{len(market_outcomes)}")
    print(f"市场数量（价格历史）：{len(price_outcomes)}\n")

    # 检查价格历史中 outcome 数量是否为 2
    bad_counts: List[str] = [slug for slug, outs in price_outcomes.items() if len(outs) != 2]
    if bad_counts:
        print("⚠️ 以下盘口在价格历史中 outcome 数量不是 2：")
        for slug in bad_counts:
            print(f"  - {slug}: {sorted(price_outcomes[slug])}")
    else:
        print("✅ 价格历史中的所有盘口均包含 2 个 outcome。")

    # 检查价格历史中的 outcome 名称是否与市场元数据一致
    mismatch: List[str] = []
    for slug, outs in price_outcomes.items():
        meta_outs = market_outcomes.get(slug, set())
        if meta_outs and outs != meta_outs:
            mismatch.append(slug)
    if mismatch:
        print("\n⚠️ 以下盘口的 outcome 名称与 markets_history.csv 不一致：")
        for slug in mismatch:
            print(f"  - {slug}: metadata={sorted(market_outcomes.get(slug, []))}, prices={sorted(price_outcomes.get(slug, []))}")
    else:
        print("\n✅ 价格历史中的 outcome 名称与 markets_history.csv 一致。")

    # 统计价格历史缺失的盘口
    missing_in_price = set(market_outcomes) - set(price_outcomes)
    if missing_in_price:
        print("\n⚠️ 以下盘口在价格历史中缺失：")
        for slug in sorted(missing_in_price):
            print(f"  - {slug}")
    else:
        print("\n✅ 所有 markets_history.csv 中的盘口均在价格历史中找到记录。")


if __name__ == "__main__":
    validate()
