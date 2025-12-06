"""拉取指定 domain 下市场的价格历史并导出 CSV。"""

from __future__ import annotations

import argparse
import csv
import os
import re
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests

# Set up the Gamma API and CLOB API hosts
GAMMA_HOST = os.environ.get("GAMMA_HOST", "https://gamma-api.polymarket.com").rstrip("/")
CLOB_HOST = os.environ.get("CLOB_HOST", "https://clob.polymarket.com").rstrip("/")


@dataclass
class MarketRecord:
    market_slug: str
    outcomes: List[str]
    token_ids: List[str]


@dataclass
class PricePoint:
    market_slug: str
    outcome: str
    timestamp: int
    price_raw: float
    price_prob: float


def fetch_json(url: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """简单的 GET 请求封装，附带清晰的错误信息。"""

    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:  # pragma: no cover - 网络依赖
        raise RuntimeError(f"Failed to fetch data from {url} -> {exc}") from exc


def _parse_timestamp(raw: Any) -> Optional[int]:
    """尽可能将 API 字段转换为 epoch 秒数。"""

    if raw is None:
        return None
    if isinstance(raw, (int, float)):
        return int(raw)
    if isinstance(raw, str):
        raw = raw.strip()
        if raw.isdigit():
            return int(raw)
        # 尝试解析 ISO 时间
        try:
            from datetime import datetime

            if raw.endswith("Z"):
                raw = raw[:-1] + "+00:00"
            return int(datetime.fromisoformat(raw).timestamp())
        except Exception:
            return None
    return None


def _normalize_outcomes(raw_outcomes: Any) -> List[str]:
    if isinstance(raw_outcomes, list):
        return [str(o).strip() for o in raw_outcomes if str(o).strip()]
    if isinstance(raw_outcomes, str):
        return [o.strip() for o in re.split(r"[|,]", raw_outcomes) if o.strip()]
    return []


def _extract_token_ids(market: Dict[str, Any]) -> List[str]:
    tokens: List[str] = []
    for key in ("tokens", "outcomeTokens", "contracts"):
        token_list = market.get(key)
        if not isinstance(token_list, Iterable):
            continue
        for token in token_list:
            if isinstance(token, dict):
                for field in ("tokenId", "id", "marketId", "conditionId"):
                    value = token.get(field)
                    if value:
                        tokens.append(str(value))
                        break
            elif token:
                tokens.append(str(token))
    # 去重，保持顺序
    seen = set()
    unique_tokens = []
    for tok in tokens:
        if tok not in seen:
            seen.add(tok)
            unique_tokens.append(tok)
    return unique_tokens


def _match_sports(market: Dict[str, Any], sports: Optional[List[str]]) -> bool:
    if not sports:
        return True

    def _as_lower_set(values: Any) -> List[str]:
        items: List[str] = []
        if isinstance(values, dict):
            values = values.values()
        if isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
            for v in values:
                if v is None:
                    continue
                items.append(str(v).lower())
        return items

    tokens: List[str] = []
    for key in ("tags", "categories", "sports", "group"):
        tokens.extend(_as_lower_set(market.get(key)))

    slug = str(market.get("slug") or "").lower()
    question = str(market.get("question") or market.get("title") or "").lower()
    tokens.append(slug)
    tokens.append(question)

    sports_lower = {s.lower() for s in sports}
    return any(any(s in token for token in tokens) for s in sports_lower)


def _market_volume(market: Dict[str, Any]) -> float:
    for key in ("volume24h", "volume", "totalVolume"):
        value = market.get(key)
        if isinstance(value, (int, float)):
            return float(value)
        if isinstance(value, str):
            try:
                return float(value)
            except ValueError:
                continue
    return 0.0


def fetch_markets(domain: str, sports: Optional[List[str]], days: int, min_volume: float) -> List[MarketRecord]:
    """分页抓取 Gamma markets，并按参数进行过滤。"""

    records: List[MarketRecord] = []
    page = 1
    start_ts = int(time.time()) - days * 86400

    while True:
        params = {
            "domain": domain,
            "page": page,
            "limit": 1000,
            "closed": False,
        }
        data = fetch_json(f"{GAMMA_HOST}/markets", params=params)
        markets: Iterable[Dict[str, Any]] = data if isinstance(data, list) else data.get("markets", [])
        market_list = list(markets)
        if not market_list:
            break

        for market in market_list:
            if not _match_sports(market, sports):
                continue

            created_ts = _parse_timestamp(
                market.get("createdAt")
                or market.get("openDate")
                or market.get("startDate")
            )
            if created_ts and created_ts < start_ts:
                continue

            if _market_volume(market) < min_volume:
                continue

            outcomes = _normalize_outcomes(
                market.get("outcomes")
                or market.get("outcomeNames")
                or market.get("labels")
            )
            token_ids = _extract_token_ids(market)
            if not outcomes or not token_ids:
                continue

            slug = market.get("slug") or market.get("url") or market.get("question")
            if not slug:
                continue

            records.append(MarketRecord(slug, outcomes, token_ids))

        if len(market_list) < 1000:
            break
        page += 1

    return records


def fetch_price_history(token_id: str, start_ts: int, end_ts: int, fidelity: int) -> List[Tuple[int, float]]:
    """获取某个 token 在指定时间区间的价格历史。"""

    params = {
        "market": token_id,
        "tokenId": token_id,
        "startTs": start_ts,
        "endTs": end_ts,
        "fidelity": fidelity,
    }
    data = fetch_json(f"{CLOB_HOST}/prices-history", params=params)
    history = data.get("history", []) if isinstance(data, dict) else []
    points: List[Tuple[int, float]] = []
    for point in history:
        if not isinstance(point, dict):
            continue
        if "t" not in point or "p" not in point:
            continue
        price_raw = point["p"]
        price_prob = price_raw / 10000 if price_raw > 1 else price_raw / 100
        points.append((int(point["t"]), float(price_prob)))
    return points


def collect_price_points(record: MarketRecord, fidelity: int, start_ts: int, end_ts: int) -> List[PricePoint]:
    price_points: List[PricePoint] = []
    for outcome, token_id in zip(record.outcomes, record.token_ids):
        history = fetch_price_history(token_id, start_ts, end_ts, fidelity)
        for timestamp, price_prob in history:
            price_points.append(
                PricePoint(
                    market_slug=record.market_slug,
                    outcome=outcome,
                    timestamp=timestamp,
                    price_raw=price_prob * 10000,
                    price_prob=price_prob,
                )
            )
    return price_points


def export_csv(price_points: List[PricePoint], output_path: str) -> None:
    fieldnames = ["market_slug", "timestamp", "outcome", "price_raw", "price_prob"]
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for point in price_points:
            writer.writerow(
                [
                    point.market_slug,
                    point.timestamp,
                    point.outcome,
                    f"{point.price_raw:.4f}",
                    f"{point.price_prob:.4f}",
                ]
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Polymarket domain scanner")
    parser.add_argument("--domain", required=True, help="Gamma domain 参数，例如 sports_all")
    parser.add_argument("--sports", help="按体育名称过滤，多个值用逗号分隔，如 NBA,NFL", default=None)
    parser.add_argument("--days", type=int, default=1, help="只保留最近 N 天创建的市场")
    parser.add_argument("--min-volume", dest="min_volume", type=float, default=0.0, help="过滤 volume24h/totalVolume")
    parser.add_argument("--fidelity", type=int, default=60, help="价格历史的粒度（秒）")
    parser.add_argument("--output", required=True, help="输出 CSV 路径")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    sports = [s.strip() for s in args.sports.split(",") if s.strip()] if args.sports else None
    end_ts = int(time.time())
    start_ts = end_ts - args.days * 86400

    records = fetch_markets(args.domain, sports, args.days, args.min_volume)
    if not records:
        print("[WARN] 未找到满足条件的市场，输出为空。")
        return

    price_points: List[PricePoint] = []
    for record in records:
        price_points.extend(collect_price_points(record, args.fidelity, start_ts, end_ts))

    export_csv(price_points, args.output)
    print(f"写入 {len(price_points)} 条价格数据到 {args.output}")


if __name__ == "__main__":
    main()
