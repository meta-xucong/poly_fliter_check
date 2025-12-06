"""
多领域批量盘口扫描与价格历史拉取工具。

功能概述：
- 通过 Gamma API 获取体育/电竞等领域的 tag，按领域筛选盘口。
- 支持配置时间范围（默认近 7 天），同时获取进行中与已结束盘口。
- 可按成交量/流动性过滤盘口，提取盘口元数据。
- 调用 CLOB prices-history 接口，为每个盘口的每个 outcome 拉取完整价格曲线。
- 结果可导出为 CSV，方便后续分析。

使用示例：
    python poly_domain_scan.py --domain sports_all --days 7 --min-volume 5000 \
        --fidelity 60 --output markets_history.csv

参数说明：
- domain：预设领域（sports_all 表示所有体育；esports 表示电竞；custom 需配合 --tags）。
- sports：当 domain=sports_all 时可用逗号分隔指定具体体育种类（NBA,CBB,Soccer 等）。
- tags：自定义 tag id/slug/label，逗号分隔，优先级最高。
- days：未显式传入开始/结束日期时，默认抓取近 N 天。
- start-date / end-date：YYYY-MM-DD 格式，覆盖 days 设置。
- min-volume：最小成交量（USD），过滤低流动性盘口；默认 0 不限制。
- fidelity：prices-history 请求的精度（分钟），数值越小说明采样越密集。
- output：输出 CSV 路径，默认 markets_history.csv。
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import requests

GAMMA_HOST = os.environ.get("GAMMA_HOST", "https://gamma-api.polymarket.com").rstrip("/")
CLOB_HOST = os.environ.get("CLOB_HOST", "https://clob.polymarket.com").rstrip("/")


@dataclass
class MarketRecord:
    event_slug: str
    event_question: str
    market_id: str
    market_slug: str
    market_question: str
    market_type: str
    outcomes: List[str]
    token_ids: List[str]
    created_at: Optional[str]
    closed_time: Optional[str]
    game_start_time: Optional[str]
    volume: float


@dataclass
class PricePoint:
    event_slug: str
    market_slug: str
    outcome: str
    timestamp: int
    price_raw: float
    price_prob: float


def _fetch_json(url: str, params: Optional[Dict] = None) -> Dict:
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"请求失败：{url} -> {exc}") from exc


def _parse_date(text: str) -> dt.datetime:
    return dt.datetime.strptime(text, "%Y-%m-%d")


def _to_unix(ts: str) -> Optional[int]:
    if not ts:
        return None
    try:
        return int(dt.datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp())
    except ValueError:
        return None


def _parse_token_ids(raw: str | Sequence[str] | None) -> List[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except json.JSONDecodeError:
            pass
        return [raw]
    return [str(x) for x in raw]


class DomainResolver:
    GENERIC_SPORT_TAGS = {1, 100639}

    def __init__(self, esports_tag_hints: Optional[List[str]] = None) -> None:
        self.esports_tag_hints = esports_tag_hints or []
        self._sports_tags: Dict[str, List[int]] = {}
        self._tag_lookup: Dict[str, int] = {}

    def load_metadata(self) -> None:
        sports = _fetch_json(f"{GAMMA_HOST}/sports") or []
        for item in sports:
            sport_name = str(item.get("sport") or "").lower()
            tag_ids = []
            tags_raw = item.get("tags")
            if isinstance(tags_raw, str):
                tag_ids = [int(t) for t in tags_raw.split(",") if t]
            elif isinstance(tags_raw, list):
                tag_ids = [int(t) for t in tags_raw if t is not None]
            if sport_name:
                self._sports_tags[sport_name] = tag_ids

        tags_resp = _fetch_json(f"{GAMMA_HOST}/tags") or []
        tags_list: Iterable[Dict] = tags_resp if isinstance(tags_resp, list) else tags_resp.get("tags", [])
        for tag in tags_list:
            tag_id = tag.get("id")
            label = str(tag.get("label") or tag.get("slug") or tag.get("name") or "").lower()
            if tag_id is not None and label:
                self._tag_lookup[label] = int(tag_id)

    def resolve(self, domain: str, sports: Optional[List[str]] = None, custom_tags: Optional[List[str]] = None) -> List[int]:
        domain = domain.lower()
        sports = [s.lower() for s in sports or []]
        custom_tags = [t.lower() for t in custom_tags or []]

        tag_ids: List[int] = []
        if domain == "custom" and custom_tags:
            for tag in custom_tags:
                if tag.isdigit():
                    tag_ids.append(int(tag))
                elif tag in self._tag_lookup:
                    tag_ids.append(self._tag_lookup[tag])
        elif domain == "esports":
            for hint in self.esports_tag_hints:
                if hint.isdigit():
                    tag_ids.append(int(hint))
                elif hint.lower() in self._tag_lookup:
                    tag_ids.append(self._tag_lookup[hint.lower()])
        else:
            # sports_all 或者指定体育种类
            if sports:
                for sp in sports:
                    ids = self._sports_tags.get(sp, [])
                    filtered = [t for t in ids if t not in self.GENERIC_SPORT_TAGS]
                    tag_ids.extend(filtered)
            else:
                for ids in self._sports_tags.values():
                    tag_ids.extend(ids)
        # 去重
        return sorted(set(tag_ids))


def fetch_markets(
    tag_ids: List[int],
    start_date: dt.datetime,
    end_date: dt.datetime,
    sports_market_types: Optional[List[str]] = None,
) -> List[Dict]:
    if not tag_ids:
        return []
    markets: List[Dict] = []
    params_base = {
        "start_date_min": start_date.strftime("%Y-%m-%d"),
        "start_date_max": end_date.strftime("%Y-%m-%d"),
    }
    if sports_market_types:
        params_base["sports_market_types"] = ",".join(sports_market_types)

    for tag in tag_ids:
        offset = 0
        while True:
            params = dict(params_base)
            params.update({"tag_id": tag, "limit": 100, "offset": offset})
            data = _fetch_json(f"{GAMMA_HOST}/markets", params=params)
            batch: Iterable[Dict] = data if isinstance(data, list) else data.get("markets", [])
            batch_list = list(batch)
            markets.extend(batch_list)
            if len(batch_list) < 100:
                break
            offset += 100
    return markets


def build_market_records(markets: List[Dict], min_volume: float) -> List[MarketRecord]:
    records: List[MarketRecord] = []
    for mk in markets:
        if not mk.get("enableOrderBook", True):
            continue
        volume = float(mk.get("volume") or mk.get("volume24hr") or 0)
        if volume < min_volume:
            continue
        outcomes = mk.get("outcomes") or mk.get("shortOutcomes") or []
        token_ids = _parse_token_ids(mk.get("clobTokenIds"))
        records.append(
            MarketRecord(
                event_slug=str(mk.get("eventSlug") or mk.get("event_slug") or ""),
                event_question=str(mk.get("eventQuestion") or mk.get("eventTitle") or mk.get("event") or ""),
                market_id=str(mk.get("id") or mk.get("_id") or ""),
                market_slug=str(mk.get("slug") or ""),
                market_question=str(mk.get("question") or mk.get("title") or ""),
                market_type=str(mk.get("sportsMarketType") or mk.get("marketType") or mk.get("type") or ""),
                outcomes=[str(o) for o in outcomes],
                token_ids=token_ids,
                created_at=mk.get("createdAt"),
                closed_time=mk.get("closedTime") or mk.get("endDate"),
                game_start_time=mk.get("gameStartTime"),
                volume=volume,
            )
        )
    return records


def fetch_price_history(token_id: str, start_ts: int, end_ts: int, fidelity: int) -> List[Tuple[int, float]]:
    params = {"market": token_id, "startTs": start_ts, "endTs": end_ts, "fidelity": fidelity}
    data = _fetch_json(f"{CLOB_HOST}/prices-history", params=params)
    history = data.get("history") or []
    result: List[Tuple[int, float]] = []
    for point in history:
        t = point.get("t")
        p = point.get("p")
        if t is None or p is None:
            continue
        try:
            result.append((int(t), float(p)))
        except (TypeError, ValueError):
            continue
    return result


def collect_price_points(record: MarketRecord, fidelity: int) -> List[PricePoint]:
    now_ts = int(time.time())
    start_ts = min(_to_unix(record.created_at) or now_ts - 7 * 86400, now_ts)
    end_ts = min(_to_unix(record.closed_time) or now_ts, now_ts)
    if end_ts <= start_ts:
        return []
    points: List[PricePoint] = []
    for outcome, token_id in zip(record.outcomes, record.token_ids):
        history = fetch_price_history(token_id, start_ts, end_ts, fidelity)
        for t, p in history:
            price_prob = p / 10000 if p > 1 else p / 100
            points.append(
                PricePoint(
                    event_slug=record.event_slug,
                    market_slug=record.market_slug,
                    outcome=outcome,
                    timestamp=t,
                    price_raw=p,
                    price_prob=price_prob,
                )
            )
        time.sleep(0.1)
    return points


def export_csv(records: List[MarketRecord], prices: List[PricePoint], path: str) -> Tuple[str, str]:
    root, ext = os.path.splitext(path)
    ext = ext or ".csv"
    markets_path = path
    prices_path = f"{root}_prices{ext}"

    with open(markets_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "event_slug",
                "event_question",
                "market_id",
                "market_slug",
                "market_question",
                "market_type",
                "outcomes",
                "token_ids",
                "created_at",
                "closed_time",
                "game_start_time",
                "volume",
            ]
        )
        for rec in records:
            writer.writerow(
                [
                    rec.event_slug,
                    rec.event_question,
                    rec.market_id,
                    rec.market_slug,
                    rec.market_question,
                    rec.market_type,
                    "|".join(rec.outcomes),
                    "|".join(rec.token_ids),
                    rec.created_at or "",
                    rec.closed_time or "",
                    rec.game_start_time or "",
                    rec.volume,
                ]
            )

    with open(prices_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["event_slug", "market_slug", "outcome", "timestamp", "price_raw", "price_prob"])
        for pt in prices:
            writer.writerow(
                [pt.event_slug, pt.market_slug, pt.outcome, pt.timestamp, pt.price_raw, pt.price_prob]
            )

    return markets_path, prices_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="按领域批量抓取 Polymarket 盘口和价格历史")
    parser.add_argument("--domain", default="sports_all", help="sports_all / esports / custom")
    parser.add_argument("--sports", help="限定体育种类，逗号分隔，例如 NBA,Soccer")
    parser.add_argument("--tags", help="自定义 tag id/slug/label，逗号分隔，仅 domain=custom 生效")
    parser.add_argument("--days", type=int, default=7, help="未指定日期范围时，默认抓取近 N 天")
    parser.add_argument("--start-date", dest="start_date", help="开始日期 YYYY-MM-DD")
    parser.add_argument("--end-date", dest="end_date", help="结束日期 YYYY-MM-DD")
    parser.add_argument("--min-volume", type=float, default=0.0, help="最小成交量 USD，过滤盘口")
    parser.add_argument("--fidelity", type=int, default=60, help="prices-history 的采样精度（分钟）")
    parser.add_argument("--output", default="markets_history.csv", help="输出 CSV 路径")
    parser.add_argument(
        "--sports-market-types",
        dest="sports_market_types",
        help="限定体育盘口类型，逗号分隔，例如 moneyline,spread,totals",
    )
    parser.add_argument(
        "--esports-tag-hints",
        dest="esports_tag_hints",
        help="电竞标签线索（id/slug/label），逗号分隔，用于 domain=esports",
    )
    return parser.parse_args()


def determine_date_range(args: argparse.Namespace) -> Tuple[dt.datetime, dt.datetime]:
    if args.start_date and args.end_date:
        start = _parse_date(args.start_date)
        end = _parse_date(args.end_date)
    else:
        end = dt.datetime.now(dt.UTC)
        start = end - dt.timedelta(days=args.days)
    return start, end


def main() -> None:
    args = parse_args()
    start_date, end_date = determine_date_range(args)

    resolver = DomainResolver(
        esports_tag_hints=args.esports_tag_hints.split(",") if args.esports_tag_hints else None
    )
    resolver.load_metadata()
    sports_list = args.sports.split(",") if args.sports else None
    custom_tags = args.tags.split(",") if args.tags else None
    tag_ids = resolver.resolve(args.domain, sports=sports_list, custom_tags=custom_tags)
    if not tag_ids:
        raise SystemExit("未能解析到有效的 tag id，请检查 domain/sports/tags 设置。")

    sports_market_types = (
        [t.strip() for t in args.sports_market_types.split(",") if t.strip()] if args.sports_market_types else None
    )
    markets_raw = fetch_markets(tag_ids, start_date, end_date, sports_market_types=sports_market_types)
    records = build_market_records(markets_raw, min_volume=args.min_volume)
    print(f"共获取盘口 {len(records)} 个，准备拉取价格历史…")

    price_points: List[PricePoint] = []
    for rec in records:
        price_points.extend(collect_price_points(rec, fidelity=args.fidelity))

    markets_path, prices_path = export_csv(records, price_points, args.output)
    print(f"完成，盘口信息写入 {markets_path}，价格历史写入 {prices_path}")


if __name__ == "__main__":
    main()
