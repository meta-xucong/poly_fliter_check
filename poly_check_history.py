"""
poly_check_history.py

交互式小工具：根据 Polymarket 市场链接获取对应事件，列出子市场并筛选相似历史盘，汇总历史结果。

核心流程：
1. 读取用户输入的 URL，解析 slug。
2. 通过 Gamma API 查询事件及其子市场，允许用户选择具体市场。
3. 基于 slug 关键词搜索其他已结算的历史事件，收集同类市场的结果。
4. 输出每个历史盘的赢家摘要，并生成汇总文案。
"""
from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional
from urllib.parse import urlparse

import requests

GAMMA_HOST = os.environ.get("GAMMA_HOST", "https://gamma-api.polymarket.com").rstrip("/")


@dataclass
class MarketResult:
    market_id: str
    question: str
    slug: str
    resolved: bool
    winning_outcome: Optional[str]
    close_time: Optional[str]


@dataclass
class EventData:
    event_id: str
    slug: str
    question: str
    markets: List[Dict]
    series_slug: Optional[str] = None
    tags: Optional[List] = None


def _fetch_json(url: str, params: Optional[Dict] = None) -> Dict:
    """简化的 GET 请求包装，抛出更友好的错误。"""
    try:
        resp = requests.get(url, params=params, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"请求失败：{url} -> {exc}") from exc


def extract_slug(raw_url: str) -> str:
    parsed = urlparse(raw_url.strip())
    path = parsed.path.rstrip("/")
    slug = path.split("/")[-1]
    if not slug:
        raise ValueError("无法从链接解析出 slug，请检查输入。")
    return slug


def fetch_event_detail(slug: str) -> Optional[EventData]:
    """优先使用 /events/slug 取到完整事件信息和盘口。"""
    try:
        data = _fetch_json(f"{GAMMA_HOST}/events/slug/{slug}")
    except RuntimeError:
        return None

    event_slug = data.get("slug") or slug
    event_id = str(data.get("id") or data.get("eventId") or data.get("_id") or "")
    question = data.get("question") or data.get("title") or event_slug
    markets = data.get("markets") or []
    return EventData(
        event_id=event_id or event_slug,
        slug=event_slug,
        question=question,
        markets=markets,
        series_slug=data.get("seriesSlug"),
        tags=data.get("tags"),
    )


def fetch_events_by_slug(slug: str) -> List[EventData]:
    """尽可能宽松地按 slug 搜索事件，优先包含 slug 直取的详情。"""
    search_params = [
        {"slug": slug},
        {"search": slug},
        {"search": slug, "closed": True},
    ]
    seen: Dict[str, EventData] = {}

    detail = fetch_event_detail(slug)
    if detail:
        key = detail.event_id or detail.slug
        seen[key] = detail

    for params in search_params:
        data = _fetch_json(f"{GAMMA_HOST}/events", params=params)
        events: Iterable[Dict] = data if isinstance(data, list) else data.get("events", [])
        for item in events:
            event_slug = item.get("slug") or item.get("url") or ""
            event_id = str(item.get("id") or item.get("eventId") or item.get("_id") or "")
            question = item.get("question") or item.get("title") or event_slug or slug
            markets = item.get("markets") or []
            key = event_id or event_slug
            if key and key not in seen:
                seen[key] = EventData(
                    event_id=event_id or key,
                    slug=event_slug or slug,
                    question=question,
                    markets=markets,
                    series_slug=item.get("seriesSlug"),
                    tags=item.get("tags"),
                )
    return list(seen.values())


def pick_event(events: List[EventData]) -> EventData:
    if len(events) == 1:
        return events[0]
    print("检测到多个事件，请输入序号选择：")
    for idx, ev in enumerate(events, start=1):
        print(f"  [{idx}] {ev.slug} | {ev.question}")
    while True:
        choice = input("事件序号：").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(events):
            return events[int(choice) - 1]
        print("无效序号，请重新输入。")


def pick_market(markets: List[Dict]) -> Dict:
    if len(markets) == 1:
        return markets[0]
    print("该事件包含多个盘口，请输入序号选择：")
    for idx, mk in enumerate(markets, start=1):
        title = mk.get("question") or mk.get("title") or mk.get("slug") or "(未命名盘口)"
        status = "已结算" if mk.get("closed") or mk.get("resolved") else "未结算"
        print(f"  [{idx}] {title} | {status}")
    while True:
        choice = input("盘口序号：").strip()
        if choice.isdigit() and 1 <= int(choice) <= len(markets):
            return markets[int(choice) - 1]
        print("无效序号，请重新输入。")


def derive_keywords(slug: str) -> List[str]:
    tokens = [t for t in re.split(r"[-_/]", slug) if t]
    keywords = [t for t in tokens if not re.fullmatch(r"\d{2,}|\d{4}-\d{2}-\d{2}", t)]
    # 过滤常见噪声
    blacklist = {"event", "sports", "market", "polymarket"}
    return [k.lower() for k in keywords if k.lower() not in blacklist]


def fetch_markets_for_event(event_id: str) -> List[Dict]:
    if not event_id:
        return []
    data = _fetch_json(f"{GAMMA_HOST}/markets", params={"eventId": event_id, "closed": True})
    markets: Iterable[Dict] = data if isinstance(data, list) else data.get("markets", [])
    return list(markets)


def fetch_historical_events(current_event: EventData, keywords: List[str]) -> List[EventData]:
    """优先按 seriesSlug 查询历史事件，失败再回退到关键字搜索。"""
    if not keywords:
        return []

    params_list = []
    if current_event.series_slug:
        params_list.append({"seriesSlug": current_event.series_slug, "closed": True, "limit": 100})
    search_phrase = " ".join(keywords)
    params_list.append({"search": search_phrase, "closed": True})

    results: Dict[str, EventData] = {}
    for params in params_list:
        data = _fetch_json(f"{GAMMA_HOST}/events", params=params)
        events: Iterable[Dict] = data if isinstance(data, list) else data.get("events", [])
        for item in events:
            event_slug = item.get("slug") or item.get("url") or ""
            event_id = str(item.get("id") or item.get("eventId") or item.get("_id") or "")
            # 避免将当前事件重复加入
            if event_id == current_event.event_id:
                continue
            question = item.get("question") or item.get("title") or event_slug
            markets = item.get("markets") or []
            slug_lower = (event_slug or "").lower()
            if params.get("seriesSlug") or all(k in slug_lower for k in keywords):
                if event_id not in results:
                    results[event_id] = EventData(
                        event_id=event_id or event_slug,
                        slug=event_slug,
                        question=question,
                        markets=markets,
                        series_slug=item.get("seriesSlug"),
                        tags=item.get("tags"),
                    )
    return list(results.values())


def parse_market_result(market: Dict) -> MarketResult:
    outcomes = market.get("outcomes") or []
    winning = market.get("outcome") or None
    # 尝试从最终价格判断赢家
    price_field = market.get("outcomePrices") or market.get("prices") or []
    if not winning and isinstance(price_field, list):
        for outcome, price in zip(outcomes, price_field):
            try:
                if float(price) >= 0.999:
                    winning = outcome
                    break
            except (TypeError, ValueError):
                continue
    return MarketResult(
        market_id=str(market.get("id") or market.get("_id") or market.get("market_id") or ""),
        question=market.get("question") or market.get("title") or "",
        slug=market.get("slug") or "",
        resolved=bool(market.get("resolved") or market.get("closed")),
        winning_outcome=winning,
        close_time=market.get("endDate") or market.get("closeTime") or market.get("closedTime"),
    )


def summarize_results(results: List[MarketResult]) -> str:
    if not results:
        return "未找到匹配的历史盘。"
    summary_lines: List[str] = []
    outcome_counter: Dict[str, int] = {}
    for res in results:
        winner = res.winning_outcome or "未知结果"
        outcome_counter[winner] = outcome_counter.get(winner, 0) + 1
        status = "已结算" if res.resolved else "未结算"
        summary_lines.append(
            f"- {res.question or res.slug} | {status} | 胜出：{winner} | 结束时间：{res.close_time or '-'}"
        )
    ranking = sorted(outcome_counter.items(), key=lambda x: x[1], reverse=True)
    headline = "；".join([f"{name} {count} 场" for name, count in ranking])
    return "历史盘结果统计：" + headline + "\n" + "\n".join(summary_lines)


def main() -> None:
    raw_url = input("请输入 Polymarket 事件链接：").strip()
    slug = extract_slug(raw_url)
    events = fetch_events_by_slug(slug)
    if not events:
        print("未找到对应事件，请检查链接或稍后重试。")
        sys.exit(1)

    event = pick_event(events)
    if not event.markets:
        event.markets = fetch_markets_for_event(event.event_id)
    if not event.markets:
        print("该事件没有找到盘口数据，可能接口暂不可用。")
        sys.exit(1)

    market = pick_market(event.markets)
    keywords = derive_keywords(event.slug or slug)
    historical_events = fetch_historical_events(event, keywords)

    historical_results: List[MarketResult] = []
    for ev in historical_events:
        if not ev.markets:
            ev.markets = fetch_markets_for_event(ev.event_id)
        for mk in ev.markets:
            historical_results.append(parse_market_result(mk))

    # 当前选中盘也加入统计，方便对比
    historical_results.insert(0, parse_market_result(market))

    summary = summarize_results(historical_results)
    print("\n==== 历史盘摘要 ====")
    print(summary)

    if len(historical_results) == 1:
        print("\n[提示] 未找到任何带盘口数据的历史事件，目前仅统计了当前选择的盘口。")


if __name__ == "__main__":
    main()
