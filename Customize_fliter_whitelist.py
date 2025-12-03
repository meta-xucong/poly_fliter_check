#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Customize_fliter_whitelist.py  ·  基于 Volatility_fliter_EOA.py 的 REST-only 极简版
- 仅用 REST /books 批量获取买一/卖一（bestBid/bestAsk），完全移除 WS 逻辑
- 保留：时间切片（突破500）、早筛后回补、流式逐个输出/详细块、诊断样本等
- 新增：高亮参数（HIGHLIGHT_*）支持命令行自定义
"""

from __future__ import annotations

import argparse
import datetime as dt
import os
import re
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import requests
except Exception as e:
    print("[FATAL] 缺少 requests，请先 pip install requests", file=sys.stderr)
    raise

# ======
# 全局常量
# ======

# Gamma 公共 API
_GAMMA_HOST = os.environ.get("GAMMA_HOST", "https://gamma-api.polymarket.com")
# CLOB 公共 REST API（仅 /books）
_POLY_HOST = os.environ.get("POLY_HOST", "https://clob.polymarket.com").rstrip("/")

# 默认时间窗口：仅抓未来 [2h, 30d] 内结束的市场，按 endDate 分片抓取
DEFAULT_MIN_END_HOURS = 2.0
DEFAULT_MAX_END_DAYS = 30

# Gamma 时间切片参数
DEFAULT_GAMMA_WINDOW_DAYS = 2
DEFAULT_GAMMA_MIN_WINDOW_HOURS = 1

# 旧格式/归档市场阈值（结束时间早于该值的视为“旧市场”）
DEFAULT_LEGACY_END_DAYS = 730  # 约 2 年

# 高亮（严格口径）参数默认值
HIGHLIGHT_MAX_HOURS = 180.0           # 180 小时内（约 7.5 天）
HIGHLIGHT_ASK_MIN = 0.90              # 卖一价下限
HIGHLIGHT_ASK_MAX = 0.99              # 卖一价上限
HIGHLIGHT_MIN_TOTAL_VOLUME = 10000.0  # 总成交量下限（USDC）
HIGHLIGHT_MAX_ASK_DIFF = 0.10         # 单边点差上限

# REST /books 默认批量大小
DEFAULT_BOOKS_BATCH_SIZE = 200

# Gamma /markets 每页上限（固定为 500）
GAMMA_PAGE_LIMIT = 500


# -------------------------------
# REST-only 客户端加载（非必需，仅用于展示 API key 前缀）
# -------------------------------

def _import_rest_client():
    try:
        from Volatility_arbitrage_main_rest_EOA import get_client as _get_client
        return _get_client
    except Exception as e:
        print(f"[WARN] 无法加载 REST 客户端：{e}", file=sys.stderr)

        def _noop():
            return None

        return _noop


get_rest_client = _import_rest_client()


# ======
# 数据结构
# ======

@dataclass
class OutcomeSnapshot:
    token_id: str
    side: str  # "YES" / "NO"
    bid: Optional[float] = None
    ask: Optional[float] = None


@dataclass
class OrderbookSide:
    bid: Optional[float] = None
    ask: Optional[float] = None


@dataclass
class MarketSnapshot:
    id: str
    slug: str
    title: str
    end_time: dt.datetime
    active: bool
    resolved: bool
    closed: bool
    acceptingOrders: bool
    liquidity: Optional[float]
    totalVolume: Optional[float]
    tags: List[str] = field(default_factory=list)
    outcomes: List[OutcomeSnapshot] = field(default_factory=list)
    yes: OrderbookSide = field(default_factory=OrderbookSide)
    no: OrderbookSide = field(default_factory=OrderbookSide)
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FilterResult:
    """用于 collect_filter_results 的结构化输出。"""
    markets: List[MarketSnapshot]
    early_rejects: List[Tuple[MarketSnapshot, str]]
    highlights: List[Tuple[MarketSnapshot, OutcomeSnapshot, float]]


# ======
# 工具函数
# ======

def _now_utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _parse_iso8601(s: str) -> Optional[dt.datetime]:
    if not s:
        return None
    try:
        # Gamma 使用 ISO8601 带 'Z'
        return dt.datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def _as_bool(x: Any) -> Optional[bool]:
    if isinstance(x, bool):
        return x
    if isinstance(x, str):
        s = x.strip().lower()
        if s in ("true", "yes", "y", "1"):
            return True
        if s in ("false", "no", "n", "0"):
            return False
    if isinstance(x, (int, float)):
        return bool(x)
    return None


def _fmt_money(x: Optional[float]) -> str:
    if x is None:
        return "-"
    try:
        return f"{x:,.2f}"
    except Exception:
        return str(x)


def _hours_until(t: Optional[dt.datetime]) -> Optional[float]:
    if t is None:
        return None
    now = _now_utc()
    try:
        delta = t - now
        return delta.total_seconds() / 3600.0
    except Exception:
        return None


def _fmt_hours_left(t: Optional[dt.datetime]) -> str:
    h = _hours_until(t)
    if h is None:
        return "-"
    return f"{h:.1f}"


def _safe_float(x: Any) -> Optional[float]:
    if x is None or x == "":
        return None
    try:
        return float(x)
    except Exception:
        return None


# ======
# Gamma /markets 抓取（带时间切片）
# ======

def fetch_markets_page(end_min: dt.datetime,
                       end_max: dt.datetime,
                       cursor: Optional[str] = None) -> Dict[str, Any]:
    """调用 Gamma /markets 一页，带时间过滤和 cursor。"""
    params = {
        "limit": GAMMA_PAGE_LIMIT,
        "endDateMin": end_min.isoformat().replace("+00:00", "Z"),
        "endDateMax": end_max.isoformat().replace("+00:00", "Z"),
    }
    if cursor:
        params["cursor"] = cursor
    url = f"{_GAMMA_HOST}/markets"
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_markets_windowed(end_min: dt.datetime,
                           end_max: dt.datetime,
                           *,
                           window_days: int = DEFAULT_GAMMA_WINDOW_DAYS,
                           min_window_hours: int = DEFAULT_GAMMA_MIN_WINDOW_HOURS) -> List[Dict[str, Any]]:
    """
    按 endDate 时间窗口递归抓取，避免单次命中 500 上限。
    - 初始窗口大小 = window_days
    - 若某窗口命中 500 条，则将该窗口继续二分，直到窗口 < min_window_hours 仍命中 500，则退化到按 endDate 再分页（cursor）
    """
    results: List[Dict[str, Any]] = []

    def _extract(payload: Any) -> Tuple[List[Dict[str, Any]], Optional[str]]:
        """提取 markets 列表和 cursor，兼容 list / dict 返回值。"""
        if isinstance(payload, list):
            return payload, None
        if isinstance(payload, dict):
            mkts = payload.get("data") or payload.get("markets") or []
            cursor = payload.get("nextCursor") or payload.get("cursor")
            return mkts, cursor
        return [], None

    def _split_window(start: dt.datetime, stop: dt.datetime):
        # 若窗口结束时间早于最小开始时间，跳过
        if stop <= start:
            return
        span_hours = (stop - start).total_seconds() / 3600.0
        if span_hours <= min_window_hours:
            # 已小于最小窗口：直接按 cursor 分页抓取
            cursor: Optional[str] = None
            while True:
                try:
                    data = fetch_markets_page(start, stop, cursor)
                except Exception as e:
                    print(f"[WARN] fetch_markets_page 失败：{e}", file=sys.stderr)
                    break
                mkts, cursor = _extract(data)
                results.extend(mkts)
                if not cursor or len(mkts) < GAMMA_PAGE_LIMIT:
                    break
            return

        # 尝试以当前窗口抓一次
        try:
            data = fetch_markets_page(start, stop, None)
        except Exception as e:
            print(f"[WARN] fetch_markets_page（window）失败：{e}", file=sys.stderr)
            return
        mkts, cursor = _extract(data)
        if len(mkts) >= GAMMA_PAGE_LIMIT:
            # 命中上限：继续二分
            mid = start + dt.timedelta(hours=span_hours / 2.0)
            _split_window(start, mid)
            _split_window(mid, stop)
        else:
            # 未命中上限，可按 cursor 继续抓完
            results.extend(mkts)
            while cursor:
                try:
                    data2 = fetch_markets_page(start, stop, cursor)
                except Exception as e:
                    print(f"[WARN] fetch_markets_page（cursor）失败：{e}", file=sys.stderr)
                    break
                mkts2, cursor = _extract(data2)
                results.extend(mkts2)
                if not cursor or len(mkts2) < GAMMA_PAGE_LIMIT:
                    break

    _split_window(end_min, end_max)
    return results


# ======
# 解析 Gamma 市场数据
# ======

def _parse_market(raw: Dict[str, Any]) -> MarketSnapshot:
    mid = str(raw.get("id") or raw.get("_id") or "")
    slug = str(raw.get("slug") or "")
    title = str(raw.get("question") or raw.get("title") or slug or mid)
    end_time = _parse_iso8601(raw.get("endDate") or raw.get("endTime") or "")

    active = bool(_as_bool(raw.get("active")))
    resolved = bool(_as_bool(raw.get("resolved")))
    closed = bool(_as_bool(raw.get("closed")))
    accepting = bool(_as_bool(raw.get("acceptingOrders") or raw.get("accepting")))

    liquidity = _safe_float(raw.get("liquidity"))
    total_volume = _safe_float(raw.get("totalVolume") or raw.get("volume"))

    tags = raw.get("tags") or []
    if not isinstance(tags, list):
        tags = []

    outcomes_raw = raw.get("outcomes") or raw.get("markets") or []
    outcomes: List[OutcomeSnapshot] = []
    yes_side = OrderbookSide()
    no_side = OrderbookSide()

    # 支持多种 outcome 结构，但这里只关心 YES/NO token_id / prices
    for o in outcomes_raw:
        token_id = str(o.get("tokenId") or o.get("id") or "")
        # side 可以是 YES/NO，也可能是 YES_TOKEN/NO_TOKEN 等，这里做一个兼容
        side_raw = str(o.get("side") or o.get("type") or "").upper()
        side = "YES" if "YES" in side_raw else "NO" if "NO" in side_raw else side_raw
        bid = _safe_float(o.get("bestBid"))
        ask = _safe_float(o.get("bestAsk"))
        snap = OutcomeSnapshot(token_id=token_id, side=side, bid=bid, ask=ask)
        outcomes.append(snap)
        if side == "YES":
            if bid is not None:
                yes_side.bid = bid if yes_side.bid is None else max(yes_side.bid, bid)
            if ask is not None:
                yes_side.ask = ask if yes_side.ask is None else min(yes_side.ask, ask)
        elif side == "NO":
            if bid is not None:
                no_side.bid = bid if no_side.bid is None else max(no_side.bid, bid)
            if ask is not None:
                no_side.ask = ask if no_side.ask is None else min(no_side.ask, ask)

    return MarketSnapshot(
        id=mid,
        slug=slug,
        title=title,
        end_time=end_time,
        active=active,
        resolved=resolved,
        closed=closed,
        acceptingOrders=accepting,
        liquidity=liquidity,
        totalVolume=total_volume,
        tags=tags,
        outcomes=outcomes,
        yes=yes_side,
        no=no_side,
        raw=raw,
    )


# ======
# REST /books 回补买一/卖一（bestBid/bestAsk）
# ======

def fetch_books_batch(token_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    调用 /books?ids=... 获取若干 token 的订单簿最佳价：
    返回结构：{tokenId: {"yes": {"bids":[...], "asks":[...]}, "no": {...}}}
    """
    if not token_ids:
        return {}
    url = f"{_POLY_HOST}/books"
    params = [("ids", tid) for tid in token_ids]
    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    data = resp.json() or {}
    if isinstance(data, list):
        # 历史遗留：/books 早期可能是 list，这里兼容一下
        d2: Dict[str, Dict[str, Any]] = {}
        for item in data:
            tid = str(item.get("id") or item.get("tokenId") or "")
            if tid:
                d2[tid] = item
        return d2
    return data


def _extract_best_from_book(book: Dict[str, Any]) -> Tuple[Optional[float], Optional[float]]:
    """
    从 book["bids"], book["asks"] 中提取最佳买价/卖价（price 最大买价、最小卖价）。
    允许 bids/asks 为 None 或空列表。
    """
    best_bid = None
    best_ask = None
    bids = book.get("bids") or []
    asks = book.get("asks") or []
    if isinstance(bids, list):
        for b in bids:
            p = _safe_float(b.get("price"))
            if p is not None:
                best_bid = p if best_bid is None else max(best_bid, p)
    if isinstance(asks, list):
        for a in asks:
            p = _safe_float(a.get("price"))
            if p is not None:
                best_ask = p if best_ask is None else min(best_ask, p)
    return best_bid, best_ask


def _rest_books_backfill(markets: List[MarketSnapshot],
                         batch_size: int = DEFAULT_BOOKS_BATCH_SIZE) -> None:
    """
    对传入的 markets，根据其 outcomes 中缺失的 bid/ask 信息，通过 /books 批量回补。
    只更新 YES/NO 的 bestBid/bestAsk。
    """
    # 收集所有 token_id
    token_ids: List[str] = []
    for ms in markets:
        for o in ms.outcomes:
            if o.token_id:
                token_ids.append(o.token_id)
    # 去重
    token_ids = sorted(set(token_ids))
    if not token_ids:
        return

    print(f"[TRACE] REST 回补：共 {len(token_ids)} 个 token_id，批量大小={batch_size}")

    # 分批拉取 /books
    books_all: Dict[str, Dict[str, Any]] = {}
    for i in range(0, len(token_ids), batch_size):
        batch = token_ids[i:i + batch_size]
        try:
            books = fetch_books_batch(batch)
        except Exception as e:
            print(f"[WARN] fetch_books_batch 失败：{e}", file=sys.stderr)
            continue
        books_all.update(books)

    # 回补到 MarketSnapshot
    for ms in markets:
        for o in ms.outcomes:
            if not o.token_id:
                continue
            b = books_all.get(o.token_id)
            if not b:
                continue
            # 根据 side 区分 yes/no
            side_raw = str(b.get("side") or b.get("type") or "").upper()
            side = "YES" if "YES" in side_raw else "NO" if "NO" in side_raw else side_raw

            # 兼容结构：有的返回 {"yes": {...}, "no": {...}}，有的直接是 {"bids":[...], "asks":[...]}
            if "yes" in b or "no" in b:
                if side == "YES":
                    node = b.get("yes") or {}
                elif side == "NO":
                    node = b.get("no") or {}
                else:
                    node = b.get("yes") or b.get("no") or {}
            else:
                node = b

            best_bid, best_ask = _extract_best_from_book(node)

            # 更新 outcome
            if best_bid is not None:
                o.bid = best_bid
            if best_ask is not None:
                o.ask = best_ask

            # 同步到 YES/NO 汇总
            if o.side == "YES":
                if best_bid is not None:
                    ms.yes.bid = best_bid if ms.yes.bid is None else max(ms.yes.bid, best_bid)
                if best_ask is not None:
                    ms.yes.ask = best_ask if ms.yes.ask is None else min(ms.yes.ask, best_ask)
            elif o.side == "NO":
                if best_bid is not None:
                    ms.no.bid = best_bid if ms.no.bid is None else max(ms.no.bid, best_bid)
                if best_ask is not None:
                    ms.no.ask = best_ask if ms.no.ask is None else min(ms.no.ask, best_ask)


# ======
# 早筛逻辑（仅看 endDate / active / resolved 等元信息）
# ======

def _early_filter_reason(ms: MarketSnapshot,
                         min_end_hours: float,
                         legacy_end_days: int) -> Tuple[bool, str]:
    """
    返回 (是否通过, 理由)。不看价格，仅根据元数据做早期过滤：
    - 必须 active=True
    - 不允许 resolved=True
    - 不允许 closed=True
    - end_time 在未来 min_end_hours 之后
    - end_time 不早于 NOW - legacy_end_days
    """
    if not ms.active:
        return False, "非 active"
    if ms.resolved:
        return False, "已 resolved"
    if ms.closed:
        return False, "已 closed"
    if ms.end_time is None:
        return False, "无 end_time"
    h = _hours_until(ms.end_time)
    if h is None:
        return False, "无法计算剩余时间"
    if h < min_end_hours:
        return False, f"剩余时间 {h:.1f}h < {min_end_hours}h"
    # 旧格式/归档：结束时间太久远
    now = _now_utc()
    if ms.end_time < now - dt.timedelta(days=legacy_end_days):
        return False, "旧市场/归档"
    # 对订单簿的最低要求在后续统一检查
    return True, "OK"


def _satisfy_basic_liquidity(ms: MarketSnapshot, allow_illiquid: bool = False) -> Tuple[bool, str]:
    """
    基础流动性/价格可用性检查：
    - 若 allow_illiquid=False：
      - 要求 YES/NO 中至少一边有 bid 或 ask
    """
    if allow_illiquid:
        return True, "允许无流动性"
    # 至少一边有报价
    yes_ok = (ms.yes.bid is not None or ms.yes.ask is not None)
    no_ok = (ms.no.bid is not None or ms.no.ask is not None)
    if not (yes_ok or no_ok):
        return False, "缺少买卖价（空簿/超时）"
    return True, "OK"


# -------------------------------
# 打印
# -------------------------------

# -------------------------------
# 关键词白名单（仅保留包含以下词条的话题）
# -------------------------------

WHITELIST_TERMS = [
    "Bitcoin","BTC","ETH","Ethereum","Sol","Solana","Doge","Dogecoin",
    "BNB","Binance","Cardano","ADA","XRP","Ripple","Matic","Polygon",
    "Crypto","Cryptocurrency","Blockchain","Token","NFT","DeFi",
    "vs","odds","score","spread","moneyline","win",
    "Esports","CS2","Cup","Arsenal","Liverpool","Chelsea",
    "EPL","PGA","Tour Championship","Scottie Scheffler",
    "Vitality","MOUZ","Falcons","The MongolZ","AL","Houston","Chicago","New York",
    "Will Elon Musk post","dota2","a dozen eggs",
]


def _build_whitelist_patterns(terms: Iterable[str]) -> List[Tuple[str, re.Pattern[str]]]:
    patterns: List[Tuple[str, re.Pattern[str]]] = []
    for term in terms:
        tl = term.lower()
        if len(tl) <= 3 and tl.isalpha():
            pat = re.compile(rf"\b{re.escape(tl)}\b", re.IGNORECASE)
        else:
            pat = re.compile(re.escape(term), re.IGNORECASE)
        patterns.append((term, pat))
    return patterns


WHITELIST_PATTERNS = _build_whitelist_patterns(WHITELIST_TERMS)


def _whitelist_match(ms: MarketSnapshot) -> Optional[str]:
    """返回命中的白名单词条；若白名单为空则返回 None（不启用过滤）。"""
    if not WHITELIST_PATTERNS:
        return None
    parts = [ms.title or "", ms.slug or ""]
    if ms.tags:
        parts.append(" ".join(ms.tags))
    haystack = " ".join(filter(None, parts))
    for term, pat in WHITELIST_PATTERNS:
        if pat.search(haystack):
            return term
    return None


def _print_snapshot(idx: int, total: int, ms: MarketSnapshot):
    print(f"[TRACE] [{idx}/{total}] 原始市场：slug={ms.slug} | 标题={ms.title}")
    st = " ".join([
        f"active={'是' if ms.active else '-'}",
        f"resolved={'是' if ms.resolved else '-'}",
        f"closed={'是' if ms.closed else '-'}",
        f"acceptingOrders={'是' if ms.acceptingOrders else '-'}",
    ])
    print(f"[TRACE]   状态：{st}")
    print(f"[TRACE]   金额：liquidity={_fmt_money(ms.liquidity)} totalVolume={_fmt_money(ms.totalVolume)}")
    print(f"[TRACE]   时间：end_time={ms.end_time} (剩余 { _fmt_hours_left(ms.end_time)}h )")
    if ms.tags:
        print(f"[TRACE]   标签：{', '.join(ms.tags)}")
    print(f"[TRACE]   YES: bid={ms.yes.bid} ask={ms.yes.ask}")
    print(f"[TRACE]   NO : bid={ms.no.bid} ask={ms.no.ask}")


def _print_singleline(ms: MarketSnapshot, reason: str):
    h = _fmt_hours_left(ms.end_time)
    yb = f"{ms.yes.bid:.3f}" if ms.yes.bid is not None else "-"
    ya = f"{ms.yes.ask:.3f}" if ms.yes.ask is not None else "-"
    nb = f"{ms.no.bid:.3f}" if ms.no.bid is not None else "-"
    na = f"{ms.no.ask:.3f}" if ms.no.ask is not None else "-"
    print(f"[RESULT] {ms.slug} | {ms.title} | YES {yb}/{ya} NO {nb}/{na} | ends_in={h}h | {reason}", flush=True)


def _highlight_outcomes(ms: MarketSnapshot,
                        max_hours: Optional[float] = None,
                        ask_min: Optional[float] = None,
                        ask_max: Optional[float] = None,
                        min_total_volume: Optional[float] = None,
                        max_ask_diff: Optional[float] = None) -> List[Tuple[OutcomeSnapshot, float]]:
    """
    高亮（严格口径）筛选条件：
    - 剩余时间 ≤ max_hours（默认 HIGHLIGHT_MAX_HOURS）
    - 卖一价 ask ∈ [ask_min, ask_max]（默认 HIGHLIGHT_ASK_MIN/HIGHLIGHT_ASK_MAX）
    - 总交易量 totalVolume ≥ min_total_volume（默认 HIGHLIGHT_MIN_TOTAL_VOLUME）
    - 同一 token 的点差 |ask - bid| ≤ max_ask_diff（YES 或 NO 任一满足即可；默认 HIGHLIGHT_MAX_ASK_DIFF）
    - 若配置了白名单：仅保留命中白名单词条的市场
    """
    mh = HIGHLIGHT_MAX_HOURS if max_hours is None else max_hours
    lo = HIGHLIGHT_ASK_MIN if ask_min is None else ask_min
    hi = HIGHLIGHT_ASK_MAX if ask_max is None else ask_max
    mv = HIGHLIGHT_MIN_TOTAL_VOLUME if min_total_volume is None else min_total_volume
    mdiff = HIGHLIGHT_MAX_ASK_DIFF if max_ask_diff is None else max_ask_diff

    hours = _hours_until(ms.end_time)
    if hours is None or hours < 0 or hours > mh:
        return []

    # 若配置了白名单：仅保留命中白名单词条的市场
    if WHITELIST_PATTERNS and _whitelist_match(ms) is None:
        return []

    # 交易量要求
    tv = getattr(ms, "totalVolume", None)
    if tv is None:
        try:
            tv = float(ms.raw.get("totalVolume") or ms.raw.get("volume") or 0)
        except Exception:
            tv = 0.0
    if tv < mv:
        return []

    # 单边点差（同一 token 内 ask-bid）约束在逐项判定中完成
    highlights: List[Tuple[OutcomeSnapshot, float]] = []
    for o in ms.outcomes:
        if o.ask is None:
            continue
        if not (lo <= o.ask <= hi):
            continue
        # 计算对应 side 的 bid
        if o.side == "YES":
            bid = ms.yes.bid
        elif o.side == "NO":
            bid = ms.no.bid
        else:
            bid = o.bid
        if bid is None:
            continue
        if abs(o.ask - bid) > mdiff:
            continue
        hours2 = _hours_until(ms.end_time) or 0.0
        highlights.append((o, hours2))
    return highlights


def _print_highlighted(highlights: List[Tuple[MarketSnapshot, OutcomeSnapshot, float]]):
    if not highlights:
        print("[HIGHLIGHT] 当前无满足高亮条件的市场。")
        return
    print("[HIGHLIGHT] 满足高亮条件的市场：")
    label = _highlight_label()
    print(f"  条件：{label}")
    for ms, snap, hours in highlights:
        print(f"  - slug={ms.slug} | 标题={ms.title}")
        print(f"      side={snap.side} token_id={snap.token_id} bid={snap.bid} ask={snap.ask} | 剩余 {hours:.1f}h")


def _highlight_label() -> str:
    base = (f"≤{int(HIGHLIGHT_MAX_HOURS)}h & ask在 {HIGHLIGHT_ASK_MIN:.3f}-{HIGHLIGHT_ASK_MAX:.3f} "
            f"& 总交易量≥{int(HIGHLIGHT_MIN_TOTAL_VOLUME)}USDC & 单边点差≤{HIGHLIGHT_MAX_ASK_DIFF:.2f}")
    if WHITELIST_PATTERNS:
        return base + " & 命中白名单"
    return base


# -------------------------------
# 面向自动化脚本的封装
# -------------------------------

def collect_filter_results(
    *,
    min_end_hours: float = DEFAULT_MIN_END_HOURS,
    max_end_days: int = DEFAULT_MAX_END_DAYS,
    gamma_window_days: int = DEFAULT_GAMMA_WINDOW_DAYS,
    gamma_min_window_hours: int = DEFAULT_GAMMA_MIN_WINDOW_HOURS,
    legacy_end_days: int = DEFAULT_LEGACY_END_DAYS,
    allow_illiquid: bool = False,
    skip_orderbook: bool = False,
    no_rest_backfill: bool = False,
    books_batch_size: int = 200,
    only: str = "",
    prefetched_markets: Optional[List[Dict[str, Any]]] = None,
) -> FilterResult:
    """执行一次筛选流程并返回结构化结果。"""

    if prefetched_markets is None:
        now = _now_utc()
        end_min = now + dt.timedelta(hours=min_end_hours)
        end_max = now + dt.timedelta(days=max_end_days)
        mkts_raw = fetch_markets_windowed(
            end_min,
            end_max,
            window_days=gamma_window_days,
            min_window_hours=gamma_min_window_hours,
        )
    else:
        mkts_raw = prefetched_markets

    only_pat = only.lower().strip()

    market_list: List[MarketSnapshot] = []
    early_rejects: List[Tuple[MarketSnapshot, str]] = []

    for raw in mkts_raw:
        title = (raw.get("question") or raw.get("title") or "")
        slug = (raw.get("slug") or "")
        if only_pat and (only_pat not in title.lower() and only_pat not in slug.lower()):
            continue
        ms = _parse_market(raw)
        # 白名单过滤：若配置了白名单，则仅保留命中的市场
        if WHITELIST_PATTERNS and _whitelist_match(ms) is None:
            early_rejects.append((ms, "未命中白名单"))
            continue
        ok, reason = _early_filter_reason(ms, min_end_hours, legacy_end_days)
        if ok:
            market_list.append(ms)
        else:
            early_rejects.append((ms, reason))

    if not skip_orderbook and market_list and (not no_rest_backfill):
        _rest_books_backfill(market_list, batch_size=books_batch_size)

    chosen: List[MarketSnapshot] = []
    highlights: List[Tuple[MarketSnapshot, OutcomeSnapshot, float]] = []
    for ms in market_list:
        ok, reason = _satisfy_basic_liquidity(ms, allow_illiquid=allow_illiquid)
        if not ok:
            early_rejects.append((ms, reason))
            continue
        chosen.append(ms)
        for snap, hours in _highlight_outcomes(ms):
            highlights.append((ms, snap, hours))

    return FilterResult(markets=chosen, early_rejects=early_rejects, highlights=highlights)


# -------------------------------
# CLI 入口
# -------------------------------

def main():
    ap = argparse.ArgumentParser(description="Polymarket 市场筛选（REST-only：/books 批量回补买一/卖一）")
    ap.add_argument("--books-batch-size", type=int, default=200, help="REST /books 批量回补的 token_id 数量上限（非流式模式）")
    ap.add_argument("--no_rest_backfill", dest="no_rest_backfill", action="store_true", help="关闭 REST 回补（诊断用，默认开启）")
    ap.add_argument("--skip-orderbook", action="store_true", help="跳过任何订单簿/价格回补（仅诊断）")
    ap.add_argument("--allow-illiquid", action="store_true", help="允许无报价市场通过（仅诊断）")

    ap.add_argument("--min-end-hours", type=float, default=DEFAULT_MIN_END_HOURS, help="仅抓取结束时间晚于该阈值（小时）的市场")
    ap.add_argument("--max-end-days", type=int, default=DEFAULT_MAX_END_DAYS, help="仅抓取结束时间在未来 N 天内的市场")
    ap.add_argument("--gamma-window-days", type=int, default=DEFAULT_GAMMA_WINDOW_DAYS, help="Gamma 时间切片的窗口大小（天），命中 500 会自动递归切分")
    ap.add_argument("--gamma-min-window-hours", type=int, default=DEFAULT_GAMMA_MIN_WINDOW_HOURS, help="Gamma 时间切片命中 500 时递归拆分的最小窗口（小时）；窗口缩到该级别仍满额会按 endDate 继续分页")

    ap.add_argument("--legacy-end-days", type=int, default=DEFAULT_LEGACY_END_DAYS, help="结束早于 N 天视为旧格式/归档（默认 730 天）")

    # 高亮（严格口径）参数：不指定时使用脚本顶部的 HIGHLIGHT_* 默认值
    ap.add_argument("--hl-max-hours", type=float, default=None,
                    help="高亮条件：剩余时间 ≤ 该阈值（小时），例如 48 表示 48 小时内")
    ap.add_argument("--hl-ask-min", type=float, default=None,
                    help="高亮条件：卖一价下限，例如 0.96 表示 96% 起")
    ap.add_argument("--hl-ask-max", type=float, default=None,
                    help="高亮条件：卖一价上限，例如 0.995 表示 99.5% 封顶")
    ap.add_argument("--hl-min-total-volume", type=float, default=None,
                    help="高亮条件：总成交量下限（USDC），例如 10000 表示 ≥1 万 USDC")
    ap.add_argument("--hl-max-ask-diff", type=float, default=None,
                    help="高亮条件：单边点差 |ask-bid| 上限，例如 0.10 表示 ≤10 个点")

    ap.add_argument("--diagnose", action="store_true", help="打印诊断信息（非流式模式下打印样本）")
    ap.add_argument("--diagnose-samples", type=int, default=30, help="诊断打印的样本数上限（非流式模式）")
    ap.add_argument("--only", type=str, default="", help="仅处理包含该子串的 slug/title（大小写不敏感）")

    # 流式输出选项
    ap.add_argument("--stream", action="store_true", help="启用流式逐个输出（按分片处理）")
    ap.add_argument("--stream-chunk-size", type=int, default=200, help="流式：每个分片的市场数量")
    ap.add_argument("--stream-books-batch-size", type=int, default=200, help="流式：每个分片内 REST /books 批量回补的 token_id 数量上限")
    ap.add_argument("--stream-verbose", action="store_true", help="流式：逐个输出详细块（默认仅单行）")
    args = ap.parse_args()

    # 若指定了高亮参数，则覆盖全局 HIGHLIGHT_*，以便后续筛选与标签展示使用
    global HIGHLIGHT_MAX_HOURS, HIGHLIGHT_ASK_MIN, HIGHLIGHT_ASK_MAX, HIGHLIGHT_MIN_TOTAL_VOLUME, HIGHLIGHT_MAX_ASK_DIFF
    if args.hl_max_hours is not None:
        HIGHLIGHT_MAX_HOURS = args.hl_max_hours
    if args.hl_ask_min is not None:
        HIGHLIGHT_ASK_MIN = args.hl_ask_min
    if args.hl_ask_max is not None:
        HIGHLIGHT_ASK_MAX = args.hl_ask_max
    if args.hl_min_total_volume is not None:
        HIGHLIGHT_MIN_TOTAL_VOLUME = args.hl_min_total_volume
    if args.hl_max_ask_diff is not None:
        HIGHLIGHT_MAX_ASK_DIFF = args.hl_max_ask_diff

    # 仅用于展示 API key 前缀
    try:
        getc = get_rest_client
        rest_client = getc() if callable(getc) else None
        api_creds = getattr(rest_client, "api_creds", None)

        def g(x, k):
            if isinstance(x, dict):
                return x.get(k)
            return getattr(x, k, None)

        ak = g(api_creds, "api_key")
        if ak:
            print(f"[INFO] 已加载 EOA API credentials：{ak[:6]}***{ak[-4:]}")
    except Exception:
        pass

    # 仅抓未来盘：时间窗口 = [now + min_end_hours, now + max_end_days]
    now = _now_utc()
    end_min = now + dt.timedelta(hours=args.min_end_hours)
    end_max = now + dt.timedelta(days=args.max_end_days)

    mkts_raw = fetch_markets_windowed(
        end_min,
        end_max,
        window_days=args.gamma_window_days,
        min_window_hours=args.gamma_min_window_hours,
    )
    print(f"[TRACE] 采用时间切片抓取完成：共获取 {len(mkts_raw)} 条（窗口={args.gamma_window_days} 天，最小窗口={args.gamma_min_window_hours} 小时）")

    only_pat = args.only.lower().strip()

    # ---------- 流式模式 ----------
    if args.stream:
        total = len(mkts_raw)
        processed = 0
        chosen_cnt = 0
        highlights: List[Tuple[MarketSnapshot, OutcomeSnapshot, float]] = []
        for s in range(0, total, args.stream_chunk_size):
            chunk_raw = mkts_raw[s:s + args.stream_chunk_size]
            # 解析 + 早筛（即时输出被拒绝的理由）
            candidates: List[MarketSnapshot] = []
            for raw in chunk_raw:
                title = (raw.get("question") or raw.get("title") or "")
                slug = (raw.get("slug") or "")
                if only_pat and (only_pat not in title.lower() and only_pat not in slug.lower()):
                    continue
                ms = _parse_market(raw)

                # 白名单过滤：若配置了白名单，则仅保留命中的市场
                if WHITELIST_PATTERNS and _whitelist_match(ms) is None:
                    reason = "未命中白名单"
                    if args.stream_verbose:
                        _print_snapshot(processed+1, total, ms)
                        print(f"[TRACE]   -> 结果：{reason}。")
                        print(f"[TRACE]   --------------------------------------------------")
                    else:
                        _print_singleline(ms, reason)
                    processed += 1
                    continue

                ok, reason = _early_filter_reason(ms, args.min_end_hours, args.legacy_end_days)
                if ok:
                    candidates.append(ms)
                else:
                    if args.stream_verbose:
                        _print_snapshot(processed+1, total, ms)
                        print(f"[TRACE]   -> 结果：{reason}。")
                        print(f"[TRACE]   --------------------------------------------------")
                    else:
                        _print_singleline(ms, reason)
                processed += 1

            # REST 回补
            if (not args.skip_orderbook) and candidates and (not args.no_rest_backfill):
                _rest_books_backfill(candidates, batch_size=args.stream_books_batch_size)

            # 二次筛选 + 高亮
            for ms in candidates:
                ok2, reason2 = _satisfy_basic_liquidity(ms, allow_illiquid=args.allow_illiquid)
                if args.stream_verbose:
                    _print_snapshot(processed+1, total, ms)
                    print(f"[TRACE]   -> 结果：{reason2}。")
                    print(f"[TRACE]   --------------------------------------------------")
                else:
                    _print_singleline(ms, reason2)
                if ok2:
                    chosen_cnt += 1
                    for snap, hours in _highlight_outcomes(ms):
                        highlights.append((ms, snap, hours))

            print(f"[STREAM] 已处理 {processed}/{total} 条，当前 chunk={s//args.stream_chunk_size+1}")

        print("")
        _print_highlighted(highlights)
        print(f"\n[INFO] 通过筛选的市场数量：{chosen_cnt}")
        return

    # ---------- 非流式模式 ----------
    fr = collect_filter_results(
        min_end_hours=args.min_end_hours,
        max_end_days=args.max_end_days,
        gamma_window_days=args.gamma_window_days,
        gamma_min_window_hours=args.gamma_min_window_hours,
        legacy_end_days=args.legacy_end_days,
        allow_illiquid=args.allow_illiquid,
        skip_orderbook=args.skip_orderbook,
        no_rest_backfill=args.no_rest_backfill,
        books_batch_size=args.books_batch_size,
        only=args.only,
        prefetched_markets=mkts_raw,
    )

    # 打印结果
    for ms, reason in fr.early_rejects[:args.diagnose_samples]:
        if args.diagnose:
            _print_snapshot(0, len(fr.early_rejects), ms)
            print(f"[TRACE]   -> 结果：{reason}")
            print(f"[TRACE]   --------------------------------------------------")
        else:
            _print_singleline(ms, reason)

    for ms in fr.markets:
        _print_singleline(ms, "通过筛选")

    print("")
    _print_highlighted(fr.highlights)
    print(f"\n[INFO] 通过筛选的市场数量：{len(fr.markets)}")


if __name__ == "__main__":
    main()
