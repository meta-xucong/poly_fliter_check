# poly_domain_scan.py 使用指南

本工具用于批量抓取 Polymarket 的盘口与价格历史，并导出 CSV 或宽表。主要流程：
1. 根据领域（体育、电竞或自定义标签）检索盘口。
2. 按成交量/流动性筛选目标盘口，保存元数据。
3. 为每个 outcome 调用 CLOB prices-history 接口拉取价格曲线。
4. 输出盘口文件与价格文件，可选生成宽表（按盘口+时间戳展开 outcome 列）。

## 常用参数
- `--domain`：领域选择，`sports_all`/`esports`/`custom`。当使用 `custom` 时需配合 `--tags`。
- `--sports`：仅 `domain=sports_all` 时生效，逗号分隔体育标签（如 `NBA,Soccer`）。
- `--tags`：自定义 tag（id/slug/label），逗号分隔，优先级最高。
- `--esports-tag-hints`：电竞标签线索，仅 `domain=esports` 时用于推断 tag。
- `--days`：未指定日期范围时使用的回溯天数，默认 7。
- `--start-date` / `--end-date`：`YYYY-MM-DD` 格式，覆盖 `--days`。
- `--min-volume`：成交量下限（USD），过滤低流动性盘口。
- `--sports-market-types`：筛选体育盘口类型（如 `moneyline,spread,totals`）。
- `--fidelity`：价格历史采样粒度（分钟），数字越小采样越密集。
- `--output`：盘口 CSV 路径；价格文件会自动追加 `_prices` 后缀。
- `--markets-csv` / `--prices-csv`：使用本地现有盘口与价格 CSV，跳过 API 抓取。
- `--wide-output`：生成 outcome 拆列的宽表，适合后续建模或可视化。

## 运行示例
- 默认示例：
  ```bash
  python poly_domain_scan.py --domain sports_all --days 7 --min-volume 5000 \
      --fidelity 60 --output markets_history.csv
  ```

- 指定路径与 NBA 高成交量筛选示例：
  ```bash
  python3 /home/trader/polymarket_api/poly_fliter_check/poly_domain_scan.py \
    --domain sports_all \
    --sports NBA \
    --days 1 \
    --min-volume 50000 \
    --fidelity 60 \
    --output /home/trader/polymarket_api/poly_fliter_check/markets_history.csv
  ```

- 电竞 DOTA2 示例（参数同上，仅切换领域与标签线索）：
  ```bash
  python3 /home/trader/polymarket_api/poly_fliter_check/poly_domain_scan.py \
    --domain esports \
    --esports-tag-hints DOTA2 \
    --days 1 \
    --min-volume 50000 \
    --fidelity 60 \
    --output /home/trader/polymarket_api/poly_fliter_check/markets_history.csv
  ```

## 输出文件说明
- 盘口文件：包含事件 slug、问题、市场类型、outcomes、token ids、创建/截止/开赛时间与成交量等。
- 价格文件：每个 outcome 的时间序列，含 `price_raw` 与换算的 `price_prob`。
- 宽表文件：以盘口+时间戳为行，`outcome_X_prob` 列为各 outcome 概率值。

## 使用建议
- 初次探索可提高 `--min-volume` 与 `--fidelity`（如 60）快速扫描，再根据需求降低采样间隔。
- 若仅需生成宽表或重复加工，可使用已有的 `--markets-csv` 与 `--prices-csv` 以节省网络请求。
- 电竞场景建议提供明确的 `--esports-tag-hints` 以提升标签解析准确度。
