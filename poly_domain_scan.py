
import csv
import json
import requests
import re
import time
from typing import List, Tuple
import os
from urllib.parse import urlparse

# Set up the Gamma API and CLOB API hosts
GAMMA_HOST = os.environ.get("GAMMA_HOST", "https://gamma-api.polymarket.com").rstrip("/")
CLOB_HOST = os.environ.get("CLOB_HOST", "https://clob.polymarket.com").rstrip("/")

# Define the MarketRecord and PricePoint classes to hold the necessary data
class MarketRecord:
    def __init__(self, market_slug, outcomes, token_ids):
        self.market_slug = market_slug
        self.outcomes = outcomes
        self.token_ids = token_ids

class PricePoint:
    def __init__(self, market_slug, outcome, timestamp, price_raw, price_prob):
        self.market_slug = market_slug
        self.outcome = outcome
        self.timestamp = timestamp
        self.price_raw = price_raw
        self.price_prob = price_prob

def fetch_json(url: str, params: Dict = None) -> Dict:
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as exc:
        raise RuntimeError(f"Failed to fetch data from {url} -> {exc}") from exc

def normalize_outcomes(outcomes: str) -> List[str]:
    # Normalize the outcomes field by splitting by '|' and removing extra characters
    return [o.strip() for o in re.split(r'[|,]', outcomes.strip()) if o.strip()]

def fetch_price_history(token_id: str, start_ts: int, end_ts: int, fidelity: int) -> List[Tuple[int, float]]:
    # Fetch the price history for a given token ID
    params = {"market": token_id, "startTs": start_ts, "endTs": end_ts, "fidelity": fidelity}
    data = fetch_json(f"{CLOB_HOST}/prices-history", params=params)
    history = data.get("history", [])
    return [(point['t'], point['p'] / 10000 if point['p'] > 1 else point['p'] / 100) for point in history if 't' in point and 'p' in point]

def collect_price_points(record: MarketRecord, fidelity: int, start_ts: int, end_ts: int) -> List[PricePoint]:
    # Collect price points for both outcomes of a given market
    price_points = []
    for outcome, token_id in zip(record.outcomes, record.token_ids):
        history = fetch_price_history(token_id, start_ts, end_ts, fidelity)
        for timestamp, price_prob in history:
            price_points.append(PricePoint(record.market_slug, outcome, timestamp, price_prob * 10000, price_prob))  # multiplying by 10000 to adjust raw price
    return price_points

def load_market_records_from_csv(path: str) -> List[MarketRecord]:
    # Load market records from a CSV file
    records = []
    with open(path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            outcomes = normalize_outcomes(row.get('outcomes', ''))
            token_ids = row.get('token_ids', '').split(',')
            records.append(MarketRecord(row['market_slug'], outcomes, token_ids))
    return records

def export_csv(records: List[MarketRecord], price_points: List[PricePoint], output_path: str) -> None:
    # Export the data to CSV format
    fieldnames = ['market_slug', 'timestamp', 'outcome', 'price_raw', 'price_prob']
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for point in price_points:
            writer.writerow([point.market_slug, point.timestamp, point.outcome, point.price_raw, point.price_prob])

def generate_report(records: List[MarketRecord], fidelity: int, start_ts: int, end_ts: int, output_path: str) -> None:
    # Generate the report by collecting price points and exporting them
    price_points = []
    for record in records:
        price_points.extend(collect_price_points(record, fidelity, start_ts, end_ts))
    export_csv(records, price_points, output_path)

# Script for running the solution
def main():
    # Define the date range for the analysis (for example, 1 week ago to now)
    start_ts = int(time.time()) - 7 * 86400  # 7 days ago
    end_ts = int(time.time())  # Current time
    
    # Update the market records file path to an absolute path
    markets_csv_path = '/home/trader/polymarket_api/poly_fliter_check/markets_history.csv'  # Updated file path
    records = load_market_records_from_csv(markets_csv_path)
    
    # Define the output CSV file path
    output_path = '/home/trader/polymarket_api/poly_fliter_check/market_price_history_report.csv'  # Output file path
    
    # Run the analysis and export the results
    generate_report(records, fidelity=60, start_ts=start_ts, end_ts=end_ts, output_path=output_path)
    print(f"Report generated and saved to {output_path}")

if __name__ == '__main__':
    main()
