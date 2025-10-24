import os
import sys
import requests

# World Bank connector
# Fetches an indicator time series for a country and date range
# Writes a readable .txt file with year → value

# Example:
# python worldbank.py BRA SP.POP.TOTL 2018:2024 output_brazil.txt
#
# Dependencies:
#   pip install requests

BASE_URL = "https://api.worldbank.org/v2"


def fetch_indicator(country_code, indicator_code, date_range="2018:2024", per_page=1000):
    url = f"{BASE_URL}/country/{country_code}/indicator/{indicator_code}"
    params = {"date": date_range, "format": "json", "per_page": per_page}
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    # API returns [metadata, data_rows]
    if not isinstance(data, list) or len(data) < 2:
        return [], {"country": country_code, "indicator": indicator_code, "date_range": date_range, "count": 0}

    meta = data[0] or {}
    rows = data[1] or []

    series = []
    for row in rows:
        series.append({
            "year": row.get("date"),
            "value": row.get("value"),
            "country": (row.get("country") or {}).get("value"),
            "indicator": (row.get("indicator") or {}).get("value"),
        })

    info = {
        "country": country_code,
        "indicator": indicator_code,
        "date_range": date_range,
        "count": len(series),
        "pages": meta.get("pages"),
    }
    return series, info


def to_text(series, meta):
    def line(title): return f"{title}\n{'=' * len(title)}"

    output = [
        line("World Bank — Indicator Data"),
        "",
        "Query Info:",
        f"- Country: {meta.get('country')}",
        f"- Indicator: {meta.get('indicator')}",
        f"- Date range: {meta.get('date_range')}",
        f"- Total rows: {meta.get('count')}",
        "",
        line("Values (year → value)"),
    ]

    if not series:
        output.append("(no data found)")
    else:
        for s in sorted(series, key=lambda x: (x.get('year') or "")):
            output.append(f"- {s.get('year')}: {s.get('value')}")

    return "\n".join(output) + "\n"


def write_txt(path, text):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text)


def main(args=None):
    args = args or sys.argv[1:]
    country = args[0] if len(args) > 0 else "BRA"
    indicator = args[1] if len(args) > 1 else "SP.POP.TOTL"
    date_range = args[2] if len(args) > 2 else "2018:2024"
    out_path = args[3] if len(args) > 3 else "worldbank_output.txt"

    data, meta = fetch_indicator(country, indicator, date_range)
    text = to_text(data, meta)
    write_txt(out_path, text)
    print(f"Saved {out_path} ({meta.get('count', 0)} rows)")


if __name__ == "__main__":
    main()

