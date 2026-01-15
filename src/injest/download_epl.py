import io
import time
import pandas as pd
import requests
from pathlib import Path

URL = "https://www.football-data.co.uk/mmz4281/2526/E0.csv"
OUT_PATH = "data/raw/epl_matches.csv"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
    "Accept": "text/csv,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

def download_csv(url: str, retries: int = 5, backoff: float = 1.5) -> bytes:
    last_err = None
    for attempt in range(1, retries + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            resp.raise_for_status()
            return resp.content
        except Exception as e:
            last_err = e
            sleep_s = backoff ** attempt
            print(f"Download failed (attempt {attempt}/{retries}): {e}")
            print(f"Retrying in {sleep_s:.1f}s...\n")
            time.sleep(sleep_s)
    raise RuntimeError(f"Failed to download after {retries} attempts. Last error: {last_err}")

def main():
    print("Downloading EPL CSV...")

    content = download_csv(URL)
    df = pd.read_csv(io.BytesIO(content))

    required = ["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Downloaded file missing columns: {missing}. Found: {df.columns.tolist()}")

    out = df[["Date", "HomeTeam", "AwayTeam", "FTHG", "FTAG"]].copy()
    out = out.rename(columns={
        "Date": "date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG": "home_goals",
        "FTAG": "away_goals",
    })

    out["date"] = pd.to_datetime(out["date"], dayfirst=True, errors="coerce")
    out = out.dropna(subset=["date", "home_team", "away_team", "home_goals", "away_goals"])

    out["home_goals"] = pd.to_numeric(out["home_goals"], errors="coerce")
    out["away_goals"] = pd.to_numeric(out["away_goals"], errors="coerce")
    out = out.dropna(subset=["home_goals", "away_goals"])

    out["home_goals"] = out["home_goals"].astype(int)
    out["away_goals"] = out["away_goals"].astype(int)

    out = (
        out.sort_values("date")
           .drop_duplicates(subset=["date", "home_team", "away_team"], keep="last")
           .reset_index(drop=True)
    )

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT_PATH, index=False)

    print(f"✅ Updated raw file: {OUT_PATH} (rows={len(out)})")
    print(out.tail(3))

if __name__ == "__main__":
    main()
