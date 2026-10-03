"""
Download the model's training sources (no Kalshi data here; see pull_big5.py for that).

    python download_data.py            # both sources
    python download_data.py --tm       # Transfermarkt only
    python download_data.py --fd       # Football-Data only

Outputs (paths from config.PATHS, all git-ignored):
  transfermarkt-datasets.duckdb   results, goal/card minutes, lineups, player values
                                  (github.com/dcaribou/transfermarkt-datasets, CC0)
  football_data_games.parquet     pre-match odds + kickoff times (football-data.co.uk via soccerdata)
"""
import argparse
import os

import requests

from config import PATHS, SPLIT

TM_URL = "https://pub-e682421888d945d684bcae8890b0ec20.r2.dev/data/transfermarkt-datasets.duckdb"
FD_LEAGUES = ["ENG-Premier League", "ESP-La Liga", "GER-Bundesliga", "ITA-Serie A", "FRA-Ligue 1"]
LAST_SEASON = 2026   # season starting in this year (2026/27)


def download_tm(path=PATHS["tm_duckdb"]):
    print(f"Transfermarkt -> {path}")
    tmp = path + ".part"
    with requests.get(TM_URL, stream=True, timeout=60) as r:
        r.raise_for_status()
        total, done = int(r.headers.get("content-length", 0)), 0
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
                done += len(chunk)
                if total:
                    print(f"\r  {done / 1e6:.0f}/{total / 1e6:.0f} MB", end="")
    os.replace(tmp, path)
    print()


def download_fd(path=PATHS["fd_games"]):
    import soccerdata as sd
    seasons = [f"{y % 100:02d}{(y + 1) % 100:02d}" for y in range(SPLIT["tm_start_season"], LAST_SEASON + 1)]
    print(f"Football-Data {seasons[0]}..{seasons[-1]} -> {path}")
    df = sd.MatchHistory(leagues=FD_LEAGUES, seasons=seasons).read_games()
    df.to_parquet(path)
    print(f"  {len(df)} games; columns include: {[c for c in df.columns][:12]} ...")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tm", action="store_true")
    ap.add_argument("--fd", action="store_true")
    a = ap.parse_args()
    both = not (a.tm or a.fd)
    if a.tm or both:
        download_tm()
    if a.fd or both:
        download_fd()
