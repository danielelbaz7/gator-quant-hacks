"""
Download the model's training sources (no Kalshi data here; see pull_big5.py for that).

    python download_data.py            # both sources
    python download_data.py --tm       # Transfermarkt only
    python download_data.py --fd       # Football-Data only (big 5, via soccerdata)
    python download_data.py --fd-extra # Football-Data CSVs for the extra leagues

Outputs (paths from config.PATHS, all git-ignored):
  transfermarkt-datasets.duckdb   results, goal/card minutes, lineups, player values
                                  (github.com/dcaribou/transfermarkt-datasets, CC0)
  football_data_games.parquet     pre-match odds + kickoff times (football-data.co.uk via soccerdata)
  football_data_extra.parquet     same for the extra leagues (direct CSV download, same column layout)
"""
import argparse
import os

import pandas as pd
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


FD_EXTRA = {"P1": "POR-Liga Portugal", "N1": "NED-Eredivisie"}   # leagues with full ESPN detail


def download_fd_extra(path=PATHS["fd_extra"]):
    import io
    frames = []
    for y in range(SPLIT["tm_start_season"], LAST_SEASON + 1):
        season = f"{y % 100:02d}{(y + 1) % 100:02d}"
        for code, league in FD_EXTRA.items():
            r = requests.get(f"https://www.football-data.co.uk/mmz4281/{season}/{code}.csv", timeout=60)
            if r.status_code != 200:
                continue
            df = pd.read_csv(io.BytesIO(r.content), encoding="latin-1", on_bad_lines="skip")
            df.columns = [c.replace("\ufeff", "").replace("ï»¿", "") for c in df.columns]
            df = df.dropna(subset=["HomeTeam", "AwayTeam", "Date"])
            day = pd.to_datetime(df.Date, dayfirst=True, errors="coerce")
            hm = df["Time"].fillna("12:00").astype(str) if "Time" in df else "12:00"   # 12:00 = no time
            date = pd.to_datetime(day.dt.strftime("%Y-%m-%d") + " " + hm, errors="coerce").fillna(day + pd.Timedelta(hours=12))
            out = df.drop(columns=["Div", "Date", "HomeTeam", "AwayTeam"] + (["Time"] if "Time" in df else []))
            out.insert(0, "date", date)
            out.insert(1, "home_team", df.HomeTeam)
            out.insert(2, "away_team", df.AwayTeam)
            out.insert(0, "season", season)
            out.insert(0, "league", league)
            frames.append(out)
            print(f"  {league} {season}: {len(out)} games")
    fd = pd.concat(frames, ignore_index=True)
    fd = fd[[c for c in fd.columns if not c.startswith("Unnamed")]]
    for c in fd.columns[5:]:
        fd[c] = pd.to_numeric(fd[c], errors="coerce") if fd[c].dtype == object else fd[c]
    fd.to_parquet(path)
    print(f"Football-Data extra leagues -> {path}: {len(fd)} games")


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
    ap.add_argument("--fd-extra", action="store_true")
    a = ap.parse_args()
    both = not (a.tm or a.fd or a.fd_extra)
    if a.tm or both:
        download_tm()
    if a.fd or both:
        download_fd()
    if a.fd_extra or both:
        download_fd_extra()
