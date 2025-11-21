from curl_cffi import requests as cf_requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
import os
import io
import sys
import time

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

FBREF_SEASONS = {
    "2023-24": "https://fbref.com/en/comps/20/2023-2024/stats/2023-2024-Bundesliga-Stats",
    "2024-25": "https://fbref.com/en/comps/20/2024-2025/stats/2024-2025-Bundesliga-Stats",
    "2025-26": "https://fbref.com/en/comps/20/2025-2026/stats/2025-2026-Bundesliga-Stats",
    "2023-24-2BL": "https://fbref.com/en/comps/33/2023-2024/stats/2023-2024-2-Bundesliga-Stats",
    "2024-25-2BL": "https://fbref.com/en/comps/33/2024-2025/stats/2024-2025-2-Bundesliga-Stats",
    "2025-26-2BL": "https://fbref.com/en/comps/33/2025-2026/stats/2025-2026-2-Bundesliga-Stats",
}

SPECIAL_TEAMS_2BL = ["Köln", "Hamburg", "Hamburger SV", "Hamburg SV"]


def get_response(url):
    """Chrome TLS fingerprint ile gerçek tarayıcı gibi istek atar (403 çözümü)."""
    time.sleep(1.5)
    resp = cf_requests.get(
        url,
        impersonate="chrome120",   # gerçek Chrome gibi görünür
        headers={
            "Accept-Language": "en-US,en;q=0.9",
            "Referer": "https://google.com",
        }
    )
    resp.raise_for_status()
    return resp


def fetch_table_from_comments(soup, table_id="stats_standard"):
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        if table_id in comment:
            table_soup = BeautifulSoup(comment, "html.parser")
            return table_soup.find("table", {"id": table_id})
    return None


def make_unique_columns(cols):
    seen = {}
    new_cols = []
    for c in cols:
        if c not in seen:
            seen[c] = 0
            new_cols.append(c)
        else:
            seen[c] += 1
            new_cols.append(f"{c}.{seen[c]}")
    return new_cols


def fetch_season_stats(season_label, url, filter_teams=None):
    print(f"🌐 Fetching: {season_label}")

    resp = get_response(url)
    soup = BeautifulSoup(resp.text, "html.parser")

    table = fetch_table_from_comments(soup)
    if table is None:
        raise ValueError(f"{season_label} tablosu bulunamadı!")

    df = pd.read_html(io.StringIO(str(table)))[0]

    if df.columns.nlevels > 1:
        df.columns = df.columns.droplevel(0)

    df.columns = make_unique_columns(df.columns)

    df = df[df["Squad"] != "League Average"]

    df["Team"] = df["Squad"]
    df["Season"] = season_label

    if "Gls" in df.columns and "Ast" in df.columns:
        df["Goal_Contribution"] = df["Gls"] + df["Ast"]

    if filter_teams:
        df = df[df["Team"].str.contains("|".join(filter_teams), case=False, na=False)]

    return df


def main():
    all_dfs = []

    for season, url in FBREF_SEASONS.items():
        print(f"\n📥 Fetching {season}...")

        filter_teams = SPECIAL_TEAMS_2BL if "2BL" in season else None

        try:
            df = fetch_season_stats(season, url, filter_teams)
            all_dfs.append(df)
            print(f"✔ {season} başarıyla çekildi ({len(df)} satır)")
        except Exception as e:
            print(f"❌ Error fetching {season}: {e}")

    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        os.makedirs("data", exist_ok=True)
        full_df.to_csv("data/fbref_team_stats_all_seasons.csv", index=False, encoding="utf-8-sig")
        print("\n[✔] Tüm veriler başarıyla kaydedildi.")
        print(full_df[["Season", "Team"]].drop_duplicates())
    else:
        print("\n[!] Veri çekilemedi.")


if __name__ == "__main__":
    main()
