import requests
from bs4 import BeautifulSoup, Comment
import pandas as pd
import os
import io
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

# FBref linkleri
FBREF_SEASONS = {
    "2023-24": "https://fbref.com/en/comps/20/2023-2024/stats/2023-2024-Bundesliga-Stats",
    "2024-25": "https://fbref.com/en/comps/20/2024-2025/stats/2024-2025-Bundesliga-Stats",
    "2025-26": "https://fbref.com/en/comps/20/2025-2026/stats/2025-2026-Bundesliga-Stats",         # eklendi
    "2023-24-2BL": "https://fbref.com/en/comps/33/2023-2024/stats/2023-2024-2-Bundesliga-Stats",
    "2024-25-2BL": "https://fbref.com/en/comps/33/2024-2025/stats/2024-2025-2-Bundesliga-Stats",
    "2025-26-2BL": "https://fbref.com/en/comps/33/2025-2026/stats/2025-2026-2-Bundesliga-Stats",     # eklendi
}

# Bundesliga 2’den sadece bu takımlar
SPECIAL_TEAMS_2BL = ["Köln", "Hamburg", "Hamburger SV", "Hamburg SV"]

def fetch_table_from_comments(soup, table_id="stats_standard"):
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    for comment in comments:
        if table_id in comment:
            table_soup = BeautifulSoup(comment, "html.parser")
            table = table_soup.find("table", {"id": table_id})
            if table:
                return table
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
    headers = {"User-Agent": "Mozilla/5.0"}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    table = fetch_table_from_comments(soup)
    if table is None:
        raise ValueError(f"{season_label} için stats_standard tablosu bulunamadı.")

    from io import StringIO
    df = pd.read_html(StringIO(str(table)))[0]

    # MultiIndex varsa düzleştir
    if df.columns.nlevels > 1:
        df.columns = df.columns.droplevel(0)

    # Benzersiz kolon isimleri
    df.columns = make_unique_columns(df.columns)

    # Gereksiz satırları çıkar
    df = df[df["Squad"] != "League Average"]
    df["Team"] = df["Squad"]
    df["Season"] = season_label

    # Ek feature: Goal Contribution
    if "Gls" in df.columns and "Ast" in df.columns:
        df["Goal_Contribution"] = df["Gls"] + df["Ast"]

    if filter_teams:
        df = df[df["Team"].str.contains("|".join(filter_teams), case=False, na=False)]

    return df

def main():
    all_dfs = []

    for season, url in FBREF_SEASONS.items():
        print(f"📥 Fetching {season} stats...")

        # Bundesliga 2 ve özel takımlar filtresi
        filter_teams = SPECIAL_TEAMS_2BL if "2BL" in season else None

        try:
            df = fetch_season_stats(season, url, filter_teams=filter_teams)
            all_dfs.append(df)
        except Exception as e:
            print(f"❌ Error fetching {season}: {e}")

    if all_dfs:
        full_df = pd.concat(all_dfs, ignore_index=True)
        os.makedirs("data", exist_ok=True)
        full_df.to_csv("data/fbref_team_stats_all_seasons.csv", index=False, encoding="utf-8-sig")
        print("[✔] FBref verileri başarıyla kaydedildi.")
        print(full_df[["Season", "Team"]].drop_duplicates())
    else:
        print("[!] Veri çekilemedi.")

if __name__ == "__main__":
    main()
