#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pandas as pd
import requests
from bs4 import BeautifulSoup, Comment
from collections import defaultdict
from io import StringIO

# ---------- SELENIUM ----------
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
import time
# -------------------------------


# ======================================================
# 🔧 1) Selenium Driver
# ======================================================
def create_driver():
    chrome_options = Options()
    chrome_options.add_argument("--headless=new")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage")
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36"
    )

    driver = webdriver.Chrome(
        service=Service(ChromeDriverManager().install()),
        options=chrome_options
    )
    return driver


# ======================================================
# 📊 2) FBref — Takım İstatistikleri (gizli JS yorumlu tablo)
# ======================================================
def get_fbref_team_stats():
    print("📊 Selenium ile FBref verisi çekiliyor...")

    url = "https://fbref.com/en/comps/20/2024-2025/stats/2024-2025-Bundesliga-Stats"

    driver = create_driver()
    driver.get(url)
    time.sleep(3)

    html = driver.page_source
    driver.quit()

    soup = BeautifulSoup(html, "html.parser")

    # tüm yorum bloklarını bul
    comments = soup.find_all(string=lambda text: isinstance(text, Comment))
    tables = []

    for c in comments:
        if "<table" in c and "</table>" in c:
            try:
                df_list = pd.read_html(StringIO(c))
                if df_list:
                    tables.append(df_list[0])
            except:
                pass

    if not tables:
        raise Exception("FBref yorumlarında hiç tablo bulunamadı!")

    # "Squad" kolonunu içeren tabloyu seç
    target_df = None
    for t in tables:
        flat_cols = [str(col).lower() for col in t.columns]
        if any("squad" in col for col in flat_cols):
            target_df = t
            break

    if target_df is None:
        raise Exception("Squad tablosu bulunamadı!")

    df = target_df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)

    df.columns = [c.replace(" ", "").strip() for c in df.columns]

    rename_map = {
        "Squad": "Team",
        "Sh": "Shots",
        "SoT": "ShotsOnTarget",
        "Gls": "Goals",
        "xG": "xG",
        "xGA": "xGA",
        "G+A": "GoalsAssists",
    }
    df = df.rename(columns=rename_map)

    keep_cols = [c for c in ["Team", "Shots", "ShotsOnTarget", "Goals", "xG", "xGA"] if c in df.columns]
    df = df[keep_cols]

    print("✔ FBref başarıyla çekildi.")
    return df


# ======================================================
# 🩼 3) Transfermarkt Sakatlık Verisi
# ======================================================
def get_transfermarkt_injuries():
    print("🩼 Transfermarkt sakatlık verisi çekiliyor...")

    url = "https://www.transfermarkt.com/bundesliga/verletztespieler/wettbewerb/L1"
    res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(res.text, "html.parser")

    table = soup.find("table", {"class": "items"})
    df = pd.read_html(str(table))[0]

    team_col = next(c for c in df.columns if "Club" in str(c))
    player_col = next(c for c in df.columns if "Name" in str(c) or "Player" in str(c))

    injuries = df[[team_col, player_col]].rename(columns={team_col: "Team", player_col: "Player"})
    injuries["InjuryCount"] = 1

    summary = injuries.groupby("Team")["InjuryCount"].sum().reset_index()

    print("✔ Transfermarkt verisi tamam.")
    return summary


# ======================================================
# 📈 4) Football-Data API — Son 5 Form
# ======================================================
def get_team_form(api_token):
    print("📈 Football-Data API form verisi çekiliyor...")

    url = "https://api.football-data.org/v4/competitions/BL1/matches"
    headers = {"X-Auth-Token": api_token}

    res = requests.get(url, headers=headers)
    res.raise_for_status()

    data = res.json()
    team_points = defaultdict(list)

    for m in data.get("matches", []):
        if m.get("status") != "FINISHED":
            continue

        home = m["homeTeam"]["name"]
        away = m["awayTeam"]["name"]
        sH = m["score"]["fullTime"]["home"]
        sA = m["score"]["fullTime"]["away"]

        if sH > sA:
            team_points[home].append(3)
            team_points[away].append(0)
        elif sA > sH:
            team_points[home].append(0)
            team_points[away].append(3)
        else:
            team_points[home].append(1)
            team_points[away].append(1)

    rows = []
    for team, pts in team_points.items():
        last5 = pts[-5:]
        avg = sum(last5) / len(last5) if last5 else 0
        rows.append({"Team": team, "Last5FormPoints": round(avg, 2)})

    print("✔ Football-Data tamam.")
    return pd.DataFrame(rows)


# ======================================================
# 🧩 5) Tüm veri kaynaklarını tek dataset'te birleştir
# ======================================================
def build_final_dataset(api_token):
    fbref = get_fbref_team_stats()
    injuries = get_transfermarkt_injuries()
    form = get_team_form(api_token)

    df = fbref.merge(injuries, on="Team", how="left")
    df = df.merge(form, on="Team", how="left")

    df = df.fillna(0)

    return df


# ======================================================
# 🚀 MAIN
# ======================================================
if __name__ == "__main__":
    API_TOKEN = "745aa92502704e82902bdd4cd5df40e4"

    try:
        final = build_final_dataset(API_TOKEN)
        final.to_excel("data/bundesliga_final_dataset.xlsx", index=False)
        print("✅ Kayıt tamamlandı: data/bundesliga_final_dataset.xlsx")
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")
