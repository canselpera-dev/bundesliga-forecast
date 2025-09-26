#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
8_5_bundesliga_dataset_builder_final.py
Robust ve UTF-8 uyumlu Bundesliga dataset builder.
FBref, Transfermarkt, Football-Data API verilerini tek datasette toplar.
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
from collections import defaultdict

# ==============================
# 1) FBref verisi
# ==============================
def get_fbref_team_stats():
    print("📊 FBref verisi çekiliyor...")
    url = "https://fbref.com/en/comps/20/Bundesliga-Stats"
    res = requests.get(url)
    soup = BeautifulSoup(res.text, "html.parser")
    table = soup.find("table", {"id": "stats_squads_standard_for"})
    df = pd.read_html(str(table))[0]

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)

    # Dinamik sütun eşleme
    cols_map = {}
    for col in df.columns:
        if "Squad" in col:
            cols_map[col] = "Team"
        elif "Sh" in col:
            cols_map[col] = "Shots"
        elif "SoT" in col:
            cols_map[col] = "ShotsOnTarget"
        elif "Gls" in col:
            cols_map[col] = "Goals"
        elif "xG" in col:
            cols_map[col] = "xG"
        elif "xGA" in col:
            cols_map[col] = "xGA"

    df = df.rename(columns=cols_map)
    available_cols = [c for c in ["Team", "Shots", "ShotsOnTarget", "Goals", "xG", "xGA"] if c in df.columns]
    df = df[available_cols]
    return df

# ==============================
# 2) Transfermarkt verisi
# ==============================
def get_transfermarkt_injuries():
    print("🩼 Transfermarkt sakatlık verisi çekiliyor...")
    url = "https://www.transfermarkt.com/bundesliga/verletztespieler/wettbewerb/L1"
    res = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    soup = BeautifulSoup(res.text, "html.parser")
    table = soup.find("table", {"class": "items"})
    df = pd.read_html(str(table))[0]

    print("🔎 Transfermarkt kolonları:", df.columns.tolist())

    # Dinamik kolon seçimi
    club_col = next((c for c in df.columns if "Club" in str(c)), None)
    player_col = next((c for c in df.columns if "Player" in str(c) or "Name" in str(c) or "Player/Position" in str(c)), None)

    if not club_col or not player_col:
        raise Exception(f"Transfermarkt tablosunda uygun Player/Club kolonları yok: {df.columns.tolist()}")

    injuries = df[[club_col, player_col]].copy()
    injuries = injuries.rename(columns={club_col: "Team", player_col: "Player"})
    injuries["InjuryCount"] = 1
    injuries_summary = injuries.groupby("Team")["InjuryCount"].sum().reset_index()
    return injuries_summary

# ==============================
# 3) Football-Data API verisi
# ==============================
def get_team_form(api_token):
    print("📈 Football-Data API'den form verisi çekiliyor...")
    url = "https://api.football-data.org/v4/competitions/BL1/matches"
    headers = {
        "X-Auth-Token": api_token,  # ASCII karakterlerden oluşmalı
        "Content-Type": "application/json; charset=utf-8"
    }
    try:
        res = requests.get(url, headers=headers)
        res.raise_for_status()
    except requests.exceptions.RequestException as e:
        raise Exception(f"Football-Data API hatası: {e}")

    data = res.json()
    team_points = defaultdict(list)

    for match in data.get("matches", []):
        if match.get("status") != "FINISHED":
            continue
        home = match["homeTeam"]["name"]
        away = match["awayTeam"]["name"]
        score_home = match["score"]["fullTime"]["home"]
        score_away = match["score"]["fullTime"]["away"]

        if score_home > score_away:
            team_points[home].append(3)
            team_points[away].append(0)
        elif score_home < score_away:
            team_points[home].append(0)
            team_points[away].append(3)
        else:
            team_points[home].append(1)
            team_points[away].append(1)

    form_data = []
    for team, points in team_points.items():
        last5 = points[-5:] if len(points) >= 5 else points
        avg_points = round(sum(last5)/len(last5), 2) if last5 else 0
        form_data.append({"Team": team, "Last5FormPoints": avg_points})
    return pd.DataFrame(form_data)

# ==============================
# 4) Datasetleri birleştir
# ==============================
def build_final_dataset(api_token):
    fbref = get_fbref_team_stats()
    injuries = get_transfermarkt_injuries()
    form = get_team_form(api_token)
    df = fbref.merge(injuries, on="Team", how="left")
    df = df.merge(form, on="Team", how="left")
    df = df.fillna(0)
    return df

# ==============================
# Main
# ==============================
if __name__ == "__main__":
    API_TOKEN = "745aa92502704e82902bdd4cd5df40e4"  # Token Türkçe karakter içermemeli!
    try:
        dataset = build_final_dataset(API_TOKEN)
        dataset.to_excel("data/bundesliga_final_dataset.xlsx", index=False)
        print("✅ Final dataset hazır ve kaydedildi: data/bundesliga_final_dataset.xlsx")
    except Exception as e:
        print(f"❌ Hata oluştu: {e}")
