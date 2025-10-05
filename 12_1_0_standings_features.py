#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
12_1_0_standings_features.py
FBref üzerinden Bundesliga 2025-2026 sezonu puan tablosu verisi toplayıcı.
Gelişmiş Metrikler: Home/Away performansı ve xG istatistikleri eklendi.
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
from io import StringIO
import datetime
import os

# ==============================
# 1) FBref'ten veri çekme
# ==============================
STANDINGS_URL = "https://fbref.com/en/comps/20/Bundesliga-Stats"

print("📊 Bundesliga puan tablosu verisi çekiliyor...")

try:
    response = requests.get(STANDINGS_URL)
    response.raise_for_status()
except Exception as e:
    raise ValueError(f"⚠️ FBref sayfasına erişilemedi: {e}")

soup = BeautifulSoup(response.text, "html.parser")

# Tablo ID'lerini bul - mevcut sezon yapısına uygun :cite[1]
overall_table = soup.find("table", {"id": "results2025-2026201_overall"})
home_table = soup.find("table", {"id": "results2025-2026201_home"})
away_table = soup.find("table", {"id": "results2025-2026201_away"})

if overall_table is None:
    raise ValueError("⚠️ Genel puan tablosu bulunamadı. FBref sayfasını kontrol edin!")

# ==============================
# 2) Tabloları DataFrame'e çevir ve birleştir
# ==============================
def parse_table(table, suffix):
    """HTML tablosunu ayrıştır ve sütunlara sonek ekle"""
    if table is None:
        return None
        
    df_temp = pd.read_html(StringIO(str(table)))[0]
    df_temp = df_temp[df_temp["Rk"].notna()]
    
    # Sütun isimlerini düzenle
    rename_cols = {
        "Squad": "team",
        "Pts": f"points_{suffix}",
        "GF": f"goals_for_{suffix}",
        "GA": f"goals_against_{suffix}",
        "GD": f"goal_diff_{suffix}",
        "W": f"wins_{suffix}",
        "D": f"draws_{suffix}", 
        "L": f"losses_{suffix}",
        "MP": f"matches_played_{suffix}",
        "xG": f"xg_{suffix}",
        "xGA": f"xga_{suffix}"
    }
    
    # Sadece mevcut sütunları yeniden adlandır
    existing_cols = {k: v for k, v in rename_cols.items() if k in df_temp.columns}
    df_temp = df_temp.rename(columns=existing_cols)
    
    # Temel sütunları seç
    keep_cols = ["team"] + list(existing_cols.values())
    return df_temp[keep_cols]

# Tüm tabloları ayrıştır
df_overall = parse_table(overall_table, "overall")
df_home = parse_table(home_table, "home")
df_away = parse_table(away_table, "away")

# Tabloları takım ismi üzerinden birleştir
df = df_overall
if df_home is not None:
    df = pd.merge(df, df_home, on="team", how="left")
if df_away is not None:
    df = pd.merge(df, df_away, on="team", how="left")

print("✅ Temel, ev ve deplasman istatistikleri birleştirildi")

# ==============================
# 3) GELİŞMİŞ METRİKLERİ HESAPLA
# ==============================
print("🧮 Gelişmiş metrikler hesaplanıyor...")

# Temel performans metrikleri :cite[1]
df["points_per_game"] = df["points_overall"] / df["matches_played_overall"]
df["goals_per_game"] = df["goals_for_overall"] / df["matches_played_overall"]
df["goals_against_per_game"] = df["goals_against_overall"] / df["matches_played_overall"]

# Ev/Deplasman performans metrikleri
if "points_home" in df.columns and "matches_played_home" in df.columns:
    df["home_points_per_game"] = df["points_home"] / df["matches_played_home"]
    df["home_goals_per_game"] = df["goals_for_home"] / df["matches_played_home"]
    df["home_win_rate"] = df["wins_home"] / df["matches_played_home"]

if "points_away" in df.columns and "matches_played_away" in df.columns:
    df["away_points_per_game"] = df["points_away"] / df["matches_played_away"]
    df["away_goals_per_game"] = df["goals_for_away"] / df["matches_played_away"] 
    df["away_win_rate"] = df["wins_away"] / df["matches_played_away"]

# Ev avantajı metriği
if "home_points_per_game" in df.columns and "away_points_per_game" in df.columns:
    df["home_advantage_ratio"] = df["home_points_per_game"] / df["away_points_per_game"]

# xG bazlı metrikler (eğer mevcutsa) :cite[1]
if "xg_overall" in df.columns:
    df["xg_per_game"] = df["xg_overall"] / df["matches_played_overall"]
    df["xga_per_game"] = df["xga_overall"] / df["matches_played_overall"]
    df["xg_efficiency"] = df["goals_for_overall"] / df["xg_overall"]  # Gol verimliliği

# Form ve istikrar metrikleri
df["win_rate"] = df["wins_overall"] / df["matches_played_overall"]
df["draw_rate"] = df["draws_overall"] / df["matches_played_overall"] 
df["loss_rate"] = df["losses_overall"] / df["matches_played_overall"]

# ==============================
# 4) ZAMAN DAMGASI ve META VERİ EKLE
# ==============================
df["data_extraction_timestamp"] = datetime.datetime.now()
df["season"] = "2025-2026"

print(f"🕐 Zaman damgası eklendi: {df['data_extraction_timestamp'].iloc[0]}")

# ==============================
# 5) KAYDET
# ==============================
# Data klasörünü kontrol et
os.makedirs('data', exist_ok=True)

output_path_csv = "data/bundesliga_standings_features.csv"
output_path_xlsx = "data/bundesliga_standings_features.xlsx"

# CSV kaydet :cite[5]
df.to_csv(output_path_csv, index=False, encoding='utf-8')

# XLSX kaydet
df.to_excel(output_path_xlsx, index=False)

print(f"✅ Bundesliga standings dataset hazır: {output_path_csv}")
print(f"📊 Toplam {len(df)} takım, {len(df.columns)} metrik")

# Özet göster
print("\n📋 İlk 5 takımın özeti:")
summary_cols = ['team', 'points_overall', 'goal_diff_overall', 'points_per_game']
if 'home_points_per_game' in df.columns:
    summary_cols.extend(['home_points_per_game', 'away_points_per_game'])
if 'xg_per_game' in df.columns:
    summary_cols.extend(['xg_per_game', 'xg_efficiency'])

print(df[summary_cols].head().round(2))

print("🚀 İşlem tamamlandı!")