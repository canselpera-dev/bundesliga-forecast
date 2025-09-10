# -*- coding: utf-8 -*-
import requests
import pandas as pd
import sys, io
import os

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")

API_TOKEN = "745aa92502704e82902bdd4cd5df40e4"
headers = {"X-Auth-Token": API_TOKEN}

def fetch_bundesliga_matches(season):
    """Belirtilen sezon için Bundesliga maçlarını API'den çeker"""
    url = f"https://api.football-data.org/v4/competitions/BL1/matches?season={season}&status=FINISHED"
    response = requests.get(url, headers=headers)

    if response.status_code != 200:
        print(f"[!] Hata: {response.status_code} - {response.text}")
        return pd.DataFrame()

    data = response.json().get("matches", [])
    df = pd.json_normalize(data)

    # Result sütununu ekle
    df["result"] = df.apply(
        lambda row: "HomeWin" if row["score.fullTime.home"] > row["score.fullTime.away"]
        else ("AwayWin" if row["score.fullTime.home"] < row["score.fullTime.away"] else "Draw"),
        axis=1
    )
    return df[[
        "id", "utcDate", "matchday",
        "homeTeam.id", "homeTeam.name",
        "awayTeam.id", "awayTeam.name",
        "score.fullTime.home", "score.fullTime.away", "result"
    ]]

# 📂 Çıktı klasörü
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

csv_path = os.path.join(output_dir, "bundesliga_matches_2023_2025.csv")

# Mevcut veriyi oku (varsa)
if os.path.exists(csv_path):
    df_existing = pd.read_csv(csv_path, encoding="utf-8-sig")
    print(f"📂 Mevcut veri yüklendi: {df_existing.shape[0]} maç")
else:
    df_existing = pd.DataFrame()
    print("📂 Mevcut veri bulunamadı, sıfırdan başlıyoruz.")

# ✅ 2023-24, 2024-25 ve 2025-26 sezonlarını çek
df_2023 = fetch_bundesliga_matches(2023)
df_2024 = fetch_bundesliga_matches(2024)
df_2025 = fetch_bundesliga_matches(2025)

# Yeni çekilen veriler
df_new = pd.concat([df_2023, df_2024, df_2025], ignore_index=True)

# 🔹 Yalnızca yeni maçları bul
if not df_existing.empty:
    existing_ids = set(df_existing["id"])
    df_only_new = df_new[~df_new["id"].isin(existing_ids)]
else:
    df_only_new = df_new.copy()

print(f"🆕 Yeni eklenen maç sayısı: {df_only_new.shape[0]}")

# 🔹 Eski + yeni veriyi birleştir (id üzerinden eşsizleştir)
df_all = pd.concat([df_existing, df_only_new], ignore_index=True)
df_all = df_all.drop_duplicates(subset=["id"], keep="last")

print("📊 Güncel toplam maç sayısı:", df_all.shape[0])

# 📌 Güncellenmiş datasetleri kaydet
df_all.to_pickle(os.path.join(output_dir, "bundesliga_matches_2023_2025.pkl"))
df_all.to_parquet(os.path.join(output_dir, "bundesliga_matches_2023_2025.parquet"), index=False)
df_all.to_csv(csv_path, index=False, encoding="utf-8-sig")
df_all.to_excel(os.path.join(output_dir, "bundesliga_matches_2023_2025.xlsx"), index=False, engine="openpyxl")

print("✅ Tüm dosyalar güncellendi.")
