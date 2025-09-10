# -*- coding: utf-8 -*-
import pandas as pd
import os

# 📂 Çıktı klasörü
output_dir = "data"

# 1️⃣ Orijinal pickle verisini oku
matches_path = os.path.join(output_dir, "bundesliga_matches_2023_2025.pkl")
df_matches = pd.read_pickle(matches_path)

# 2️⃣ Feature engineering sonrası veriyi oku
final_path = os.path.join(output_dir, "bundesliga_matches_2023_2025_final_fe.pkl")
df_final = pd.read_pickle(final_path)

print("============================================================")
print("📊 Veri Kıyaslama: Orijinal vs Final")
print("------------------------------------------------------------")
print(f"Orijinal veri (pickle) maç sayısı   : {len(df_matches)}")
print(f"Final veri (feature engineered) sayısı: {len(df_final)}")
print("------------------------------------------------------------")

# 3️⃣ Orijinal son 5 maç
print("\n⚽ Orijinal dataset (son 5 maç):")
print(df_matches.sort_values("utcDate").tail(5)[
    ["utcDate","homeTeam.name","awayTeam.name","result"]
])

# 4️⃣ Final son 5 maç
print("\n⚽ Final dataset (son 5 maç):")
print(df_final.sort_values("utcDate").tail(5)[
    ["utcDate","homeTeam.name","awayTeam.name","result"]
])

# 5️⃣ Orijinalde olup finalde olmayan maçları bul
merged = df_matches.merge(
    df_final, on="id", how="left", indicator=True
)
missing = merged[merged["_merge"] == "left_only"]

print("\n❌ Orijinalde olup finalde olmayan maçlar:")
if missing.empty:
    print("YOK ✅ - Tüm maçlar final dataset'te var")
else:
    print(missing[["utcDate_x","homeTeam.name_x","awayTeam.name_x","result_x"]])
print("============================================================")
