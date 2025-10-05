#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_pipeline.py
==================================
Bundesliga Tahmin Sistemi için tam pipeline.
1) Veri toplama ve feature engineering
2) Dataset oluşturma
3) Model eğitimi
4) Streamlit arayüzünü başlatma
"""

import subprocess

# Çalıştırılacak Python scriptleri sırasıyla
scripts = [
    "1_bundesliga_matches_2023_2025.py",
    "2_bundesliga2_updade.py",
    "3_bundesliga_feature_pipeline.py",
    "4_transfermakts_scraper_team_value.py",
    "5_fbref_scraper.py",
    "6_curent_bundesliga_players2.py",
    "7_bundesliga_mapping_pipeline.py",
    "8_1_bundesliga_dataset_builder.py",
    "8_bundesliga_final_dataset.py",
    "9_new_model_training.py"
]

def run_scripts():
    for script in scripts:
        print(f"\n🚀 Çalıştırılıyor: {script}")
        subprocess.run(["python", script], check=True)

    print("\n✅ Tüm pipeline başarıyla tamamlandı!")
    print("🚀 Şimdi Streamlit arayüzü başlatılıyor...")

    # Streamlit uygulamasını başlat
    subprocess.run(["streamlit", "run", "app.py"], check=True)


if __name__ == "__main__":
    run_scripts()
