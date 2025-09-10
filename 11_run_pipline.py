#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
run_pipeline.py
Tüm Bundesliga tahminleme pipeline'ını sırayla çalıştırır.
"""

import subprocess
import sys

# Çalışma sırasına göre dosyalar
pipeline_steps = [
    "1_bundesliga_matches_2023_2025.py",
    "2_bundesliga2_uptade.py",
    "3_bundesliga_feature_pipeline.py",
    "4_transfermatks_scraper_team_value.py",
    "5_fbref_scraper.py",
    "6_curent_bundesliga_players2.py",
    "7_bundesliga_mapping_pipeline.py",
    "8_new_model_training.py",
    "9_prediction_woking.py",
]

def run_step(step):
    print("=" * 70)
    print(f"🚀 Çalıştırılıyor: {step}")
    print("=" * 70)
    try:
        subprocess.run([sys.executable, step], check=True)
        print(f"✅ Tamamlandı: {step}\n")
    except subprocess.CalledProcessError as e:
        print(f"❌ HATA: {step} dosyasında sorun oluştu!")
        sys.exit(1)

def main():
    print("🏆 Bundesliga Tahminleme Pipeline Başlatılıyor...\n")
    for step in pipeline_steps:
        run_step(step)
    print("🎯 Pipeline başarıyla tamamlandı!")

if __name__ == "__main__":
    main()
