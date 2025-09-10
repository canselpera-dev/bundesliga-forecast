import pandas as pd
import numpy as np

# ----------------------------
# Dosya yolu
# ----------------------------
input_path = 'data/merged_squad_stats.xlsx'
output_path = 'data/calculate_player_rating.xlsx'

# Excel dosyasını oku
final = pd.read_excel(input_path)

# ----------------------------
# Sabitler
# ----------------------------
standard_rating = 50
STARTER_WEIGHT = 0.7
SUB_WEIGHT = 0.3

# ----------------------------
# Pozisyon grupları
# ----------------------------
def pos_group(pos):
    if pos in ['GK']:
        return 'GK'
    elif pos in ['DF']:
        return 'DF'
    elif pos in ['MF']:
        return 'MF'
    elif pos in ['FW']:
        return 'FW'
    return 'MF'

# ----------------------------
# PlayerRating hesaplama fonksiyonu
# ----------------------------
def calculate_player_rating(row):
    try:
        # Başlangıç rating
        rating = standard_rating

        # Performans puanını hesapla
        if pd.notna(row.get('Gls')) and pd.notna(row.get('Ast')):
            raw_rating = row['Gls']*4 + row['Ast']*3 + row.get('xG',0)*2 + row.get('xAG',0)*1.5
        elif row['Pos'] == 'GK':
            raw_rating = row.get('Min',0)/90
        else:
            raw_rating = 0

        # Starter/Sub ağırlığı uygula
        weight = 1
        if 'Status' in row:
            if row['Status'] == 'Starter':
                weight = STARTER_WEIGHT
            elif row['Status'] == 'Sub':
                weight = SUB_WEIGHT

        # Final rating = standard + weighted performans
        rating += raw_rating * weight

    except:
        rating = standard_rating

    return rating

# ----------------------------
# PlayerRating sütunu ekle
# ----------------------------
final['PlayerRating'] = final.apply(calculate_player_rating, axis=1)

# ----------------------------
# Excel olarak kaydet
# ----------------------------
final.to_excel(output_path, index=False)
print(f"✅ PlayerRating sütunu eklendi ve dosya kaydedildi: {output_path}")

# Kontrol amaçlı ilk 5 satır
print(final.head())
