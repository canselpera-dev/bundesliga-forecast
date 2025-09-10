import os
import pandas as pd

DATA_DIR = "data"
OUTPUT_FILE = "data/bundesliga_complete_dataset.xlsx"

# 1️⃣ Tüm feature’ları ekliyoruz, player bazlı ratingleri de dahil
SELECTED_FEATURES = [
    'Home_AvgRating', 'Away_AvgRating', 'Rating_Diff', 'Total_AvgRating',
    'Home_Form', 'Away_Form', 'Form_Diff', 'IsDerby',
    'homeTeam_GoalsScored_5', 'homeTeam_GoalsConceded_5',
    'awayTeam_GoalsScored_5', 'awayTeam_GoalsConceded_5',
    'homeTeam_Momentum', 'awayTeam_Momentum',
    'Home_GK_Rating', 'Home_DF_Rating', 'Home_MF_Rating', 'Home_FW_Rating',
    'Away_GK_Rating', 'Away_DF_Rating', 'Away_MF_Rating', 'Away_FW_Rating',
    # Player bazlı labellar
    'Player', 'Team', 'Pos', 'PlayerRating'
]

# Mapping: dosyadaki mevcut isim -> eğitim kodunun beklediği isim
column_mapping = {
    'Home_GoalsScored_5': 'homeTeam_GoalsScored_5',
    'Home_GoalsConceded_5': 'homeTeam_GoalsConceded_5',
    'Away_GoalsScored_5': 'awayTeam_GoalsScored_5',
    'Away_GoalsConceded_5': 'awayTeam_GoalsConceded_5',
    'Home_Momentum': 'homeTeam_Momentum',
    'Away_Momentum': 'awayTeam_Momentum'
}

# 2️⃣ Tüm dosyaları oku ve tek bir dictionary içinde sakla
all_data = {}
for file in os.listdir(DATA_DIR):
    if file.endswith((".xlsx", ".xls", ".csv")):
        path = os.path.join(DATA_DIR, file)
        try:
            if file.endswith(".csv"):
                df = pd.read_csv(path)
            else:
                df = pd.read_excel(path)
            df.rename(columns=column_mapping, inplace=True)
            all_data[file] = df
        except Exception as e:
            print(f"Hata {file}: {e}")

# 3️⃣ En büyük veri setini baz al (satır sayısı en çok olan)
base_file = max(all_data.items(), key=lambda x: len(x[1]))[0]
df_base = all_data[base_file].copy()
print(f"📝 Baz alınan dosya: {base_file} ({len(df_base)} satır)")

# 4️⃣ Eksik sütunları diğer dosyalardan doldur
for feat in SELECTED_FEATURES:
    if feat not in df_base.columns:
        filled = False
        for df in all_data.values():
            if feat in df.columns:
                df_base[feat] = df[feat].reindex(df_base.index)
                filled = True
                print(f"✅ {feat} sütunu diğer dosyadan dolduruldu")
                break
        if not filled:
            df_base[feat] = 0  # Eğer hiçbirsinde yoksa 0 ile doldur
            print(f"⚠️ {feat} hiçbir dosyada bulunamadı, 0 ile dolduruldu")

# 5️⃣ Son hali kaydet
df_base.to_excel(OUTPUT_FILE, index=False)
print(f"\n🎯 Tüm feature’lar tamamlandı ve kaydedildi: {OUTPUT_FILE}")
