import os
import pandas as pd

data_folder = "data"

# Normalizasyon fonksiyonu (örnek)
def normalize_label(val):
    if pd.isna(val):
        return val
    return val.strip().lower().replace(" ", "_")  # kendi normalizasyon kurallarına göre değiştir

# Tüm dosyaları kontrol et
for file_name in os.listdir(data_folder):
    file_path = os.path.join(data_folder, file_name)
    
    # Sadece Excel ve CSV dosyalarını al
    if not (file_path.endswith(".csv") or file_path.endswith(".xlsx")):
        continue

    # Dosyayı oku
    if file_path.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    # Sadece 'Player' sütunu varsa normalize et
    if 'Player' in df.columns:
        df['Player'] = df['Player'].apply(normalize_label)
        print(f"{file_name} dosyasında normalize edildi.")
        
        # Dosyayı tekrar kaydet
        if file_path.endswith(".csv"):
            df.to_csv(file_path, index=False)
        else:
            df.to_excel(file_path, index=False)
    else:
        print(f"{file_name} dosyasında 'Player' sütunu yok, atlandı.")
