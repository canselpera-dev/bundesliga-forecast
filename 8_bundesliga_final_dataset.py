from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import os
import time
import sys
import io
import unicodedata
import re
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# ------------------------------
# 1. Transfermarkt Takım Verilerini Çek
# ------------------------------
def get_transfermarkt_data():
    URL = "https://www.transfermarkt.com/bundesliga/marktwerteverein/wettbewerb/L1"
    
    options = Options()
    options.add_argument("--headless")
    options.add_argument("--disable-gpu")
    options.add_argument("--no-sandbox")
    options.add_argument("--window-size=1920,1080")
    options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    service = Service()
    driver = webdriver.Chrome(service=service, options=options)
    data = []

    try:
        print("[ℹ] Transfermarkt sayfasına erişiliyor...")
        driver.get(URL)

        # Cookie kabul et
        try:
            WebDriverWait(driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, '//button[contains(., "Accept")] | //button[contains(., "Kabul")] | //button[contains(., "Accept all")]'))
            ).click()
            print("[✔] Cookie kabul edildi")
            time.sleep(2)
        except:
            print("[ℹ] Cookie butonu bulunamadı veya tıklanamadı")

        # Tablonun yüklenmesini bekle
        WebDriverWait(driver, 20).until(
            EC.presence_of_element_located((By.CLASS_NAME, "items"))
        )
        print("[✔] Tablo yüklendi")
        time.sleep(5)  # Ekstra bekleme süresi

        # Sayfa kaynağını al
        soup = BeautifulSoup(driver.page_source, "html.parser")
        
        # Tabloyu bul
        table = soup.find("table", {"class": "items"})
        if not table:
            print("[❌] Tablo bulunamadı!")
            return pd.DataFrame()

        print(f"[ℹ] Tablo bulundu, satırlar işleniyor...")
        
        # Satırları işle
        for row in table.select("tbody tr"):
            cols = row.find_all("td")
            if len(cols) >= 7:
                try:
                    club = cols[1].get_text(strip=True)
                    current_value_str = cols[4].get_text(strip=True)
                    previous_value_str = cols[5].get_text(strip=True)
                    pct_change_str = cols[6].get_text(strip=True)

                    def parse_currency_value(value_str):
                        if not value_str or value_str == "-":
                            return None
                        value_str = value_str.replace("€", "").replace(",", "").strip()
                        if "m" in value_str:
                            return float(value_str.replace("m", "")) * 1_000_000
                        elif "k" in value_str:
                            return float(value_str.replace("k", "")) * 1_000
                        else:
                            try:
                                return float(value_str)
                            except ValueError:
                                return None

                    current_value = parse_currency_value(current_value_str)
                    previous_value = parse_currency_value(previous_value_str)
                    pct_change = None
                    
                    if pct_change_str and "%" in pct_change_str:
                        try:
                            pct_change = float(pct_change_str.replace("%", "").replace("+", "").strip())
                        except:
                            pass

                    # Yaş bilgisini al
                    age_str = cols[3].get_text(strip=True)
                    age = None
                    try:
                        age = float(age_str.replace(",", "."))
                    except:
                        pass

                    if club and current_value is not None:
                        data.append({
                            "club": club,
                            "current_value_eur": current_value,
                            "previous_value_eur": previous_value,
                            "value_change_pct": pct_change,
                            "squad_avg_age": age,
                            "league": "Bundesliga"
                        })
                        print(f"[✔] {club} eklendi: {current_value/1_000_000:.1f}M €")
                except Exception as e:
                    print(f"[❌] Satır işlenirken hata: {e}")
                    continue
                    
    except Exception as e:
        print(f"[❌] Sayfa yüklenirken hata oluştu: {e}")
        return pd.DataFrame()
        
    finally:
        driver.quit()
        print("[ℹ] Browser kapatıldı")

    df_team_values = pd.DataFrame(data)
    
    if not df_team_values.empty:
        df_team_values['absolute_change'] = df_team_values['current_value_eur'] - df_team_values['previous_value_eur']
        df_team_values['log_current_value'] = df_team_values['current_value_eur'].apply(lambda x: round(np.log(x), 2))
        print(f"[✔] {len(df_team_values)} takım verisi başarıyla alındı")
    else:
        print("[❌] Hiç takım verisi alınamadı!")
    
    return df_team_values

# Transfermarkt verilerini al
df_team_values = get_transfermarkt_data()

# Eğer Transfermarkt'tan veri alınamazsa, manuel veri kullan
if df_team_values.empty:
    print("[ℹ] Transfermarkt'tan veri alınamadı, manuel veri kullanılıyor...")
    
    # Manuel takım verileri
    manual_team_data = [
        {"club": "FC Bayern München", "current_value_eur": 980000000, "previous_value_eur": 950000000, "value_change_pct": 3.2, "squad_avg_age": 26.8},
        {"club": "Bayer 04 Leverkusen", "current_value_eur": 620000000, "previous_value_eur": 580000000, "value_change_pct": 6.9, "squad_avg_age": 25.2},
        {"club": "Borussia Dortmund", "current_value_eur": 470000000, "previous_value_eur": 450000000, "value_change_pct": 4.4, "squad_avg_age": 24.9},
        {"club": "RB Leipzig", "current_value_eur": 520000000, "previous_value_eur": 500000000, "value_change_pct": 4.0, "squad_avg_age": 24.1},
        {"club": "Eintracht Frankfurt", "current_value_eur": 280000000, "previous_value_eur": 260000000, "value_change_pct": 7.7, "squad_avg_age": 26.3},
        {"club": "VfB Stuttgart", "current_value_eur": 220000000, "previous_value_eur": 200000000, "value_change_pct": 10.0, "squad_avg_age": 25.7},
        {"club": "Borussia Mönchengladbach", "current_value_eur": 250000000, "previous_value_eur": 240000000, "value_change_pct": 4.2, "squad_avg_age": 26.0},
        {"club": "VfL Wolfsburg", "current_value_eur": 230000000, "previous_value_eur": 220000000, "value_change_pct": 4.5, "squad_avg_age": 25.8},
        {"club": "SC Freiburg", "current_value_eur": 180000000, "previous_value_eur": 170000000, "value_change_pct": 5.9, "squad_avg_age": 26.5},
        {"club": "1. FSV Mainz 05", "current_value_eur": 120000000, "previous_value_eur": 115000000, "value_change_pct": 4.3, "squad_avg_age": 25.9},
        {"club": "TSG 1899 Hoffenheim", "current_value_eur": 190000000, "previous_value_eur": 185000000, "value_change_pct": 2.7, "squad_avg_age": 25.3},
        {"club": "1. FC Union Berlin", "current_value_eur": 150000000, "previous_value_eur": 140000000, "value_change_pct": 7.1, "squad_avg_age": 27.1},
        {"club": "FC Augsburg", "current_value_eur": 130000000, "previous_value_eur": 125000000, "value_change_pct": 4.0, "squad_avg_age": 26.2},
        {"club": "Werder Bremen", "current_value_eur": 110000000, "previous_value_eur": 105000000, "value_change_pct": 4.8, "squad_avg_age": 26.4},
        {"club": "1. FC Köln", "current_value_eur": 100000000, "previous_value_eur": 95000000, "value_change_pct": 5.3, "squad_avg_age": 26.7},
        {"club": "FC St. Pauli", "current_value_eur": 45000000, "previous_value_eur": 40000000, "value_change_pct": 12.5, "squad_avg_age": 26.9},
        {"club": "1. FC Heidenheim 1846", "current_value_eur": 40000000, "previous_value_eur": 35000000, "value_change_pct": 14.3, "squad_avg_age": 27.2},
        {"club": "Hamburger SV", "current_value_eur": 70000000, "previous_value_eur": 65000000, "value_change_pct": 7.7, "squad_avg_age": 26.8}
    ]
    
    df_team_values = pd.DataFrame(manual_team_data)
    df_team_values['league'] = "Bundesliga"
    df_team_values['absolute_change'] = df_team_values['current_value_eur'] - df_team_values['previous_value_eur']
    df_team_values['log_current_value'] = df_team_values['current_value_eur'].apply(lambda x: round(np.log(x), 2))
    print("[✔] Manuel takım verileri oluşturuldu")

print(f"\n[📊] Takım Verileri:\n{df_team_values[['club', 'current_value_eur']]}")

# ------------------------------
# 2. Geliştirilmiş Takım İsimi Normalizasyonu
# ------------------------------
def improved_normalize_name(name):
    if pd.isna(name):
        return None
    
    name = name.lower().strip()
    # Özel karakterleri düzelt
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
    
    # Yaygın önekleri kaldır
    prefixes = ['fc ', '1. ', 'borussia ', 'sv ', 'tsg ', 'sc ', 'vfl ', 'fsv ', '1.']
    for prefix in prefixes:
        name = name.replace(prefix, '')
    
    # Fazla boşlukları temizle
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

# 18 Bundesliga takımı için tam mapping
expanded_mapping = {
    # Bayern Munich
    "bayern munchen": "fc bayern munchen",
    "bayern": "fc bayern munchen",
    "munchen": "fc bayern munchen",
    "fc bayern": "fc bayern munchen",
    
    # Bayer Leverkusen
    "bayer leverkusen": "bayer 04 leverkusen",
    "leverkusen": "bayer 04 leverkusen",
    "bayer 04": "bayer 04 leverkusen",
    
    # Eintracht Frankfurt
    "eintracht frankfurt": "eintracht frankfurt",
    "frankfurt": "eintracht frankfurt",
    "eintracht": "eintracht frankfurt",
    
    # Borussia Dortmund
    "borussia dortmund": "borussia dortmund",
    "dortmund": "borussia dortmund",
    "bvb": "borussia dortmund",
    
    # SC Freiburg
    "freiburg": "sc freiburg",
    "sc freiburg": "sc freiburg",
    
    # Mainz 05
    "mainz 05": "1. fsv mainz 05",
    "mainz": "1. fsv mainz 05",
    "fsv mainz": "1. fsv mainz 05",
    
    # RB Leipzig
    "rb leipzig": "rb leipzig",
    "leipzig": "rb leipzig",
    "rb leipzg": "rb leipzig",
    
    # Werder Bremen
    "werder bremen": "sv werder bremen",
    "bremen": "sv werder bremen",
    "werder": "sv werder bremen",
    
    # Stuttgart
    "vfb stuttgart": "vfb stuttgart",
    "stuttgart": "vfb stuttgart",
    "vfb": "vfb stuttgart",
    
    # Borussia Mönchengladbach
    "monchengladbach": "borussia monchengladbach",
    "gladbach": "borussia monchengladbach",
    "borussia mg": "borussia monchengladbach",
    "mgladbach": "borussia monchengladbach",
    
    # Wolfsburg
    "wolfsburg": "vfl wolfsburg",
    "vfl wolfsburg": "vfl wolfsburg",
    
    # Augsburg
    "augsburg": "fc augsburg",
    "fc augsburg": "fc augsburg",
    
    # Union Berlin
    "union berlin": "1. fc union berlin",
    "union": "1. fc union berlin",
    "fc union": "1. fc union berlin",
    
    # St. Pauli
    "st pauli": "fc st. pauli",
    "pauli": "fc st. pauli",
    "fc st pauli": "fc st. pauli",
    
    # Hoffenheim
    "hoffenheim": "tsg 1899 hoffenheim",
    "tsg hoffenheim": "tsg 1899 hoffenheim",
    "tsg": "tsg 1899 hoffenheim",
    
    # Heidenheim
    "heidenheim": "1. fc heidenheim 1846",
    "heidenheim 1846": "1. fc heidenheim 1846",
    "fc heidenheim": "1. fc heidenheim 1846",
    
    # Köln
    "koln": "1. fc koln",
    "cologne": "1. fc koln",
    "fc koln": "1. fc koln",
    
    # Hamburg
    "hamburger sv": "hamburger sv",
    "hamburg": "hamburger sv",
    "hsv": "hamburger sv"
}

# Transfermarkt verilerini normalize et
df_team_values['club_norm'] = df_team_values['club'].apply(improved_normalize_name)
df_team_values['club_norm'] = df_team_values['club_norm'].replace(expanded_mapping)

# ------------------------------
# 3. Maç Verisini Yükle
# ------------------------------
try:
    matches_path = "data/bundesliga_matches_2023_2025_final_fe.pkl"
    df_matches = pd.read_pickle(matches_path)
    print(f"[✔] Maç verisi yüklendi: {matches_path}, {len(df_matches)} kayıt")
    
    # İlk birkaç maçı göster
    print(f"\n[📋] İlk 5 maç:")
    print(df_matches[['homeTeam.name', 'awayTeam.name', 'utcDate']].head())
    
except FileNotFoundError:
    print(f"[❌] Hata: {matches_path} dosyası bulunamadı!")
    print("[ℹ] Örnek maç verisi oluşturuluyor...")
    
    # Örnek maç verisi oluştur
    sample_matches = [
        {"homeTeam.name": "FC Bayern München", "awayTeam.name": "Borussia Dortmund", "utcDate": "2024-01-15"},
        {"homeTeam.name": "Bayer 04 Leverkusen", "awayTeam.name": "RB Leipzig", "utcDate": "2024-01-16"},
        {"homeTeam.name": "Eintracht Frankfurt", "awayTeam.name": "VfB Stuttgart", "utcDate": "2024-01-17"}
    ]
    df_matches = pd.DataFrame(sample_matches)
    
except Exception as e:
    print(f"[❌] Maç verisi yüklenirken hata oluştu: {e}")
    sys.exit(1)

# Maç verilerini normalize et
df_matches['home_norm'] = df_matches['homeTeam.name'].apply(improved_normalize_name)
df_matches['away_norm'] = df_matches['awayTeam.name'].apply(improved_normalize_name)
df_matches['home_norm'] = df_matches['home_norm'].replace(expanded_mapping)
df_matches['away_norm'] = df_matches['away_norm'].replace(expanded_mapping)

print(f"\n[🔍] Normalize edilmiş takım isimleri:")
print("Home teams:", df_matches['home_norm'].unique())
print("Away teams:", df_matches['away_norm'].unique())

# ------------------------------
# 4. Mapping ve Veri Birleştirme
# ------------------------------
df_team_values_indexed = df_team_values.set_index('club_norm')
df_final = df_matches.copy()

# Takım değerlerini eşleştir
for side in ["home", "away"]:
    df_final[f'{side}_current_value_eur'] = df_final[f'{side}_norm'].map(df_team_values_indexed['current_value_eur'])
    df_final[f'{side}_previous_value_eur'] = df_final[f'{side}_norm'].map(df_team_values_indexed['previous_value_eur'])
    df_final[f'{side}_value_change_pct'] = df_final[f'{side}_norm'].map(df_team_values_indexed['value_change_pct'])
    df_final[f'{side}_squad_avg_age'] = df_final[f'{side}_norm'].map(df_team_values_indexed['squad_avg_age'])
    df_final[f'{side}_absolute_change'] = df_final[f'{side}_norm'].map(df_team_values_indexed['absolute_change'])
    df_final[f'{side}_log_current_value'] = df_final[f'{side}_norm'].map(df_team_values_indexed['log_current_value'])

print(f"\n[📊] Birleştirme sonrası veri boyutu: {df_final.shape}")

# ------------------------------
# 5. NaN Yönetimi
# ------------------------------
print("\n[🔍] NaN Değer Analizi:")
print(df_final.isnull().sum())

# Eksik değerleri doldur
numeric_cols = ['current_value_eur', 'previous_value_eur', 'value_change_pct', 'squad_avg_age', 'absolute_change', 'log_current_value']

for side in ["home", "away"]:
    for col in numeric_cols:
        full_col = f'{side}_{col}'
        if full_col in df_final.columns:
            # Lig ortalaması ile doldur
            league_avg = df_team_values[col].mean()
            df_final[full_col] = df_final[full_col].fillna(league_avg)
            print(f"[ℹ] {full_col} sütunundaki NaN değerler lig ortalaması ile dolduruldu: {league_avg:.2f}")

# Son durumu kontrol et
print(f"\n[🔍] Son NaN Durumu:")
print(df_final.isnull().sum())

# ------------------------------
# 6. Sonuç ve Kaydetme
# ------------------------------
print(f"\n[✔] İşlem tamamlandı!")
print(f"[✔] Toplam kayıt: {len(df_final)}")
print(f"[✔] Toplam sütun: {len(df_final.columns)}")
print(f"[✔] NaN değer sayısı: {df_final.isnull().sum().sum()}")

# Sonucu göster
print(f"\n[📋] İlk 5 kayıt:")
print(df_final.head())

# Kaydet
os.makedirs("data", exist_ok=True)
df_final.to_pickle("data/bundesliga_matches_2023_2025_final_fe_team_values_cleaned.pkl")
df_final.to_csv("data/bundesliga_matches_2023_2025_final_fe_team_values_cleaned.csv", index=False)
df_final.to_excel("data/bundesliga_matches_2023_2025_final_fe_team_values_cleaned.xlsx", index=False)

print(f"\n[💾] Kaydedildi: .pkl, .csv ve .xlsx")

# Eşleşen takımları göster
matched_teams = set(df_team_values['club_norm'])
print(f"\n[📊] Eşleşen {len(matched_teams)} Takım:")
for i, team in enumerate(sorted(matched_teams), 1):
    team_value = df_team_values[df_team_values['club_norm'] == team]['current_value_eur'].values[0]
    print(f"{i:2d}. {team:25s} → {team_value/1_000_000:6.1f}M €")