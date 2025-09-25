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
        time.sleep(5)

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
# 2. Bundesliga Final Dataset Verilerini Yükle
# ------------------------------
def load_bundesliga_final_dataset():
    """Bundesliga final dataset dosyasını yükler ve işler"""
    try:
        dataset_path = "data/bundesliga_final_dataset.xlsx"
        if not os.path.exists(dataset_path):
            dataset_path = "bundesliga_final_dataset.xlsx"
            
        df_final_dataset = pd.read_excel(dataset_path)
        print(f"[✔] Bundesliga final dataset yüklendi: {dataset_path}")
        print(f"[📊] Dataset boyutu: {df_final_dataset.shape}")
        print(f"[📋] Sütunlar: {list(df_final_dataset.columns)}")
        
        print(f"\n[📋] Bundesliga Final Dataset İçeriği:")
        print(df_final_dataset.head(18))
        
        return df_final_dataset
        
    except Exception as e:
        print(f"[❌] Bundesliga final dataset yüklenirken hata: {e}")
        return None

# Bundesliga final dataset'i yükle
df_bundesliga_final = load_bundesliga_final_dataset()

# ------------------------------
# 3. Geliştirilmiş Takım İsimi Normalizasyonu
# ------------------------------
def improved_normalize_name(name):
    if pd.isna(name):
        return None
    
    name = name.lower().strip()
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
    
    prefixes = ['fc ', '1. ', 'borussia ', 'sv ', 'tsg ', 'sc ', 'vfl ', 'fsv ', '1.']
    for prefix in prefixes:
        name = name.replace(prefix, '')
    
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

# 18 Bundesliga takımı için tam mapping
expanded_mapping = {
    "bayern munchen": "fc bayern munchen",
    "bayern": "fc bayern munchen",
    "munchen": "fc bayern munchen",
    "fc bayern": "fc bayern munchen",
    "bayern munich": "fc bayern munchen",
    
    "bayer leverkusen": "bayer 04 leverkusen",
    "leverkusen": "bayer 04 leverkusen",
    "bayer 04": "bayer 04 leverkusen",
    
    "eintracht frankfurt": "eintracht frankfurt",
    "frankfurt": "eintracht frankfurt",
    "eintracht": "eintracht frankfurt",
    "eint frankfurt": "eintracht frankfurt",
    
    "borussia dortmund": "borussia dortmund",
    "dortmund": "borussia dortmund",
    "bvb": "borussia dortmund",
    
    "freiburg": "sc freiburg",
    "sc freiburg": "sc freiburg",
    
    "mainz 05": "1. fsv mainz 05",
    "mainz": "1. fsv mainz 05",
    "fsv mainz": "1. fsv mainz 05",
    "fmainz 05": "1. fsv mainz 05",
    
    "rb leipzig": "rb leipzig",
    "leipzig": "rb leipzig",
    "rb leipzg": "rb leipzig",
    
    "werder bremen": "sv werder bremen",
    "bremen": "sv werder bremen",
    "werder": "sv werder bremen",
    
    "vfb stuttgart": "vfb stuttgart",
    "stuttgart": "vfb stuttgart",
    "vfb": "vfb stuttgart",
    
    "monchengladbach": "borussia monchengladbach",
    "gladbach": "borussia monchengladbach",
    "borussia mg": "borussia monchengladbach",
    "mgladbach": "borussia monchengladbach",
    "borussia monchengladbach": "borussia monchengladbach",
    
    "wolfsburg": "vfl wolfsburg",
    "vfl wolfsburg": "vfl wolfsburg",
    
    "augsburg": "fc augsburg",
    "fc augsburg": "fc augsburg",
    
    "union berlin": "1. fc union berlin",
    "union": "1. fc union berlin",
    "fc union": "1. fc union berlin",
    
    "st pauli": "fc st. pauli",
    "pauli": "fc st. pauli",
    "fc st pauli": "fc st. pauli",
    "st. pauli": "fc st. pauli",
    
    "hoffenheim": "tsg 1899 hoffenheim",
    "tsg hoffenheim": "tsg 1899 hoffenheim",
    "tsg": "tsg 1899 hoffenheim",
    "1899 hoffenheim": "tsg 1899 hoffenheim",
    
    "heidenheim": "1. fc heidenheim 1846",
    "heidenheim 1846": "1. fc heidenheim 1846",
    "fc heidenheim": "1. fc heidenheim 1846",
    
    "koln": "1. fc koln",
    "cologne": "1. fc koln",
    "fc koln": "1. fc koln",
    "1. fc koln": "1. fc koln",
    
    "hamburger sv": "hamburger sv",
    "hamburg": "hamburger sv",
    "hsv": "hamburger sv"
}

# Transfermarkt verilerini normalize et
df_team_values['club_norm'] = df_team_values['club'].apply(improved_normalize_name)
df_team_values['club_norm'] = df_team_values['club_norm'].replace(expanded_mapping)

# Bundesliga final dataset'i normalize et
if df_bundesliga_final is not None:
    df_bundesliga_final['Team_norm'] = df_bundesliga_final['Team'].apply(improved_normalize_name)
    df_bundesliga_final['Team_norm'] = df_bundesliga_final['Team_norm'].replace(expanded_mapping)
    print(f"\n[🔍] Normalize edilmiş Bundesliga Final Dataset takımları:")
    print(df_bundesliga_final[['Team', 'Team_norm']].head(18))

# ------------------------------
# 4. Maç Verisini Yükle
# ------------------------------
try:
    matches_path = "data/bundesliga_matches_2023_2025_final_fe.pkl"
    df_matches = pd.read_pickle(matches_path)
    print(f"[✔] Maç verisi yüklendi: {matches_path}, {len(df_matches)} kayıt")
    
    print(f"\n[📋] İlk 5 maç:")
    print(df_matches[['homeTeam.name', 'awayTeam.name', 'utcDate']].head())
    
except FileNotFoundError:
    print(f"[❌] Hata: {matches_path} dosyası bulunamadı!")
    sys.exit(1)
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
# 5. Geliştirilmiş H2H Feature Engineering
# ------------------------------
def calculate_h2h_features(df):
    print("[ℹ] H2H özellikleri hesaplanıyor...")
    
    df = df.sort_values('utcDate')
    
    h2h_features = [
        'h2h_home_wins', 'h2h_away_wins', 'h2h_draws', 
        'h2h_home_goals', 'h2h_away_goals', 'h2h_matches_count'
    ]
    
    for feature in h2h_features:
        df[feature] = 0
    
    for idx, row in df.iterrows():
        home_team = row['home_norm']
        away_team = row['away_norm']
        current_date = row['utcDate']
        
        past_matches = df[
            (df['utcDate'] < current_date) & 
            (((df['home_norm'] == home_team) & (df['away_norm'] == away_team)) | 
             ((df['home_norm'] == away_team) & (df['away_norm'] == home_team)))
        ]
        
        if len(past_matches) > 0:
            home_wins = len(past_matches[
                ((past_matches['home_norm'] == home_team) & (past_matches['result'] == 'H')) |
                ((past_matches['away_norm'] == home_team) & (past_matches['result'] == 'A'))
            ])
            
            away_wins = len(past_matches[
                ((past_matches['home_norm'] == away_team) & (past_matches['result'] == 'H')) |
                ((past_matches['away_norm'] == away_team) & (past_matches['result'] == 'A'))
            ])
            
            draws = len(past_matches[past_matches['result'] == 'D'])
            
            home_goals = 0
            away_goals = 0
            
            for _, match in past_matches.iterrows():
                if match['home_norm'] == home_team:
                    home_goals += match.get('score.fullTime.home', 0)
                    away_goals += match.get('score.fullTime.away', 0)
                else:
                    home_goals += match.get('score.fullTime.away', 0)
                    away_goals += match.get('score.fullTime.home', 0)
            
            df.at[idx, 'h2h_home_wins'] = home_wins
            df.at[idx, 'h2h_away_wins'] = away_wins
            df.at[idx, 'h2h_draws'] = draws
            df.at[idx, 'h2h_home_goals'] = home_goals
            df.at[idx, 'h2h_away_goals'] = away_goals
            df.at[idx, 'h2h_matches_count'] = len(past_matches)
    
    df['h2h_win_ratio'] = df.apply(
        lambda x: x['h2h_home_wins'] / x['h2h_matches_count'] if x['h2h_matches_count'] > 0 else 0.5, 
        axis=1
    )
    df['h2h_goal_difference'] = df['h2h_home_goals'] - df['h2h_away_goals']
    df['h2h_avg_goals'] = df.apply(
        lambda x: (x['h2h_home_goals'] + x['h2h_away_goals']) / x['h2h_matches_count'] if x['h2h_matches_count'] > 0 else 2.5, 
        axis=1
    )
    
    return df

df_matches = calculate_h2h_features(df_matches)

# ------------------------------
# 6. Form Özelliklerini Geliştir
# ------------------------------
def improve_form_features(df):
    print("[ℹ] Form özellikleri geliştiriliyor...")
    
    form_columns = ['home_form', 'away_form']
    for col in form_columns:
        if col in df.columns:
            df[col] = df[col].fillna(0)
            df[col] = df[col].apply(lambda x: max(0, x))
    
    return df

df_matches = improve_form_features(df_matches)

# ------------------------------
# 7. isDerby Özelliği Ekleme
# ------------------------------
def add_derby_feature(df):
    print("[ℹ] Derby özelliği ekleniyor...")
    
    derby_matches = [
        ("fc bayern munchen", "borussia dortmund"),
        ("1. fc union berlin", "hertha berlin"),
        ("1. fc koln", "borussia monchengladbach"),
        ("1. fc koln", "bayer 04 leverkusen"),
        ("borussia monchengladbach", "bayer 04 leverkusen"),
        ("vfb stuttgart", "sc freiburg"),
        ("vfb stuttgart", "tsg 1899 hoffenheim"),
        ("sc freiburg", "tsg 1899 hoffenheim"),
        ("borussia dortmund", "vfl bochum"),
        ("borussia dortmund", "bayer 04 leverkusen"),
        ("sv werder bremen", "vfl wolfsburg")
    ]
    
    df['isDerby'] = 0
    df['derbyType'] = "Normal"
    
    for idx, row in df.iterrows():
        home_team = row['home_norm']
        away_team = row['away_norm']
        
        for derby in derby_matches:
            if (home_team == derby[0] and away_team == derby[1]) or \
               (home_team == derby[1] and away_team == derby[0]):
                df.at[idx, 'isDerby'] = 1
                df.at[idx, 'derbyType'] = f"{derby[0]} vs {derby[1]}"
                break
    
    return df

df_matches = add_derby_feature(df_matches)

# ------------------------------
# 8. Geliştirilmiş Veri Birleştirme
# ------------------------------
def improved_data_merging(df_matches, df_team_values, df_bundesliga_final):
    print("[ℹ] Geliştirilmiş veri birleştirme işlemi...")
    
    df_final = df_matches.copy()
    df_team_values_indexed = df_team_values.set_index('club_norm')
    
    for side in ["home", "away"]:
        df_final[f'{side}_current_value_eur'] = df_final[f'{side}_norm'].map(df_team_values_indexed['current_value_eur'])
        df_final[f'{side}_previous_value_eur'] = df_final[f'{side}_norm'].map(df_team_values_indexed['previous_value_eur'])
        df_final[f'{side}_value_change_pct'] = df_final[f'{side}_norm'].map(df_team_values_indexed['value_change_pct'])
        df_final[f'{side}_squad_avg_age'] = df_final[f'{side}_norm'].map(df_team_values_indexed['squad_avg_age'])
        df_final[f'{side}_absolute_change'] = df_final[f'{side}_norm'].map(df_team_values_indexed['absolute_change'])
        df_final[f'{side}_log_current_value'] = df_final[f'{side}_norm'].map(df_team_values_indexed['log_current_value'])

    if df_bundesliga_final is not None:
        df_bundesliga_final_indexed = df_bundesliga_final.set_index('Team_norm')
        
        for side in ["home", "away"]:
            df_final[f'{side}_goals'] = df_final[f'{side}_norm'].map(df_bundesliga_final_indexed['Goals'])
            df_final[f'{side}_xg'] = df_final[f'{side}_norm'].map(df_bundesliga_final_indexed['xG'])
            df_final[f'{side}_injury_count'] = df_final[f'{side}_norm'].map(df_bundesliga_final_indexed['InjuryCount'])
            df_final[f'{side}_last5_form_points'] = df_final[f'{side}_norm'].map(df_bundesliga_final_indexed['Last5FormPoints'])
        
        print("[✔] Bundesliga final dataset verileri başarıyla eklendi")
    else:
        for side in ["home", "away"]:
            df_final[f'{side}_goals'] = np.nan
            df_final[f'{side}_xg'] = np.nan
            df_final[f'{side}_injury_count'] = np.nan
            df_final[f'{side}_last5_form_points'] = np.nan
        
        print("[ℹ] Bundesliga final dataset yüklenemedi, sütunlar NaN olarak eklendi")

    return df_final

df_final = improved_data_merging(df_matches, df_team_values, df_bundesliga_final)
print(f"\n[📊] Birleştirme sonrası veri boyutu: {df_final.shape}")

# ------------------------------
# 9. Geliştirilmiş NaN Yönetimi
# ------------------------------
def improved_nan_management(df_final, df_team_values, df_bundesliga_final):
    print("\n[🔍] NaN Değer Analizi:")
    print(df_final.isnull().sum())

    numeric_cols = ['current_value_eur', 'previous_value_eur', 'value_change_pct', 
                   'squad_avg_age', 'absolute_change', 'log_current_value']

    for side in ["home", "away"]:
        for col in numeric_cols:
            full_col = f'{side}_{col}'
            if full_col in df_final.columns:
                league_avg = df_team_values[col].mean()
                df_final[full_col] = df_final[full_col].fillna(league_avg)
                print(f"[ℹ] {full_col} sütunundaki NaN değerler lig ortalaması ile dolduruldu: {league_avg:.2f}")

    if df_bundesliga_final is not None:
        bundesliga_cols = ['Goals', 'xG', 'InjuryCount', 'Last5FormPoints']
        for side in ["home", "away"]:
            for col in bundesliga_cols:
                full_col = f'{side}_{col.lower().replace("count", "").replace("points", "").replace(" ", "_")}'
                if full_col in df_final.columns:
                    if col in df_bundesliga_final.columns:
                        league_avg = df_bundesliga_final[col].mean()
                        df_final[full_col] = df_final[full_col].fillna(league_avg)
                        print(f"[ℹ] {full_col} sütunundaki NaN değerler lig ortalaması ile dolduruldu: {league_avg:.2f}")

    h2h_cols = ['h2h_home_wins', 'h2h_away_wins', 'h2h_draws', 'h2h_home_goals', 
                'h2h_away_goals', 'h2h_matches_count', 'h2h_win_ratio', 
                'h2h_goal_difference', 'h2h_avg_goals']

    for col in h2h_cols:
        if col in df_final.columns:
            if 'win_ratio' in col:
                df_final[col] = df_final[col].fillna(0.5)
            elif 'avg_goals' in col:
                df_final[col] = df_final[col].fillna(2.5)
            else:
                df_final[col] = df_final[col].fillna(0)

    form_cols = ['home_form', 'away_form', 'home_last5_form_points', 'away_last5_form_points']
    for col in form_cols:
        if col in df_final.columns:
            df_final[col] = df_final[col].fillna(0)

    injury_cols = ['home_injury_count', 'away_injury_count']
    for col in injury_cols:
        if col in df_final.columns:
            df_final[col] = df_final[col].fillna(0)

    return df_final

df_final = improved_nan_management(df_final, df_team_values, df_bundesliga_final)

print(f"\n[🔍] Son NaN Durumu:")
print(df_final.isnull().sum())

# ------------------------------
# 10. Geliştirilmiş Ek Özellik Mühendisliği
# ------------------------------
def create_improved_features(df):
    print("[ℹ] Geliştirilmiş ek özellikler oluşturuluyor...")
    
    df['value_difference'] = df['home_current_value_eur'] - df['away_current_value_eur']
    df['value_ratio'] = df['home_current_value_eur'] / df['away_current_value_eur'].replace(0, 1)
    
    if 'home_goals' in df.columns and 'away_goals' in df.columns:
        df['goals_difference'] = df['home_goals'] - df['away_goals']
        df['goals_ratio'] = df['home_goals'] / df['away_goals'].replace(0, 0.1)
    
    if 'home_xg' in df.columns and 'away_xg' in df.columns:
        df['xg_difference'] = df['home_xg'] - df['away_xg']
        df['xg_ratio'] = df['home_xg'] / df['away_xg'].replace(0, 0.1)
    
    if 'home_last5_form_points' in df.columns and 'away_last5_form_points' in df.columns:
        df['form_difference'] = df['home_last5_form_points'] - df['away_last5_form_points']
        df['form_difference'] = df['form_difference'].fillna(0)
    
    if 'home_injury_count' in df.columns and 'away_injury_count' in df.columns:
        df['injury_difference'] = df['home_injury_count'] - df['away_injury_count']
        df['injury_difference'] = df['injury_difference'].fillna(0)
    
    df['age_difference'] = df['home_squad_avg_age'] - df['away_squad_avg_age']
    
    df['home_power_index'] = (df['home_log_current_value'] * 0.7) + (df['home_last5_form_points'] * 0.3)
    df['away_power_index'] = (df['away_log_current_value'] * 0.7) + (df['away_last5_form_points'] * 0.3)
    df['power_difference'] = df['home_power_index'] - df['away_power_index']
    
    df['performance_ratio'] = df['home_goals'] / df['home_xg'].replace(0, 0.1)
    
    return df

df_final = create_improved_features(df_final)

# ------------------------------
# 11. Son Kontroller ve Kaydetme
# ------------------------------
print(f"\n[✔] İşlem tamamlandı!")
print(f"[✔] Toplam kayıt: {len(df_final)}")
print(f"[✔] Toplam sütun: {len(df_final.columns)}")
print(f"[✔] NaN değer sayısı: {df_final.isnull().sum().sum()}")

print(f"\n[🔍] Detaylı NaN Analizi:")
nan_summary = df_final.isnull().sum()
nan_columns = nan_summary[nan_summary > 0]
if len(nan_columns) > 0:
    print("NaN içeren sütunlar:")
    for col, count in nan_columns.items():
        print(f"  {col}: {count} NaN ({count/len(df_final)*100:.1f}%)")
else:
    print("✅ Hiç NaN değer kalmadı!")

print(f"\n[📋] İlk 5 kayıt:")
print(df_final.head())

print(f"\n[📊] Tüm Sütunlar ({len(df_final.columns)} adet):")
for i, col in enumerate(sorted(df_final.columns), 1):
    print(f"{i:2d}. {col}")

# Kaydet - DEĞİŞTİRİLEN KISIM
os.makedirs("data", exist_ok=True)
output_files = [
    "data/bundesliga_matches_2023_2025_final_fe_team_values_cleaned.pkl",
    "data/bundesliga_matches_2023_2025_final_fe_team_values_cleaned.csv",
    "data/bundesliga_matches_2023_2025_final_fe_team_values_cleaned.xlsx"
]

for file_path in output_files:
    try:
        if file_path.endswith('.pkl'):
            df_final.to_pickle(file_path)
        elif file_path.endswith('.csv'):
            df_final.to_csv(file_path, index=False, encoding='utf-8-sig')
        elif file_path.endswith('.xlsx'):
            df_final.to_excel(file_path, index=False)
        print(f"[💾] Kaydedildi: {file_path}")
    except Exception as e:
        print(f"[❌] {file_path} kaydedilirken hata: {e}")

# ------------------------------
# 12. İstatistiksel Özet
# ------------------------------
print(f"\n[📈] İSTATİSTİKSEL ÖZET")

matched_teams = set(df_team_values['club_norm'])
print(f"\n[📊] Eşleşen {len(matched_teams)} Takım:")
for i, team in enumerate(sorted(matched_teams), 1):
    team_value = df_team_values[df_team_values['club_norm'] == team]['current_value_eur'].values[0]
    print(f"{i:2d}. {team:25s} → {team_value/1_000_000:6.1f}M €")

if df_bundesliga_final is not None:
    print(f"\n[📊] Bundesliga Final Dataset Takımları:")
    for i, row in df_bundesliga_final.iterrows():
        team_norm = row.get('Team_norm', 'Bilinmiyor')
        goals = row.get('Goals', 0)
        xg = row.get('xG', 0)
        injury = row.get('InjuryCount', 0)
        form = row.get('Last5FormPoints', 0)
        print(f"{i+1:2d}. {team_norm:25s} → G:{goals}, xG:{xg:.1f}, Inj:{injury}, Form:{form}")

derby_count = df_final['isDerby'].sum()
print(f"\n[⚽] Derby Maç Sayısı: {derby_count}")
if derby_count > 0:
    print("\n[📊] Derby Türleri:")
    print(df_final[df_final['isDerby'] == 1]['derbyType'].value_counts())

print(f"\n[📈] ÖZELLİK İSTATİSTİKLERİ:")
important_features = [
    'value_difference', 'goals_difference', 'xg_difference', 
    'form_difference', 'power_difference', 'h2h_win_ratio',
    'home_power_index', 'away_power_index'
]

for feature in important_features:
    if feature in df_final.columns:
        print(f"{feature:20s} → Min: {df_final[feature].min():7.2f}, Max: {df_final[feature].max():7.2f}, Mean: {df_final[feature].mean():7.2f}")

total_cells = df_final.shape[0] * df_final.shape[1]
nan_cells = df_final.isnull().sum().sum()
data_quality = ((total_cells - nan_cells) / total_cells) * 100

print(f"\n[✅] VERİ KALİTESİ RAPORU:")
print(f"Toplam hücre sayısı: {total_cells}")
print(f"NaN hücre sayısı: {nan_cells}")
print(f"Veri kalitesi: {data_quality:.1f}%")

print(f"\n[🎉] Tüm işlemler başarıyla tamamlandı!")