import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import json
from sklearn.base import BaseEstimator, TransformerMixin
from openpyxl import load_workbook
import itertools
import random

# ========== KONFİGÜRASYON ==========
RANDOM_STATE = 42
MODEL_PATH = "models/bundesliga_model_final.pkl"
FEATURE_INFO_PATH = "models/feature_info.pkl"
DATA_PATH = "data/bundesliga_matches_2023_2025_final_fe_team_values_cleaned.xlsx"
PLAYER_DATA_PATH = "data/final_bundesliga_dataset_complete.xlsx"
PREDICTION_HISTORY_PATH = "data/prediction_history.xlsx"

TOP_N_STARTERS = 11
TOP_N_SUBS = 7
STARTER_WEIGHT = 0.7
SUB_WEIGHT = 0.3

SELECTED_FEATURES = [
    'Home_AvgRating', 'Away_AvgRating', 'Rating_Diff', 'Total_AvgRating',
    'home_form', 'away_form', 'Form_Diff', 'IsDerby',
    'homeTeam_GoalsScored_5', 'homeTeam_GoalsConceded_5',
    'awayTeam_GoalsScored_5', 'awayTeam_GoalsConceded_5',
    'homeTeam_Momentum', 'awayTeam_Momentum',
    'Home_GK_Rating', 'Home_DF_Rating', 'Home_MF_Rating', 'Home_FW_Rating',
    'Away_GK_Rating', 'Away_DF_Rating', 'Away_MF_Rating', 'Away_FW_Rating',
    'home_current_value_eur', 'away_current_value_eur',
    'home_squad_avg_age', 'away_squad_avg_age',
    'home_value_change_pct', 'away_value_change_pct'
]

TEAM_NAME_MAPPING = {
    # 1. FC Köln
    "fc köln": "1. FC Köln",
    "1. fc koln": "1. FC Köln",
    "1. fc köln": "1. FC Köln",
    "1.fc köln": "1. FC Köln",
    "1.fc koln": "1. FC Köln",
    "fc koln": "1. FC Köln",
    "fc köln": "1. FC Köln",
    "1. fc köln": "1. FC Köln",
    "1. fc koln": "1. FC Köln",
    "1. fc cologne": "1. FC Köln",
    "cologne": "1. FC Köln",
    "koeln": "1. FC Köln",
    "köln": "1. FC Köln",

    # Bayern
    "fc bayern münchen": "Bayern Munich",
    "fc bayern munchen": "Bayern Munich",
    "bayern münchen": "Bayern Munich",
    "bayern munchen": "Bayern Munich",
    "fc bayern": "Bayern Munich",
    "bayern": "Bayern Munich",
    "bayern münih": "Bayern Munich",
    "bayern munih": "Bayern Munich",

    # Borussia Dortmund
    "borussia dortmund": "Borussia Dortmund",
    "bvb dortmund": "Borussia Dortmund",
    "bvb": "Borussia Dortmund",
    "dortmund": "Borussia Dortmund",

    # RB Leipzig
    "rb leipzig": "RB Leipzig",
    "rasenballsport leipzig": "RB Leipzig",
    "rasenball leipzig": "RB Leipzig",
    "leipzig": "RB Leipzig",

    # Bayer Leverkusen
    "bayer 04 leverkusen": "Bayer Leverkusen",
    "bayer leverkusen": "Bayer Leverkusen",
    "leverkusen": "Bayer Leverkusen",

    # VfB Stuttgart
    "vfb stuttgart": "VfB Stuttgart",
    "vfb stuttgart": "VfB Stuttgart",
    "stuttgart": "VfB Stuttgart",

    # VfL Wolfsburg
    "vfl wolfsburg": "VfL Wolfsburg",
    "wolfsburg": "VfL Wolfsburg",

    # Eintracht Frankfurt
    "eintracht frankfurt": "Eintracht Frankfurt",
    "eintracht frankfurt": "Eintracht Frankfurt",
    "frankfurt": "Eintracht Frankfurt",

    # SC Freiburg
    "sc freiburg": "SC Freiburg",
    "freiburg": "SC Freiburg",

    # Werder Bremen
    "sv werder bremen": "Werder Bremen",
    "werder bremen": "Werder Bremen",
    "bremen": "Werder Bremen",

    # Borussia Mönchengladbach
    "borussia mönchengladbach": "Borussia M'gladbach",
    "bor. mönchengladbach": "Borussia M'gladbach",
    "borussia monchengladbach": "Borussia M'gladbach",
    "borussia m'gladbach": "Borussia M'gladbach",
    "mönchengladbach": "Borussia M'gladbach",
    "monchengladbach": "Borussia M'gladbach",
    "gladbach": "Borussia M'gladbach",

    # VfL Bochum
    "vfl bochum 1848": "Bochum",
    "vfl bochum": "Bochum",
    "bochum": "Bochum",

    # FC Augsburg
    "fc augsburg": "FC Augsburg",
    "augsburg": "FC Augsburg",

    # 1. FC Heidenheim
    "1. fc heidenheim 1846": "Heidenheim",
    "heidenheim 1846": "Heidenheim",
    "heidenheim": "Heidenheim",

    # 1. FC Union Berlin
    "1. fc union berlin": "Union Berlin",
    "union berlin": "Union Berlin",
    "union berlin": "Union Berlin",

    # Mainz 05
    "1. fsv mainz 05": "Mainz 05",
    "mainz 05": "Mainz 05",
    "mainz": "Mainz 05",

    # TSG Hoffenheim
    "1899 hoffenheim": "Hoffenheim",
    "tsg 1899 hoffenheim": "Hoffenheim",
    "tsg hoffenheim": "Hoffenheim",
    "hoffenheim": "Hoffenheim",

    # Darmstadt 98
    "sv darmstadt 98": "Darmstadt",
    "darmstadt 98": "Darmstadt",
    "darmstadt": "Darmstadt",

    # Diğer takımlar
    "schalke 04": "Schalke 04",
    "hamburger sv": "Hamburger SV",
    "hamburg": "Hamburger SV",
    "hannover 96": "Hannover 96",
    "hannover": "Hannover 96",
    "energie cottbus": "Energie Cottbus",
    "cottbus": "Energie Cottbus",
    "fc st. pauli": "FC St. Pauli",
    "st. pauli": "FC St. Pauli",
    "holstein kiel": "Holstein Kiel",
    "kiel": "Holstein Kiel",
    "fc nürnberg": "1. FC Nürnberg",
    "nürnberg": "1. FC Nürnberg",
    "nuremberg": "1. FC Nürnberg",
    "1. fc nürnberg": "1. FC Nürnberg",
    "1. fc nuremberg": "1. FC Nürnberg",
    "fc nuremberg": "1. FC Nürnberg",
    "greuther fürth": "Greuther Fürth",
    "greuther furth": "Greuther Fürth",
    "fürth": "Greuther Fürth",
    "furth": "Greuther Fürth",
    "arminia bielefeld": "Arminia Bielefeld",
    "bielefeld": "Arminia Bielefeld",
    "vfl osnabrück": "VfL Osnabrück",
    "osnabrück": "VfL Osnabrück",
    "osnabruck": "VfL Osnabrück",
    "vfl osnabruck": "VfL Osnabrück",
    "fc würzburger kickers": "FC Würzburger Kickers",
    "würzburger kickers": "FC Würzburger Kickers",
    "wurzburger kickers": "FC Würzburger Kickers",
    "fc wurzburger kickers": "FC Würzburger Kickers",
    "würzburg": "FC Würzburger Kickers",
    "wurzburg": "FC Würzburger Kickers",
    "fc ingolstadt 04": "FC Ingolstadt 04",
    "ingolstadt": "FC Ingolstadt 04",
    "fc ingolstadt": "FC Ingolstadt 04",
    "fc erzgebirge aue": "FC Erzgebirge Aue",
    "erzgebirge aue": "FC Erzgebirge Aue",
    "aue": "FC Erzgebirge Aue",
    "fc sankt pauli": "FC St. Pauli",  # alternatif yazım
    "sankt pauli": "FC St. Pauli",
}

# ========== ÖZEL TRANSFORMERLAR ==========
class FeatureSelector(BaseEstimator, TransformerMixin):
    """Önemli feature'ları seçmek için transformer"""
    def __init__(self, features_to_keep):
        self.features_to_keep = features_to_keep
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.features_to_keep]

# ========== YENİ EKLENEN FONKSİYONLAR ==========
def select_starting_and_bench(players):
    """En iyi 11 ve yedek 7 oyuncuyu seçer ve ratinglerini hesaplar"""
    if len(players) < 11:
        print(f"⚠️ Takımda sadece {len(players)} oyuncu var. 11 oyuncu olmalıydı.")
        starters = players
        bench = []
    else:
        starters = players[:11]
        bench = players[11:18]  # Sonraki 7 oyuncu

    # Rating değerlerini kontrol et ve sayısal yap
    start_ratings = []
    for p in starters:
        rating = p.get('rating', 65)
        if isinstance(rating, str):
            try:
                rating = float(rating)
            except ValueError:
                rating = 65
        start_ratings.append(rating)
    
    bench_ratings = []
    for p in bench:
        rating = p.get('rating', 65)
        if isinstance(rating, str):
            try:
                rating = float(rating)
            except ValueError:
                rating = 65
        bench_ratings.append(rating)

    start_rating = sum(start_ratings) / len(start_ratings) if start_ratings else 65.0
    bench_rating = sum(bench_ratings) / len(bench_ratings) if bench_ratings else 65.0

    # Oyuncu ID'lerini döndür (eğer orijinal DataFrame'deki indekslere ihtiyaç varsa)
    start_idx = [p['id'] for p in starters]
    bench_idx = [p['id'] for p in bench]

    return start_idx, bench_idx, start_rating, bench_rating

def calculate_ratings_from_lineup(lineup):
    """
    lineup: Her biri 'rating' anahtarına sahip oyuncu sözlüklerinden oluşan liste.
    İlk 11'i và sonraki 7 yedeği alır, rating ortalamalarını hesaplar.
    """
    if len(lineup) < 11:
        print(f"⚠️ Kadroda sadece {len(lineup)} oyuncu var. 11 oyuncu olmalıydı.")
        starters = lineup
        bench = []
    else:
        starters = lineup[:11]
        bench = lineup[11:18]  # 18 oyuncuya kadar yedek

    # Rating değerlerini kontrol et ve sayısal yap
    start_ratings = []
    for p in starters:
        rating = p.get('rating', 65)
        if isinstance(rating, str):
            try:
                rating = float(rating)
            except ValueError:
                rating = 65
        start_ratings.append(rating)
    
    bench_ratings = []
    for p in bench:
        rating = p.get('rating', 65)
        if isinstance(rating, str):
            try:
                rating = float(rating)
            except ValueError:
                rating = 65
        bench_ratings.append(rating)

    avg_start = sum(start_ratings) / len(start_ratings) if start_ratings else 65
    avg_bench = sum(bench_ratings) / len(bench_ratings) if bench_ratings else 65

    return avg_start, avg_bench

# ========== VERİ HAZIRLAMA FONKSİYONLARI ==========
def load_player_data(path=PLAYER_DATA_PATH):
    """Oyuncu verilerini yükler ve ratingleri hazırlar"""
    try:
        df = pd.read_excel(path)
        print(f"✅ Oyuncu verisi yüklendi: {len(df)} oyuncu")
        
        # Rating hesaplama - daha güvenli bir yaklaşım
        rating_columns = ['Rating', 'Overall', 'PlayerRating', 'fbref__Goal_Contribution', 'rating']
        found_rating_col = None
        
        for col in rating_columns:
            if col in df.columns:
                found_rating_col = col
                break
        
        if found_rating_col:
            # Rating değerlerini sayısal yap ve NaN'leri ortalama ile doldur
            df['PlayerRating'] = pd.to_numeric(df[found_rating_col], errors='coerce')
            avg_rating = df['PlayerRating'].mean()
            if pd.isna(avg_rating):
                avg_rating = 65.0
            df['PlayerRating'].fillna(avg_rating, inplace=True)
            
            # Ratingleri 0-100 skalasına normalize et (eğer değilse)
            max_rating = df['PlayerRating'].max()
            if max_rating > 100:
                df['PlayerRating'] = (df['PlayerRating'] / max_rating) * 100
            elif max_rating < 50:  # Ratingler çok düşükse
                df['PlayerRating'] = df['PlayerRating'] * (100 / max_rating)
                
            print(f"ℹ️ Rating sütunu: {found_rating_col}, Ortalama: {avg_rating:.1f}, Max: {max_rating:.1f}")
        else:
            # Varsayılan rating
            df['PlayerRating'] = 65.0
            print("⚠️ Rating sütunu bulunamadı, varsayılan değer kullanılıyor (65.0)")
        
        # Takım isimlerini normalize et ve eşleştir
        if 'Team' in df.columns:
            df['Team'] = df['Team'].astype(str).str.strip().str.lower()
            # Takım isimlerini eşleştir
            df['Team'] = df['Team'].apply(lambda x: TEAM_NAME_MAPPING.get(x, x.title()))
        elif 'fbref__Squad' in df.columns:
            df['Team'] = df['fbref__Squad'].astype(str).str.strip().str.lower()
            df['Team'] = df['Team'].apply(lambda x: TEAM_NAME_MAPPING.get(x, x.title()))
        else:
            print("⚠️ Takım sütunu bulunamadı!")
        
        # Pozisyon bilgisi
        if 'Position' in df.columns:
            df['Pos'] = df['Position'].astype(str).str.upper().str.strip()
        elif 'fbreref__Pos' in df.columns:
            df['Pos'] = df['fbref__Pos'].astype(str).str.upper().str.strip()
        else:
            # Pozisyon bilgisi yoksa, varsayılan olarak 'MF' ata
            df['Pos'] = 'MF'
            print("⚠️ Pozisyon sütunu bulunamadı, varsayılan olarak 'MF' atandı")
        
        # Oyuncu isimleri
        if 'Player' not in df.columns:
            # Alternatif isim sütunlarını kontrol et
            name_columns = ['Player', 'Name', 'player_name', 'fbreref__Player']
            for col in name_columns:
                if col in df.columns:
                    df['Player'] = df[col]
                    break
            if 'Player' not in df.columns:
                df['Player'] = ['Player_' + str(i) for i in range(len(df))]
                print("⚠️ Oyuncu isim sütunu bulunamadı, varsayılan isimler atandı")
        
        return df
    
    except Exception as e:
        print(f"❌ Oyuncu verisi yüklenirken hata: {e}")
        # Boş bir DataFrame döndür
        return pd.DataFrame(columns=['Player', 'Team', 'Pos', 'PlayerRating'])

def pos_group(pos_str):
    if not isinstance(pos_str, str):
        return 'MF'
    p = pos_str.upper()
    if 'GK' in p or p == 'G' or 'GOALKEEPER' in p: return 'GK'
    if p.startswith('D') or 'DF' in p or 'DEFENDER' in p or 'BACK' in p: return 'DF'
    if p.startswith('M') or 'MF' in p or 'MIDFIELDER' in p or 'MIDFIELD' in p: return 'MF'
    if p.startswith('F') or 'FW' in p or 'ST' in p or 'CF' in p or 'FORWARD' in p or 'WINGER' in p: return 'FW'
    return 'MF'

def team_players_dict(df_players):
    d = {}
    for team in df_players['Team'].unique():
        # Takım isimlerini normalize et ve eşleştir
        team_normalized = team.strip().lower()
        team_normalized = TEAM_NAME_MAPPING.get(team_normalized, team.title())
        
        team_df = df_players[df_players['Team'] == team].copy().reset_index(drop=True)
        if 'PlayerRating' not in team_df.columns:
            print(f"⚠️ Takım {team_normalized} içinde 'PlayerRating' sütunu yok!")
            team_df['PlayerRating'] = 65.0
        
        d[team_normalized] = team_df
        print(f"ℹ️ Takım eklendi: {team_normalized} ({len(team_df)} oyuncu)")
    return d

def avg_of_selected_players(df_team, idxs):
    if len(idxs) == 0: 
        return 65.0, {'GK': 65.0, 'DF': 65.0, 'MF': 65.0, 'FW': 65.0}
    
    try:
        sel = df_team.loc[idxs]
        ratings = sel['PlayerRating'].dropna()
        overall = ratings.mean() if not ratings.empty else 65.0
        
        pos_means = {}
        for pos in ['GK', 'DF', 'MF', 'FW']:
            pos_players = sel[sel['Pos'].apply(pos_group) == pos]
            if len(pos_players) > 0:
                pos_means[pos] = pos_players['PlayerRating'].mean()
            else:
                pos_means[pos] = 65.0
        
        return overall, pos_means
    except Exception as e:
        print(f"⚠️ Oyuncu ortalaması hesaplanırken hata: {e}")
        return 65.0, {'GK': 65.0, 'DF': 65.0, 'MF': 65.0, 'FW': 65.0}

def select_topn_by_rating(df_team, n):
    if 'PlayerRating' not in df_team.columns or df_team.empty:
        return []
    
    try:
        return df_team['PlayerRating'].dropna().sort_values(ascending=False).index.tolist()[:n]
    except Exception as e:
        print(f"⚠️ En iyi oyuncular seçilirken hata: {e}")
        return []

def compute_team_rating_from_lineup(df_team, starter_idxs, sub_idxs,
                                    starter_weight=STARTER_WEIGHT, sub_weight=SUB_WEIGHT):
    try:
        # Eğer DataFrame boşsa veya oyuncu yoksa varsayılan değer döndür
        if df_team.empty or (len(starter_idxs) == 0 and len(sub_idxs) == 0):
            print(f"⚠️ Takım verisi boş veya oyuncu yok. Varsayılan değer döndürülüyor.")
            return 65.0, {'GK': 65.0, 'DF': 65.0, 'MF': 65.0, 'FW': 65.0}
        
        # ID'lerin DataFrame'de var olduğundan emin ol
        valid_starter_idxs = [idx for idx in starter_idxs if idx in df_team.index]
        valid_sub_idxs = [idx for idx in sub_idxs if idx in df_team.index]
        
        print(f"🔢 Hesaplanan indeksler - Başlangıç: {valid_starter_idxs}, Yedek: {valid_sub_idxs}")
        
        starter_mean, starter_pos = avg_of_selected_players(df_team, valid_starter_idxs)
        sub_mean, sub_pos = avg_of_selected_players(df_team, valid_sub_idxs)

        print(f"⭐ Başlangıç ortalaması: {starter_mean}, Yedek ortalaması: {sub_mean}")

        if np.isnan(starter_mean) and not np.isnan(sub_mean): 
            team_rating = sub_mean
        elif np.isnan(sub_mean) and not np.isnan(starter_mean): 
            team_rating = starter_mean
        elif np.isnan(starter_mean) and np.isnan(sub_mean): 
            team_rating = 65.0
        else: 
            team_rating = (starter_mean * starter_weight) + (sub_mean * sub_weight)

        pos_combined = {}
        for pos in ['GK', 'DF', 'MF', 'FW']:
            s = starter_pos.get(pos, np.nan)
            b = sub_pos.get(pos, np.nan)
            
            if pd.isna(s) and not pd.isna(b): 
                pos_combined[pos] = b
            elif pd.isna(b) and not pd.isna(s): 
                pos_combined[pos] = s
            elif pd.isna(s) and pd.isna(b): 
                pos_combined[pos] = 65.0
            else: 
                pos_combined[pos] = (s * starter_weight) + (b * sub_weight)
                
        return team_rating, pos_combined
    except Exception as e:
        print(f"⚠️ Takım ratingi hesaplanırken hata: {e}")
        return 65.0, {'GK': 65.0, 'DF': 65.0, 'MF': 65.0, 'FW': 65.0}

def calculate_form_features_single(team, team_matches, current_date):
    """Tek takım için form özelliklerini hesaplar"""
    if team_matches.empty:
        print(f"⚠️ {team} için maç verisi bulunamadı.")
        return 0.5, 0, 0, 0
    
    try:
        # Timezone uyumsuzluğunu çöz - her iki tarafı da timezone'dan arındır
        if hasattr(current_date, 'tz'):
            current_date = current_date.tz_localize(None)
        
        # Team_matches'taki tarihleri de timezone'dan arındır
        team_matches = team_matches.copy()
        if 'Date' in team_matches.columns and len(team_matches) > 0 and hasattr(team_matches['Date'].iloc[0], 'tz'):
            team_matches['Date'] = team_matches['Date'].dt.tz_localize(None)
        
        # Son 5 maçı filtrele
        past_matches = team_matches[team_matches['Date'] < current_date]
        if len(past_matches) < 1:
            print(f"⚠️ {team} için geçmiş maç bulunamadı. Mevcut tarih: {current_date}, En eski maç: {team_matches['Date'].min()}")
            return 0.5, 0, 0, 0
        
        last_5 = past_matches.tail(5)
        
        points = 0
        goals_scored = 0
        goals_conceded = 0
        
        for _, match in last_5.iterrows():
            is_home = match['HomeTeam'] == team
            
            # Skor bilgilerini al
            home_goals = match.get('score.fullTime.home', match.get('HomeGoals', 0))
            away_goals = match.get('score.fullTime.away', match.get('AwayGoals', 0))
            
            if is_home:
                goals_scored += home_goals
                goals_conceded += away_goals
                
                if home_goals > away_goals:
                    points += 3
                elif home_goals == away_goals:
                    points += 1
            else:
                goals_scored += away_goals
                goals_conceded += home_goals
                
                if away_goals > home_goals:
                    points += 3
                elif away_goals == home_goals:
                    points += 1
        
        form = points / 15  # Maksimum 15 puan
        momentum = goals_scored - goals_conceded
        
        return max(0.1, min(0.9, form)), goals_scored, goals_conceded, momentum
    except Exception as e:
        print(f"⚠️ Form hesaplanırken hata: {e}")
        return 0.5, 0, 0, 0

def calculate_missing_features_single(row, df_players, all_matches=None):
    """Tek bir maç için eksik özellikleri hesaplar"""
    home_team = row.get('HomeTeam')
    away_team = row.get('AwayTeam')
    match_date = row.get('Date', datetime.now())
    
    # Varsayılan değerler
    feature_defaults = {
        'home_form': 0.5, 'away_form': 0.5, 'Form_Diff': 0,
        'homeTeam_GoalsScored_5': 0, 'homeTeam_GoalsConceded_5': 0,
        'awayTeam_GoalsScored_5': 0, 'awayTeam_GoalsConceded_5': 0,
        'homeTeam_Momentum': 0, 'awayTeam_Momentum': 0,
        'Home_AvgRating': 65.0, 'Away_AvgRating': 65.0,
        'Rating_Diff': 0, 'Total_AvgRating': 130.0,
        'Home_GK_Rating': 65.0, 'Home_DF_Rating': 65.0,
        'Home_MF_Rating': 65.0, 'Home_FW_Rating': 65.0,
        'Away_GK_Rating': 65.0, 'Away_DF_Rating': 65.0,
        'Away_MF_Rating': 65.0, 'Away_FW_Rating': 65.0,
        'IsDerby': 0,
        'home_current_value_eur': 0, 'away_current_value_eur': 0,
        'home_squad_avg_age': 0, 'away_squad_avg_age': 0,
        'home_value_change_pct': 0, 'away_value_change_pct': 0
    }
    
    result = feature_defaults.copy()
    
    # Takım ratinglerini hesapla
    team_dict = team_players_dict(df_players)
    
    # Takım ratinglerini hesapla (eğer halen varsayılan değerlerdeyse veya yoksa)
    for team_type, team_name in [('Home', home_team), ('Away', away_team)]:
        rating_key = f'{team_type}_AvgRating'
        # Eğer row'da rating değeri var ve 0 değilse, onu kullan
        if rating_key in row and not pd.isna(row.get(rating_key)) and row.get(rating_key) != 0:
            result[rating_key] = row[rating_key]
            # Pozisyon ratinglerini de row'dan almak için
            for pos in ['GK', 'DF', 'MF', 'FW']:
                pos_key = f'{team_type}_{pos}_Rating'
                if pos_key in row and not pd.isna(row.get(pos_key)) and row.get(pos_key) != 0:
                    result[pos_key] = row[pos_key]
        else:
            # Hesaplama yap
            if team_name in team_dict:
                df_team = team_dict[team_name]
                starters = select_topn_by_rating(df_team, TOP_N_STARTERS)
                subs = select_topn_by_rating(df_team, TOP_N_SUBS)
                
                rating, pos_ratings = compute_team_rating_from_lineup(df_team, starters, subs)
                
                result[rating_key] = rating
                for pos in ['GK', 'DF', 'MF', 'FW']:
                    pos_key = f'{team_type}_{pos}_Rating'
                    result[pos_key] = pos_ratings.get(pos, 65.0)
    
    # Form ve momentum hesapla (eğer geçmiş maç verisi varsa)
    if all_matches is not None and home_team and away_team:
        try:
            home_matches = all_matches[(all_matches['HomeTeam'] == home_team) | 
                                      (all_matches['AwayTeam'] == home_team)]
            away_matches = all_matches[(all_matches['HomeTeam'] == away_team) | 
                                      (all_matches['AwayTeam'] == away_team)]
            
            home_form, home_gs, home_gc, home_mom = calculate_form_features_single(
                home_team, home_matches, match_date)
            away_form, away_gs, away_gc, away_mom = calculate_form_features_single(
                away_team, away_matches, match_date)
            
            result.update({
                'home_form': home_form,
                'away_form': away_form,
                'Form_Diff': home_form - away_form,
                'homeTeam_GoalsScored_5': home_gs,
                'homeTeam_GoalsConceded_5': home_gc,
                'awayTeam_GoalsScored_5': away_gs,
                'awayTeam_GoalsConceded_5': away_gc,
                'homeTeam_Momentum': home_mom,
                'awayTeam_Momentum': away_mom
            })
        except Exception as e:
            print(f"⚠️ Form hesaplanırken hata: {e}")
    
    # Derby kontrolü
    big_teams = ['Bayern Munich', 'Borussia Dortmund', 'Schalke 04', 'Hamburg SV', 
                'Borussia Mönchengladbach', 'Bayer Leverkusen', 'VfB Stuttgart']
    if home_team in big_teams and away_team in big_teams:
        result['IsDerby'] = 1
    
    # Diğer rating farkları
    result['Rating_Diff'] = result['Home_AvgRating'] - result['Away_AvgRating']
    result['Total_AvgRating'] = result['Home_AvgRating'] + result['Away_AvgRating']
    
    return result

def prepare_and_enrich_dataset(df_matches, df_players):
    """
    Eksik özellikleri otomatik olarak hesaplayarak dataseti zenginleştirir
    """
    print("🔧 Veri hazırlama và zenginleştirme başlıyor...")
    
    # 1. Takım isimlerini standartlaştır
    if 'homeTeam.name' in df_matches.columns and 'HomeTeam' not in df_matches.columns:
        df_matches['HomeTeam'] = df_matches['homeTeam.name']
    if 'awayTeam.name' in df_matches.columns and 'AwayTeam' not in df_matches.columns:
        df_matches['AwayTeam'] = df_matches['awayTeam.name']
    
    # Takım isimlerini normalize et
    df_matches['HomeTeam'] = df_matches['HomeTeam'].astype(str).str.strip().str.lower()
    df_matches['AwayTeam'] = df_matches['AwayTeam'].astype(str).str.strip().str.lower()
    
    # Takım isimlerini eşleştir
    df_matches['HomeTeam'] = df_matches['HomeTeam'].apply(lambda x: TEAM_NAME_MAPPING.get(x, x.title()))
    df_matches['AwayTeam'] = df_matches['AwayTeam'].apply(lambda x: TEAM_NAME_MAPPING.get(x, x.title()))
    
    # 2. Tarih sütununu işle
    date_columns = ['utcDate', 'Date', 'date']
    for col in date_columns:
        if col in df_matches.columns:
            df_matches['Date'] = pd.to_datetime(df_matches[col], errors='coerce')
            break
    
    if 'Date' not in df_matches.columns:
        df_matches['Date'] = datetime.now()
        print("⚠️ Tarih sütunu bulunamadı, bugünün tarihi kullanılıyor")
    
    df_matches = df_matches.sort_values('Date').reset_index(drop=True)
    
    # 3. IsDerby sütunu yoksa oluştur (basit mantık)
    if 'IsDerby' not in df_matches.columns:
        print("⚠️ IsDerby sütunu bulunamadı, basit mantıkla oluşturuluyor...")
        # Büyük takımlar arası maçları derby olarak işaretle
        big_teams = ['Bayern Munich', 'Borussia Dortmund', 'Schalke 04', 'Hamburg SV', 
                    'Borussia Mönchengladbach', 'Bayer Leverkusen', 'VfB Stuttgart']
        
        def is_derby(home_team, away_team):
            if home_team in big_teams and away_team in big_teams:
                return 1
            return 0
        
        df_matches['IsDerby'] = df_matches.apply(
            lambda row: is_derby(row.get('HomeTeam', ''), row.get('AwayTeam', '')), axis=1
        )
    
    # 4. Takım ratinglerini hesapla
    print("🔁 Takım ratingleri hesaplanıyor...")
    df_matches = compute_ratings_for_matches(df_matches, df_players)
    
    # 5. Eksik özellikleri kontrol et ve gerekirse hesapla
    df_matches = calculate_missing_features(df_matches)
    
    print("✅ Veri zenginleştirme tamamlandı!")
    return df_matches

def calculate_missing_features(df):
    """Eksik özellikleri kontrol et ve gerekirse hesapla"""
    print("🔍 Eksik özellikler kontrol ediliyor...")
    
    # Özellikleri và varsayılan değerleri
    feature_defaults = {
        'home_form': 0.5,
        'away_form': 0.5,
        'Form_Diff': 0,
        'homeTeam_GoalsScored_5': 0,
        'homeTeam_GoalsConceded_5': 0,
        'awayTeam_GoalsScored_5': 0,
        'awayTeam_GoalsConceded_5': 0,
        'homeTeam_Momentum': 0,
        'awayTeam_Momentum': 0,
        'Home_AvgRating': 65.0,
        'Away_AvgRating': 65.0,
        'Rating_Diff': 0,
        'Total_AvgRating': 130.0,
        'Home_GK_Rating': 65.0,
        'Home_DF_Rating': 65.0,
        'Home_MF_Rating': 65.0,
        'Home_FW_Rating': 65.0,
        'Away_GK_Rating': 65.0,
        'Away_DF_Rating': 65.0,
        'Away_MF_Rating': 65.0,
        'Away_FW_Rating': 65.0,
        'home_current_value_eur': 0,
        'away_current_value_eur': 0,
        'home_squad_avg_age': 0,
        'away_squad_avg_age': 0,
        'home_value_change_pct': 0,
        'away_value_change_pct': 0
    }
    
    # Eksik özellikleri kontrol et ve doldur
    for feature, default_value in feature_defaults.items():
        if feature not in df.columns:
            print(f"   ⚠️ {feature} bulunamadı, varsayılan değerle dolduruluyor: {default_value}")
            df[feature] = default_value
        elif df[feature].isnull().any():
            null_count = df[feature].isnull().sum()
            print(f"   ⚠️ {feature} içinde {null_count} boş değer, varsayılan değerle dolduruluyor")
            df[feature].fillna(default_value, inplace=True)
    
    # Rating_Diff ve Total_AvgRating'i güncelle (eğer hesaplanabilirse)
    if 'Home_AvgRating' in df.columns and 'Away_AvgRating' in df.columns:
        df['Rating_Diff'] = df['Home_AvgRating'] - df['Away_AvgRating']
        df['Total_AvgRating'] = df['Home_AvgRating'] + df['Away_AvgRating']
    
    return df

def compute_ratings_for_matches(df_matches, df_players):
    df_players = df_players.copy()
    team_dict = team_players_dict(df_players)
    
    # Takım isimlerini eşleştir
    if 'homeTeam.name' in df_matches.columns and 'HomeTeam' not in df_matches.columns:
        df_matches['HomeTeam'] = df_matches['homeTeam.name']
    if 'awayTeam.name' in df_matches.columns and 'AwayTeam' not in df_matches.columns:
        df_matches['AwayTeam'] = df_matches['awayTeam.name']

    # Takım isimlerini normalize et
    df_matches['HomeTeam'] = df_matches['HomeTeam'].astype(str).str.strip().str.lower()
    df_matches['AwayTeam'] = df_matches['AwayTeam'].astype(str).str.strip().str.lower()
    
    # Takım isimlerini eşleştir
    df_matches['HomeTeam'] = df_matches['HomeTeam'].apply(lambda x: TEAM_NAME_MAPPING.get(x, x.title()))
    df_matches['AwayTeam'] = df_matches['AwayTeam'].apply(lambda x: TEAM_NAME_MAPPING.get(x, x.title()))

    cols_to_add = ['Home_AvgRating','Away_AvgRating','Total_AvgRating','Rating_Diff',
                   'Home_GK_Rating','Home_DF_Rating','Home_MF_Rating','Home_FW_Rating',
                   'Away_GK_Rating','Away_DF_Rating','Away_MF_Rating','Away_FW_Rating']
    for c in cols_to_add:
        if c not in df_matches.columns: 
            df_matches[c] = np.nan

    for idx, row in df_matches.iterrows():
        home = row.get('HomeTeam')
        away = row.get('AwayTeam')
        
        df_home = team_dict.get(home, pd.DataFrame())
        df_away = team_dict.get(away, pd.DataFrame())

        print(f"🏠 {home} takımının oyuncu sayısı: {len(df_home)}, 🏃 {away} takımının oyuncu sayısı: {len(df_away)}")
        
        # Lineup verisi yoksa en iyi oyuncuları seç
        home_starters = select_topn_by_rating(df_home, TOP_N_STARTERS)
        away_starters = select_topn_by_rating(df_away, TOP_N_STARTERS)
        

        home_subs = []
        away_subs = []

        if len(home_subs) == 0 and not df_home.empty:
            all_idxs = df_home['PlayerRating'].dropna().sort_values(ascending=False).index.tolist() if 'PlayerRating' in df_home else []
            home_subs = [i for i in all_idxs if i not in home_starters][:TOP_N_SUBS]
        
        if len(away_subs) == 0 and not df_away.empty:
            all_idxs = df_away['PlayerRating'].dropna().sort_values(ascending=False).index.tolist() if 'PlayerRating' in df_away else []
            away_subs = [i for i in all_idxs if i not in away_starters][:TOP_N_SUBS]

        h_rating, h_pos = compute_team_rating_from_lineup(df_home, home_starters, home_subs)
        a_rating, a_pos = compute_team_rating_from_lineup(df_away, away_starters, away_subs)

        df_matches.at[idx, 'Home_AvgRating'] = h_rating
        df_matches.at[idx, 'Away_AvgRating'] = a_rating
        
        if not pd.isna(h_rating) and not pd.isna(a_rating):
            df_matches.at[idx, 'Total_AvgRating'] = h_rating + a_rating
            df_matches.at[idx, 'Rating_Diff'] = h_rating - a_rating

        df_matches.at[idx, 'Home_GK_Rating'] = h_pos.get('GK', np.nan)
        df_matches.at[idx, 'Home_DF_Rating'] = h_pos.get('DF', np.nan)
        df_matches.at[idx, 'Home_MF_Rating'] = h_pos.get('MF', np.nan)
        df_matches.at[idx, 'Home_FW_Rating'] = h_pos.get('FW', np.nan)

        df_matches.at[idx, 'Away_GK_Rating'] = a_pos.get('GK', np.nan)
        df_matches.at[idx, 'Away_DF_Rating'] = a_pos.get('DF', np.nan)
        df_matches.at[idx, 'Away_MF_Rating'] = a_pos.get('MF', np.nan)
        df_matches.at[idx, 'Away_FW_Rating'] = a_pos.get('FW', np.nan)

    global_avg = df_players['PlayerRating'].mean() if 'PlayerRating' in df_players and not df_players.empty else 65.0
    df_matches['Home_AvgRating'].fillna(global_avg, inplace=True)
    df_matches['Away_AvgRating'].fillna(global_avg, inplace=True)
    df_matches['Total_AvgRating'].fillna(df_matches['Home_AvgRating'] + df_matches['Away_AvgRating'], inplace=True)
    df_matches['Rating_Diff'].fillna(df_matches['Home_AvgRating'] - df_matches['Away_AvgRating'], inplace=True)

    for pos in ['GK', 'DF', 'MF', 'FW']:
        if 'PlayerRating' in df_players and not df_players.empty:
            pos_players = df_players[df_players['Pos'].apply(pos_group) == pos]
            pos_mean = pos_players['PlayerRating'].mean() if not pos_players.empty else global_avg
        else:
            pos_mean = global_avg
            
        df_matches[f'Home_{pos}_Rating'].fillna(pos_mean, inplace=True)
        df_matches[f'Away_{pos}_Rating'].fillna(pos_mean, inplace=True)

    return df_matches

# ========== TAHMIN SINIFI ==========
class BundesligaPredictor:
    # ... (diğer metodlar aynı)

    def prepare_match_features(self, home_team, away_team, match_date=None, additional_features=None):
        if match_date is None:
            match_date = datetime.now()
        
        # Temel maç bilgisi
        match_data = {
            'HomeTeam': home_team,
            'AwayTeam': away_team,
            'Date': match_date
        }
        
        # Ek özellikleri burada ekliyoruz
        if additional_features:
            match_data.update(additional_features)
        
        # DataFrame oluştur
        match_df = pd.DataFrame([match_data])
        
        # Eksik özellikleri hesapla
        features = calculate_missing_features_single(
            match_df.iloc[0], self.df_players, self.all_matches
        )

        # additional_features varsa onları override et
        if additional_features:
            for key, val in additional_features.items():
                features[key] = val

        # np.float64 değerleri float'a çevir
        for key, value in features.items():
            if isinstance(value, np.float64):
                features[key] = float(value)

        print("DEBUG >> FEATURES SON HAL:", features)

        # Artık DataFrame oluştur
        feature_df = pd.DataFrame([features])

        # Model için gerekli feature'ları belirle
        if hasattr(self.model, 'feature_names_in_'):
            required_features = self.model.feature_names_in_.tolist()
            print(f"ℹ️ Modelin beklediği feature'lar: {required_features}")
        elif self.feature_info and 'important_features' in self.feature_info:
            required_features = self.feature_info['important_features']
        else:
            required_features = SELECTED_FEATURES

        # Eksik feature'ları kontrol et ve doldur
        for feature in required_features:
            if feature not in feature_df.columns:
                # Varsayılan değerlerle doldur
                default_values = {
                    'home_form': 0.5, 'away_form': 0.5, 'Form_Diff': 0,
                    'homeTeam_GoalsScored_5': 0, 'homeTeam_GoalsConceded_5': 0,
                    'awayTeam_GoalsScored_5': 0, 'awayTeam_GoalsConceded_5': 0,
                    'homeTeam_Momentum': 0, 'awayTeam_Momentum': 0,
                    'Home_AvgRating': 65.0, 'Away_AvgRating': 65.0,
                    'Rating_Diff': 0, 'Total_AvgRating': 130.0,
                    'Home_GK_Rating': 65.0, 'Home_DF_Rating': 65.0,
                    'Home_MF_Rating': 65.0, 'Home_FW_Rating': 65.0,
                    'Away_GK_Rating': 65.0, 'Away_DF_Rating': 65.0,
                    'Away_MF_Rating': 65.0, 'Away_FW_Rating': 65.0,
                    'IsDerby': 0,
                    'home_current_value_eur': 0, 'away_current_value_eur': 0,
                    'home_squad_avg_age': 0, 'away_squad_avg_age': 0,
                    'home_value_change_pct': 0, 'away_value_change_pct': 0
                }
                feature_df[feature] = default_values.get(feature, 0)
            elif feature_df[feature].isnull().any():
                # Boş değerleri doldur
                default_value = 0
                if 'Rating' in feature:
                    default_value = 65.0
                elif 'form' in feature:
                    default_value = 0.5
                feature_df[feature].fillna(default_value, inplace=True)
        
        # Modelin beklediği feature'ları seç
        feature_df = feature_df[required_features]
        
        return feature_df

    def predict_match(self, home_team, away_team, match_date=None, additional_features=None):
        """Tek maç tahmini yapar"""
        if self.model is None:
            return {"error": "Model yüklenemedi"}
        
        try:
            # Feature'ları hazırla
            features_df = self.prepare_match_features(home_team, away_team, match_date, additional_features)
            
            print(f"🔢 Tahmin için hazırlanan feature'lar: {features_df.columns.tolist()}")
            print(f"🔢 Feature sayısı: {len(features_df.columns)}")
            print(f"🔢 Feature değerleri: {features_df.values}")
            
            # Tahmin yap
            prediction = self.model.predict(features_df)
            probabilities = self.model.predict_proba(features_df)
            
            # Sonuçları formatla
            result_map = {0: 'Draw', 1: 'HomeWin', 2: 'AwayWin'}
            prediction_label = result_map[prediction[0]]
            
            # Olasılıkları formatla
            prob_dict = {
                'Draw': float(probabilities[0][0]),
                'HomeWin': float(probabilities[0][1]),
                'AwayWin': float(probabilities[0][2])
            }
            
            # Feature importance (eğer mevcutsa)
            feature_importance = {}
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                feature_names = features_df.columns
                for i, feature in enumerate(feature_names):
                    feature_importance[feature] = float(importances[i])
            elif hasattr(self.model, 'steps') and hasattr(self.model.named_steps['lgbm'], 'feature_importances_'):
                importances = self.model.named_steps['lgbm'].feature_importances_
                feature_names = features_df.columns
                for i, feature in enumerate(feature_names):
                    feature_importance[feature] = float(importances[i])
            
            return {
                'prediction': prediction_label,
                'probabilities': prob_dict,
                'features': features_df.iloc[0].to_dict(),
                'feature_importance': feature_importance,
                'confidence': max(prob_dict.values())
            }
            
        except Exception as e:
            return {"error": f"Tahmin yapılırken hata: {str(e)}"}
    
    def predict_multiple_matches(self, matches_list):
        """Çoklu maç tahmini yapar"""
        results = []
        for match in matches_list:
            home_team = match.get('home_team')
            away_team = match.get('away_team')
            match_date = match.get('date')
            additional_features = match.get('additional_features', {})
            
            if home_team and away_team:
                result = self.predict_match(home_team, away_team, match_date, additional_features)
                result['match_info'] = {
                    'home_team': home_team,
                    'away_team': away_team,
                    'date': match_date
                }
                results.append(result)
        
        return results
    
    def get_team_ratings(self, team_name):
        """Takım ratinglerini getirir"""
        if self.team_dict is None or team_name not in self.team_dict:
            return None
        
        df_team = self.team_dict[team_name]
        starters = select_topn_by_rating(df_team, TOP_N_STARTERS)
        subs = select_topn_by_rating(df_team, TOP_N_SUBS)
        
        rating, pos_ratings = compute_team_rating_from_lineup(df_team, starters, subs)
        
        return {
            'overall_rating': rating,
            'position_ratings': pos_ratings,
            'top_players': df_team.loc[starters][['Player', 'PlayerRating', 'Pos']].to_dict('records') if len(starters) > 0 else []
        }
    
    def analyze_feature_contributions(self, home_team, away_team):
        """Feature'ların tahmine katkısını analiz eder"""
        if self.model is None:
            return None
        
        features_df = self.prepare_match_features(home_team, away_team)
        
        # SHAP değerlerini hesapla (eğer mevcutsa)
        try:
            import shap
            explainer = shap.TreeExplainer(self.model.named_steps['lgbm'])
            shap_values = explainer.shap_values(features_df)
            
            contributions = {}
            for i, class_idx in enumerate([0, 1, 2]):  # Draw, HomeWin, AwayWin
                class_contrib = {}
                for j, feature in enumerate(features_df.columns):
                    class_contrib[feature] = float(shap_values[i][0][j])
                contributions[['Draw', 'HomeWin', 'AwayWin'][class_idx]] = class_contrib
            
            return contributions
        except:
            # SHAP yoksa feature importance kullan
            if hasattr(self.model.named_steps['lgbm'], 'feature_importances_'):
                importances = self.model.named_steps['lgbm'].feature_importances_
                feature_importance = {}
                for i, feature in enumerate(features_df.columns):
                    feature_importance[feature] = float(importances[i])
                return {'feature_importance': feature_importance}
            return None

    def get_team_form(self, team_name, num_matches=5):
        """Takımın form durumunu getirir"""
        if self.all_matches is None:
            return None
        
        team_matches = self.all_matches[(self.all_matches['HomeTeam'] == team_name) | 
                                      (self.all_matches['AwayTeam'] == team_name)]
        
        if team_matches.empty:
            return None
        
        # Son maçları al
        recent_matches = team_matches.sort_values('Date', ascending=False).head(num_matches)
        
        form_data = []
        for _, match in recent_matches.iterrows():
            is_home = match['HomeTeam'] == team_name
            opponent = match['AwayTeam'] if is_home else match['HomeTeam']
            
            # Skor bilgilerini al
            home_goals = match.get('score.fullTime.home', match.get('HomeGoals', 0))
            away_goals = match.get('score.fullTime.away', match.get('AwayGoals', 0))
            
            if is_home:
                goals_for = home_goals
                goals_against = away_goals
            else:
                goals_for = away_goals
                goals_against = home_goals
            
            # Sonucu belirle
            if goals_for > goals_against:
                result = 'W'
            elif goals_for < goals_against:
                result = 'L'
            else:
                result = 'D'
            
            form_data.append({
                'date': match['Date'],
                'opponent': opponent,
                'result': result,
                'score': f"{goals_for}-{goals_against}",
                'is_home': is_home
            })
        
        return form_data

# ========== KULLANICI ARAYÜZÜ FONKSİYONLARI ==========
def display_teams(team_dict):
    """Tüm takımları listeler"""
    print("\n🏆 Mevcut Takımlar:")
    print("=" * 50)
    teams = sorted(team_dict.keys())
    for i, team in enumerate(teams, 1):
        print(f"{i:2d}. {team}")
    return teams

def display_players(team_dict, team_name):
    """Bir takımın oyuncularını listeler"""
    if team_name not in team_dict:
        print(f"⚠️ {team_name} takımı bulunamadı!")
        return []
    
    df_team = team_dict[team_name]
    print(f"\n👥 {team_name} Kadrosu:")
    print("=" * 60)
    print(f"{'ID':<4} {'İsim':<25} {'Pozisyon':<10} {'Rating':<6}")
    print("-" * 60)
    
    players = []
    for idx, row in df_team.iterrows():
        player_id = idx
        name = row.get('Player', 'Bilinmiyor')
        pos = row.get('Pos', 'N/A')
        rating = row.get('PlayerRating', 0)
        print(f"{player_id:<4} {name:<25} {pos:<10} {rating:<6.1f}")
        players.append({
            'id': player_id,
            'name': name,
            'pos': pos,
            'rating': rating
        })
    
    return players

def select_players_interactive(players, player_type="başlangıç"):
    """Kullanıcıdan oyuncu seçimi alır"""
    print(f"\n🔢 {player_type.title()} oyuncularını seçin (ID'leri virgülle ayırarak girin):")
    selected_ids = input("Seçimleriniz: ").strip()
    
    if not selected_ids:
        print("⚠️ Hiç oyuncu seçilmedi, en iyi oyuncular otomatik seçilecek")
        return []
    
    try:
        selected_ids = [int(id_str.strip()) for id_str in selected_ids.split(',')]
        valid_players = []
        invalid_ids = []
        
        for player_id in selected_ids:
            player = next((p for p in players if p['id'] == player_id), None)
            if player:
                valid_players.append(player)
            else:
                invalid_ids.append(player_id)
        
        if invalid_ids:
            print(f"⚠️ Geçersiz ID'ler: {invalid_ids}")
        
        return valid_players
    except ValueError:
        print("⚠️ Geçersiz giriş! Lütfen sayıları virgülle ayırarak girin.")
        return []

def save_prediction_to_excel(prediction_result, home_team, away_team, home_lineup, away_lineup, filename=PREDICTION_HISTORY_PATH):
    """Tahmin sonuçlarını Excel dosyasına kaydeder"""
    try:
        # Tahmin bilgilerini hazırla
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        sheet_name = f"Pred_{timestamp}"
        
        # Ana tahmin bilgileri
        prediction_data = {
            'Tarih': [datetime.now().strftime("%Y-%m-%d %H:%M:%S")],
            'Ev Sahibi': [home_team],
            'Deplasman': [away_team],
            'Tahmin': [prediction_result['prediction']],
            'Güven': [f"{prediction_result['confidence']*100:.1f}%"],
            'Beraberlik Olasılığı': [f"{prediction_result['probabilities']['Draw']*100:.1f}%"],
            'Ev Sahibi Kazanma Olasılığı': [f"{prediction_result['probabilities']['HomeWin']*100:.1f}%"],
            'Deplasman Kazanma Olasılığı': [f"{prediction_result['probabilities']['AwayWin']*100:.1f}%"]
        }
        
        # Özellik önemlilikleri
        features = prediction_result.get('feature_importance', {})
        if features:
            top_features = dict(sorted(features.items(), key=lambda x: x[1], reverse=True)[:5])
            
            for i, (feature, importance) in enumerate(top_features.items(), 1):
                prediction_data[f'Önemli Özellik {i}'] = [f"{feature} ({importance:.4f})"]
        
        # DataFrame oluştur
        df_prediction = pd.DataFrame(prediction_data)
        
        # Kadro bilgilerini hazırla
        home_lineup_data = {
            'Ev Sahibi Kadrosu': [p['name'] for p in home_lineup[:11]],
            'Pozisyon': [p['pos'] for p in home_lineup[:11]],
            'Rating': [p['rating'] for p in home_lineup[:11]]
        }
        
        away_lineup_data = {
            'Deplasman Kadrosu': [p['name'] for p in away_lineup[:11]],
            'Pozisyon': [p['pos'] for p in away_lineup[:11]],
            'Rating': [p['rating'] for p in away_lineup[:11]]
        }
        
        # Yedek oyuncular
        home_subs = home_lineup[11:] if len(home_lineup) > 11 else []
        away_subs = away_lineup[11:] if len(away_lineup) > 11 else []
        
        if home_subs:
            home_lineup_data['Ev Sahibi Yedekler'] = [p['name'] for p in home_subs] + [''] * (11 - len(home_subs))
            home_lineup_data['Yedek Pozisyon'] = [p['pos'] for p in home_subs] + [''] * (11 - len(home_subs))
            home_lineup_data['Yedek Rating'] = [p['rating'] for p in home_subs] + [''] * (11 - len(home_subs))
        
        if away_subs:
            away_lineup_data['Deplasman Yedekler'] = [p['name'] for p in away_subs] + [''] * (11 - len(away_subs))
            away_lineup_data['Yedek Pozisyon'] = [p['pos'] for p in away_subs] + [''] * (11 - len(away_subs))
            away_lineup_data['Yedek Rating'] = [p['rating'] for p in away_subs] + [''] * (11 - len(away_subs))
        
        df_home_lineup = pd.DataFrame(home_lineup_data)
        df_away_lineup = pd.DataFrame(away_lineup_data)
        
        # Excel dosyasına yaz
        if os.path.exists(filename):
            with pd.ExcelWriter(filename, engine='openpyxl', mode='a', if_sheet_exists='replace') as writer:
                df_prediction.to_excel(writer, sheet_name=sheet_name, index=False)
                df_home_lineup.to_excel(writer, sheet_name=sheet_name, startrow=len(df_prediction) + 2, index=False)
                df_away_lineup.to_excel(writer, sheet_name=sheet_name, startrow=len(df_prediction) + len(df_home_lineup) + 4, index=False)
        else:
            with pd.ExcelWriter(filename, engine='openpyxl') as writer:
                df_prediction.to_excel(writer, sheet_name=sheet_name, index=False)
                df_home_lineup.to_excel(writer, sheet_name=sheet_name, startrow=len(df_prediction) + 2, index=False)
                df_away_lineup.to_excel(writer, sheet_name=sheet_name, startrow=len(df_prediction) + len(df_home_lineup) + 4, index=False)
        
        print(f"✅ Tahmin '{sheet_name}' sayfasına kaydedildi: {filename}")
        
    except Exception as e:
        print(f"⚠️ Tahmin kaydedilirken hata: {e}")

# ========== GÖRSELLEŞTİRME FONKSİYONLARI ==========
def visualize_prediction(result, save_path=None):
    """Tahmin sonuçlarını görselleştirir"""
    if 'error' in result:
        print(f"Hata: {result['error']}")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    # 1. Olasılık grafiği (yüzde olarak)
    probs = result['probabilities']
    probs_percent = {k: v*100 for k, v in probs.items()}
    ax1.bar(probs_percent.keys(), probs_percent.values(), color=['blue', 'green', 'red'])
    ax1.set_title('Maç Sonucu Olasılıkları (%)')
    ax1.set_ylabel('Olasılık (%)')
    ax1.set_ylim(0, 100)
    
    # Yüzde değerlerini çubukların üzerine yaz
    for i, (outcome, prob) in enumerate(probs_percent.items()):
        ax1.text(i, prob + 1, f'{prob:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # 2. Feature importance
    if 'feature_importance' in result and result['feature_importance']:
        fi_data = result['feature_importance']
        top_features = dict(sorted(fi_data.items(), key=lambda x: x[1], reverse=True)[:10])
        ax2.barh(list(top_features.keys()), list(top_features.values()))
        ax2.set_title('En Önemli 10 Özellik')
        ax2.set_xlabel('Önem Derecesi')
    
    # 3. Takım rating karşılaştırması
    features = result['features']

    # Burada her zaman Home_AvgRating / Away_AvgRating kullanıyoruz
    home_rating = features.get('Home_AvgRating', 0.0)
    away_rating = features.get('Away_AvgRating', 0.0)

    # DEBUG printleri
    print("DEBUG >> result['features']:", features)
    print("DEBUG >> home_rating:", home_rating)
    print("DEBUG >> away_rating:", away_rating)

    # Rating değerleri 0 ise, additional_features'tan al
    if home_rating == 0.0 and 'additional_features' in result:
        home_rating = result['additional_features'].get('Home_AvgRating', 0.0)
    if away_rating == 0.0 and 'additional_features' in result:
        away_rating = result['additional_features'].get('Away_AvgRating', 0.0)

    print("👥 Takım İstatistikleri:")
    print(f"   ⭐ Ev Sahibi Rating: {home_rating:.2f}")
    print(f"   ⭐ Deplasman Rating: {away_rating:.2f}")
    print(f"   📈 Ev Sahibi Formu: {features.get('home_form', 0.0)*100:.1f}%")
    print(f"   📈 Deplasman Formu: {features.get('away_form', 0.0)*100:.1f}%")
    print(f"   ⚽ Ev Sahibi Momentum: {features.get('homeTeam_Momentum', 0.0):.1f}")
    print(f"   ⚽ Deplasman Momentum: {features.get('awayTeam_Momentum', 0.0):.1f}")

    # Rating karşılaştırma grafiği
    ax3.bar(['Ev Sahibi', 'Deplasman'], [home_rating, away_rating], color=['blue', 'red'])
    ax3.set_title('Takım Rating Karşılaştırması')
    ax3.set_ylabel('Rating')

    
    # Rating değerlerini çubukların üzerine yaz
    ax3.text(0, home_rating + 0.5, f'{home_rating:.1f}', ha='center', va='bottom', fontweight='bold')
    ax3.text(1, away_rating + 0.5, f'{away_rating:.1f}', ha='center', va='bottom', fontweight='bold')
    
    # 4. Form ve momentum
    form_data = {
        'Ev Sahibi Form': features.get('home_form', 0) * 100,  # Yüzdeye çevir
        'Deplasman Form': features.get('away_form', 0) * 100,
        'Ev Momentum': features.get('homeTeam_Momentum', 0),
        'Dep Momentum': features.get('awayTeam_Momentum', 0)
    }
    ax4.bar(form_data.keys(), form_data.values())
    ax4.set_title('Form ve Momentum Karşılaştırması')
    ax4.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def display_prediction_details(result):
    """Tahmin detaylarını terminalde gösterir"""
    if 'error' in result:
        print(f"❌ Hata: {result['error']}")
        return
    
    print("\n" + "="*60)
    print("🎯 TAHMİN SONUÇLARI")
    print("="*60)
    
    # Temel tahmin bilgileri
    probs = result['probabilities']
    print(f"\n📊 Sonuç Olasılıkları (%):")
    print(f"   • Beraberlik: {probs['Draw']*100:.1f}%")
    print(f"   • Ev Sahibi Kazanır: {probs['HomeWin']*100:.1f}%")
    print(f"   • Deplasman Kazanır: {probs['AwayWin']*100:.1f}%")
    print(f"   🔮 Tahmin: {result['prediction']} ({result['confidence']*100:.1f}% güven)")
    
    # Özellik önemlilikleri
    if 'feature_importance' in result and result['feature_importance']:
        print(f"\n📈 En Önemli Özellikler:")
        features = result['feature_importance']
        top_features = sorted(features.items(), key=lambda x: x[1], reverse=True)[:5]
        
        for i, (feature, importance) in enumerate(top_features, 1):
            print(f"   {i}. {feature}: {importance:.4f}")
    
    # Takım istatistikleri
    features = result['features']
    print(f"\n👥 Takım İstatistikleri:")
    print(f"   ⭐ Ev Sahibi Rating: {features.get('Home_AvgRating', 0):.1f}")
    print(f"   ⭐ Deplasman Rating: {features.get('Away_AvgRating', 0):.1f}")
    print(f"   📈 Ev Sahibi Formu: {features.get('home_form', 0)*100:.1f}%")
    print(f"   📈 Deplasman Formu: {features.get('away_form', 0)*100:.1f}%")
    print(f"   ⚽ Ev Sahibi Momentum: {features.get('homeTeam_Momentum', 0):.1f}")
    print(f"   ⚽ Deplasman Momentum: {features.get('awayTeam_Momentum', 0):.1f}")

# ========== ANA TAHMİN FONKSİYONU ==========
def main():
    """Ana tahmin fonksiyonu - Kullanıcı etkileşimli"""
    print("🏆 Bundesliga Tahmin Sistemi")
    print("=" * 50)
    
    # Predictor'ü başlat
    predictor = BundesligaPredictor()
    
    # Geçmiş maç verilerini yükle (form hesaplamaları için)
    predictor.load_historical_matches(DATA_PATH)
    
    if not predictor.team_dict:
        print("❌ Takım verisi yüklenemedi. Lütfen oyuncu veri dosyasını kontrol edin.")
        return
    
    # Takımları listele
    teams = display_teams(predictor.team_dict)
    
    # Ev sahibi takım seçimi
    try:
        home_choice = int(input("\n🏠 Ev sahibi takım numarasını girin: ")) - 1
        home_team = teams[home_choice]
        print(f"✅ Ev sahibi: {home_team}")
    except (ValueError, IndexError):
        print("⚠️ Geçersiz seçim! Varsayılan olarak 'Bayern Munich' seçildi.")
        home_team = "Bayern Munich"
    
    # Deplasman takımı seçimi
    try:
        away_choice = int(input("✈️ Deplasman takımı numarasını girin: ")) - 1
        away_team = teams[away_choice]
        print(f"✅ Deplasman: {away_team}")
    except (ValueError, IndexError):
        print("⚠️ Geçersiz seçim! Varsayılan olarak 'Borussia Dortmund' seçildi.")
        away_team = "Borussia Dortmund"
    
    # Ev sahibi kadrosunu göster ve seçim yap
    home_players = display_players(predictor.team_dict, home_team)
    home_starters = select_players_interactive(home_players, "başlangıç")
    home_subs = select_players_interactive(home_players, "yedek")
    home_lineup = home_starters + home_subs
    
    # Deplasman kadrosunu göster ve seçim yap
    away_players = display_players(predictor.team_dict, away_team)
    away_starters = select_players_interactive(away_players, "başlangıç")
    away_subs = select_players_interactive(away_players, "yedek")
    away_lineup = away_starters + away_subs
    
    # Kadroları otomatik doldur (eğer kullanıcı seçmediyse)
    if not home_lineup:
        df_home = predictor.team_dict[home_team]
        home_lineup = [{'id': idx, 'name': row['Player'], 'pos': row.get('Pos', 'N/A'), 
                       'rating': row.get('PlayerRating', 65)} 
                      for idx, row in df_home.nlargest(18, 'PlayerRating').iterrows()]
        print("ℹ️ Ev sahibi kadrosu en iyi oyuncularla otomatik dolduruldu")
    
    if not away_lineup:
        df_away = predictor.team_dict[away_team]
        away_lineup = [{'id': idx, 'name': row['Player'], 'pos': row.get('Pos', 'N/A'), 
                       'rating': row.get('PlayerRating', 65)} 
                      for idx, row in df_away.nlargest(18, 'PlayerRating').iterrows()]
        print("ℹ️ Deplasman kadrosu en iyi oyuncularla otomatik dolduruldu")
    
    # Tahmin yap
    print(f"\n🔮 Tahmin yapılıyor: {home_team} vs {away_team}")
    
    # YENİ EKLENEN KOD: Kadro seçimi ve rating hesaplama
    home_start_idx, home_bench_idx, home_rating, home_bench_rating = select_starting_and_bench(home_lineup)
    away_start_idx, away_bench_idx, away_rating, away_bench_rating = select_starting_and_bench(away_lineup)
    
    # Debug çıktısı
    print(f"⭐ Ev Sahibi Rating (11): {home_rating:.2f}, Yedek Ortalaması: {home_bench_rating:.2f}")
    print(f"⭐ Deplasman Rating (11): {away_rating:.2f}, Yedek Ortalaması: {away_bench_rating:.2f}")
    
    # Ek özelliklerle tahmin yap
    additional_features = {
    'Home_AvgRating': home_rating,
    'Away_AvgRating': away_rating,
    'Rating_Diff': home_rating - away_rating,
    'Total_AvgRating': home_rating + away_rating,
    # Yedek ortalamalarını da ekle
    'Home_BenchRating': home_bench_rating,
    'Away_BenchRating': away_bench_rating,
    # Pozisyon bazlı ratingleri gerçek kadrodan hesapla
    'Home_GK_Rating': next((p['rating'] for p in home_lineup if p['pos'].startswith('GK')), 65.0),
    'Home_DF_Rating': np.mean([p['rating'] for p in home_lineup if 'BACK' in p['pos'] or 'CB' in p['pos'] or p['pos'].startswith('D')]) if any('BACK' in p['pos'] or 'CB' in p['pos'] or p['pos'].startswith('D') for p in home_lineup) else 65.0,
    'Home_MF_Rating': np.mean([p['rating'] for p in home_lineup if 'MID' in p['pos'] or p['pos'].startswith('M')]) if any('MID' in p['pos'] or p['pos'].startswith('M') for p in home_lineup) else 65.0,
    'Home_FW_Rating': np.mean([p['rating'] for p in home_lineup if 'FORWARD' in p['pos'] or 'WING' in p['pos'] or p['pos'].startswith('F')]) if any('FORWARD' in p['pos'] or 'WING' in p['pos'] or p['pos'].startswith('F') for p in home_lineup) else 65.0,
    'Away_GK_Rating': next((p['rating'] for p in away_lineup if p['pos'].startswith('GK')), 65.0),
    'Away_DF_Rating': np.mean([p['rating'] for p in away_lineup if 'BACK' in p['pos'] or 'CB' in p['pos'] or p['pos'].startswith('D')]) if any('BACK' in p['pos'] or 'CB' in p['pos'] or p['pos'].startswith('D') for p in away_lineup) else 65.0,
    'Away_MF_Rating': np.mean([p['rating'] for p in away_lineup if 'MID' in p['pos'] or p['pos'].startswith('M')]) if any('MID' in p['pos'] or p['pos'].startswith('M') for p in away_lineup) else 65.0,
    'Away_FW_Rating': np.mean([p['rating'] for p in away_lineup if 'FORWARD' in p['pos'] or 'WING' in p['pos'] or p['pos'].startswith('F')]) if any('FORWARD' in p['pos'] or 'WING' in p['pos'] or p['pos'].startswith('F') for p in away_lineup) else 65.0,
    }

    
    result = predictor.predict_match(home_team, away_team, additional_features=additional_features)
    
    if 'error' in result:
        print(f"❌ Tahmin hatası: {result['error']}")
        return
    
    result['match_info'] = {
        'home_team': home_team,
        'away_team': away_team,
        'date': datetime.now()
    }
    
    # Additional features'ı da sonuçlara ekle
    result['additional_features'] = additional_features
    
    # Sonuçları göster
    display_prediction_details(result)
    
    # Görselleştirme
    visualize_prediction(result, f"tahmin_{home_team}_vs_{away_team}.png")
    
    # Excel'e kaydet
    save_prediction_to_excel(result, home_team, away_team, home_lineup, away_lineup)
    
    # Takım formlarını göster
    print(f"\n📊 Takım Form Durumları:")
    print("=" * 40)
    
    for team in [home_team, away_team]:
        form = predictor.get_team_form(team)
        if form:
            print(f"\n{team} Son 5 Maç:")
            wins = sum(1 for m in form if m['result'] == 'W')
            draws = sum(1 for m in form if m['result'] == 'D')
            losses = sum(1 for m in form if m['result'] == 'L')
            print(f"   📈 Form: {wins} Galibiyet, {draws} Beraberlik, {losses} Mağlubiyet")
            
            for i, match in enumerate(form[:3], 1):  # Son 3 maç
                print(f"   {i}. {match['date'].strftime('%d.%m.%Y')} {match['opponent']} "
                      f"{match['result']} ({match['score']}) {'(E)' if match['is_home'] else '(D)'}")
        else:
            print(f"\n{team} için form bilgisi bulunamadı")

if __name__ == "__main__":
    main()