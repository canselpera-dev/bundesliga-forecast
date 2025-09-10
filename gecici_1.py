#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bundesliga_predictor.py - Geliştirilmiş Sürüm
Eğitim kodundaki tüm gelişmiş özellikleri içeren interaktif tahmin aracı
"""

import os
import re
import joblib
import difflib
import unicodedata
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

warnings.filterwarnings("ignore")

# ---------- CONFIG ----------
RANDOM_STATE = 42
DATA_PATH = "data/bundesliga_matches_2023_2025_final_fe_team_values_cleaned.xlsx"
PLAYER_DATA_PATH = "data/final_bundesliga_dataset_complete.xlsx"
MODEL_PATH = "models/bundesliga_model_final.pkl"
FEATURE_INFO_PATH = "models/feature_info.pkl"
PRED_HISTORY_PATH = "data/prediction_history.xlsx"

# Kullanıcının verdiği "güncel 18 takım" listesi
USER_TEAM_LIST = [
    "Bayern Münih", "Eintracht Frankfurt", "Köln", "Borussia Dortmund", "St. Pauli",
    "Wolfsburg", "Augsburg", "Stuttgart", "Hoffenheim", "Union Berlin",
    "RB Leipzig", "Bayer Leverkusen", "Mainz 05", "Mönchengladbach",
    "Hamburg", "Werder Bremen", "Heidenheim", "Freiburg"
]

# MODEL hedef label mapping
CLASS_LABELS = {0: "Draw", 1: "HomeWin", 2: "AwayWin"}

# Takım değerleri için varsayılan değerler
TOP_N_STARTERS = 11
TOP_N_SUBS = 7
STARTER_WEIGHT = 0.7
SUB_WEIGHT = 0.3

# Özellik listesi - SELECTED_FEATURES tanımı eklendi
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

# ---------- UTIL: isim normalizasyonu & fuzzy mapping ----------
def normalize_text(s):
    """Basit normalizasyon: lower, unicode->ascii, sadece alfanümerik+space."""
    if s is None:
        return ""
    s = str(s)
    s = s.strip().lower()
    s = unicodedata.normalize('NFKD', s)
    s = s.encode('ascii', 'ignore').decode('ascii')
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def best_match(target, candidates, cutoff=0.6):
    """
    difflib.get_close_matches ile en iyi eşleşmeyi döndür (yoksa None).
    """
    if not target or len(candidates) == 0:
        return None, 0.0
    nm = normalize_text(target)
    cand_norm = {normalize_text(c): c for c in candidates}
    matches = difflib.get_close_matches(nm, list(cand_norm.keys()), n=1, cutoff=cutoff)
    if matches:
        key = matches[0]
        score = difflib.SequenceMatcher(a=nm, b=key).ratio()
        return cand_norm[key], score
    # fallback: substring match
    for k_norm, k_orig in cand_norm.items():
        if nm in k_norm or k_norm in nm:
            score = difflib.SequenceMatcher(a=nm, b=k_norm).ratio()
            return k_orig, score
    # no match
    return None, 0.0

# ---------- DATA LOADING HELPERS ----------
def load_player_data(path=PLAYER_DATA_PATH):
    """Oyuncu datasetini yükler ve 'PlayerRating','Player','Team','Pos' sütunlarını normalize eder."""
    df = pd.read_excel(path)
    
    # Rating hesaplama
    if 'Rating' in df.columns:
        df['PlayerRating'] = df['Rating']
    elif 'fbref__Goal_Contribution' in df.columns:
        df['PlayerRating'] = df['fbref__Goal_Contribution'] * 2 + df.get('fbref__Min', 0) / 90 * 0.5
    else:
        df['PlayerRating'] = 65.0
    
    # Takım isimlerini normalize et
    if 'Team' in df.columns:
        df['Team'] = df['Team'].astype(str).str.strip()
    elif 'fbref__Squad' in df.columns:
        df['Team'] = df['fbref__Squad'].astype(str).str.strip()
    
    # Pozisyon bilgisi
    if 'Position' in df.columns:
        df['Pos'] = df['Position'].astype(str).str.upper().str.strip()
    elif 'fbref__Pos' in df.columns:
        df['Pos'] = df['fbref__Pos'].astype(str).str.upper().str.strip()
    
    # Value & Age (opsiyonel)
    if 'Value' not in df.columns:
        for alt in ['value_eur', 'market_value_eur', 'current_value_eur', 'value']:
            if alt in df.columns:
                df['Value'] = df[alt]
                break
    
    if 'Age' not in df.columns:
        for alt in ['age', 'Age_y', 'player_age']:
            if alt in df.columns:
                df['Age'] = df[alt]
                break
    
    return df

def team_players_dict(df_players):
    d = {}
    if 'Team' not in df_players.columns:
        return d
    for team in sorted(df_players['Team'].dropna().unique()):
        team_df = df_players[df_players['Team'] == team].copy().reset_index(drop=True)
        d[team] = team_df
    return d

# ---------- MATCHES normalization (momentum için) ----------
def standardize_matches_df(df):
    """
    Gelen maç dataframe'ini "Date","Home Team","Away Team","Home Goals","Away Goals","Result" sütunlarına dönüştürür.
    """
    df2 = df.copy()
    cols = {c: c for c in df2.columns}
    
    # Sütun isimlerini standardize et
    date_candidates = ['Date','date','utcDate','match_date', 'kickoff', 'kickoff_time']
    home_candidates = ['HomeTeam','homeTeam','home_team','home_team_name','Home Team','home']
    away_candidates = ['AwayTeam','awayTeam','away_team','away_team_name','Away Team','away']
    hg_candidates = ['HomeGoals','FTHG','home_goals','score.fullTime.home','home_goals','home_score']
    ag_candidates = ['AwayGoals','FTAG','away_goals','score.fullTime.away','away_goals','away_score']
    res_candidates = ['Result','FTR','result','winner','outcome']

    def find_col(cands):
        for c in cands:
            if c in df2.columns:
                return c
        lowcols = {col.lower():col for col in df2.columns}
        for c in cands:
            if c.lower() in lowcols:
                return lowcols[c.lower()]
        return None

    date_col = find_col(date_candidates)
    home_col = find_col(home_candidates)
    away_col = find_col(away_candidates)
    hg_col = find_col(hg_candidates)
    ag_col = find_col(ag_candidates)
    res_col = find_col(res_candidates)

    # Build standardized df
    out = pd.DataFrame()
    if date_col is not None:
        out['Date'] = pd.to_datetime(df2[date_col])
    else:
        out['Date'] = pd.NaT

    out['Home Team'] = df2[home_col] if home_col is not None else df2.get('homeTeam.name', df2.get('homeTeam', df2.iloc[:,0]))
    out['Away Team'] = df2[away_col] if away_col is not None else df2.get('awayTeam.name', df2.get('awayTeam', df2.iloc[:,1] if df2.shape[1]>1 else df2.iloc[:,0]))

    if hg_col is not None:
        out['Home Goals'] = pd.to_numeric(df2[hg_col], errors='coerce').fillna(0).astype(int)
    else:
        out['Home Goals'] = 0
    if ag_col is not None:
        out['Away Goals'] = pd.to_numeric(df2[ag_col], errors='coerce').fillna(0).astype(int)
    else:
        out['Away Goals'] = 0

    # Result normalization
    if res_col is not None:
        def map_res(v):
            if pd.isna(v): return 'D'
            s = str(v).strip().upper()
            if s in ['H','HOME','1']: return 'H'
            if s in ['A','AWAY','2']: return 'A'
            return 'D'
        out['Result'] = df2[res_col].apply(map_res)
    else:
        def infer_result(row):
            if row['Home Goals'] > row['Away Goals']: return 'H'
            if row['Home Goals'] < row['Away Goals']: return 'A'
            return 'D'
        out['Result'] = out.apply(infer_result, axis=1)

    # Format Date nicely
    try:
        out['Date'] = pd.to_datetime(out['Date'])
    except Exception:
        pass

    # Normalize team names as strings
    out['Home Team'] = out['Home Team'].astype(str).str.strip()
    out['Away Team'] = out['Away Team'].astype(str).str.strip()
    
    # Result_Numeric ekle
    def get_result_numeric(row):
        if row['Result'] == 'H':
            return 1  # Home win
        elif row['Result'] == 'A':
            return 2  # Away win
        else:
            return 0  # Draw
    
    out['Result_Numeric'] = out.apply(get_result_numeric, axis=1)
    
    return out

# ---------- VERİ HAZIRLAMA VE ZENGİNLEŞTİRME ----------
def prepare_and_enrich_dataset(df_matches, df_players):
    """
    Eksik özellikleri otomatik olarak hesaplayarak dataseti zenginleştirir
    """
    print("🔧 Veri hazırlama ve zenginleştirme başlıyor...")
    
    # Takım isimlerini standartlaştır
    if 'homeTeam.name' in df_matches.columns and 'HomeTeam' not in df_matches.columns:
        df_matches['HomeTeam'] = df_matches['homeTeam.name']
    if 'awayTeam.name' in df_matches.columns and 'AwayTeam' not in df_matches.columns:
        df_matches['AwayTeam'] = df_matches['awayTeam.name']
    
    # IsDerby sütunu yoksa oluştur
    if 'IsDerby' not in df_matches.columns:
        big_teams = ['Bayern Munich', 'Borussia Dortmund', 'Schalke 04', 'Hamburg SV', 
                    'Borussia Mönchengladbach', 'Bayer Leverkusen', 'VfB Stuttgart']
        
        def is_derby(home_team, away_team):
            if home_team in big_teams and away_team in big_teams:
                return 1
            return 0
        
        df_matches['IsDerby'] = df_matches.apply(
            lambda row: is_derby(row.get('HomeTeam', ''), row.get('AwayTeam', '')), axis=1
        )
    
    # Form ve momentum özelliklerini hesapla
    df_matches = calculate_form_features(df_matches)
    
    # Takım ratinglerini hesapla
    df_matches = compute_ratings_for_matches(df_matches, df_players)
    
    # Eksik özellikleri kontrol et ve gerekirse hesapla
    df_matches = calculate_missing_features(df_matches)
    
    print("✅ Veri zenginleştirme tamamlandı!")
    return df_matches

def calculate_form_features(df):
    """Form ve momentum özelliklerini hesaplar"""
    print("📊 Form ve momentum özellikleri hesaplanıyor...")
    
    # Takım listesi
    teams = pd.concat([df['Home Team'], df['Away Team']]).unique()
    
    # Yeni özellikleri başlat
    for col in ['home_form', 'away_form', 'homeTeam_GoalsScored_5', 'homeTeam_GoalsConceded_5',
                'awayTeam_GoalsScored_5', 'awayTeam_GoalsConceded_5', 'homeTeam_Momentum', 'awayTeam_Momentum']:
        if col not in df.columns:
            df[col] = 0.0
    
    # Her takım için form hesapla
    for team in teams:
        if team is None or pd.isna(team):
            continue
            
        team_matches = df[(df['Home Team'] == team) | (df['Away Team'] == team)].copy()
        if len(team_matches) == 0:
            continue
            
        team_matches = team_matches.sort_values('Date').reset_index(drop=True)
        
        for i, (idx, match) in enumerate(team_matches.iterrows()):
            if i < 5:  # İlk 5 maç için yeterli veri yok
                form = 0.5  # Nötr form
                goals_scored_5 = 0
                goals_conceded_5 = 0
            else:
                # Son 5 maçı al
                last_5 = team_matches.iloc[max(0, i-5):i]
                points = 0
                goals_scored_5 = 0
                goals_conceded_5 = 0
                
                for _, m in last_5.iterrows():
                    if m['Home Team'] == team:
                        home_goals = m.get('Home Goals', 0)
                        away_goals = m.get('Away Goals', 0)
                        goals_scored_5 += home_goals
                        goals_conceded_5 += away_goals
                        
                        if home_goals > away_goals:
                            points += 3  # Galibiyet
                        elif home_goals == away_goals:
                            points += 1  # Beraberlik
                    else:
                        home_goals = m.get('Home Goals', 0)
                        away_goals = m.get('Away Goals', 0)
                        goals_scored_5 += away_goals
                        goals_conceded_5 += home_goals
                        
                        if away_goals > home_goals:
                            points += 3  # Galibiyet
                        elif away_goals == home_goals:
                            points += 1  # Beraberlik
                
                # Formu 0-1 arasında normalize et (maksimum 15 puan üzerinden)
                form = points / 15 if points > 0 else 0.3
                
                # Momentum (gol averajı)
                momentum = goals_scored_5 - goals_conceded_5
            
            # Değerleri dataframe'e yaz
            if team_matches.iloc[i]['Home Team'] == team:
                df.loc[idx, 'home_form'] = form
                df.loc[idx, 'homeTeam_GoalsScored_5'] = goals_scored_5
                df.loc[idx, 'homeTeam_GoalsConceded_5'] = goals_conceded_5
                df.loc[idx, 'homeTeam_Momentum'] = goals_scored_5 - goals_conceded_5
            else:
                df.loc[idx, 'away_form'] = form
                df.loc[idx, 'awayTeam_GoalsScored_5'] = goals_scored_5
                df.loc[idx, 'awayTeam_GoalsConceded_5'] = goals_conceded_5
                df.loc[idx, 'awayTeam_Momentum'] = goals_scored_5 - goals_conceded_5
    
    # Form farkı
    df['Form_Diff'] = df['home_form'] - df['away_form']
    
    return df

def calculate_missing_features(df):
    """Eksik özellikleri kontrol et ve gerekirse hesapla"""
    print("🔍 Eksik özellikler kontrol ediliyor...")
    
    # Özellikleri and varsayılan değerleri
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
        'home_current_value_eur': 0.0,
        'away_current_value_eur': 0.0,
        'home_squad_avg_age': 25.0,
        'away_squad_avg_age': 25.0,
        'home_value_change_pct': 0.0,
        'away_value_change_pct': 0.0,
        'IsDerby': 0
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

# ---------- last5 + momentum ----------
def get_last_matches_and_momentum(team_name, matches_df, n=5):
    """
    Team'in son n maçını alır ve momentum'u (0-100) döndürür + summary.
    """
    if matches_df is None or matches_df.empty:
        return {"momentum": 0, "summary": "(Veri yok)"}

    team_norm = normalize_text(team_name)
    df = matches_df.copy()
    
    # ensure Date is datetime
    if 'Date' in df.columns:
        try:
            df['Date'] = pd.to_datetime(df['Date'])
        except Exception:
            pass
    
    # filter rows where team played
    played = df[(df['Home Team'].astype(str).str.lower().str.strip().apply(normalize_text) == team_norm) |
                (df['Away Team'].astype(str).str.lower().str.strip().apply(normalize_text) == team_norm)].copy()

    if played.empty:
        return {"momentum": 0, "summary": "(Veri yok)"}

    played = played.sort_values('Date', ascending=False).head(n)

    points = 0
    win = draw = loss = 0
    lines = []
    for _, row in played.iterrows():
        home = normalize_text(row['Home Team'])
        away = normalize_text(row['Away Team'])
        hg = int(row.get('Home Goals', 0))
        ag = int(row.get('Away Goals', 0))

        # identify if our team was home or away
        is_home = (home == team_norm)
        opponent = row['Away Team'] if is_home else row['Home Team']
        score_str = f"({hg}-{ag})"

        res = row.get('Result', 'D')
        if isinstance(res, str):
            r = res.strip().upper()
            if r in ['H','HOME','1']:
                match_result = 'H'
            elif r in ['A','AWAY','2']:
                match_result = 'A'
            else:
                if hg > ag: match_result = 'H'
                elif hg < ag: match_result = 'A'
                else: match_result = 'D'
        else:
            if hg > ag: match_result = 'H'
            elif hg < ag: match_result = 'A'
            else: match_result = 'D'

        # get team's result
        if match_result == 'D':
            team_res = 'D'
            points += 1
            draw += 1
        else:
            if (match_result == 'H' and is_home) or (match_result == 'A' and not is_home):
                team_res = 'W'; points += 3; win += 1
            else:
                team_res = 'L'; loss += 1

        loc = 'E' if is_home else 'D'
        date_str = row['Date'].strftime("%d.%m.%Y") if (not pd.isna(row['Date'])) else "TAR"
        lines.append(f"{date_str} {opponent} {team_res} {score_str} ({loc})")

    max_points = n * 3
    momentum = round(points / max_points * 100, 1) if max_points > 0 else 0
    summary = f"📈 Form: {win} Galibiyet, {draw} Beraberlik, {loss} Mağlubiyet\n"
    for i, l in enumerate(lines, 1):
        summary += f"   {i}. {l}\n"
    return {"momentum": momentum, "summary": summary}

# ---------- roster & rating helpers ----------
def pos_group(pos_str):
    if not isinstance(pos_str, str):
        return 'MF'
    p = pos_str.upper()
    if 'GK' in p or p == 'G' or 'GOAL' in p: return 'GK'
    if p.startswith('D') or 'BACK' in p or 'DEFENDER' in p or 'DEFENSIVE' in p: return 'DF'
    if p.startswith('M') or 'MID' in p or 'CENTRAL' in p: return 'MF'
    if p.startswith('F') or 'FW' in p or 'FORWARD' in p or 'STRIKER' in p or 'WINGER' in p: return 'FW'
    return 'MF'

def select_topn_by_rating(df_team, n):
    if df_team is None or df_team.empty or 'PlayerRating' not in df_team.columns:
        return []
    return df_team['PlayerRating'].dropna().sort_values(ascending=False).index.tolist()[:n]

def avg_of_selected_players(df_team, idxs):
    if df_team is None or len(idxs) == 0:
        return np.nan, {'GK':np.nan,'DF':np.nan,'MF':np.nan,'FW':np.nan}
    sel = df_team.loc[idxs]
    ratings = sel['PlayerRating'].dropna()
    overall = ratings.mean() if not ratings.empty else np.nan
    pos_means = {}
    for pos in ['GK','DF','MF','FW']:
        pos_means[pos] = sel[sel.get('Pos', '').apply(lambda x: pos_group(x) == pos)]['PlayerRating'].dropna().mean() if 'PlayerRating' in sel else np.nan
    return overall, pos_means

def compute_team_rating_from_lineup(df_team, starter_idxs, sub_idxs, starter_weight=0.7, sub_weight=0.3):
    starter_mean, starter_pos = avg_of_selected_players(df_team, starter_idxs)
    sub_mean, sub_pos = avg_of_selected_players(df_team, sub_idxs)
    if np.isnan(starter_mean) and not np.isnan(sub_mean):
        team_rating = sub_mean
    elif np.isnan(sub_mean) and not np.isnan(starter_mean):
        team_rating = starter_mean
    elif np.isnan(starter_mean) and np.isnan(sub_mean):
        team_rating = np.nan
    else:
        team_rating = (starter_mean * starter_weight) + (sub_mean * sub_weight)
    pos_combined = {}
    for pos in ['GK','DF','MF','FW']:
        s = starter_pos.get(pos, np.nan)
        b = sub_pos.get(pos, np.nan)
        if pd.isna(s) and not pd.isna(b): pos_combined[pos] = b
        elif pd.isna(b) and not pd.isna(s): pos_combined[pos] = s
        elif pd.isna(s) and pd.isna(b): pos_combined[pos] = np.nan
        else: pos_combined[pos] = (s * starter_weight) + (b * sub_weight)
    return team_rating, pos_combined

# ---------- MATCH FEATURE ENGINEERING ----------
def compute_ratings_for_matches(df_matches, df_players):
    df_players = df_players.copy()
    team_dict = team_players_dict(df_players)
    
    # Takım isimlerini eşleştir
    if 'homeTeam.name' in df_matches.columns and 'HomeTeam' not in df_matches.columns:
        df_matches['HomeTeam'] = df_matches['homeTeam.name']
    if 'awayTeam.name' in df_matches.columns and 'AwayTeam' not in df_matches.columns:
        df_matches['AwayTeam'] = df_matches['awayTeam.name']

    cols_to_add = ['Home_AvgRating','Away_AvgRating','Total_AvgRating','Rating_Diff',
                   'Home_GK_Rating','Home_DF_Rating','Home_MF_Rating','Home_FW_Rating',
                   'Away_GK_Rating','Away_DF_Rating','Away_MF_Rating','Away_FW_Rating']
    for c in cols_to_add:
        if c not in df_matches.columns: df_matches[c]=np.nan

    for idx,row in df_matches.iterrows():
        home=row.get('HomeTeam'); away=row.get('AwayTeam')
        df_home=team_dict.get(home,pd.DataFrame()); df_away=team_dict.get(away,pd.DataFrame())
        home_starters=[]; home_subs=[]; away_starters=[]; away_subs=[]

        # Lineup verisi yoksa en iyi oyuncuları seç
        home_starters=select_topn_by_rating(df_home,TOP_N_STARTERS)
        away_starters=select_topn_by_rating(df_away,TOP_N_STARTERS)

        if len(home_subs)==0:
            all_idxs=df_home['PlayerRating'].dropna().sort_values(ascending=False).index.tolist() if 'PlayerRating' in df_home else []
            home_subs=[i for i in all_idxs if i not in home_starters][:TOP_N_SUBS]
        if len(away_subs)==0:
            all_idxs=df_away['PlayerRating'].dropna().sort_values(ascending=False).index.tolist() if 'PlayerRating' in df_away else []
            away_subs=[i for i in all_idxs if i not in away_starters][:TOP_N_SUBS]

        h_rating,h_pos=compute_team_rating_from_lineup(df_home,home_starters,home_subs)
        a_rating,a_pos=compute_team_rating_from_lineup(df_away,away_starters,away_subs)

        df_matches.at[idx,'Home_AvgRating']=h_rating
        df_matches.at[idx,'Away_AvgRating']=a_rating
        if not pd.isna(h_rating) and not pd.isna(a_rating):
            df_matches.at[idx,'Total_AvgRating']=h_rating+a_rating
            df_matches.at[idx,'Rating_Diff']=h_rating-a_rating

        df_matches.at[idx,'Home_GK_Rating']=h_pos.get('GK',np.nan)
        df_matches.at[idx,'Home_DF_Rating']=h_pos.get('DF',np.nan)
        df_matches.at[idx,'Home_MF_Rating']=h_pos.get('MF',np.nan)
        df_matches.at[idx,'Home_FW_Rating']=h_pos.get('FW',np.nan)

        df_matches.at[idx,'Away_GK_Rating']=a_pos.get('GK',np.nan)
        df_matches.at[idx,'Away_DF_Rating']=a_pos.get('DF',np.nan)
        df_matches.at[idx,'Away_MF_Rating']=a_pos.get('MF',np.nan)
        df_matches.at[idx,'Away_FW_Rating']=a_pos.get('FW',np.nan)

    global_avg=df_players['PlayerRating'].mean() if 'PlayerRating' in df_players else 65.0
    df_matches['Home_AvgRating'].fillna(global_avg,inplace=True)
    df_matches['Away_AvgRating'].fillna(global_avg,inplace=True)
    df_matches['Total_AvgRating'].fillna(df_matches['Home_AvgRating']+df_matches['Away_AvgRating'],inplace=True)
    df_matches['Rating_Diff'].fillna(df_matches['Home_AvgRating']-df_matches['Away_AvgRating'],inplace=True)

    for pos in ['GK','DF','MF','FW']:
        pos_mean = df_players[df_players['Pos'].apply(pos_group)==pos]['PlayerRating'].mean() if 'PlayerRating' in df_players else global_avg
        df_matches[f'Home_{pos}_Rating'].fillna(pos_mean,inplace=True)
        df_matches[f'Away_{pos}_Rating'].fillna(pos_mean,inplace=True)

    return df_matches

# ---------- feature builder & model prediction ----------
def build_feature_row(home_team, away_team, home_team_df, away_team_df,
                      home_start_idxs, home_sub_idxs, away_start_idxs, away_sub_idxs,
                      matches_std):
    """Feature sözlüğü oluşturur (model input)."""
    # Ratings
    home_rating, home_pos = compute_team_rating_from_lineup(home_team_df, home_start_idxs, home_sub_idxs)
    away_rating, away_pos = compute_team_rating_from_lineup(away_team_df, away_start_idxs, away_sub_idxs)

    # Bench averages
    home_bench_idxs = [i for i in (home_sub_idxs if home_sub_idxs else [])]
    away_bench_idxs = [i for i in (away_sub_idxs if away_sub_idxs else [])]
    home_bench_mean = home_team_df.loc[home_bench_idxs]['PlayerRating'].mean() if len(home_bench_idxs)>0 else np.nan
    away_bench_mean = away_team_df.loc[away_bench_idxs]['PlayerRating'].mean() if len(away_bench_idxs)>0 else np.nan

    # Momentum & last5 forms
    home_form_info = get_last_matches_and_momentum(home_team, matches_std, n=5)
    away_form_info = get_last_matches_and_momentum(away_team, matches_std, n=5)

    # Form_Diff (use normalized 0-1 scale as training did)
    home_form_norm = home_form_info['momentum'] / 100.0
    away_form_norm = away_form_info['momentum'] / 100.0
    form_diff = (home_form_norm - away_form_norm)

    # Goals scored/conceded last 5 - attempt to compute from matches_std
    def goals_5(team):
        d = matches_std.copy()
        tnorm = normalize_text(team)
        played = d[(d['Home Team'].apply(normalize_text) == tnorm) | (d['Away Team'].apply(normalize_text) == tnorm)].sort_values('Date', ascending=False).head(5)
        if played.empty:
            return 0, 0
        gs = gc = 0
        for _, r in played.iterrows():
            hg = int(r.get('Home Goals', 0))
            ag = int(r.get('Away Goals', 0))
            if normalize_text(r['Home Team']) == tnorm:
                gs += hg; gc += ag
            else:
                gs += ag; gc += hg
        return gs, gc

    h_gs5, h_gc5 = goals_5(home_team)
    a_gs5, a_gc5 = goals_5(away_team)

    # Squad ages and values (if exist in team dfs)
    def squad_average_age(team_df):
        if team_df is None or team_df.empty: return np.nan
        if 'Age' in team_df.columns:
            try:
                return float(team_df['Age'].dropna().astype(float).mean())
            except Exception:
                return np.nan
        return np.nan

    def squad_value_sum(team_df):
        if team_df is None or team_df.empty: return np.nan
        for c in ['Value', 'value_eur', 'market_value_eur', 'current_value_eur', 'value']:
            if c in team_df.columns:
                vals = pd.to_numeric(team_df[c], errors='coerce').fillna(0)
                return float(vals.sum())
        return np.nan

    home_age = squad_average_age(home_team_df)
    away_age = squad_average_age(away_team_df)
    home_value = squad_value_sum(home_team_df)
    away_value = squad_value_sum(away_team_df)

    # IsDerby heuristic (aynı şehir / common big-team pairs)
    big_teams = ['Bayern Munich','Borussia Dortmund','Schalke 04','Hamburger SV','Borussia Mönchengladbach','Bayer Leverkusen','VfB Stuttgart']
    try:
        is_derby = 1 if (home_team in big_teams and away_team in big_teams) else 0
    except Exception:
        is_derby = 0

    # pos-based ratings
    feature_row = {}
    feature_row['Home_AvgRating'] = float(home_rating) if not pd.isna(home_rating) else float(np.nanmean([home_team_df['PlayerRating'].mean() if 'PlayerRating' in home_team_df else 65.0]))
    feature_row['Away_AvgRating'] = float(away_rating) if not pd.isna(away_rating) else float(np.nanmean([away_team_df['PlayerRating'].mean() if 'PlayerRating' in away_team_df else 65.0]))
    feature_row['Total_AvgRating'] = feature_row['Home_AvgRating'] + feature_row['Away_AvgRating']
    feature_row['Rating_Diff'] = feature_row['Home_AvgRating'] - feature_row['Away_AvgRating']

    # pos ratings
    feature_row['Home_GK_Rating'] = float(home_pos.get('GK', np.nan))
    feature_row['Home_DF_Rating'] = float(home_pos.get('DF', np.nan))
    feature_row['Home_MF_Rating'] = float(home_pos.get('MF', np.nan))
    feature_row['Home_FW_Rating'] = float(home_pos.get('FW', np.nan))

    feature_row['Away_GK_Rating'] = float(away_pos.get('GK', np.nan))
    feature_row['Away_DF_Rating'] = float(away_pos.get('DF', np.nan))
    feature_row['Away_MF_Rating'] = float(away_pos.get('MF', np.nan))
    feature_row['Away_FW_Rating'] = float(away_pos.get('FW', np.nan))

    # form/momentum/goals
    feature_row['home_form'] = home_form_norm
    feature_row['away_form'] = away_form_norm
    feature_row['Form_Diff'] = form_diff
    feature_row['homeTeam_GoalsScored_5'] = int(h_gs5)
    feature_row['homeTeam_GoalsConceded_5'] = int(h_gc5)
    feature_row['awayTeam_GoalsScored_5'] = int(a_gs5)
    feature_row['awayTeam_GoalsConceded_5'] = int(a_gc5)
    feature_row['homeTeam_Momentum'] = int(round(home_form_info['momentum']))
    feature_row['awayTeam_Momentum'] = int(round(away_form_info['momentum']))

    # extras
    feature_row['IsDerby'] = int(is_derby)
    feature_row['home_current_value_eur'] = float(home_value) if not pd.isna(home_value) else 0.0
    feature_row['away_current_value_eur'] = float(away_value) if not pd.isna(away_value) else 0.0
    feature_row['home_squad_avg_age'] = float(home_age) if not pd.isna(home_age) else 0.0
    feature_row['away_squad_avg_age'] = float(away_age) if not pd.isna(away_age) else 0.0
    feature_row['home_value_change_pct'] = 0.0
    feature_row['away_value_change_pct'] = 0.0

    # Bench ratings (extra fields used in prints)
    feature_row['Home_BenchRating'] = float(home_bench_mean) if not pd.isna(home_bench_mean) else np.nan
    feature_row['Away_BenchRating'] = float(away_bench_mean) if not pd.isna(away_bench_mean) else np.nan

    # ensure all default features exist
    for f in SELECTED_FEATURES:
        if f not in feature_row:
            feature_row[f] = 0.0

    return feature_row, home_form_info, away_form_info

# ---------- save prediction history ----------
def append_prediction_history(row_dict, path=PRED_HISTORY_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df_new = pd.DataFrame([row_dict])
    if os.path.exists(path):
        try:
            df_existing = pd.read_excel(path)
            df_out = pd.concat([df_existing, df_new], ignore_index=True)
            df_out.to_excel(path, index=False)
        except Exception:
            df_new.to_excel(path, index=False)
    else:
        df_new.to_excel(path, index=False)
    return path

# ---------- FEATURE IMPORTANCE ANALIZI ----------
def analyze_feature_importance(model, feature_names, top_n=15):
    """Feature importance analizi ve görselleştirme"""
    if hasattr(model.named_steps['lgbm'], 'feature_importances_'):
        importances = model.named_steps['lgbm'].feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\n🏆 Feature Importance Ranking:")
        for f in range(min(top_n, len(feature_names))):
            print(f"{f+1}. {feature_names[indices[f]]} ({importances[indices[f]]:.4f})")
        
        # Görselleştirme
        plt.figure(figsize=(12, 8))
        plt.title("Feature Importances")
        plt.bar(range(min(top_n, len(feature_names))), 
                importances[indices[:top_n]], align="center")
        plt.xticks(range(min(top_n, len(feature_names))), 
                  [feature_names[i] for i in indices[:top_n]], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('models/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Önemli feature'ları seç (ortalama üstündekiler)
        importance_threshold = np.mean(importances)
        important_features = [feature_names[i] for i in indices if importances[i] > importance_threshold]
        
        print(f"\n📈 Importance threshold: {importance_threshold:.4f}")
        print(f"✅ Seçilen önemli feature sayısı: {len(important_features)}/{len(feature_names)}")
        
        return important_features
    
    print("⚠️ Feature importance bulunamadı")
    return feature_names

# ---------- MAIN interactive flow ----------
def main():
    print("\n🏆 Bundesliga Predictor - Geliştirilmiş İnteraktif Tahmin")
    print("=" * 60)
    
    # 1) yükle verileri
    print("🔁 Veri yükleniyor...")
    players_df = load_player_data(PLAYER_DATA_PATH)
    
    try:
        matches_raw = pd.read_excel(DATA_PATH)
        matches_raw.columns = [col.strip().replace(' ', '_') for col in matches_raw.columns]
    except Exception:
        try:
            matches_raw = pd.read_csv(DATA_PATH)
        except Exception:
            matches_raw = pd.DataFrame()

    # Veriyi zenginleştir
    matches_std = prepare_and_enrich_dataset(matches_raw, players_df) if not matches_raw.empty else pd.DataFrame()

    # 2) dataset'ten bulunan takım isimleri
    team_names_candidates = set()
    if 'Team' in players_df.columns:
        team_names_candidates.update(players_df['Team'].dropna().unique().tolist())
    if not matches_std.empty:
        team_names_candidates.update(matches_std['Home Team'].dropna().unique().tolist())
        team_names_candidates.update(matches_std['Away Team'].dropna().unique().tolist())

    team_names_candidates = sorted([t for t in team_names_candidates if str(t).strip()!=''])
    if len(team_names_candidates) == 0:
        print("❗ HATA: Dataset içinde takım isimleri bulunamadı. PLAYER_DATA_PATH ve DATA_PATH yolunu kontrol et.")
        return

    # 3) user-provided 18 teams ile eşleştirme (fuzzy)
    mapping = {}
    scores = {}
    for u in USER_TEAM_LIST:
        match, score = best_match(u, team_names_candidates, cutoff=0.5)
        mapping[u] = match if match is not None else None
        scores[u] = score

    # Print mapping summary
    print("\n🔎 Sağladığın 18 takımı dataset'teki isimlerle eşleştiriyorum (otomatik):")
    for i, u in enumerate(USER_TEAM_LIST, 1):
        mapped = mapping[u] if mapping[u] is not None else "— (eşleşme bulunamadı)"
        sc = scores[u]
        print(f" {i:2d}. {u}  ->  {mapped}  (score={sc:.2f})")

    # Build final team list to show the user (unique)
    final_teams = sorted(set([m for m in mapping.values() if m is not None]))
    if len(final_teams) == 0:
        print("❗ Hiçbir eşleşme bulunamadı. Dataset'teki tüm takımları göstereceğim.")
        final_teams = team_names_candidates

    # If dataset has more than 18 (e.g. 20) show only final_teams (mapped) else show all
    print("\n🏆 Mevcut Takımlar (dataset isimleri):")
    print("==================================================")
    for i, t in enumerate(final_teams, 1):
        print(f"{i:2d}. {t}")
    print("")

    # 4) Kullanıcıdan ev sahibi & deplasman seçimi
    def ask_team(prompt):
        while True:
            try:
                val = input(prompt).strip()
                if val == "":
                    print("Lütfen bir numara gir.")
                    continue
                idx = int(val)
                if 1 <= idx <= len(final_teams):
                    return final_teams[idx-1]
                else:
                    print("Geçersiz numara, listedeki bir numarayı girin.")
            except ValueError:
                print("Geçersiz giriş. Sadece sayı girin.")
    home_team = ask_team("🏠 Ev sahibi takım numarasını girin: ")
    print(f"✅ Ev sahibi: {home_team}")
    away_team = ask_team("✈️ Deplasman takımı numarasını girin: ")
    print(f"✅ Deplasman: {away_team}")

    # 5) takımların oyuncu listelerini hazırla
    team_dict = team_players_dict(players_df)
    home_team_df = team_dict.get(home_team, pd.DataFrame())
    away_team_df = team_dict.get(away_team, pd.DataFrame())

    def print_team_roster(tname, tdf):
        print(f"\n👥 {tname} Kadrosu:")
        print("============================================================")
        print(f"{'ID':4s} {'Isim':28s} {'Pozisyon':14s} {'Rating':>6s}")
        print("-"*60)
        if tdf is None or tdf.empty:
            print("(Takım verisi yok)")
            return
        for idx, row in tdf.reset_index().iterrows():
            name = str(row.get('Player', 'Player')).strip()[:26].ljust(26)
            pos = str(row.get('Pos', '')).strip()[:14].ljust(14)
            rating = row.get('PlayerRating', 0)
            print(f"{int(row['index']):4d} {name} {pos} {float(rating):6.1f}")

    print_team_roster(home_team, home_team_df)
    print("")
    print_team_roster(away_team, away_team_df)

    # 6) Kadro seçim inputları (IDs)
    def ask_idxs(prompt, valid_idxs, min_n=1, max_n=11):
        print(prompt + " (boş bırakırsan otomatik seçilecek)")
        s = input("Seçimleriniz (örn. 0,1,2...): ").strip()
        if s == "":
            return []
        try:
            parts = [int(x.strip()) for x in re.split(r'[,\s]+', s) if x.strip()!='']
            parts = [p for p in parts if p in valid_idxs]
            if len(parts) < min_n:
                print(f"Uyarı: En az {min_n} seçim bekleniyor, otomatik dolgu yapılacak.")
            return parts
        except Exception:
            print("Giriş okunamadı, otomatik seçim yapılacak.")
            return []

    home_valid_idxs = home_team_df.index.tolist() if (home_team_df is not None) else []
    away_valid_idxs = away_team_df.index.tolist() if (away_team_df is not None) else []

    home_start_idxs = ask_idxs(f"\n🔢 {home_team} Başlangıç oyuncularını seçin (ID'leri virgülle, aralık 11 için):", home_valid_idxs, min_n=0, max_n=11)
    home_sub_idxs = ask_idxs(f"\n🔢 {home_team} Yedek oyuncularını seçin (ID'leri virgülle):", home_valid_idxs, min_n=0, max_n=7)

    away_start_idxs = ask_idxs(f"\n🔢 {away_team} Başlangıç oyuncularını seçin (ID'leri virgülle, aralık 11 için):", away_valid_idxs, min_n=0, max_n=11)
    away_sub_idxs = ask_idxs(f"\n🔢 {away_team} Yedek oyuncularını seçin (ID'leri virgülle):", away_valid_idxs, min_n=0, max_n=7)

    # Auto-fill if empty: en iyi oyuncuları sırala
    if not home_start_idxs:
        home_start_idxs = select_topn_by_rating(home_team_df, 11)
        print("ℹ️ Ev sahibi kadrosu en iyi oyuncularla otomatik dolduruldu")
    if not home_sub_idxs:
        # bench: next best 7
        all_idxs = select_topn_by_rating(home_team_df, 18)
        home_sub_idxs = [i for i in all_idxs if i not in home_start_idxs][:7]
    if not away_start_idxs:
        away_start_idxs = select_topn_by_rating(away_team_df, 11)
        print("ℹ️ Deplasman kadrosu en iyi oyuncularla otomatik dolduruldu")
    if not away_sub_idxs:
        all_idxs = select_topn_by_rating(away_team_df, 18)
        away_sub_idxs = [i for i in all_idxs if i not in away_start_idxs][:7]

    # 7) Feature oluştur + momentum hesaplama
    print(f"\n🔮 Tahmin yapılıyor: {home_team} vs {away_team}")
    feat_row, home_form_info, away_form_info = build_feature_row(
        home_team, away_team,
        home_team_df, away_team_df,
        home_start_idxs, home_sub_idxs,
        away_start_idxs, away_sub_idxs,
        matches_std
    )

    # kısa ekip özetleri
    print(f"⭐ Ev Sahibi Rating (11): {feat_row['Home_AvgRating']:.2f}, Yedek Ortalaması: {feat_row.get('Home_BenchRating', 0):.2f}")
    print(f"⭐ Deplasman Rating (11): {feat_row['Away_AvgRating']:.2f}, Yedek Ortalaması: {feat_row.get('Away_BenchRating', 0):.2f}")

    # 8) Modeli yükle ve tahmin et
    model = None
    feature_list = SELECTED_FEATURES
    try:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
            # feature info
            if os.path.exists(FEATURE_INFO_PATH):
                f_info = joblib.load(FEATURE_INFO_PATH)
                feature_list = f_info.get('important_features', f_info.get('all_features', SELECTED_FEATURES))
                best_params = f_info.get('best_params', {})
                print(f"📋 Model hiperparametreleri: {best_params}")
    except Exception as e:
        print("⚠️ Model yüklenirken hata:", e)
        model = None

    # prepare dataframe of features (model expects columns in feature_list)
    X_row = {}
    for f in feature_list:
        X_row[f] = feat_row.get(f, 0.0)
    X_df = pd.DataFrame([X_row])

    if model is not None:
        try:
            proba = model.predict_proba(X_df)[0]
            # model.classes_ gives class ordering
            classes_order = getattr(model.named_steps['lgbm'], 'classes_', None)
            if classes_order is None:
                classes_order = np.array([0,1,2])
            # Map probs to 0,1,2 ordering
            prob_map = {int(cl): float(p) for cl,p in zip(classes_order, proba)}
            probs_ordered = [prob_map.get(0,0.0), prob_map.get(1,0.0), prob_map.get(2,0.0)]
            pred_idx = int(np.argmax(probs_ordered))
            pred_label = CLASS_LABELS.get(pred_idx, str(pred_idx))
            confidence = np.max(probs_ordered)
            
            # Güven eşiği kontrolü
            if confidence < 0.6:
                print("⚠️ Düşük güvenilirlik: Tahmin sonucu düşük güvenilirlik seviyesinde")
                
        except Exception as e:
            print("⚠️ Model tahmini sırasında hata:", e)
            model = None

    if model is None:
        # fallback heuristic using rating diff (sigmoid)
        rd = feat_row['Rating_Diff']
        def sigmoid(x): return 1.0 / (1.0 + np.exp(-x/10.0))
        away_prob = sigmoid(-rd)
        home_prob = sigmoid(rd)
        draw_prob = 0.10
        total = home_prob + away_prob
        home_prob = (home_prob/(total)) * (1-draw_prob)
        away_prob = (away_prob/(total)) * (1-draw_prob)
        probs_ordered = [draw_prob, home_prob, away_prob]
        pred_idx = int(np.argmax(probs_ordered))
        pred_label = CLASS_LABELS[pred_idx]
        confidence = np.max(probs_ordered)

    # print result nicely
    print("\n" + "="*60)
    print("🎯 TAHMİN SONUÇLARI")
    print("="*60 + "\n")
    print("📊 Sonuç Olasılıkları (%):")
    print(f"   • Beraberlik: {probs_ordered[0]*100:4.1f}%")
    print(f"   • Ev Sahibi Kazanır: {probs_ordered[1]*100:4.1f}%")
    print(f"   • Deplasman Kazanır: {probs_ordered[2]*100:4.1f}%")
    print(f"   🔮 Tahmin: {pred_label} ({confidence*100:.1f}% güven)\n")

    # Feature importance göster
    try:
        if model is not None and hasattr(model.named_steps['lgbm'], 'feature_importances_'):
            importances = model.named_steps['lgbm'].feature_importances_
            feat_names = feature_list
            idxs = np.argsort(importances)[::-1]
            
            print("📊 Özellik Önem Sıralaması:")
            for rank in range(min(5, len(feat_names))):
                print(f"   {rank+1}. {feat_names[idxs[rank]]}: {importances[idxs[rank]]:.4f}")
    except Exception as e:
        print("⚠️ Feature importance görselleştirme hatası:", e)

    # Team stats print
    print("\n👥 Takım İstatistikleri:")
    print(f"   ⭐ Ev Sahibi Rating: {feat_row['Home_AvgRating']:.2f}")
    print(f"   ⭐ Deplasman Rating: {feat_row['Away_AvgRating']:.2f}")
    print(f"   📈 Ev Sahibi Formu: {home_form_info['momentum']:.1f}%")
    print(f"   📈 Deplasman Formu: {away_form_info['momentum']:.1f}%")
    print(f"   ⚽ Ev Sahibi Momentum: {feat_row['homeTeam_Momentum']}")
    print(f"   ⚽ Deplasman Momentum: {feat_row['awayTeam_Momentum']}")
    print(f"   💰 Ev Değeri (EUR): {feat_row['home_current_value_eur']}")
    print(f"   💰 Dep Değeri (EUR): {feat_row['away_current_value_eur']}")
    print(f"   👶 Ev Yaş Ort.: {feat_row['home_squad_avg_age']}")
    print(f"   👶 Dep Yaş Ort.: {feat_row['away_squad_avg_age']}")

    # Son 5 maç özetleri
    print("\n📊 Takım Form Durumları:")
    print("========================================\n")
    print(f"{home_team} Son 5 Maç:")
    print(home_form_info['summary'])
    print(f"{away_team} Son 5 Maç:")
    print(away_form_info['summary'])

    # 9) Kaydet
    timestamp = datetime.utcnow().strftime("Pred_%Y-%m-%d_%H-%M-%S")
    row_to_save = {
        'timestamp': timestamp,
        'home_team': home_team,
        'away_team': away_team,
        'home_start_idxs': ",".join(map(str, home_start_idxs)),
        'home_sub_idxs': ",".join(map(str, home_sub_idxs)),
        'away_start_idxs': ",".join(map(str, away_start_idxs)),
        'away_sub_idxs': ",".join(map(str, away_sub_idxs)),
        'home_rating': feat_row['Home_AvgRating'],
        'away_rating': feat_row['Away_AvgRating'],
        'home_bench_rating': feat_row.get('Home_BenchRating', np.nan),
        'away_bench_rating': feat_row.get('Away_BenchRating', np.nan),
        'home_momentum': feat_row['homeTeam_Momentum'],
        'away_momentum': feat_row['awayTeam_Momentum'],
        'pred_label': pred_label,
        'prob_draw': float(probs_ordered[0]),
        'prob_home': float(probs_ordered[1]),
        'prob_away': float(probs_ordered[2]),
        'confidence': float(confidence)
    }
    # add features used
    for k,v in feat_row.items():
        if k not in row_to_save:
            row_to_save[f"f_{k}"] = v

    saved_path = append_prediction_history(row_to_save, PRED_HISTORY_PATH)
    print(f"\n✅ Tahmin '{timestamp}' kaydedildi: {saved_path}")

    print("\n🎉 Tahmin tamamlandı. İyi analizler!")

if __name__ == "__main__":
    main()