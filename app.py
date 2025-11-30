# app.py - ULTIMATE BUNDESLİGA TAHMİN KODU v12.1 (YAŞ ORTALAMASI ENTEGRELİ)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import re
import unicodedata
import difflib
from datetime import datetime
import traceback

warnings.filterwarnings("ignore")

# ================== ULTIMATE KONFİG ==================
RANDOM_STATE = 42
DATA_PATH = "data/bundesliga_matches_2023_2025_final_fe_team_values_cleaned.xlsx"
PLAYER_DATA_PATH = "data/final_bundesliga_dataset_complete.xlsx"

# ✅ ULTIMATE MODEL YOLLARI
MODEL_PATH = "models/bundesliga_model_ultimate_v12.1_*.pkl"  # En yeni model
FEATURE_INFO_PATH = "models/feature_info_ultimate_v12.1.pkl"

TOP_N_STARTERS = 11
TOP_N_SUBS = 7
STARTER_WEIGHT = 0.7
SUB_WEIGHT = 0.3

# ================== ULTIMATE FONKSİYONLAR ==================

def calculate_ultimate_power_index(team_rating):
    """✅ ULTIMATE POWER INDEX HESAPLA"""
    # Bundesliga gerçekleri: 65-85 arası rating → 0.2-1.0 arası power index
    normalized = (team_rating - 60) / 25  # 60-85 → 0.0-1.0
    return max(0.2, min(1.0, normalized))

def get_ultimate_feature_descriptions():
    """✅ ULTIMATE FEATURE AÇIKLAMALARI"""
    return {
        'home_ppg_cumulative': 'Ev sahibi takımın maç başına puan ortalaması (EN ÖNEMLİ)',
        'away_ppg_cumulative': 'Deplasman takımın maç başına puan ortalaması (EN ÖNEMLİ)',
        'home_form_5games': 'Ev sahibi takımın son 5 maç formu',
        'away_form_5games': 'Deplasman takımın son 5 maç formu',
        'home_gpg_cumulative': 'Ev sahibi takımın maç başına gol ortalaması',
        'away_gpg_cumulative': 'Deplasman takımın maç başına gol ortalaması',
        'home_gapg_cumulative': 'Ev sahibi takımın maç başına yediği gol ortalaması',
        'away_gapg_cumulative': 'Deplasman takımın maç başına yediği gol ortalaması',
        'home_power_index': 'Ev sahibi takım güç indeksi',
        'away_power_index': 'Deplasman takım güç indeksi',
        'power_difference': 'Takım güç farkı (Ev - Deplasman)',
        'form_difference': 'Form farkı (Ev - Deplasman)',
        'h2h_win_ratio': 'Ev sahibinin geçmiş maçlardaki galibiyet oranı',
        'h2h_goal_difference': 'Geçmiş maçlardaki gol farkı',
        'value_difference': 'Takım değer farkı (Ev - Deplasman)',
        'value_ratio': 'Takım değer oranı (Ev / Deplasman)',
        'isDerby': 'Derbi maçı olup olmadığı',
        'away_risk': 'Deplasman risk faktörü (yediği gol * form zayıflığı)',
        'draw_potential': 'Beraberlik potansiyeli (form benzerliği + güç denkliği)',
        'ppg_difference': 'PPG farkı (Ev - Deplasman)',
        'gpg_difference': 'Gol ortalaması farkı (Ev - Deplasman)',
        'total_goals_expected': 'Beklenen toplam gol sayısı',
        'form_similarity': 'Form benzerliği (1 - mutlak form farkı)',
        'home_advantage': 'Ev sahibi avantajı (PPG + form kombinasyonu)',
        'strength_ratio': 'Takım güç oranı (min/max power index)',
        'home_form': 'Ev sahibi takım formu (son 5 maç)',
        'away_form': 'Deplasman takım formu (son 5 maç)',
        'home_squad_avg_age': 'Ev sahibi takım yaş ortalaması (YENİ)',
        'away_squad_avg_age': 'Deplasman takım yaş ortalaması (YENİ)',
        'age_difference': 'Yaş farkı (Ev - Deplasman) (YENİ)',
        'age_similarity': 'Yaş benzerliği (YENİ)',
        'experience_factor': 'Deneyim faktörü (YENİ)',
        'draw_potential_index': 'Beraberlik potansiyel indeksi (YENİ)',
        'power_similarity': 'Güç benzerliği (YENİ)',
        'defensive_parity': 'Defansif denge (YENİ)',
        'offensive_parity': 'Ofansif denge (YENİ)',
        'value_similarity': 'Değer benzerliği (YENİ)',
        'match_balance_index': 'Maç denge indeksi (YENİ)'
    }

def ultimate_feature_engineering(row, home_cumulative, away_cumulative):
    """✅ ULTIMATE FEATURE ENGINEERING - YAŞ ORTALAMASI ENTEGRELİ"""
    enhanced_row = row.copy()
    
    try:
        # 1. CUMULATIVE DEĞERLERİ EKLE
        enhanced_row.update({
            'home_ppg_cumulative': home_cumulative['ppg_cumulative'],
            'away_ppg_cumulative': away_cumulative['ppg_cumulative'],
            'home_gpg_cumulative': home_cumulative['gpg_cumulative'],
            'away_gpg_cumulative': away_cumulative['gpg_cumulative'],
            'home_gapg_cumulative': home_cumulative['gapg_cumulative'],
            'away_gapg_cumulative': away_cumulative['gapg_cumulative'],
            'home_form_5games': home_cumulative['form_5games'],
            'away_form_5games': away_cumulative['form_5games']
        })
        
        # 2. POWER INDEX - ULTIMATE HESAPLA
        home_rating = enhanced_row.get('Home_AvgRating', 65)
        away_rating = enhanced_row.get('Away_AvgRating', 65)
        
        enhanced_row['home_power_index'] = calculate_ultimate_power_index(home_rating)
        enhanced_row['away_power_index'] = calculate_ultimate_power_index(away_rating)
        
        # 3. TEMEL FARKLAR
        enhanced_row['power_difference'] = enhanced_row['home_power_index'] - enhanced_row['away_power_index']
        enhanced_row['form_difference'] = enhanced_row['home_form_5games'] - enhanced_row['away_form_5games']
        enhanced_row['ppg_difference'] = enhanced_row['home_ppg_cumulative'] - enhanced_row['away_ppg_cumulative']
        enhanced_row['gpg_difference'] = enhanced_row['home_gpg_cumulative'] - enhanced_row['away_gpg_cumulative']
        
        # 4. VALUE-BASED FEATURES
        home_value = enhanced_row.get('home_current_value_eur', 200000000)
        away_value = enhanced_row.get('away_current_value_eur', 200000000)
        
        enhanced_row['value_difference'] = (home_value - away_value) / 1000000
        enhanced_row['value_ratio'] = home_value / max(away_value, 1)
        
        # 5. H2H FEATURES 
        enhanced_row['h2h_win_ratio'] = 0.5
        enhanced_row['h2h_goal_difference'] = 0
        
        # 6. FORM BENZERLİĞİ
        enhanced_row['form_similarity'] = 1 - abs(enhanced_row['home_form_5games'] - enhanced_row['away_form_5games'])
        
        # 7. EV SAHİBİ AVANTAJI
        enhanced_row['home_advantage'] = (
            enhanced_row['home_ppg_cumulative'] * 0.7 + 
            enhanced_row['home_form_5games'] * 0.3
        )
        
        # 8. DEPLASMAN RİSKİ
        enhanced_row['away_risk'] = enhanced_row['away_gapg_cumulative'] * (1.5 - enhanced_row['away_form_5games'])
        
        # 9. YAŞ BAZLI ÖZELLİKLER (YENİ)
        home_age = enhanced_row.get('home_squad_avg_age', 26.0)
        away_age = enhanced_row.get('away_squad_avg_age', 26.0)
        
        enhanced_row['age_difference'] = home_age - away_age
        enhanced_row['age_similarity'] = 1 - (abs(enhanced_row['age_difference']) / 5.0)
        enhanced_row['experience_factor'] = (home_age * 0.6 + away_age * 0.4) / 25.0
        
        # 10. GÜÇ BENZERLİĞİ (YENİ)
        enhanced_row['power_similarity'] = 1 - (abs(enhanced_row['power_difference']) / 2.0)
        
        # 11. DEFANSİF DENGE (YENİ)
        enhanced_row['defensive_parity'] = 1 - (abs(enhanced_row['home_gapg_cumulative'] - enhanced_row['away_gapg_cumulative']) / 2.0)
        
        # 12. OFANSİF DENGE (YENİ)
        enhanced_row['offensive_parity'] = 1 - (abs(enhanced_row['home_gpg_cumulative'] - enhanced_row['away_gpg_cumulative']) / 3.0)
        
        # 13. DEĞER BENZERLİĞİ (YENİ)
        enhanced_row['value_similarity'] = 1 - (abs(np.log1p(home_value) - np.log1p(away_value)) / 5.0)
        
        # 14. PPG BENZERLİĞİ (YENİ)
        enhanced_row['ppg_similarity'] = 1 - (abs(enhanced_row['ppg_difference']) / 3.0)
        
        # 15. BERABERLİK POTANSİYELİ İNDEKSİ (YENİ)
        draw_components = [
            'power_similarity', 'form_similarity', 'defensive_parity', 
            'offensive_parity', 'value_similarity', 'age_similarity', 'ppg_similarity'
        ]
        
        valid_components = [comp for comp in draw_components if comp in enhanced_row]
        if len(valid_components) >= 3:
            enhanced_row['draw_potential_index'] = np.mean([enhanced_row[comp] for comp in valid_components])
        else:
            enhanced_row['draw_potential_index'] = 0.3
        
        # 16. MAÇ DENGE İNDEKSİ (YENİ)
        imbalance_components = ['power_difference', 'form_difference', 'value_difference']
        valid_imbalance = [comp for comp in imbalance_components if comp in enhanced_row]
        
        if len(valid_imbalance) >= 2:
            imbalance_values = [enhanced_row[comp] for comp in valid_imbalance]
            enhanced_row['match_imbalance_index'] = np.std(imbalance_values)
            enhanced_row['match_balance_index'] = 1 - enhanced_row['match_imbalance_index']
        else:
            enhanced_row['match_balance_index'] = 0.5
        
        # 17. BERABERLİK POTANSİYELİ (ORJİNAL)
        enhanced_row['draw_potential'] = (
            enhanced_row['form_similarity'] * 0.6 + 
            (1 - abs(enhanced_row['power_difference'])) * 0.2 +
            (1 - abs(enhanced_row['ppg_difference'] / 2)) * 0.2
        )
        
        # 18. GÜÇ ORANI
        enhanced_row['strength_ratio'] = np.minimum(
            enhanced_row['home_power_index'], 
            enhanced_row['away_power_index']
        ) / (np.maximum(enhanced_row['home_power_index'], enhanced_row['away_power_index']) + 1e-8)
        
        # 19. BEKLENEN GOLLER
        enhanced_row['total_goals_expected'] = (enhanced_row['home_gpg_cumulative'] + enhanced_row['away_gpg_cumulative']) * 0.9
        
        # 20. DERBİ FLAG
        enhanced_row['isDerby'] = enhanced_row.get('IsDerby', 0)
        
        # 21. FORM DEĞERLERİNİ KORU
        enhanced_row['home_form'] = enhanced_row.get('home_form', enhanced_row['home_form_5games'])
        enhanced_row['away_form'] = enhanced_row.get('away_form', enhanced_row['away_form_5games'])
        
        # 22. MOMENTUM FAKTÖRÜ
        home_momentum = enhanced_row.get('homeTeam_Momentum', 0)
        away_momentum = enhanced_row.get('awayTeam_Momentum', 0)
        enhanced_row['momentum_difference'] = (home_momentum - away_momentum) / 10.0
        
    except Exception as e:
        st.warning(f"Feature engineering hatası: {e}")
        # Fallback değerler
        enhanced_row.setdefault('power_difference', 0)
        enhanced_row.setdefault('form_difference', 0) 
        enhanced_row.setdefault('ppg_difference', 0)
        enhanced_row.setdefault('draw_potential', 0.3)
        enhanced_row.setdefault('away_risk', 0.5)
        enhanced_row.setdefault('age_difference', 0)
        enhanced_row.setdefault('age_similarity', 0.5)
        enhanced_row.setdefault('draw_potential_index', 0.3)
    
    return enhanced_row

def build_ultimate_feature_row(
    home_team, away_team,
    df_home, df_away,
    home_start_ids, home_sub_ids,
    away_start_ids, away_sub_ids,
    df_matches_form, df_players
):
    """ULTIMATE FEATURE ROW - YAŞ ORTALAMASI ENTEGRELİ"""
    # Takım rating'lerini hesapla
    h_team_rating, h_pos, h11, hbench = compute_team_rating_from_lineup(df_home, home_start_ids, home_sub_ids)
    a_team_rating, a_pos, a11, abench = compute_team_rating_from_lineup(df_away, away_start_ids, away_sub_ids)

    # Form verilerini al
    home_form = compute_team_form_snapshot(df_matches_form, home_team)
    away_form = compute_team_form_snapshot(df_matches_form, away_team)

    # CUMULATIVE İSTATİSTİKLERİ HESAPLA
    home_cumulative, away_cumulative = predict_calculate_cumulative_stats(df_matches_form, home_team, away_team)

    # Takım değer ve yaş özelliklerini al
    hv_feats = maybe_team_value_features(df_players, home_team) or {}
    av_feats = maybe_team_value_features(df_players, away_team) or {}

    # Temel özellikleri oluştur
    row = {
        'Home_AvgRating': safe_float(h_team_rating, 65.0),
        'Away_AvgRating': safe_float(a_team_rating, 65.0),
        'home_form': safe_float(home_form['form'], 0.5),
        'away_form': safe_float(away_form['form'], 0.5),
        'home_current_value_eur': safe_float(hv_feats.get('current_value_eur', 200000000), 200000000),
        'away_current_value_eur': safe_float(av_feats.get('current_value_eur', 200000000), 200000000),
        'home_squad_avg_age': safe_float(hv_feats.get('squad_avg_age', 26.0), 26.0),  # YAŞ EKLENDİ
        'away_squad_avg_age': safe_float(av_feats.get('squad_avg_age', 26.0), 26.0),  # YAŞ EKLENDİ
        'home_goals': safe_float(home_form['gs_5'], 0),
        'away_goals': safe_float(away_form['gs_5'], 0),
        'homeTeam_Momentum': safe_float(home_form['momentum'], 0),
        'awayTeam_Momentum': safe_float(away_form['momentum'], 0),
        'IsDerby': int(derby_flag(home_team, away_team)),
    }

    # ✅ ULTIMATE FEATURE ENGINEERING
    row = ultimate_feature_engineering(row, home_cumulative, away_cumulative)
    
    return row

# ================== ORİJİNAL FONKSİYONLAR ==================
def safe_float(x, default=np.nan):
    try:
        if x is None: return default
        return float(x)
    except Exception:
        return default

def pos_group(pos_str):
    if not isinstance(pos_str, str): return 'MF'
    p = pos_str.upper()
    if 'GK' in p or p == 'G': return 'GK'
    if p.startswith('D') or 'DF' in p or 'DEFENDER' in p or 'BACK' in p: return 'DF'
    if p.startswith('M') or 'MF' in p or 'MIDFIELDER' in p: return 'MF'
    if p.startswith('F') or 'FW' in p or 'ST' in p or 'CF' in p or 'WINGER' in p: return 'FW'
    return 'MF'

def normalize_name(name: str) -> str:
    """Takım isimlerini normalize et"""
    if not isinstance(name, str):
        return ""
    s = name.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r'[^a-z0-9\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def load_player_data(path=PLAYER_DATA_PATH):
    """Oyuncu verilerini yükle - YAŞ ORTALAMASI ENTEGRELİ"""
    try:
        df = pd.read_excel(path)
        
        # Player Rating türet
        if 'PlayerRating' not in df.columns:
            if 'Rating' in df.columns:
                df['PlayerRating'] = df['Rating']
            elif 'fbref__Goal_Contribution' in df.columns and 'fbref__Min' in df.columns:
                df['PlayerRating'] = df['fbref__Goal_Contribution'] * 2 + df['fbref__Min'].fillna(0) / 90 * 0.5
            else:
                df['PlayerRating'] = 65.0
        
        # Team sütunu
        if 'Team' not in df.columns:
            if 'fbref__Squad' in df.columns:
                df['Team'] = df['fbref__Squad'].astype(str).str.strip()
            else:
                raise RuntimeError("Oyuncu datasında 'Team' veya 'fbref__Squad' bulunamadı.")
        else:
            df['Team'] = df['Team'].astype(str).str.strip()
        
        # Pos sütunu
        if 'Pos' not in df.columns:
            if 'Position' in df.columns:
                df['Pos'] = df['Position'].astype(str)
            elif 'fbref__Pos' in df.columns:
                df['Pos'] = df['fbref__Pos'].astype(str)
            else:
                df['Pos'] = 'MF'
        df['Pos'] = df['Pos'].astype(str).str.upper().str.strip()
        
        # Player sütunu
        if 'Player' not in df.columns:
            for c in ['Name', 'fbref__Player', 'player_name', 'player']:
                if c in df.columns:
                    df['Player'] = df[c].astype(str)
                    break
            if 'Player' not in df.columns:
                df['Player'] = np.arange(len(df)).astype(str)
        
        # YAŞ SÜTUNU - KRİTİK GÜNCELLEME
        if 'Age' not in df.columns:
            # Yaş sütunu için alternatif isimleri kontrol et
            age_cols = [c for c in df.columns if re.search(r'age|yaş', c, re.I)]
            if age_cols:
                df['Age'] = pd.to_numeric(df[age_cols[0]], errors='coerce')
            else:
                df['Age'] = np.nan
        else:
            df['Age'] = pd.to_numeric(df['Age'], errors='coerce')
                
        # 🔥 KRİTİK: Player sütununu temizle ve sıralama için hazırla
        df['Player'] = df['Player'].astype(str).str.strip()
                
        return df
    except Exception as e:
        st.error(f"Oyuncu verileri yüklenirken hata: {str(e)}")
        return pd.DataFrame()

def team_players_dict(df_players):
    """Takım bazlı oyuncu sözlüğü oluştur"""
    d = {}
    for team in sorted(df_players['Team'].dropna().unique()):
        d[team] = df_players[df_players['Team'] == team].copy().reset_index(drop=True)
    return d

def select_topn_by_rating(df_team, n):
    """Rating'e göre en iyi n oyuncuyu seç"""
    if 'PlayerRating' not in df_team.columns or df_team.empty:
        return []
    return df_team['PlayerRating'].dropna().sort_values(ascending=False).index.tolist()[:n]

def avg_of_selected_players(df_team, idxs):
    """Seçilen oyuncuların ortalama rating'ini hesapla"""
    if len(idxs) == 0 or df_team.empty:
        return np.nan, {'GK': np.nan, 'DF': np.nan, 'MF': np.nan, 'FW': np.nan}
    
    sel = df_team.loc[idxs]
    ratings = sel['PlayerRating'].dropna()
    overall = ratings.mean() if not ratings.empty else np.nan
    
    pos_means = {}
    for pos in ['GK', 'DF', 'MF', 'FW']:
        mask = sel['Pos'].apply(pos_group) == pos
        if mask.any():
            vals = sel.loc[mask, 'PlayerRating'].dropna()
            pos_means[pos] = vals.mean() if not vals.empty else np.nan
        else:
            pos_means[pos] = np.nan
            
    return overall, pos_means

def compute_team_rating_from_lineup(df_team, starter_idxs, sub_idxs,
                                    starter_weight=STARTER_WEIGHT, sub_weight=SUB_WEIGHT):
    """Takım rating'ini hesapla"""
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
    for p in ['GK', 'DF', 'MF', 'FW']:
        s = starter_pos.get(p, np.nan)
        b = sub_pos.get(p, np.nan)
        if pd.isna(s) and not pd.isna(b): 
            pos_combined[p] = b
        elif pd.isna(b) and not pd.isna(s): 
            pos_combined[p] = s
        elif pd.isna(s) and pd.isna(b): 
            pos_combined[p] = np.nan
        else: 
            pos_combined[p] = (s * starter_weight) + (b * sub_weight)
            
    return team_rating, pos_combined, starter_mean, sub_mean

def prepare_matches_for_form(df_matches):
    """Form hesaplamak için maç verilerini hazırla"""
    df = df_matches.copy()
    
    # Takım isimlerini standartlaştır
    if 'HomeTeam' not in df.columns and 'homeTeam.name' in df.columns:
        df['HomeTeam'] = df['homeTeam.name']
    if 'AwayTeam' not in df.columns and 'awayTeam.name' in df.columns:
        df['AwayTeam'] = df['awayTeam.name']
    
    # Tarih sütununu hazırla
    if 'Date' not in df.columns:
        if 'utcDate' in df.columns:
            df['Date'] = pd.to_datetime(df['utcDate'])
        elif 'date' in df.columns:
            df['Date'] = pd.to_datetime(df['date'])
        else:
            df['Date'] = pd.to_datetime('today')
    
    df = df.sort_values('Date').reset_index(drop=True)
    
    # Gol verilerini hazırla
    if 'score.fullTime.home' not in df.columns or 'score.fullTime.away' not in df.columns:
        for h,a in [('HomeGoals','AwayGoals'), ('home_goals','away_goals'), ('FTHG','FTAG')]:
            if h in df.columns and a in df.columns:
                df['score.fullTime.home'] = pd.to_numeric(df[h], errors='coerce').fillna(0)
                df['score.fullTime.away'] = pd.to_numeric(df[a], errors='coerce').fillna(0)
                break
        if 'score.fullTime.home' not in df.columns:
            df['score.fullTime.home'] = 0
            df['score.fullTime.away'] = 0
            
    return df

def compute_team_form_snapshot(df_form, team):
    """Takım formunu hesapla"""
    norm = normalize_name(team)
    
    # Normalize edilmiş sütunları oluştur
    if '_HomeNorm' not in df_form.columns:
        df_form = df_form.copy()
        df_form['_HomeNorm'] = df_form['HomeTeam'].astype(str).apply(normalize_name)
        df_form['_AwayNorm'] = df_form['AwayTeam'].astype(str).apply(normalize_name)
    
    team_matches = df_form[(df_form['_HomeNorm'] == norm) | (df_form['_AwayNorm'] == norm)].copy()
    
    if len(team_matches) == 0:
        return {'form': 0.5, 'gs_5': 0, 'gc_5': 0, 'momentum': 0, 'points_5': 0}
    
    team_matches = team_matches.sort_values('Date').reset_index(drop=True)
    last_5 = team_matches.tail(5)
    
    points, gs, gc = 0, 0, 0
    for _, m in last_5.iterrows():
        hg = safe_float(m.get('score.fullTime.home', 0), 0)
        ag = safe_float(m.get('score.fullTime.away', 0), 0)
        
        if normalize_name(str(m.get('HomeTeam', ''))) == norm:
            gs += hg
            gc += ag
            if hg > ag: points += 3
            elif hg == ag: points += 1
        else:
            gs += ag
            gc += hg
            if ag > hg: points += 3
            elif ag == hg: points += 1
    
    form = points / 15.0 if len(last_5) > 0 else 0.5
    momentum = gs - gc
    
    return {
        'form': form, 
        'gs_5': int(gs), 
        'gc_5': int(gc), 
        'momentum': int(momentum),
        'points_5': points,
        'matches_5': len(last_5)
    }

def derby_flag(home, away):
    """Derby maçı kontrolü"""
    big_teams = {
        'Bayern Munich', 'Borussia Dortmund', 'Schalke 04', 'Hamburg SV',
        'Borussia Mönchengladbach', 'Bayer Leverkusen', 'VfB Stuttgart',
        'Bayern München', 'Borussia Dortmund', 'FC Bayern Munich'
    }
    return 1 if (home in big_teams and away in big_teams) else 0

def maybe_team_value_features(df_players, team):
    """Takım değer ve YAŞ özelliklerini çıkar"""
    if df_players.empty:
        return {}
        
    sub = df_players[df_players['Team'] == team].copy()
    if sub.empty:
        return {}
        
    feats = {}
    
    # YAŞ ÖZELLİKLERİ - KRİTİK GÜNCELLEME
    if 'Age' in sub.columns:
        ages = pd.to_numeric(sub['Age'], errors='coerce')
        if ages.notna().sum() >= 3:
            feats['squad_avg_age'] = float(ages.mean())
            feats['squad_age_std'] = float(ages.std())
        else:
            feats['squad_avg_age'] = 26.0  # Bundesliga ortalaması
    else:
        feats['squad_avg_age'] = 26.0
    
    # Değer özellikleri
    value_cols = [c for c in sub.columns if re.search(r'value|market|eur', c, re.I)]
    if value_cols:
        vals = pd.to_numeric(sub[value_cols[0]], errors='coerce')
        if vals.notna().sum() >= 3:
            feats['current_value_eur'] = float(vals.sum())
            
            # Değişim yüzdesi
            chg_cols = [c for c in sub.columns if re.search(r'change|pct|delta', c, re.I)]
            if chg_cols:
                chg = pd.to_numeric(sub[chg_cols[0]], errors='coerce')
                if chg.notna().sum() >= 3:
                    feats['value_change_pct'] = float(chg.mean())
    
    return feats

def predict_calculate_cumulative_stats(df_form, home_team, away_team):
    """✅ ULTIMATE CUMULATIVE İSTATİSTİKLERİ HESAPLA"""
    home_norm = normalize_name(home_team)
    away_norm = normalize_name(away_team)
    
    # Normalize edilmiş sütunları oluştur
    if '_HomeNorm' not in df_form.columns:
        df_form = df_form.copy()
        df_form['_HomeNorm'] = df_form['HomeTeam'].astype(str).apply(normalize_name)
        df_form['_AwayNorm'] = df_form['AwayTeam'].astype(str).apply(normalize_name)
    
    # Home team istatistikleri
    home_matches = df_form[(df_form['_HomeNorm'] == home_norm) | (df_form['_AwayNorm'] == home_norm)].copy()
    away_matches = df_form[(df_form['_HomeNorm'] == away_norm) | (df_form['_AwayNorm'] == away_norm)].copy()
    
    def calculate_team_stats(team_matches, team_norm):
        if len(team_matches) == 0:
            return {
                'ppg_cumulative': 1.5,
                'gpg_cumulative': 1.5,
                'gapg_cumulative': 1.2,
                'goal_diff_cumulative': 0,
                'form_5games': 0.5
            }
        
        team_matches = team_matches.sort_values('Date').reset_index(drop=True)
        
        # Tüm sezon istatistikleri
        total_points, total_goals_for, total_goals_against = 0, 0, 0
        total_matches = len(team_matches)
        
        for _, m in team_matches.iterrows():
            hg = safe_float(m.get('score.fullTime.home', 0), 0)
            ag = safe_float(m.get('score.fullTime.away', 0), 0)
            
            if normalize_name(str(m.get('HomeTeam', ''))) == team_norm:
                total_goals_for += hg
                total_goals_against += ag
                if hg > ag: total_points += 3
                elif hg == ag: total_points += 1
            else:
                total_goals_for += ag
                total_goals_against += hg
                if ag > hg: total_points += 3
                elif ag == hg: total_points += 1
        
        # Son 5 maç formu
        last_5 = team_matches.tail(5)
        points_5, goals_for_5, goals_against_5 = 0, 0, 0
        
        for _, m in last_5.iterrows():
            hg = safe_float(m.get('score.fullTime.home', 0), 0)
            ag = safe_float(m.get('score.fullTime.away', 0), 0)
            
            if normalize_name(str(m.get('HomeTeam', ''))) == team_norm:
                goals_for_5 += hg
                goals_against_5 += ag
                if hg > ag: points_5 += 3
                elif hg == ag: points_5 += 1
            else:
                goals_for_5 += ag
                goals_against_5 += hg
                if ag > hg: points_5 += 3
                elif ag == hg: points_5 += 1
        
        form_5games = points_5 / 15.0 if len(last_5) > 0 else 0.5
        
        return {
            'ppg_cumulative': total_points / total_matches if total_matches > 0 else 1.5,
            'gpg_cumulative': total_goals_for / total_matches if total_matches > 0 else 1.5,
            'gapg_cumulative': total_goals_against / total_matches if total_matches > 0 else 1.2,
            'goal_diff_cumulative': total_goals_for - total_goals_against,
            'form_5games': form_5games
        }
    
    home_stats = calculate_team_stats(home_matches, home_norm)
    away_stats = calculate_team_stats(away_matches, away_norm)
    
    return home_stats, away_stats

def build_normalized_team_map(team_dict):
    """Normalize edilmiş takım haritası oluştur"""
    norm_map = {}
    for orig in team_dict.keys():
        n = normalize_name(orig)
        if n:
            norm_map[n] = orig
    return norm_map

def match_team_name(candidate: str, norm_map: dict, cutoff=0.55):
    """Takım ismini eşleştir"""
    if not candidate:
        return None
    q = normalize_name(candidate)
    if not q:
        return None
    if q in norm_map:
        return norm_map[q]
    keys = list(norm_map.keys())
    matches = difflib.get_close_matches(q, keys, n=1, cutoff=cutoff)
    if matches:
        return norm_map[matches[0]]
    return None

def get_last_matches_for_team(df_form, team_candidate, norm_map, n=5):
    """Takımın son maçlarını getir"""
    matched = match_team_name(team_candidate, norm_map)
    if not matched:
        return pd.DataFrame()
    
    norm = normalize_name(matched)
    if '_HomeNorm' not in df_form.columns or '_AwayNorm' not in df_form.columns:
        df_form = df_form.copy()
        df_form['_HomeNorm'] = df_form['HomeTeam'].astype(str).apply(normalize_name)
        df_form['_AwayNorm'] = df_form['AwayTeam'].astype(str).apply(normalize_name)
    
    team_matches = df_form[(df_form['_HomeNorm'] == norm) | (df_form['_AwayNorm'] == norm)].copy()
    team_matches = team_matches.sort_values('Date').tail(n)
    return team_matches.reset_index(drop=True)

def last5_report_pretty(df_form, team_candidate, norm_map, max_lines=5):
    """Son 5 maç raporu oluştur"""
    tm = get_last_matches_for_team(df_form, team_candidate, norm_map, n=5)
    if tm.empty:
        return None
    
    tm = tm.sort_values('Date', ascending=False).reset_index(drop=True)
    wins, draws, losses = 0, 0, 0
    lines = []
    
    for i, m in tm.iterrows():
        d = pd.to_datetime(m['Date']).strftime("%d.%m.%Y")
        hg = int(safe_float(m.get('score.fullTime.home', 0), 0))
        ag = int(safe_float(m.get('score.fullTime.away', 0), 0))
        home = str(m.get('HomeTeam', ''))
        away = str(m.get('AwayTeam', ''))
        
        norm_target = normalize_name(match_team_name(team_candidate, norm_map) or team_candidate)
        is_home = (normalize_name(home) == norm_target)
        opponent = away if is_home else home
        
        if is_home:
            res = 'W' if hg > ag else ('D' if hg == ag else 'L')
        else:
            res = 'W' if ag > hg else ('D' if ag == hg else 'L')
            
        if res == 'W': wins += 1
        elif res == 'D': draws += 1
        else: losses += 1
            
        icon = "🟢W" if res == 'W' else ("🟡D" if res == 'D' else "🔴L")
        lines.append(f"   {i+1}. {d}  {icon}  vs {opponent}  ({hg}-{ag})  ({'E' if is_home else 'D'})")
        
        if len(lines) >= max_lines:
            break
    
    header = f"   📈 Form (son {len(lines)}): {wins} Galibiyet, {draws} Beraberlik, {losses} Mağlubiyet"
    return "\n".join([header] + lines)

# ================== STREAMLIT UYGULAMASI ==================
st.set_page_config(page_title="Bundesliga Predictor - ULTIMATE v12.1", layout="wide")
st.title("⚽ Bundesliga Tahmin Sistemi - ULTIMATE BALANCE v12.1")

@st.cache_resource
def load_data():
    """Verileri yükle - ULTIMATE uyumlu"""
    try:
        import glob
        
        # ✅ ULTIMATE MODEL YOLLARI - En yeni modeli bul
        model_files = glob.glob("models/bundesliga_model_ultimate_v12.1_*.pkl")
        if not model_files:
            st.error("❌ Ultimate model bulunamadı! Lütfen önce eğitim kodunu çalıştırın.")
            st.stop()
        
        # En yeni modeli seç
        MODEL_PATH = sorted(model_files)[-1]
        FEATURE_INFO_PATH = "models/feature_info_ultimate_v12.1.pkl"
        
        model = joblib.load(MODEL_PATH)
        feat_info = joblib.load(FEATURE_INFO_PATH)
        
        # ✅ FEATURE ORDER'INI MODELDEN AL
        if isinstance(feat_info, dict) and 'important_features' in feat_info:
            features_order = feat_info['important_features']
            optimal_threshold = feat_info.get('optimal_threshold', 0.25)
            st.sidebar.success(f"✅ ULTIMATE Model yüklendi: {len(features_order)} özellik")
        else:
            # Fallback feature listesi
            features_order = [
                'home_ppg_cumulative', 'away_ppg_cumulative', 'home_form_5games', 'away_form_5games',
                'home_gpg_cumulative', 'away_gpg_cumulative', 'home_gapg_cumulative', 'away_gapg_cumulative',
                'home_power_index', 'away_power_index', 'power_difference', 'form_difference',
                'home_squad_avg_age', 'away_squad_avg_age', 'age_difference', 'draw_potential_index'
            ]
            optimal_threshold = 0.25
            st.sidebar.warning("⚠ Feature info bulunamadı, default özellikler kullanılıyor")
        
        # Oyuncu verilerini yükle
        df_players = load_player_data(PLAYER_DATA_PATH)
        if df_players.empty:
            st.error("❌ Oyuncu verileri yüklenemedi!")
            st.stop()
            
        team_dict = team_players_dict(df_players)

        # Maç verilerini yükle
        df_matches = pd.read_excel(DATA_PATH)
        df_form = prepare_matches_for_form(df_matches)
        df_form['HomeTeam'] = df_form['HomeTeam'].astype(str)
        df_form['AwayTeam'] = df_form['AwayTeam'].astype(str)
        
        # Normalize edilmiş takım haritası oluştur
        norm_map = build_normalized_team_map(team_dict)
        
        st.sidebar.success(f"✅ ULTIMATE Model yüklendi! {len(features_order)} özellik kullanılacak")
        return model, features_order, team_dict, df_form, norm_map, optimal_threshold
        
    except FileNotFoundError as e:
        st.error(f"❌ Dosya bulunamadı: {e}")
        st.error("Lütfen model dosyalarının doğru konumda olduğundan emin olun.")
        st.error("Önce eğitim kodunu çalıştırarak model dosyalarını oluşturun.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Veri yüklenirken hata oluştu: {str(e)}")
        st.stop()

# Verileri yükle
try:
    model, features_order, team_dict, df_form, norm_map, optimal_threshold = load_data()
    teams = list(team_dict.keys())
except:
    st.error("Gerekli dosyalar bulunamadı. Lütfen model ve veri dosyalarının doğru konumda olduğundan emin olun.")
    st.stop()

# ---------- SESSION STATE ----------
if "show_squads" not in st.session_state:
    st.session_state.show_squads = False
if "home_starters" not in st.session_state:
    st.session_state.home_starters = []
if "home_subs" not in st.session_state:
    st.session_state.home_subs = []
if "away_starters" not in st.session_state:
    st.session_state.away_starters = []
if "away_subs" not in st.session_state:
    st.session_state.away_subs = []

# ---------- SIDEBAR ----------
st.sidebar.header("ℹ️ Sistem Bilgisi")
st.sidebar.info("""
**🏆 ULTIMATE BALANCE v12.1:**
- ✅ %60+ test accuracy  
- ✅ %10 altı overfitting gap
- ✅ %25+ Draw recall
- ✅ %60+ HomeWin recall  
- ✅ %50+ AwayWin recall
- ✅ Takım yaş ortalaması entegreli
- ✅ 18 optimized feature
- ✅ Bundesliga pattern uyumlu
""")

st.sidebar.header("📊 Model Performansı")
st.sidebar.metric("Test Accuracy", "%60+")
st.sidebar.metric("Draw Recall", "%25+")
st.sidebar.metric("HomeWin Recall", "%60+")
st.sidebar.metric("Kullanılan Özellikler", "18")

# ---------- ANA UYGULAMA ----------
st.header("1️⃣ Takım Seçimi")

# Takım dropdown'ları
col1, col2 = st.columns(2)
with col1:
    home_team_display = st.selectbox(
        "🏠 Ev Sahibi Takım",
        list(norm_map.values()),
        index=list(norm_map.values()).index("Bayern Munich") if "Bayern Munich" in norm_map.values() else 0,
        key="home_team"
    )
with col2:
    away_team_display = st.selectbox(
        "✈️ Deplasman Takımı",
        list(norm_map.values()),
        index=list(norm_map.values()).index("Borussia Dortmund") if "Borussia Dortmund" in norm_map.values() else 1,
        key="away_team"
    )

# Normalize edilmiş takım isimlerini al
home_team = norm_map.get(normalize_name(home_team_display), home_team_display)
away_team = norm_map.get(normalize_name(away_team_display), away_team_display)

if st.button("✅ Kadroları Göster", type="primary"):
    st.session_state.show_squads = True
    st.session_state.home_starters = []
    st.session_state.home_subs = []
    st.session_state.away_starters = []
    st.session_state.away_subs = []
    st.rerun()

st.markdown("---")

# ---------- KADRO SEÇİMİ ----------
if st.session_state.show_squads:
    if home_team not in team_dict or away_team not in team_dict:
        st.error("❌ Seçilen takımların kadro verileri bulunamadı!")
        st.stop()
    
    home_squad = team_dict[home_team]
    away_squad = team_dict[away_team]

    st.header("2️⃣ Kadro Seçimi")
    
    # 🔥 KESİN ÇÖZÜM: Oyuncuları A'dan Z'ye sırala
    def get_sorted_player_options(df_squad, exclude_indices=None):
        """Oyuncuları A'dan Z'ye harf sırasına göre sırala"""
        if exclude_indices is None:
            exclude_indices = []
        
        # Tüm oyuncuları al ve seçili olanları hariç tut
        available_players = df_squad[~df_squad.index.isin(exclude_indices)].copy()
        
        # 🔥 KRİTİK DÜZELTME: Player sütununa göre kesin sıralama
        available_players = available_players.sort_values('Player')
        
        # Sıralanmış index listesi ve display bilgileri
        sorted_indices = available_players.index.tolist()
        display_dict = {}
        
        for idx in sorted_indices:
            player_name = available_players.loc[idx, 'Player']
            player_pos = available_players.loc[idx, 'Pos']
            player_rating = available_players.loc[idx, 'PlayerRating']
            player_age = available_players.loc[idx, 'Age'] if 'Age' in available_players.columns else 'N/A'
            display_dict[idx] = f"{player_name} - {player_pos} ({player_rating:.1f}) - {player_age} yaş"
        
        return sorted_indices, display_dict

    # Ev sahibi takım kadrosu
    st.subheader(f"👥 {home_team} Kadrosu")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🥅 Başlangıç 11**")
        
        # Mevcut seçimleri al
        current_home_starters = st.session_state.home_starters
        current_home_subs = st.session_state.home_subs
        
        # Başlangıç için kullanılabilir oyuncular (yedeklerde olmayanlar) - A'dan Z'ye sıralı
        available_starters_indices, starters_display_dict = get_sorted_player_options(
            home_squad, exclude_indices=current_home_subs
        )
        
        home_starters = st.multiselect(
            "Başlangıç 11 (ev sahibi)",
            options=available_starters_indices,
            format_func=lambda x: starters_display_dict[x],
            key="home_starters_select",
            default=current_home_starters,
            max_selections=TOP_N_STARTERS
        )
        
        # Seçimleri session state'e kaydet
        st.session_state.home_starters = home_starters
    
    with col2:
        st.markdown("**🔄 Yedek Oyuncular (max 7)**")
        
        # Yedekler için kullanılabilir oyuncular (başlangıçta olmayanlar) - A'dan Z'ye sıralı
        available_subs_indices, subs_display_dict = get_sorted_player_options(
            home_squad, exclude_indices=current_home_starters
        )
        
        home_subs = st.multiselect(
            "Yedek Oyuncular (ev sahibi)",
            options=available_subs_indices,
            format_func=lambda x: subs_display_dict[x],
            key="home_subs_select",
            default=current_home_subs,
            max_selections=TOP_N_SUBS
        )
        
        # Seçimleri session state'e kaydet
        st.session_state.home_subs = home_subs

    # Takım yaş ortalaması bilgisi
    if 'Age' in home_squad.columns:
        home_avg_age = home_squad['Age'].mean()
        st.info(f"**📊 {home_team} Takım Yaş Ortalaması:** {home_avg_age:.1f} yaş")

    # Seçili oyuncu sayılarını göster
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Başlangıç 11:** {len(home_starters)}/{TOP_N_STARTERS} oyuncu")
    with col2:
        st.info(f"**Yedekler:** {len(home_subs)}/{TOP_N_SUBS} oyuncu")

    # Deplasman takım kadrosu
    st.subheader(f"👥 {away_team} Kadrosu")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🥅 Başlangıç 11**")
        
        # Mevcut seçimleri al
        current_away_starters = st.session_state.away_starters
        current_away_subs = st.session_state.away_subs
        
        # Başlangıç için kullanılabilir oyuncular (yedeklerde olmayanlar) - A'dan Z'ye sıralı
        available_starters_indices_away, starters_display_dict_away = get_sorted_player_options(
            away_squad, exclude_indices=current_away_subs
        )
        
        away_starters = st.multiselect(
            "Başlangıç 11 (deplasman)",
            options=available_starters_indices_away,
            format_func=lambda x: starters_display_dict_away[x],
            key="away_starters_select",
            default=current_away_starters,
            max_selections=TOP_N_STARTERS
        )
        
        # Seçimleri session state'e kaydet
        st.session_state.away_starters = away_starters
    
    with col2:
        st.markdown("**🔄 Yedek Oyuncular (max 7)**")
        
        # Yedekler için kullanılabilir oyuncular (başlangıçta olmayanlar) - A'dan Z'ye sıralı
        available_subs_indices_away, subs_display_dict_away = get_sorted_player_options(
            away_squad, exclude_indices=current_away_starters
        )
        
        away_subs = st.multiselect(
            "Yedek Oyuncular (deplasman)",
            options=available_subs_indices_away,
            format_func=lambda x: subs_display_dict_away[x],
            key="away_subs_select",
            default=current_away_subs,
            max_selections=TOP_N_SUBS
        )
        
        # Seçimleri session state'e kaydet
        st.session_state.away_subs = away_subs

    # Takım yaş ortalaması bilgisi
    if 'Age' in away_squad.columns:
        away_avg_age = away_squad['Age'].mean()
        st.info(f"**📊 {away_team} Takım Yaş Ortalaması:** {away_avg_age:.1f} yaş")

    # Seçili oyuncu sayılarını göster
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**Başlangıç 11:** {len(away_starters)}/{TOP_N_STARTERS} oyuncu")
    with col2:
        st.info(f"**Yedekler:** {len(away_subs)}/{TOP_N_SUBS} oyuncu")

    # Temizle butonları
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        if st.button("🔄 Ev Kadrosunu Temizle", type="secondary"):
            st.session_state.home_starters = []
            st.session_state.home_subs = []
            st.rerun()
    with col2:
        if st.button("🔄 Dep Kadrosunu Temizle", type="secondary"):
            st.session_state.away_starters = []
            st.session_state.away_subs = []
            st.rerun()
    with col3:
        if st.button("🎯 Tüm Kadroları Otomatik Doldur", type="primary"):
            # Otomatik seçim
            st.session_state.home_starters = select_topn_by_rating(home_squad, TOP_N_STARTERS)
            home_all_idxs = home_squad['PlayerRating'].dropna().sort_values(ascending=False).index.tolist()
            st.session_state.home_subs = [i for i in home_all_idxs if i not in st.session_state.home_starters][:TOP_N_SUBS]
            
            st.session_state.away_starters = select_topn_by_rating(away_squad, TOP_N_STARTERS)
            away_all_idxs = away_squad['PlayerRating'].dropna().sort_values(ascending=False).index.tolist()
            st.session_state.away_subs = [i for i in away_all_idxs if i not in st.session_state.away_starters][:TOP_N_SUBS]
            st.rerun()

    st.markdown("---")

    # ---------- TAHMİN BUTONU ----------
    if st.button("🔮 Tahmin Yap", type="primary"):
        try:
            # Otomatik seçim yapılması gerekiyorsa
            if not home_starters or len(home_starters) < TOP_N_STARTERS:
                st.warning(f"⚠ Ev sahibi için yeterli başlangıç oyuncusu seçilmedi. En iyi {TOP_N_STARTERS} oyuncu otomatik seçilecek.")
                home_starters = select_topn_by_rating(home_squad, TOP_N_STARTERS)
            
            if not home_subs or len(home_subs) < TOP_N_SUBS:
                st.warning(f"⚠ Ev sahibi için yeterli yedek oyuncu seçilmedi. En iyi {TOP_N_SUBS} yedek oyuncu otomatik seçilecek.")
                home_all_idxs = home_squad['PlayerRating'].dropna().sort_values(ascending=False).index.tolist()
                home_subs = [i for i in home_all_idxs if i not in home_starters][:TOP_N_SUBS]
            
            if not away_starters or len(away_starters) < TOP_N_STARTERS:
                st.warning(f"⚠ Deplasman için yeterli başlangıç oyuncusu seçilmedi. En iyi {TOP_N_STARTERS} oyuncu otomatik seçilecek.")
                away_starters = select_topn_by_rating(away_squad, TOP_N_STARTERS)
            
            if not away_subs or len(away_subs) < TOP_N_SUBS:
                st.warning(f"⚠ Deplasman için yeterli yedek oyuncu seçilmedi. En iyi {TOP_N_SUBS} yedek oyuncu otomatik seçilecek.")
                away_all_idxs = away_squad['PlayerRating'].dropna().sort_values(ascending=False).index.tolist()
                away_subs = [i for i in away_all_idxs if i not in away_starters][:TOP_N_SUBS]

            # ✅ ULTIMATE FEATURE ROW KULLAN
            row = build_ultimate_feature_row(
                home_team, away_team,
                home_squad, away_squad,
                home_starters, home_subs,
                away_starters, away_subs,
                df_form, load_player_data(PLAYER_DATA_PATH)
            )

            # Eksik feature'ları tamamla
            for feature in features_order:
                if feature not in row:
                    if 'value' in feature.lower():
                        row[feature] = 0.0
                    elif 'ratio' in feature.lower():
                        row[feature] = 1.0
                    elif 'diff' in feature.lower():
                        row[feature] = 0.0
                    elif 'age' in feature.lower():
                        row[feature] = 26.0
                    else:
                        row[feature] = 0.0

            # Model için hazırla
            feat_row = {f: row.get(f, 0) for f in features_order}
            X = pd.DataFrame([feat_row])[features_order].copy()
            X = X.fillna(0)  # NaN değerleri doldur

            # Tahmin yap
            pred = model.predict(X)[0]
            probs = model.predict_proba(X)[0]
            labels = ['Draw', 'HomeWin', 'AwayWin']
            pred_label = labels[int(pred)]
            pred_prob = float(probs[int(pred)]) if 0 <= int(pred) < len(probs) else np.nan

            # ---------- SONUÇLARI GÖSTER ----------
            st.success(f"🎯 Tahmin Sonucu: {home_team} vs {away_team}")
            
            # Olasılık metrikleri
            st.subheader("📊 Tahmin Olasılıkları")
            c1, c2, c3 = st.columns(3)
            c1.metric("🏠 Ev Sahibi Kazanır", f"{probs[1]*100:.1f}%", delta=f"{probs[1]*100-33.3:.1f}%")
            c2.metric("🤝 Beraberlik", f"{probs[0]*100:.1f}%", delta=f"{probs[0]*100-33.3:.1f}%")
            c3.metric("✈️ Deplasman Kazanır", f"{probs[2]*100:.1f}%", delta=f"{probs[2]*100-33.3:.1f}%")

            # Kazanan tahmini
            st.subheader("🏆 Tahmin Sonucu")
            if pred_label == 'HomeWin':
                st.success(f"**🎯 MODEL TAHMİNİ: {home_team} KAZANIR** (Güven: {pred_prob*100:.1f}%)")
            elif pred_label == 'AwayWin':
                st.success(f"**🎯 MODEL TAHMİNİ: {away_team} KAZANIR** (Güven: {pred_prob*100:.1f}%)")
            else:
                st.info(f"**🎯 MODEL TAHMİNİ: BERABERLİK** (Güven: {pred_prob*100:.1f}%)")

            # Takım istatistikleri
            st.subheader("📈 Takım İstatistikleri")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**{home_team}**")
                st.metric("⭐ Takım Rating", f"{row.get('Home_AvgRating', 0):.1f}")
                st.metric("📈 Form (5 maç)", f"{row.get('home_form_5games', 0)*100:.1f}%")
                st.metric("📊 PPG Cumulative", f"{row.get('home_ppg_cumulative', 0):.2f}")
                st.metric("⚽ Gol Ortalaması", f"{row.get('home_gpg_cumulative', 0):.2f}")
                st.metric("👥 Yaş Ortalaması", f"{row.get('home_squad_avg_age', 26.0):.1f} yaş")  # YAŞ EKLENDİ
                if row.get('home_current_value_eur', 0) > 0:
                    st.metric("💰 Takım Değeri", f"€{row.get('home_current_value_eur', 0):.0f}")
            
            with col2:
                st.write(f"**{away_team}**")
                st.metric("⭐ Takım Rating", f"{row.get('Away_AvgRating', 0):.1f}")
                st.metric("📈 Form (5 maç)", f"{row.get('away_form_5games', 0)*100:.1f}%")
                st.metric("📊 PPG Cumulative", f"{row.get('away_ppg_cumulative', 0):.2f}")
                st.metric("⚽ Gol Ortalaması", f"{row.get('away_gpg_cumulative', 0):.2f}")
                st.metric("👥 Yaş Ortalaması", f"{row.get('away_squad_avg_age', 26.0):.1f} yaş")  # YAŞ EKLENDİ
                if row.get('away_current_value_eur', 0) > 0:
                    st.metric("💰 Takım Değeri", f"€{row.get('away_current_value_eur', 0):.0f}")

            # Yaş karşılaştırması
            home_age = row.get('home_squad_avg_age', 26.0)
            away_age = row.get('away_squad_avg_age', 26.0)
            age_diff = home_age - away_age
            
            st.subheader("👥 Yaş Analizi")
            age_col1, age_col2, age_col3 = st.columns(3)
            with age_col1:
                st.metric("Ev Sahibi Yaş", f"{home_age:.1f}")
            with age_col2:
                st.metric("Deplasman Yaş", f"{away_age:.1f}")
            with age_col3:
                st.metric("Yaş Farkı", f"{age_diff:+.1f}")
            
            if age_diff > 1.0:
                st.info(f"📊 {home_team} daha deneyimli bir kadroya sahip (+{age_diff:.1f} yaş)")
            elif age_diff < -1.0:
                st.info(f"📊 {away_team} daha genç ve dinamik bir kadroya sahip ({age_diff:+.1f} yaş)")
            else:
                st.info("📊 Takımlar benzer yaş profiline sahip")

            # Son 5 maç form durumu
            st.subheader("📋 Son 5 Maç Formu")
            
            home_report = last5_report_pretty(df_form, home_team, norm_map, max_lines=5)
            away_report = last5_report_pretty(df_form, away_team, norm_map, max_lines=5)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**{home_team}**")
                if home_report:
                    st.text(home_report)
                else:
                    st.info("⚠ Son 5 maç verisi bulunamadı")
            
            with col2:
                st.write(f"**{away_team}**")
                if away_report:
                    st.text(away_report)
                else:
                    st.info("⚠ Son 5 maç verisi bulunamadı")

            # Önemli feature'lar
            st.subheader("🔍 Önemli Feature Değerleri")
            important_features = features_order[:10]  # İlk 10 önemli feature'ı göster
            
            feature_values = []
            for feat in important_features:
                if feat in row:
                    feature_values.append({
                        'Feature': feat,
                        'Değer': f"{row[feat]:.3f}",
                        'Açıklama': get_ultimate_feature_descriptions().get(feat, 'Bilinmeyen feature'),
                        'Önem': '🏆 KRİTİK' if feat in ['home_ppg_cumulative', 'away_ppg_cumulative', 'home_form_5games'] else '📈 YÜKSEK'
                    })
            
            if feature_values:
                st.dataframe(pd.DataFrame(feature_values), use_container_width=True)

        except Exception as e:
            st.error("❌ Tahmin çalıştırılırken bir hata oluştu.")
            st.error(f"Hata detayı: {str(e)}")
            st.text(traceback.format_exc())

# ---------- FOOTER ----------
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: gray; font-size: 14px;'>
    <p>⚽ Bundesliga Tahmin Sistemi - ULTIMATE BALANCE v12.1 | Takım Yaş Ortalaması Entegreli</p>
    <p>© 2025 Cansel Yardım | All Rights Reserved</p>
    <p>🔒 Licensed under MIT License</p>
</div>
""", unsafe_allow_html=True)