# app.py - GÜNCELLENMİŞ TAM KOD
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

# ================== GÜNCELLENMİŞ KONFİG ==================
RANDOM_STATE = 42
DATA_PATH = "data/bundesliga_matches_2023_2025_final_fe_team_values_cleaned.xlsx"
PLAYER_DATA_PATH = "data/final_bundesliga_dataset_complete.xlsx"
MODEL_PATH = "models/bundesliga_model_overfitting_fixed.pkl"
FEATURE_INFO_PATH = "models/feature_info_overfitting_fixed.pkl"

TOP_N_STARTERS = 11
TOP_N_SUBS = 7
STARTER_WEIGHT = 0.7
SUB_WEIGHT = 0.3

# Varsayılan feature listesi (model yüklenmezse kullanılacak)
DEFAULT_FEATURES = [
    'value_difference', 'form_difference', 'away_form', 'power_sum', 
    'power_difference', 'home_form', 'power_ratio', 'goals_ratio', 
    'xg_difference', 'xg_ratio', 'h2h_avg_goals', 'value_ratio', 
    'age_difference', 'h2h_away_goals', 'h2h_goal_difference'
]

# ================== YARDIMCI FONKSİYONLAR ==================
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

def get_feature_description(feature_name):
    """Feature açıklamalarını getir"""
    descriptions = {
        'value_difference': 'Takım değer farkı (Ev - Deplasman)',
        'form_difference': 'Form farkı (Ev - Deplasman)',
        'power_difference': 'Güç indeksi farkı',
        'power_ratio': 'Güç oranı (Ev / Deplasman)',
        'goals_ratio': 'Gol oranı (Ev / Deplasman)',
        'age_difference': 'Yaş farkı (Ev - Deplasman)',
        'away_form': 'Deplasman takım formu',
        'home_form': 'Ev sahibi takım formu',
        'power_sum': 'Toplam güç indeksi',
        'xg_difference': 'Expected goals farkı',
        'xg_ratio': 'Expected goals oranı',
        'h2h_avg_goals': 'H2H ortalama gol sayısı',
        'value_ratio': 'Takım değer oranı',
        'h2h_away_goals': 'H2H deplasman golleri',
        'h2h_goal_difference': 'H2H gol farkı'
    }
    return descriptions.get(feature_name, 'Bilinmeyen feature')

def load_player_data(path=PLAYER_DATA_PATH):
    """Oyuncu verilerini yükle"""
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
        
        # Yaş sütunu
        if 'Age' not in df.columns:
            if 'fbref__Age' in df.columns:
                df['Age'] = pd.to_numeric(df['fbref__Age'], errors='coerce')
            else:
                df['Age'] = np.nan
                
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
        return {'form': 0.5, 'gs_5': 0, 'gc_5': 0, 'momentum': 0}
    
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
    
    return {'form': form, 'gs_5': int(gs), 'gc_5': int(gc), 'momentum': int(momentum)}

def derby_flag(home, away):
    """Derby maçı kontrolü"""
    big_teams = {
        'Bayern Munich', 'Borussia Dortmund', 'Schalke 04', 'Hamburg SV',
        'Borussia Mönchengladbach', 'Bayer Leverkusen', 'VfB Stuttgart',
        'Bayern München', 'Borussia Dortmund', 'FC Bayern Munich'
    }
    return 1 if (home in big_teams and away in big_teams) else 0

def maybe_team_value_features(df_players, team):
    """Takım değer özelliklerini çıkar"""
    if df_players.empty:
        return {}
        
    sub = df_players[df_players['Team'] == team].copy()
    if sub.empty:
        return {}
        
    feats = {}
    
    # Yaş özellikleri
    age_cols = [c for c in sub.columns if re.search(r'age', c, re.I)]
    if age_cols:
        ages = pd.to_numeric(sub[age_cols[0]], errors='coerce')
        if ages.notna().sum() >= 3:
            feats['squad_avg_age'] = float(ages.mean())
    
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

def enhanced_feature_engineering(row):
    """Gelişmiş feature engineering"""
    enhanced_row = row.copy()
    
    try:
        # 1. Value-based özellikler
        if all(k in row for k in ['home_current_value_eur', 'away_current_value_eur']):
            enhanced_row['value_difference'] = row['home_current_value_eur'] - row['away_current_value_eur']
            enhanced_row['value_ratio'] = row['home_current_value_eur'] / max(row['away_current_value_eur'], 1)
        
        # 2. Form-based özellikler  
        if all(k in row for k in ['home_form', 'away_form']):
            enhanced_row['form_difference'] = row['home_form'] - row['away_form']
            enhanced_row['form_sum'] = row['home_form'] + row['away_form']
        
        # 3. Power metrics
        home_power = row.get('Home_AvgRating', 65) * row.get('home_form', 0.5)
        away_power = row.get('Away_AvgRating', 65) * row.get('away_form', 0.5)
        
        enhanced_row['home_power_index'] = home_power
        enhanced_row['away_power_index'] = away_power
        enhanced_row['power_difference'] = home_power - away_power
        enhanced_row['power_sum'] = home_power + away_power
        enhanced_row['power_ratio'] = home_power / max(away_power, 1)
        
        # 4. Goal-based özellikler
        if all(k in row for k in ['home_goals', 'away_goals']):
            enhanced_row['goals_difference'] = row['home_goals'] - row['away_goals']
            enhanced_row['goals_ratio'] = row['home_goals'] / max(row['away_goals'], 1)
        
        # 5. Age difference
        if all(k in row for k in ['home_squad_avg_age', 'away_squad_avg_age']):
            enhanced_row['age_difference'] = row['home_squad_avg_age'] - row['away_squad_avg_age']
        
        # 6. Varsayılan değerler (eğer veri yoksa)
        enhanced_row.setdefault('h2h_avg_goals', 2.5)
        enhanced_row.setdefault('h2h_away_goals', 0)
        enhanced_row.setdefault('h2h_goal_difference', 0)
        enhanced_row.setdefault('xg_difference', 0)
        enhanced_row.setdefault('xg_ratio', 1.0)
        enhanced_row.setdefault('isDerby', row.get('IsDerby', 0))
        enhanced_row.setdefault('injury_difference', 0)
        enhanced_row.setdefault('performance_ratio', 1.0)
        
    except Exception as e:
        st.warning(f"Feature engineering hatası: {e}")
    
    return enhanced_row

def build_feature_row(
    home_team, away_team,
    df_home, df_away,
    home_start_ids, home_sub_ids,
    away_start_ids, away_sub_ids,
    df_matches_form, df_players
):
    """Model için feature satırı oluştur"""
    # Takım rating'lerini hesapla
    h_team_rating, h_pos, h11, hbench = compute_team_rating_from_lineup(df_home, home_start_ids, home_sub_ids)
    a_team_rating, a_pos, a11, abench = compute_team_rating_from_lineup(df_away, away_start_ids, away_sub_ids)

    # Form verilerini al
    home_form = compute_team_form_snapshot(df_matches_form, home_team)
    away_form = compute_team_form_snapshot(df_matches_form, away_team)

    # Takım değer özelliklerini al
    hv_feats = maybe_team_value_features(df_players, home_team) or {}
    av_feats = maybe_team_value_features(df_players, away_team) or {}

    # Temel özellikleri oluştur
    row = {
    'Home_AvgRating': safe_float(h_team_rating, 65.0),
    'Away_AvgRating': safe_float(a_team_rating, 65.0),
    'home_form': safe_float(home_form['form'], 0.5),
    'away_form': safe_float(away_form['form'], 0.5),
    'home_current_value_eur': safe_float(hv_feats.get('current_value_eur', 0.0), 0.0),
    'away_current_value_eur': safe_float(av_feats.get('current_value_eur', 0.0), 0.0),
    'home_squad_avg_age': safe_float(hv_feats.get('squad_avg_age', 0.0), 0.0),
    'away_squad_avg_age': safe_float(av_feats.get('squad_avg_age', 0.0), 0.0),
    'home_goals': safe_float(home_form['gs_5'], 0),
    'away_goals': safe_float(away_form['gs_5'], 0),
    'homeTeam_Momentum': safe_float(home_form['momentum'], 0),
    'awayTeam_Momentum': safe_float(away_form['momentum'], 0),
    'IsDerby': int(derby_flag(home_team, away_team)),

    # Ek form özellikleri
    'home_last5_form_points': safe_float(home_form['gs_5'] - home_form['gc_5'] + 5, 5),
    'away_last5_form_points': safe_float(away_form['gs_5'] - away_form['gc_5'] + 5, 5),
    }

    
    # Enhanced feature engineering uygula
    row = enhanced_feature_engineering(row)
    
    return row

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
st.set_page_config(page_title="Bundesliga Predictor - Geliştirilmiş", layout="wide")
st.title("⚽ Bundesliga Tahmin Sistemi (Stable)")

@st.cache_resource
def load_data():
    """Verileri yükle"""
    try:
        # Model ve feature bilgilerini yükle
        model = joblib.load(MODEL_PATH)
        feat_info = joblib.load(FEATURE_INFO_PATH)
        
        # Feature order'ı al
        if isinstance(feat_info, dict) and 'important_features' in feat_info:
            features_order = feat_info['important_features']
        else:
            features_order = DEFAULT_FEATURES
            st.sidebar.warning("⚠ Feature info bulunamadı, varsayılan özellikler kullanılıyor")
        
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
        
        st.sidebar.success(f"✅ Model yüklendi! {len(features_order)} özellik kullanılacak")
        return model, features_order, team_dict, df_form, norm_map
        
    except FileNotFoundError as e:
        st.error(f"❌ Dosya bulunamadı: {e}")
        st.error("Lütfen model dosyalarının doğru konumda olduğundan emin olun.")
        st.stop()
    except Exception as e:
        st.error(f"❌ Veri yüklenirken hata oluştu: {str(e)}")
        st.stop()

# Verileri yükle
try:
    model, features_order, team_dict, df_form, norm_map = load_data()
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
# st.sidebar.header("ℹ️ Sistem Bilgisi")
# st.sidebar.info("""
# **Model Özellikleri:**
# - ✅ Overfitting önleyici
# - ✅ %54.7 test accuracy  
# - ✅ 15 önemli feature
# - ✅ Gelişmiş regularizasyon
# """)

# st.sidebar.header("📊 Model Performansı")
# st.sidebar.metric("Test Doğruluk", "%54.7")
# st.sidebar.metric("Overfitting Gap", "%8.3")
# st.sidebar.metric("Kullanılan Özellikler", len(features_order))

# Eğer sidebar tamamen gizlensin istersen:
hide_sidebar = """
    <style>
        [data-testid="stSidebar"] {display: none;}
        [data-testid="stSidebarNav"] {display: none;}
    </style>
"""
st.markdown(hide_sidebar, unsafe_allow_html=True)

# ---------- ANA UYGULAMA ----------
st.header("1️⃣ Takım Seçimi")

# Takım dropdown'ları
# Bochum ve Holstein Kiel'i hariç tut
exclude_norm = [normalize_name("vfl bochum"), normalize_name("Holstein Kiel")]
teams_display = [t for t in norm_map.values() if normalize_name(t) not in exclude_norm]

col1, col2 = st.columns(2)
with col1:
    home_team_display = st.selectbox(
        "🏠 Ev Sahibi Takım",
        teams_display,
        index=teams_display.index("Bayern Munich") if "Bayern Munich" in teams_display else 0,
        key="home_team"
    )
with col2:
    away_team_display = st.selectbox(
        "✈️ Deplasman Takımı",
        teams_display,
        index=teams_display.index("Borussia Dortmund") if "Borussia Dortmund" in teams_display else 1,
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
    
    # Ev sahibi takım kadrosu
    st.subheader(f"👥 {home_team} Kadrosu")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🥅 Başlangıç 11**")
        home_starters = st.multiselect(
            "Başlangıç 11 (ev sahibi)",
            options=list(home_squad.index),
            format_func=lambda x: f"{home_squad.loc[x, 'Player']} - {home_squad.loc[x, 'Pos']} ({home_squad.loc[x, 'PlayerRating']:.1f})",
            key="home_starters_select",
            default=st.session_state.home_starters
        )
    
    with col2:
        st.markdown("**🔄 Yedek Oyuncular (max 7)**")
        home_subs = st.multiselect(
            "Yedek Oyuncular (ev sahibi)",
            options=list(home_squad.index),
            format_func=lambda x: f"{home_squad.loc[x, 'Player']} - {home_squad.loc[x, 'Pos']} ({home_squad.loc[x, 'PlayerRating']:.1f})",
            key="home_subs_select",
            default=st.session_state.home_subs
        )

    # Deplasman takım kadrosu
    st.subheader(f"👥 {away_team} Kadrosu")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🥅 Başlangıç 11**")
        away_starters = st.multiselect(
            "Başlangıç 11 (deplasman)",
            options=list(away_squad.index),
            format_func=lambda x: f"{away_squad.loc[x, 'Player']} - {away_squad.loc[x, 'Pos']} ({away_squad.loc[x, 'PlayerRating']:.1f})",
            key="away_starters_select",
            default=st.session_state.away_starters
        )
    
    with col2:
        st.markdown("**🔄 Yedek Oyuncular (max 7)**")
        away_subs = st.multiselect(
            "Yedek Oyuncular (deplasman)",
            options=list(away_squad.index),
            format_func=lambda x: f"{away_squad.loc[x, 'Player']} - {away_squad.loc[x, 'Pos']} ({away_squad.loc[x, 'PlayerRating']:.1f})",
            key="away_subs_select",
            default=st.session_state.away_subs
        )

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

            # Özellik satırı oluştur
            row = build_feature_row(
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
                st.metric("📈 Form", f"{row.get('home_form', 0)*100:.1f}%")
                st.metric("⚽ Momentum", row.get('homeTeam_Momentum', 0))
                if row.get('home_current_value_eur', 0) > 0:
                    st.metric("💰 Takım Değeri", f"€{row.get('home_current_value_eur', 0):.0f}")
            
            with col2:
                st.write(f"**{away_team}**")
                st.metric("⭐ Takım Rating", f"{row.get('Away_AvgRating', 0):.1f}")
                st.metric("📈 Form", f"{row.get('away_form', 0)*100:.1f}%")
                st.metric("⚽ Momentum", row.get('awayTeam_Momentum', 0))
                if row.get('away_current_value_eur', 0) > 0:
                    st.metric("💰 Takım Değeri", f"€{row.get('away_current_value_eur', 0):.0f}")

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

            # Seçili oyuncular
            st.subheader("👥 Seçili Oyuncular")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**{home_team} - Başlangıç 11:**")
                if home_starters:
                    home_starters_df = home_squad.loc[home_starters, ["Player", "Pos", "PlayerRating"]].copy()
                    home_starters_df.columns = ["İsim", "Pozisyon", "Rating"]
                    home_starters_df["Rating"] = home_starters_df["Rating"].round(1)
                    st.dataframe(home_starters_df.reset_index(drop=True), use_container_width=True)
                else:
                    st.info("Başlangıç 11 seçilmedi")
                
                st.write(f"**{home_team} - Yedekler:**")
                if home_subs:
                    home_subs_df = home_squad.loc[home_subs, ["Player", "Pos", "PlayerRating"]].copy()
                    home_subs_df.columns = ["İsim", "Pozisyon", "Rating"]
                    home_subs_df["Rating"] = home_subs_df["Rating"].round(1)
                    st.dataframe(home_subs_df.reset_index(drop=True), use_container_width=True)
                else:
                    st.info("Yedek oyuncu seçilmedi")
            
            with col2:
                st.write(f"**{away_team} - Başlangıç 11:**")
                if away_starters:
                    away_starters_df = away_squad.loc[away_starters, ["Player", "Pos", "PlayerRating"]].copy()
                    away_starters_df.columns = ["İsim", "Pozisyon", "Rating"]
                    away_starters_df["Rating"] = away_starters_df["Rating"].round(1)
                    st.dataframe(away_starters_df.reset_index(drop=True), use_container_width=True)
                else:
                    st.info("Başlangıç 11 seçilmedi")
                
                st.write(f"**{away_team} - Yedekler:**")
                if away_subs:
                    away_subs_df = away_squad.loc[away_subs, ["Player", "Pos", "PlayerRating"]].copy()
                    away_subs_df.columns = ["İsim", "Pozisyon", "Rating"]
                    away_subs_df["Rating"] = away_subs_df["Rating"].round(1)
                    st.dataframe(away_subs_df.reset_index(drop=True), use_container_width=True)
                else:
                    st.info("Yedek oyuncu seçilmedi")

            # Önemli feature'lar
            st.subheader("🔍 Önemli Feature Değerleri")
            important_features = [
                'value_difference', 'form_difference', 'power_difference', 
                'power_ratio', 'goals_ratio', 'age_difference'
            ]
            
            feature_values = []
            for feat in important_features:
                if feat in row:
                    feature_values.append({
                        'Feature': feat,
                        'Değer': f"{row[feat]:.3f}",
                        'Açıklama': get_feature_description(feat)
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
    <p>⚽ Bundesliga Tahmin Sistemi v1.0</p>
    <p>© 2025 Cansel Yardım | All Rights Reserved</p>
    <p>🔒 Licensed under MIT License</p>
</div>

""", unsafe_allow_html=True)

# Debug için feature listesini göster (opsiyonel)
if st.sidebar.checkbox("🔧 Debug Modu"):
    st.sidebar.subheader("Debug Bilgileri")
    st.sidebar.write(f"Kullanılacak Feature'lar ({len(features_order)}):")
    for i, feat in enumerate(features_order[:10]):  # İlk 10'u göster
        st.sidebar.write(f"{i+1}. {feat}")
    if len(features_order) > 10:
        st.sidebar.write(f"... ve {len(features_order) - 10} feature daha")