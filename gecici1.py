#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bundesliga Tahmin Sistemi - ULTIMATE FIX v15.0
Streamlit Tahmin Uygulaması - MODÜLER & KULLANICI DOSTU
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import warnings
import re
import unicodedata
import difflib
from datetime import datetime, timedelta
import traceback
import os

warnings.filterwarnings("ignore")

# ================== KONFİGÜRASYON ==================
DATA_PATH = "data/bundesliga_matches_2023_2025_final_fe_team_values_cleaned.pkl"
PLAYER_DATA_PATH = "data/final_bundesliga_dataset_complete.xlsx"
MODEL_PATH = "models/bundesliga_model_ultimate_v12.1_20251122_204623.pkl"
FEATURE_INFO_PATH = "models/feature_info_ultimate_v12.1.pkl"

TOP_N_STARTERS = 11
TOP_N_SUBS = 7

# ================== TAHMİN SİSTEMİ ==================
def simple_predict(model, X):
    """BASİT ve ETKİLİ tahmin sistemi - Beraberliği minimize et"""
    y_pred_proba = model.predict_proba(X)
    
    y_pred_custom = []
    for proba in y_pred_proba:
        draw_prob = proba[0]
        home_prob = proba[1] 
        away_prob = proba[2]
        
        # EV SAHİBİNE GÜÇLÜ AVANTAJ (%15 bonus)
        home_prob_boosted = home_prob * 1.15
        
        # ÇOK BASİT KURAL: En yüksek olasılık kazanan, BERABERLİK ÇOK NADİR
        beraberlik_kosul = (
            draw_prob > 0.40 and
            draw_prob > home_prob_boosted * 1.20 and
            draw_prob > away_prob * 1.20 and
            abs(home_prob_boosted - away_prob) < 0.05
        )
        
        if beraberlik_kosul:
            y_pred_custom.append(0)  # Draw
        else:
            # NORMAL DURUM: En yüksek olasılık kazanan
            if home_prob_boosted > away_prob and home_prob_boosted > draw_prob:
                y_pred_custom.append(1)  # HomeWin
            elif away_prob > home_prob_boosted and away_prob > draw_prob:
                y_pred_custom.append(2)  # AwayWin
            else:
                # Son çare: En yüksek olasılık (beraberlik hariç)
                if home_prob_boosted >= away_prob:
                    y_pred_custom.append(1)  # HomeWin
                else:
                    y_pred_custom.append(2)  # AwayWin
    
    return np.array(y_pred_custom), y_pred_proba

def normalize_name(name: str) -> str:
    """Takım isimlerini normalize et"""
    if not isinstance(name, str):
        return ""
    s = name.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r'[^a-z0-9\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    
    mapping = {
        'bayern munchen': 'fc bayern munchen',
        'bayern munich': 'fc bayern munchen', 
        'bayer leverkusen': 'bayer 04 leverkusen',
        'eintracht frankfurt': 'eintracht frankfurt',
        'borussia dortmund': 'borussia dortmund',
        'freiburg': 'sc freiburg',
        'mainz 05': '1. fsv mainz 05',
        'rb leipzig': 'rb leipzig',
        'werder bremen': 'sv werder bremen',
        'vfb stuttgart': 'vfb stuttgart',
        'monchengladbach': 'borussia monchengladbach',
        'wolfsburg': 'vfl wolfsburg',
        'augsburg': 'fc augsburg',
        'union berlin': '1. fc union berlin',
        'hoffenheim': 'tsg 1899 hoffenheim',
        'heidenheim': '1. fc heidenheim 1846',
        'koln': '1. fc koln',
        'bochum': 'bochum 1848',
        'darmstadt': 'darmstadt 98',
        'st pauli': 'fc st. pauli'
    }
    
    for key, value in mapping.items():
        if key in s:
            return value
    
    return s

# ================== STREAMLIT UYGULAMASI ==================
st.set_page_config(
    page_title="Bundesliga Predictor - ULTIMATE FIX v15.0", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# ================== SİSTEM YÜKLEME FONKSİYONLARI ==================
@st.cache_data(show_spinner=False)
def load_player_data():
    try:
        if not os.path.exists(PLAYER_DATA_PATH):
            st.error(f"❌ Oyuncu veri dosyası bulunamadı: {PLAYER_DATA_PATH}")
            return pd.DataFrame()
            
        df = pd.read_excel(PLAYER_DATA_PATH)
        df.columns = [col.strip() for col in df.columns]
        
        if 'PlayerRating' not in df.columns:
            df['PlayerRating'] = 65.0
        
        if 'Team' not in df.columns:
            if 'fbref__Squad' in df.columns:
                df['Team'] = df['fbref__Squad'].astype(str).str.strip()
            else:
                return pd.DataFrame()
        else:
            df['Team'] = df['Team'].astype(str).str.strip()
        
        if 'Pos' not in df.columns:
            df['Pos'] = 'MF'
        df['Pos'] = df['Pos'].astype(str).str.upper().str.strip()
        
        if 'Player' not in df.columns:
            df['Player'] = np.arange(len(df)).astype(str)
        
        df['Player'] = df['Player'].astype(str).str.strip()
        return df
        
    except Exception as e:
        st.error(f"❌ Oyuncu verileri yüklenirken hata: {str(e)}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def load_match_data():
    try:
        if not os.path.exists(DATA_PATH):
            st.error(f"❌ Maç veri dosyası bulunamadı: {DATA_PATH}")
            return pd.DataFrame()
            
        if DATA_PATH.endswith('.pkl'):
            return pd.read_pickle(DATA_PATH)
        else:
            return pd.read_excel(DATA_PATH)
    except Exception as e:
        st.error(f"❌ Maç verileri yüklenirken hata: {str(e)}")
        return pd.DataFrame()

@st.cache_resource(show_spinner=False)
def load_model_and_features():
    try:
        if not os.path.exists(MODEL_PATH):
            st.error(f"❌ Model dosyası bulunamadı: {MODEL_PATH}")
            return None, []
            
        model = joblib.load(MODEL_PATH)
        
        if os.path.exists(FEATURE_INFO_PATH):
            feat_info = joblib.load(FEATURE_INFO_PATH)
            if isinstance(feat_info, dict) and 'important_features' in feat_info:
                return model, feat_info['important_features']
        
        features_order = [
            'home_ppg_cumulative', 'away_ppg_cumulative', 'home_form_5games', 'away_form_5games',
            'home_gpg_cumulative', 'away_gpg_cumulative', 'home_gapg_cumulative', 'away_gapg_cumulative',
            'home_power_index', 'away_power_index', 'power_difference', 'form_difference'
        ]
        return model, features_order
        
    except Exception as e:
        st.error(f"❌ Model yüklenirken hata: {str(e)}")
        return None, []

def initialize_system():
    with st.spinner("🔄 Sistem başlatılıyor..."):
        model, features_order = load_model_and_features()
        if model is None:
            st.stop()
            
        df_players = load_player_data()
        if df_players.empty:
            st.error("❌ Oyuncu verileri yüklenemedi!")
            st.stop()
            
        team_dict = {}
        for team in sorted(df_players['Team'].dropna().unique()):
            team_dict[team] = df_players[df_players['Team'] == team].copy().reset_index(drop=True)
            
        df_matches = load_match_data()
        if df_matches.empty:
            st.error("❌ Maç verileri yüklenemedi!")
            st.stop()
            
        norm_map = {}
        for orig in team_dict.keys():
            n = normalize_name(orig)
            if n:
                norm_map[n] = orig
                
        st.sidebar.success(f"✅ Sistem hazır! {len(team_dict)} takım")
        
        return model, features_order, team_dict, df_matches, norm_map

# ================== SESSION STATE ==================
def initialize_session_state():
    defaults = {
        "show_squads": False,
        "home_starters": [],
        "home_subs": [], 
        "away_starters": [],
        "away_subs": [],
        "auto_fill_triggered": False,
        "last_prediction": None
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# ================== YARDIMCI FONKSİYONLAR ==================
def safe_float(x, default=np.nan):
    try:
        if x is None: return default
        return float(x)
    except Exception:
        return default

def select_topn_by_rating(df_team, n):
    if 'PlayerRating' not in df_team.columns or df_team.empty:
        return []
    return df_team['PlayerRating'].dropna().sort_values(ascending=False).index.tolist()[:n]

def compute_team_rating_from_lineup(df_team, starter_idxs, sub_idxs):
    if len(starter_idxs) == 0 and len(sub_idxs) == 0:
        return 65.0, {}, 65.0, 65.0
    
    starter_ratings = df_team.loc[starter_idxs, 'PlayerRating'].dropna()
    sub_ratings = df_team.loc[sub_idxs, 'PlayerRating'].dropna()
    
    starter_mean = starter_ratings.mean() if not starter_ratings.empty else 65.0
    sub_mean = sub_ratings.mean() if not sub_ratings.empty else 65.0
    
    team_rating = (starter_mean * 0.7) + (sub_mean * 0.3)
    
    return team_rating, {}, starter_mean, sub_mean

def calculate_realistic_power_index(team_rating):
    if pd.isna(team_rating):
        return 0.5
    normalized = (team_rating - 60) / 25
    return max(0.2, min(1.0, normalized))

def prepare_matches_for_form(df_matches):
    df = df_matches.copy()
    
    if 'HomeTeam' not in df.columns and 'homeTeam.name' in df.columns:
        df['HomeTeam'] = df['homeTeam.name']
    if 'AwayTeam' not in df.columns and 'awayTeam.name' in df.columns:
        df['AwayTeam'] = df['awayTeam.name']
    
    if 'Date' not in df.columns:
        df['Date'] = pd.to_datetime('today')
    
    df = df.sort_values('Date').reset_index(drop=True)
    
    if 'score.fullTime.home' not in df.columns:
        df['score.fullTime.home'] = 0
        df['score.fullTime.away'] = 0
            
    return df

def compute_team_form_snapshot(df_form, team):
    norm = normalize_name(team)
    
    if '_HomeNorm' not in df_form.columns:
        df_form = df_form.copy()
        df_form['_HomeNorm'] = df_form['HomeTeam'].astype(str).apply(normalize_name)
        df_form['_AwayNorm'] = df_form['AwayTeam'].astype(str).apply(normalize_name)
    
    team_matches = df_form[(df_form['_HomeNorm'] == norm) | (df_form['_AwayNorm'] == norm)].copy()
    
    if len(team_matches) == 0:
        return {'form': 0.5, 'points_5': 0}
    
    team_matches = team_matches.sort_values('Date').reset_index(drop=True)
    last_5 = team_matches.tail(5)
    
    points = 0
    for _, m in last_5.iterrows():
        hg = safe_float(m.get('score.fullTime.home', 0), 0)
        ag = safe_float(m.get('score.fullTime.away', 0), 0)
        
        if normalize_name(str(m.get('HomeTeam', ''))) == norm:
            if hg > ag: points += 3
            elif hg == ag: points += 1
        else:
            if ag > hg: points += 3
            elif ag == hg: points += 1
    
    form = points / 15.0 if len(last_5) > 0 else 0.5
    
    return {'form': form, 'points_5': points}

def predict_calculate_cumulative_stats(df_form, home_team, away_team):
    home_norm = normalize_name(home_team)
    away_norm = normalize_name(away_team)
    
    if '_HomeNorm' not in df_form.columns:
        df_form = df_form.copy()
        df_form['_HomeNorm'] = df_form['HomeTeam'].astype(str).apply(normalize_name)
        df_form['_AwayNorm'] = df_form['AwayTeam'].astype(str).apply(normalize_name)
    
    home_matches = df_form[(df_form['_HomeNorm'] == home_norm) | (df_form['_AwayNorm'] == home_norm)].copy()
    away_matches = df_form[(df_form['_HomeNorm'] == away_norm) | (df_form['_AwayNorm'] == away_norm)].copy()
    
    def calculate_team_stats(team_matches, team_norm):
        if len(team_matches) == 0:
            return {'ppg_cumulative': 1.5, 'gpg_cumulative': 1.5, 'gapg_cumulative': 1.2, 'form_5games': 0.5}
        
        team_matches = team_matches.sort_values('Date').reset_index(drop=True)
        
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
        
        last_5 = team_matches.tail(5)
        points_5 = 0
        
        for _, m in last_5.iterrows():
            hg = safe_float(m.get('score.fullTime.home', 0), 0)
            ag = safe_float(m.get('score.fullTime.away', 0), 0)
            
            if normalize_name(str(m.get('HomeTeam', ''))) == team_norm:
                if hg > ag: points_5 += 3
                elif hg == ag: points_5 += 1
            else:
                if ag > hg: points_5 += 3
                elif ag == hg: points_5 += 1
        
        form_5games = points_5 / 15.0 if len(last_5) > 0 else 0.5
        
        return {
            'ppg_cumulative': total_points / total_matches if total_matches > 0 else 1.5,
            'gpg_cumulative': total_goals_for / total_matches if total_matches > 0 else 1.5,
            'gapg_cumulative': total_goals_against / total_matches if total_matches > 0 else 1.2,
            'form_5games': form_5games
        }
    
    home_stats = calculate_team_stats(home_matches, home_norm)
    away_stats = calculate_team_stats(away_matches, away_norm)
    
    return home_stats, away_stats

def build_simple_feature_row(home_team, away_team, home_squad, away_squad,
                            home_starters, home_subs, away_starters, away_subs, df_form):
    
    h_team_rating, _, _, _ = compute_team_rating_from_lineup(home_squad, home_starters, home_subs)
    a_team_rating, _, _, _ = compute_team_rating_from_lineup(away_squad, away_starters, away_subs)

    home_cumulative, away_cumulative = predict_calculate_cumulative_stats(df_form, home_team, away_team)

    row = {
        'home_ppg_cumulative': safe_float(home_cumulative['ppg_cumulative'], 1.5),
        'away_ppg_cumulative': safe_float(away_cumulative['ppg_cumulative'], 1.5),
        'home_form_5games': safe_float(home_cumulative['form_5games'], 0.5),
        'away_form_5games': safe_float(away_cumulative['form_5games'], 0.5),
        'home_gpg_cumulative': safe_float(home_cumulative['gpg_cumulative'], 1.5),
        'away_gpg_cumulative': safe_float(away_cumulative['gpg_cumulative'], 1.5),
        'home_gapg_cumulative': safe_float(home_cumulative['gapg_cumulative'], 1.2),
        'away_gapg_cumulative': safe_float(away_cumulative['gapg_cumulative'], 1.2),
        'home_power_index': calculate_realistic_power_index(safe_float(h_team_rating, 65)),
        'away_power_index': calculate_realistic_power_index(safe_float(a_team_rating, 65)),
    }

    # Basit feature engineering
    row['power_difference'] = row['home_power_index'] - row['away_power_index']
    row['form_difference'] = row['home_form_5games'] - row['away_form_5games']
    
    return row

def match_team_name(candidate: str, norm_map: dict, cutoff=0.55):
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
    tm = get_last_matches_for_team(df_form, team_candidate, norm_map, n=5)
    if tm.empty:
        return None, []
    
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
            
        icon = "🟢" if res == 'W' else ("🟡" if res == 'D' else "🔴")
        venue = "🏠" if is_home else "✈️"
        lines.append(f"{icon} {venue} {opponent[:15]:15} {hg}-{ag}")
        
        if len(lines) >= max_lines:
            break
    
    header = f"**Son {len(lines)} maç: {wins} Galibiyet, {draws} Beraberlik, {losses} Mağlubiyet**"
    return header, lines

# ================== YENİ FORM GÖSTERİM FONKSİYONLARI ==================
def display_team_form_metrics(df_form, team_name, norm_map):
    """Takım form metriklerini göster"""
    stats = compute_team_form_snapshot(df_form, team_name)
    last_matches = get_last_matches_for_team(df_form, team_name, norm_map, 5)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        form_percentage = stats['form'] * 100
        st.metric("📈 Son 5 Maç Formu", f"{form_percentage:.1f}%")
    
    with col2:
        points = stats['points_5']
        st.metric("🏆 Son 5 Maç Puanı", f"{points}/15")
    
    with col3:
        if not last_matches.empty:
            goals_for = 0
            goals_against = 0
            for _, match in last_matches.iterrows():
                is_home = normalize_name(str(match['HomeTeam'])) == normalize_name(team_name)
                if is_home:
                    goals_for += safe_float(match.get('score.fullTime.home', 0), 0)
                    goals_against += safe_float(match.get('score.fullTime.away', 0), 0)
                else:
                    goals_for += safe_float(match.get('score.fullTime.away', 0), 0)
                    goals_against += safe_float(match.get('score.fullTime.home', 0), 0)
            st.metric("⚽ Gol Ortalaması", f"{(goals_for/5):.1f}")
    
    with col4:
        if not last_matches.empty:
            goals_for = 0
            goals_against = 0
            for _, match in last_matches.iterrows():
                is_home = normalize_name(str(match['HomeTeam'])) == normalize_name(team_name)
                if is_home:
                    goals_for += safe_float(match.get('score.fullTime.home', 0), 0)
                    goals_against += safe_float(match.get('score.fullTime.away', 0), 0)
                else:
                    goals_for += safe_float(match.get('score.fullTime.away', 0), 0)
                    goals_against += safe_float(match.get('score.fullTime.home', 0), 0)
            st.metric("🧱 Y. Gol Ortalaması", f"{(goals_against/5):.1f}")

def display_comparison_metrics(home_stats, away_stats, home_team, away_team):
    """Takım karşılaştırma metriklerini göster"""
    st.subheader("📊 Takım Karşılaştırması")
    
    # Ana metrikler
    col1, col2, col3 = st.columns(3)
    
    with col1:
        form_diff = home_stats['form_5games'] - away_stats['form_5games']
        form_status = "🏠 Ev Sahibi" if form_diff > 0 else "✈️ Deplasman" if form_diff < 0 else "⚖️ Eşit"
        st.metric("📈 Form Farkı", f"{abs(form_diff*100):.1f}%", form_status)
    
    with col2:
        ppg_diff = home_stats['ppg_cumulative'] - away_stats['ppg_cumulative']
        ppg_status = "🏠 Ev Sahibi" if ppg_diff > 0 else "✈️ Deplasman" if ppg_diff < 0 else "⚖️ Eşit"
        st.metric("📊 PPG Farkı", f"{abs(ppg_diff):.2f}", ppg_status)
    
    with col3:
        gpg_diff = home_stats['gpg_cumulative'] - away_stats['gpg_cumulative']
        gpg_status = "🏠 Ev Sahibi" if gpg_diff > 0 else "✈️ Deplasman" if gpg_diff < 0 else "⚖️ Eşit"
        st.metric("⚽ Gol Farkı", f"{abs(gpg_diff):.2f}", gpg_status)
    
    # Detaylı karşılaştırma
    st.markdown("##### Detaylı Takım İstatistikleri")
    detail_col1, detail_col2, detail_col3 = st.columns(3)
    
    with detail_col1:
        st.markdown(f"**{home_team}**")
        st.metric("Form", f"{home_stats['form_5games']*100:.1f}%")
        st.metric("PPG", f"{home_stats['ppg_cumulative']:.2f}")
        st.metric("Gol/maç", f"{home_stats['gpg_cumulative']:.2f}")
        st.metric("Y. Gol/maç", f"{home_stats['gapg_cumulative']:.2f}")
    
    with detail_col2:
        st.markdown("**Farklar**")
        form_diff_pct = (home_stats['form_5games'] - away_stats['form_5games']) * 100
        st.metric("Form Farkı", f"{form_diff_pct:+.1f}%")
        
        ppg_diff = home_stats['ppg_cumulative'] - away_stats['ppg_cumulative']
        st.metric("PPG Farkı", f"{ppg_diff:+.2f}")
        
        gpg_diff = home_stats['gpg_cumulative'] - away_stats['gpg_cumulative']
        st.metric("Gol Farkı", f"{gpg_diff:+.2f}")
        
        gapg_diff = home_stats['gapg_cumulative'] - away_stats['gapg_cumulative']
        st.metric("Y. Gol Farkı", f"{gapg_diff:+.2f}")
    
    with detail_col3:
        st.markdown(f"**{away_team}**")
        st.metric("Form", f"{away_stats['form_5games']*100:.1f}%")
        st.metric("PPG", f"{away_stats['ppg_cumulative']:.2f}")
        st.metric("Gol/maç", f"{away_stats['gpg_cumulative']:.2f}")
        st.metric("Y. Gol/maç", f"{away_stats['gapg_cumulative']:.2f}")

# ================== KADRO YÖNETİMİ FONKSİYONLARI ==================
def get_sorted_player_options(df_squad):
    """Oyuncu seçeneklerini rating'e göre sırala"""
    available_players = df_squad.copy()
    available_players = available_players.sort_values('PlayerRating', ascending=False)
    
    sorted_indices = available_players.index.tolist()
    display_dict = {}
    
    for idx in sorted_indices:
        player_name = available_players.loc[idx, 'Player']
        player_pos = available_players.loc[idx, 'Pos']
        player_rating = available_players.loc[idx, 'PlayerRating']
        display_dict[idx] = f"⭐ {player_rating:.1f} | {player_pos} | {player_name}"
    
    return sorted_indices, display_dict

def display_team_squad_selection(team_name, squad_df, starters_key, subs_key):
    """Takım kadro seçim arayüzünü göster"""
    st.subheader(f"👥 {team_name} Kadrosu")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**🥅 Başlangıç 11**")
        available_starters_indices, starters_display_dict = get_sorted_player_options(squad_df)
        
        starters = st.multiselect(
            "Başlangıç 11 seçin",
            options=available_starters_indices,
            format_func=lambda x: starters_display_dict[x],
            key=f"{starters_key}_multiselect",
            max_selections=TOP_N_STARTERS
        )
        st.session_state[starters_key] = starters
        
    with col2:
        st.markdown("**🔄 Yedek Oyuncular**")
        available_subs_indices, subs_display_dict = get_sorted_player_options(squad_df)
        
        subs = st.multiselect(
            "Yedek oyuncuları seçin",
            options=available_subs_indices,
            format_func=lambda x: subs_display_dict[x],
            key=f"{subs_key}_multiselect",
            max_selections=TOP_N_SUBS
        )
        st.session_state[subs_key] = subs
    
    # Kadro istatistikleri
    if starters:
        starter_ratings = squad_df.loc[starters, 'PlayerRating'].mean()
        st.metric("🥅 Başlangıç 11 Ortalama Rating", f"{starter_ratings:.1f}")
    
    if subs:
        sub_ratings = squad_df.loc[subs, 'PlayerRating'].mean()
        st.metric("🔄 Yedekler Ortalama Rating", f"{sub_ratings:.1f}")

def auto_fill_squads(home_squad, away_squad):
    """Tüm kadroları otomatik doldur"""
    home_starters_auto = select_topn_by_rating(home_squad, TOP_N_STARTERS)
    home_all_idxs = home_squad['PlayerRating'].dropna().sort_values(ascending=False).index.tolist()
    home_subs_auto = [i for i in home_all_idxs if i not in home_starters_auto][:TOP_N_SUBS]
    
    away_starters_auto = select_topn_by_rating(away_squad, TOP_N_STARTERS)
    away_all_idxs = away_squad['PlayerRating'].dropna().sort_values(ascending=False).index.tolist()
    away_subs_auto = [i for i in away_all_idxs if i not in away_starters_auto][:TOP_N_SUBS]
    
    st.session_state.home_starters = home_starters_auto
    st.session_state.home_subs = home_subs_auto
    st.session_state.away_starters = away_starters_auto
    st.session_state.away_subs = away_subs_auto
    st.session_state.auto_fill_triggered = True

# ================== TAHMİN GÖSTERİM FONKSİYONLARI ==================
def display_prediction_results(pred, probs, home_team, away_team, row):
    """Tahmin sonuçlarını göster"""
    labels = ['Draw', 'HomeWin', 'AwayWin']
    pred_label = labels[int(pred)]
    pred_prob = float(probs[0][int(pred)])

    # Tahmin olasılıkları
    st.success("🎯 TAHMİN SONUCU")
    
    st.subheader("📊 Tahmin Olasılıkları")
    c1, c2, c3 = st.columns(3)
    
    actual_home_prob = probs[0][1] * 100
    actual_draw_prob = probs[0][0] * 100
    actual_away_prob = probs[0][2] * 100
    
    boosted_home_prob = min(100, actual_home_prob * 1.15)
    
    c1.metric("🏠 Ev Sahibi", f"{actual_home_prob:.1f}%", 
             delta=f"+{boosted_home_prob - actual_home_prob:.1f}% (bonus)", delta_color="normal")
    c2.metric("🤝 Beraberlik", f"{actual_draw_prob:.1f}%")
    c3.metric("✈️ Deplasman", f"{actual_away_prob:.1f}%")

    # Model tahmini
    st.subheader("🏆 MODEL TAHMİNİ")
    if pred_label == 'HomeWin':
        st.success(f"**🎯 {home_team} KAZANIR** (Güven: {pred_prob*100:.1f}%)")
        st.balloons()
    elif pred_label == 'AwayWin':
        st.success(f"**🎯 {away_team} KAZANIR** (Güven: {pred_prob*100:.1f}%)")
        st.balloons()
    else:
        st.info(f"**🎯 BERABERLİK** (Güven: {pred_prob*100:.1f}%)")
        st.warning("⚠ BERABERLİK SADECE Ç�OK ÖZEL KOŞULLARDA TAHMİN EDİLİR!")

    # Tahmin kuralları
    st.info("""
    **🔍 TAHMİN KURALLARI:**
    - Ev sahibine %15 bonus uygulanır
    - Beraberlik SADECE:
      * Beraberlik olasılığı > %40
      * Beraberlik diğerlerinden %20 daha yüksek  
      * Takımlar arasında maksimum %5 fark
    - Diğer durumlarda: EN YÜKSEK OLASILIK KAZANAN
    """)

# ================== ANA UYGULAMA ==================
def main():
    initialize_session_state()
    
    st.title("⚽ Bundesliga Tahmin Sistemi - ULTIMATE FIX v15.0")
    
    # Sidebar
    st.sidebar.header("ℹ️ Sistem Bilgisi")
    st.sidebar.info("""
    **🏆 ULTIMATE FIX v15.0:**
    - ✅ MODÜLER ve KULLANICI DOSTU
    - ✅ DETAYLI FORM METRİKLERİ
    - ✅ TAKIM KARŞILAŞTIRMALARI
    - ✅ GELİŞMİŞ KADRO YÖNETİMİ
    - ✅ BASİT ve ETKİLİ TAHMİN
    """)

    st.sidebar.header("📊 Tahmin Kuralları")
    st.sidebar.metric("Ev Sahibi Bonusu", "%15")
    st.sidebar.metric("Beraberlik Eşiği", "%40")
    st.sidebar.metric("Beraberlik Üstünlük", "%20")
    st.sidebar.metric("Takım Yakınlık", "%5")

    try:
        model, features_order, team_dict, df_matches, norm_map = initialize_system()
        df_form = prepare_matches_for_form(df_matches)
    except Exception as e:
        st.error(f"❌ Sistem başlatılırken hata: {str(e)}")
        st.stop()

    # Takım Seçimi Bölümü
    st.header("1️⃣ Takım Seçimi")

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

    home_team = norm_map.get(normalize_name(home_team_display), home_team_display)
    away_team = norm_map.get(normalize_name(away_team_display), away_team_display)

    # Takım Form Gösterimi
    st.subheader("📈 Takım Form Durumu")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### 🏠 {home_team}")
        home_report_header, home_report_lines = last5_report_pretty(df_form, home_team, norm_map, 5)
        if home_report_header:
            st.markdown(home_report_header)
            for line in home_report_lines:
                st.write(line)
        else:
            st.info("⚠ Son maç verisi bulunamadı")
        
        # Form metrikleri
        display_team_form_metrics(df_form, home_team, norm_map)
        
    with col2:
        st.markdown(f"### ✈️ {away_team}")
        away_report_header, away_report_lines = last5_report_pretty(df_form, away_team, norm_map, 5)
        if away_report_header:
            st.markdown(away_report_header)
            for line in away_report_lines:
                st.write(line)
        else:
            st.info("⚠ Son maç verisi bulunamadı")
        
        # Form metrikleri
        display_team_form_metrics(df_form, away_team, norm_map)

    # Karşılaştırma metrikleri
    home_stats, away_stats = predict_calculate_cumulative_stats(df_form, home_team, away_team)
    display_comparison_metrics(home_stats, away_stats, home_team, away_team)

    st.markdown("---")

    # Kadro Seçimi Bölümü
    if st.button("✅ Kadroları Göster & Tahmin Yap", type="primary", use_container_width=True):
        st.session_state.show_squads = True
        st.rerun()

    if st.session_state.show_squads:
        if home_team not in team_dict or away_team not in team_dict:
            st.error("❌ Seçilen takımların kadro verileri bulunamadı!")
            st.stop()
        
        home_squad = team_dict[home_team]
        away_squad = team_dict[away_team]

        st.header("2️⃣ Kadro Seçimi")
        
        # Ev sahibi takım kadrosu
        display_team_squad_selection(home_team, home_squad, "home_starters", "home_subs")
        
        st.markdown("---")
        
        # Deplasman takım kadrosu
        display_team_squad_selection(away_team, away_squad, "away_starters", "away_subs")

        # Otomatik doldurma butonu
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🎯 Tüm Kadroları Otomatik Doldur", type="primary", use_container_width=True):
                auto_fill_squads(home_squad, away_squad)
                st.rerun()

        st.markdown("---")

        # Tahmin butonu
        if st.button("🔮 TAHMİNİ ÇALIŞTIR", type="primary", use_container_width=True):
            with st.spinner("🏃‍♂️ Tahmin yapılıyor..."):
                try:
                    home_starters = st.session_state.home_starters
                    home_subs = st.session_state.home_subs
                    away_starters = st.session_state.away_starters
                    away_subs = st.session_state.away_subs

                    # Eksik kadroları otomatik tamamla
                    if not home_starters or len(home_starters) < TOP_N_STARTERS:
                        home_starters = select_topn_by_rating(home_squad, TOP_N_STARTERS)
                    
                    if not home_subs or len(home_subs) < TOP_N_SUBS:
                        home_all_idxs = home_squad['PlayerRating'].dropna().sort_values(ascending=False).index.tolist()
                        home_subs = [i for i in home_all_idxs if i not in home_starters][:TOP_N_SUBS]
                    
                    if not away_starters or len(away_starters) < TOP_N_STARTERS:
                        away_starters = select_topn_by_rating(away_squad, TOP_N_STARTERS)
                    
                    if not away_subs or len(away_subs) < TOP_N_SUBS:
                        away_all_idxs = away_squad['PlayerRating'].dropna().sort_values(ascending=False).index.tolist()
                        away_subs = [i for i in away_all_idxs if i not in away_starters][:TOP_N_SUBS]

                    # Özellik vektörünü oluştur
                    row = build_simple_feature_row(
                        home_team, away_team,
                        home_squad, away_squad,
                        home_starters, home_subs,
                        away_starters, away_subs, df_form
                    )

                    for feature in features_order:
                        if feature not in row:
                            row[feature] = 0.0

                    feat_row = {f: row.get(f, 0) for f in features_order}
                    X = pd.DataFrame([feat_row])[features_order].copy()
                    X = X.fillna(0)

                    # Tahmin yap
                    pred, probs = simple_predict(model, X)
                    pred = pred[0]
                    
                    # Sonuçları göster
                    display_prediction_results(pred, probs, home_team, away_team, row)
                    
                    # Takım rating karşılaştırması
                    st.subheader("⭐ Takım Rating Karşılaştırması")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        h_rating, _, h_starter_avg, h_sub_avg = compute_team_rating_from_lineup(home_squad, home_starters, home_subs)
                        st.metric("🏠 Takım Rating", f"{safe_float(h_rating, 0):.1f}")
                        st.metric("🥅 Başlangıç Ort.", f"{h_starter_avg:.1f}")
                        st.metric("🔄 Yedek Ort.", f"{h_sub_avg:.1f}")
                    
                    with col2:
                        a_rating, _, a_starter_avg, a_sub_avg = compute_team_rating_from_lineup(away_squad, away_starters, away_subs)
                        st.metric("✈️ Takım Rating", f"{safe_float(a_rating, 0):.1f}")
                        st.metric("🥅 Başlangıç Ort.", f"{a_starter_avg:.1f}")
                        st.metric("🔄 Yedek Ort.", f"{a_sub_avg:.1f}")

                except Exception as e:
                    st.error("❌ Tahmin çalıştırılırken bir hata oluştu.")
                    st.error(f"Hata detayı: {str(e)}")

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: gray; font-size: 14px;'>
        <p>⚽ Bundesliga Tahmin Sistemi - ULTIMATE FIX v15.0 | Modüler & Kullanıcı Dostu</p>
        <p>© 2025 Cansel Yardım | All Rights Reserved</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()