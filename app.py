# app.py
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

# ================== KONFİG ==================
RANDOM_STATE = 42
DATA_PATH = "data/bundesliga_matches_2023_2025_final_fe_team_values_cleaned.xlsx"
PLAYER_DATA_PATH = "data/final_bundesliga_dataset_complete.xlsx"
MODEL_PATH = "models/bundesliga_model_final.pkl"
FEATURE_INFO_PATH = "models/feature_info.pkl"

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

# ================== YARDIMCILAR ==================
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
    """Takım isimlerini normalize et (küçük harf, aksansız, fazlalık boşluk kaldır)."""
    if not isinstance(name, str):
        return ""
    s = name.strip().lower()
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r'[^a-z0-9\s]', '', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def load_player_data(path=PLAYER_DATA_PATH):
    df = pd.read_excel(path)
    # Player Rating türet
    if 'PlayerRating' not in df.columns:
        if 'Rating' in df.columns:
            df['PlayerRating'] = df['Rating']
        elif 'fbref__Goal_Contribution' in df.columns and 'fbref__Min' in df.columns:
            df['PlayerRating'] = df['fbref__Goal_Contribution'] * 2 + df['fbref__Min'].fillna(0) / 90 * 0.5
        else:
            df['PlayerRating'] = 65.0
    # Team
    if 'Team' not in df.columns:
        if 'fbref__Squad' in df.columns:
            df['Team'] = df['fbref__Squad'].astype(str).str.strip()
        else:
            raise RuntimeError("Oyuncu datasında 'Team' veya 'fbref__Squad' bulunamadı.")
    else:
        df['Team'] = df['Team'].astype(str).str.strip()
    # Pos
    if 'Pos' not in df.columns:
        if 'Position' in df.columns:
            df['Pos'] = df['Position'].astype(str)
        elif 'fbref__Pos' in df.columns:
            df['Pos'] = df['fbref__Pos'].astype(str)
        else:
            df['Pos'] = 'MF'
    df['Pos'] = df['Pos'].astype(str).str.upper().str.strip()
    # Name
    if 'Player' not in df.columns:
        for c in ['Name', 'fbref__Player', 'player_name', 'player']:
            if c in df.columns:
                df['Player'] = df[c].astype(str)
                break
        if 'Player' not in df.columns:
            df['Player'] = np.arange(len(df)).astype(str)
    # Yaş (opsiyonel)
    if 'Age' not in df.columns:
        if 'fbref__Age' in df.columns:
            df['Age'] = pd.to_numeric(df['fbref__Age'], errors='coerce')
        else:
            df['Age'] = np.nan
    return df

def team_players_dict(df_players):
    d = {}
    for team in sorted(df_players['Team'].dropna().unique()):
        d[team] = df_players[df_players['Team'] == team].copy().reset_index(drop=True)
    return d

def select_topn_by_rating(df_team, n):
    if 'PlayerRating' not in df_team.columns: return []
    return df_team['PlayerRating'].dropna().sort_values(ascending=False).index.tolist()[:n]

def avg_of_selected_players(df_team, idxs):
    if len(idxs) == 0:
        return np.nan, {'GK': np.nan, 'DF': np.nan, 'MF': np.nan, 'FW': np.nan}
    sel = df_team.loc[idxs]
    ratings = sel['PlayerRating'].dropna()
    overall = ratings.mean() if not ratings.empty else np.nan
    pos_means = {}
    for pos in ['GK', 'DF', 'MF', 'FW']:
        mask = sel['Pos'].apply(pos_group) == pos
        vals = sel.loc[mask, 'PlayerRating'].dropna()
        pos_means[pos] = vals.mean() if not vals.empty else np.nan
    return overall, pos_means

def compute_team_rating_from_lineup(df_team, starter_idxs, sub_idxs,
                                    starter_weight=STARTER_WEIGHT, sub_weight=SUB_WEIGHT):
    starter_mean, starter_pos = avg_of_selected_players(df_team, starter_idxs)
    sub_mean, sub_pos = avg_of_selected_players(df_team, sub_idxs)
    if np.isnan(starter_mean) and not np.isnan(sub_mean): team_rating = sub_mean
    elif np.isnan(sub_mean) and not np.isnan(starter_mean): team_rating = starter_mean
    elif np.isnan(starter_mean) and np.isnan(sub_mean): team_rating = np.nan
    else: team_rating = (starter_mean * starter_weight) + (sub_mean * sub_weight)
    pos_combined = {}
    for p in ['GK', 'DF', 'MF', 'FW']:
        s = starter_pos.get(p, np.nan)
        b = sub_pos.get(p, np.nan)
        if pd.isna(s) and not pd.isna(b): pos_combined[p] = b
        elif pd.isna(b) and not pd.isna(s): pos_combined[p] = s
        elif pd.isna(s) and pd.isna(b): pos_combined[p] = np.nan
        else: pos_combined[p] = (s * starter_weight) + (b * sub_weight)
    return team_rating, pos_combined, starter_mean, sub_mean

def prepare_matches_for_form(df_matches):
    df = df_matches.copy()
    if 'HomeTeam' not in df.columns and 'homeTeam.name' in df.columns:
        df['HomeTeam'] = df['homeTeam.name']
    if 'AwayTeam' not in df.columns and 'awayTeam.name' in df.columns:
        df['AwayTeam'] = df['awayTeam.name']
    if 'Date' not in df.columns:
        if 'utcDate' in df.columns:
            df['Date'] = pd.to_datetime(df['utcDate'])
        else:
            if 'date' in df.columns:
                df['Date'] = pd.to_datetime(df['date'])
            else:
                df['Date'] = pd.to_datetime('today')
    df = df.sort_values('Date').reset_index(drop=True)
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
    norm = normalize_name(team)
    team_matches = df_form[(df_form['_HomeNorm'] == norm) | (df_form['_AwayNorm'] == norm)].copy()
    team_matches = team_matches.sort_values('Date').reset_index(drop=True)
    if len(team_matches) == 0:
        return {
            'form': 0.5, 'gs_5': 0, 'gc_5': 0, 'momentum': 0
        }
    last_5 = team_matches.tail(5)
    points = 0
    gs, gc = 0, 0
    for _, m in last_5.iterrows():
        hg = safe_float(m.get('score.fullTime.home', 0), 0)
        ag = safe_float(m.get('score.fullTime.away', 0), 0)
        if normalize_name(m.get('HomeTeam', '')) == norm:
            gs += hg; gc += ag
            if hg > ag: points += 3
            elif hg == ag: points += 1
        else:
            gs += ag; gc += hg
            if ag > hg: points += 3
            elif ag == hg: points += 1
    form = (points / 15.0) if points > 0 else 0.3
    momentum = gs - gc
    return {
        'form': form, 'gs_5': int(gs), 'gc_5': int(gc), 'momentum': int(momentum)
    }

def derby_flag(home, away):
    big_teams = {'Bayern Munich', 'Borussia Dortmund', 'Schalke 04', 'Hamburg SV',
                 'Borussia Mönchengladbach', 'Bayer Leverkusen', 'VfB Stuttgart'}
    return 1 if (home in big_teams and away in big_teams) else 0

def maybe_team_value_features(df_players, team):
    sub = df_players[df_players['Team'] == team].copy()
    feats = {}
    value_cols = [c for c in sub.columns if re.search(r'value|market|eur', c, re.I)]
    age_cols = [c for c in sub.columns if re.search(r'age', c, re.I)]
    if age_cols:
        ages = pd.to_numeric(sub[age_cols[0]], errors='coerce')
        if ages.notna().sum() >= 3:
            feats['squad_avg_age'] = float(ages.mean())
    if value_cols:
        vals = pd.to_numeric(sub[value_cols[0]], errors='coerce')
        if vals.notna().sum() >= 3:
            feats['current_value_eur'] = float(vals.sum())
            chg_cols = [c for c in sub.columns if re.search(r'change|pct|delta', c, re.I)]
            if chg_cols:
                chg = pd.to_numeric(sub[chg_cols[0]], errors='coerce')
                if chg.notna().sum() >= 3:
                    feats['value_change_pct'] = float(chg.mean())
    return feats

def build_feature_row(
    home_team, away_team,
    df_home, df_away,
    home_start_ids, home_sub_ids,
    away_start_ids, away_sub_ids,
    df_matches_form, df_players
):
    h_team_rating, h_pos, h11, hbench = compute_team_rating_from_lineup(df_home, home_start_ids, home_sub_ids)
    a_team_rating, a_pos, a11, abench = compute_team_rating_from_lineup(df_away, away_start_ids, away_sub_ids)

    home_form = compute_team_form_snapshot(df_matches_form, home_team)
    away_form = compute_team_form_snapshot(df_matches_form, away_team)

    hv_feats = maybe_team_value_features(df_players, home_team) or {}
    av_feats = maybe_team_value_features(df_players, away_team) or {}

    row = {
        'Home_AvgRating': safe_float(h_team_rating, 65.0),
        'Away_AvgRating': safe_float(a_team_rating, 65.0),
        'Rating_Diff': safe_float(h_team_rating - a_team_rating, 0.0),
        'Total_AvgRating': safe_float(h_team_rating + a_team_rating, 130.0),

        'home_form': safe_float(home_form['form'], 0.5),
        'away_form': safe_float(away_form['form'], 0.5),
        'Form_Diff': safe_float(home_form['form'] - away_form['form'], 0.0),

        'homeTeam_GoalsScored_5': int(home_form['gs_5']),
        'homeTeam_GoalsConceded_5': int(home_form['gc_5']),
        'awayTeam_GoalsScored_5': int(away_form['gs_5']),
        'awayTeam_GoalsConceded_5': int(away_form['gc_5']),

        'homeTeam_Momentum': int(home_form['momentum']),
        'awayTeam_Momentum': int(away_form['momentum']),

        'Home_GK_Rating': safe_float(h_pos.get('GK'), 65.0),
        'Home_DF_Rating': safe_float(h_pos.get('DF'), 65.0),
        'Home_MF_Rating': safe_float(h_pos.get('MF'), 65.0),
        'Home_FW_Rating': safe_float(h_pos.get('FW'), 65.0),

        'Away_GK_Rating': safe_float(a_pos.get('GK'), 65.0),
        'Away_DF_Rating': safe_float(a_pos.get('DF'), 65.0),
        'Away_MF_Rating': safe_float(a_pos.get('MF'), 65.0),
        'Away_FW_Rating': safe_float(a_pos.get('FW'), 65.0),

        'IsDerby': int(derby_flag(home_team, away_team)),

        'home_current_value_eur': safe_float(hv_feats.get('current_value_eur', 0.0), 0.0),
        'away_current_value_eur': safe_float(av_feats.get('current_value_eur', 0.0), 0.0),
        'home_squad_avg_age': safe_float(hv_feats.get('squad_avg_age', 0.0), 0.0),
        'away_squad_avg_age': safe_float(av_feats.get('squad_avg_age', 0.0), 0.0),
        'home_value_change_pct': safe_float(hv_feats.get('value_change_pct', 0.0), 0.0),
        'away_value_change_pct': safe_float(av_feats.get('value_change_pct', 0.0), 0.0),

        'Home_BenchRating': safe_float(hbench, np.nan),
        'Away_BenchRating': safe_float(abench, np.nan),
        '_HomeXI': safe_float(h11, np.nan),
        '_AwayXI': safe_float(a11, np.nan),
    }
    return row

def build_normalized_team_map(team_dict):
    norm_map = {}
    for orig in team_dict.keys():
        n = normalize_name(orig)
        if n:
            norm_map[n] = orig
    return norm_map

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
st.set_page_config(page_title="Bundesliga Predictor", layout="wide")
st.title("⚽ Bundesliga Tahmin Sistemi")

@st.cache_resource
def load_data():
    try:
        model = joblib.load(MODEL_PATH)
        feat_info = joblib.load(FEATURE_INFO_PATH)
        
        if isinstance(feat_info, dict) and 'important_features' in feat_info and feat_info['important_features']:
            features_order = feat_info['important_features']
        else:
            features_order = SELECTED_FEATURES[:]

        df_players = load_player_data(PLAYER_DATA_PATH)
        team_dict = team_players_dict(df_players)

        df_matches = pd.read_excel(DATA_PATH)
        df_form = prepare_matches_for_form(df_matches)
        df_form['HomeTeam'] = df_form['HomeTeam'].astype(str)
        df_form['AwayTeam'] = df_form['AwayTeam'].astype(str)
        df_form['_HomeNorm'] = df_form['HomeTeam'].apply(normalize_name)
        df_form['_AwayNorm'] = df_form['AwayTeam'].apply(normalize_name)

        norm_map = build_normalized_team_map(team_dict)
        
        return model, features_order, team_dict, df_form, norm_map
    except Exception as e:
        st.error(f"Veri yüklenirken hata oluştu: {str(e)}")
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

# ---------- STEP 1: Takım seçimi ----------
# Dropdown’da orijinal isimleri gösterelim
teams_display = list(norm_map.values())

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

# Arka planda normalize edilmiş karşılıklarını kullan
home_team = norm_map[normalize_name(home_team_display)]
away_team = norm_map[normalize_name(away_team_display)]


if st.button("✅ Kadroları Göster"):
    st.session_state.show_squads = True
    st.session_state.home_starters = []
    st.session_state.home_subs = []
    st.session_state.away_starters = []
    st.session_state.away_subs = []

st.markdown("---")

# ---------- STEP 2: Kadro & Tahmin ----------
if st.session_state.show_squads:
    home_squad = team_dict[home_team]
    away_squad = team_dict[away_team]

    st.header("2️⃣ Kadro Seçimi")
    
    # Ev sahibi takım kadrosu
    st.subheader(f"👥 {home_team} Kadrosu")
    home_df_display = home_squad[['Player', 'Pos', 'PlayerRating']].copy()
    home_df_display.columns = ['İsim', 'Pozisyon', 'Rating']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Başlangıç 11**")
        home_starters = st.multiselect(
            "Başlangıç 11 (ev sahibi)",
            options=list(home_squad.index),
            format_func=lambda x: f"{home_squad.loc[x, 'Player']} - {home_squad.loc[x, 'Pos']} ({home_squad.loc[x, 'PlayerRating']:.1f})",
            key="home_starters",
            default=st.session_state.home_starters
        )
    
    with col2:
        st.markdown("**Yedek Oyuncular (max 7)**")
        home_subs = st.multiselect(
            "Yedek Oyuncular (ev sahibi)",
            options=list(home_squad.index),
            format_func=lambda x: f"{home_squad.loc[x, 'Player']} - {home_squad.loc[x, 'Pos']} ({home_squad.loc[x, 'PlayerRating']:.1f})",
            key="home_subs",
            default=st.session_state.home_subs
        )

    # Deplasman takım kadrosu
    st.subheader(f"👥 {away_team} Kadrosu")
    away_df_display = away_squad[['Player', 'Pos', 'PlayerRating']].copy()
    away_df_display.columns = ['İsim', 'Pozisyon', 'Rating']
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Başlangıç 11**")
        away_starters = st.multiselect(
            "Başlangıç 11 (deplasman)",
            options=list(away_squad.index),
            format_func=lambda x: f"{away_squad.loc[x, 'Player']} - {away_squad.loc[x, 'Pos']} ({away_squad.loc[x, 'PlayerRating']:.1f})",
            key="away_starters",
            default=st.session_state.away_starters
        )
    
    with col2:
        st.markdown("**Yedek Oyuncular (max 7)**")
        away_subs = st.multiselect(
            "Yedek Oyuncular (deplasman)",
            options=list(away_squad.index),
            format_func=lambda x: f"{away_squad.loc[x, 'Player']} - {away_squad.loc[x, 'Pos']} ({away_squad.loc[x, 'PlayerRating']:.1f})",
            key="away_subs",
            default=st.session_state.away_subs
        )

    st.markdown("---")

    if st.button("🔮 Tahmin Yap"):
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

            # Model için hazırla
            feat_row = {f: row.get(f, 0) for f in features_order}
            X = pd.DataFrame([feat_row])[features_order].copy()

            # Tahmin yap
            pred = model.predict(X)[0]
            probs = model.predict_proba(X)[0]
            labels = ['Draw', 'HomeWin', 'AwayWin']
            pred_label = labels[int(pred)]
            pred_prob = float(probs[int(pred)]) if 0 <= int(pred) < len(probs) else np.nan

            # Sonuçları göster
            st.success(f"🔮 Tahmin: {home_team} vs {away_team}")
            
            # Olasılık metrikleri
            c1, c2, c3 = st.columns(3)
            c1.metric("🏠 Ev Sahibi Kazanır", f"{probs[1]*100:.1f}%")
            c2.metric("🤝 Beraberlik", f"{probs[0]*100:.1f}%")
            c3.metric("✈️ Deplasman Kazanır", f"{probs[2]*100:.1f}%")

            # Takım istatistikleri
            st.subheader("📊 Takım İstatistikleri")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**{home_team}**")
                st.write(f"⭐ Rating: {row['Home_AvgRating']:.2f}")
                st.write(f"📈 Form: {row['home_form']*100:.1f}%")
                st.write(f"⚽ Momentum: {row['homeTeam_Momentum']}")
                if row.get('home_current_value_eur', 0) > 0:
                    st.write(f"💰 Değer: {row['home_current_value_eur']:.0f} EUR")
            
            with col2:
                st.write(f"**{away_team}**")
                st.write(f"⭐ Rating: {row['Away_AvgRating']:.2f}")
                st.write(f"📈 Form: {row['away_form']*100:.1f}%")
                st.write(f"⚽ Momentum: {row['awayTeam_Momentum']}")
                if row.get('away_current_value_eur', 0) > 0:
                    st.write(f"💰 Değer: {row['away_current_value_eur']:.0f} EUR")

            # Son 5 maç form durumu
            st.subheader("📊 Takım Form Durumları")
            
            home_report = last5_report_pretty(df_form, home_team, norm_map, max_lines=5)
            if home_report is None:
                st.write(f"**{home_team} Son 5 Maç:**")
                st.write("⚠ Veri bulunamadı")
            else:
                st.write(f"**{home_team} Son 5 Maç:**")
                st.write(home_report)

            away_report = last5_report_pretty(df_form, away_team, norm_map, max_lines=5)
            if away_report is None:
                st.write(f"**{away_team} Son 5 Maç:**")
                st.write("⚠ Veri bulunamadı")
            else:
                st.write(f"**{away_team} Son 5 Maç:**")
                st.write(away_report)

            # Seçili oyuncular
            st.subheader("📋 Seçili Oyuncular")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write(f"**{home_team} - Başlangıç 11:**")
                home_starters_players = home_squad.loc[home_starters, ["Player", "Pos", "PlayerRating"]].copy()
                home_starters_players.columns = ["İsim", "Pozisyon", "Rating"]
                home_starters_players["Rating"] = home_starters_players["Rating"].round(1)
                st.dataframe(home_starters_players.reset_index(drop=True))
                
                st.write(f"**{home_team} - Yedek Oyuncular:**")
                home_subs_players = home_squad.loc[home_subs, ["Player", "Pos", "PlayerRating"]].copy()
                home_subs_players.columns = ["İsim", "Pozisyon", "Rating"]
                home_subs_players["Rating"] = home_subs_players["Rating"].round(1)
                st.dataframe(home_subs_players.reset_index(drop=True))
            
            with col2:
                st.write(f"**{away_team} - Başlangıç 11:**")
                away_starters_players = away_squad.loc[away_starters, ["Player", "Pos", "PlayerRating"]].copy()
                away_starters_players.columns = ["İsim", "Pozisyon", "Rating"]
                away_starters_players["Rating"] = away_starters_players["Rating"].round(1)
                st.dataframe(away_starters_players.reset_index(drop=True))
                
                st.write(f"**{away_team} - Yedek Oyuncular:**")
                away_subs_players = away_squad.loc[away_subs, ["Player", "Pos", "PlayerRating"]].copy()
                away_subs_players.columns = ["İsim", "Pozisyon", "Rating"]
                away_subs_players["Rating"] = away_subs_players["Rating"].round(1)
                st.dataframe(away_subs_players.reset_index(drop=True))

        except Exception as e:
            st.error("Tahmin çalıştırılırken bir hata oluştu.")
            st.error(f"Hata: {str(e)}")
            st.text(traceback.format_exc())