# bundesliga_predictor.py
# Tek dosyalık, interaktif Bundesliga maç tahmin motoru
# Model: models/bundesliga_model_final.pkl
# Özellik bilgisi: models/feature_info.pkl
# Veri: data/bundesliga_matches_2023_2025_final_fe_team_values_cleaned.xlsx
#       data/final_bundesliga_dataset_complete.xlsx

import os
import re
import sys
import joblib
import warnings
warnings.filterwarnings("ignore")

import difflib
import unicodedata

import numpy as np
import pandas as pd
from datetime import datetime

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

# Eğitimde kullandığın feature listesi (yine de feature_info'dan okumaya çalışacağız)
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
    s = re.sub(r'[^a-z0-9\s]', '', s)  # noktalama/özel karakter temizle
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
        # olası isim kolonları
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

# --------- Form ve momentum hesapları (son 5 maç) ---------
def prepare_matches_for_form(df_matches):
    # kolon isimlerini normalize et
    df = df_matches.copy()
    if 'HomeTeam' not in df.columns and 'homeTeam.name' in df.columns:
        df['HomeTeam'] = df['homeTeam.name']
    if 'AwayTeam' not in df.columns and 'awayTeam.name' in df.columns:
        df['AwayTeam'] = df['awayTeam.name']
    # tarih
    if 'Date' not in df.columns:
        if 'utcDate' in df.columns:
            df['Date'] = pd.to_datetime(df['utcDate'])
        else:
            # fallback: varsa 'date'
            if 'date' in df.columns:
                df['Date'] = pd.to_datetime(df['date'])
            else:
                df['Date'] = pd.to_datetime('today')
    df = df.sort_values('Date').reset_index(drop=True)
    # skor kolonları
    if 'score.fullTime.home' not in df.columns or 'score.fullTime.away' not in df.columns:
        # varsa alternatif skor kolonlarını dene
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
    """Takımın en son an itibarıyla 5 maçlık form, golleri ve momentumunu döndürür."""
    # df_form içinde normalizasyonlu kolonlar oluşturulmuş olmalı (main() içinde yapıyoruz)
    norm = normalize_name(team)
    team_matches = df_form[(df_form['_HomeNorm'] == norm) | (df_form['_AwayNorm'] == norm)].copy()
    team_matches = team_matches.sort_values('Date').reset_index(drop=True)
    if len(team_matches) == 0:
        return {
            'form': 0.5, 'gs_5': 0, 'gc_5': 0, 'momentum': 0
        }
    # son 5
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
    """Eğer veri varsa değer/yaş bilgilerini hesaplar; yoksa None döner."""
    sub = df_players[df_players['Team'] == team].copy()
    feats = {}
    # takım toplam değeri / değişim yüzdesi benzeri kolonlar varsa kullan
    # potansiyel kolon isimleri:
    value_cols = [c for c in sub.columns if re.search(r'value|market|eur', c, re.I)]
    age_cols = [c for c in sub.columns if re.search(r'age', c, re.I)]
    # ortalama yaş
    if age_cols:
        ages = pd.to_numeric(sub[age_cols[0]], errors='coerce')
        if ages.notna().sum() >= 3:
            feats['squad_avg_age'] = float(ages.mean())
    # toplam/ortalama değer
    if value_cols:
        vals = pd.to_numeric(sub[value_cols[0]], errors='coerce')
        if vals.notna().sum() >= 3:
            feats['current_value_eur'] = float(vals.sum())
            # değişim yüzdesi için genelde veri yok; varsa almaya çalış
            chg_cols = [c for c in sub.columns if re.search(r'change|pct|delta', c, re.I)]
            if chg_cols:
                chg = pd.to_numeric(sub[chg_cols[0]], errors='coerce')
                if chg.notna().sum() >= 3:
                    feats['value_change_pct'] = float(chg.mean())
    return feats

# ================== ANA AKIŞ ==================
def print_team_list(team_dict):
    teams = list(team_dict.keys())
    print("\n🏆 Mevcut Takımlar:")
    print("==================================================")
    for i, t in enumerate(teams, 1):
        print(f"{i:2d}. {t}")
    return teams

def show_squad(team_name, df_team):
    print(f"\n👥 {team_name} Kadrosu:")
    print("============================================================")
    print("ID   İsim                          Pozisyon        Rating")
    print("------------------------------------------------------------")
    for i, row in df_team.reset_index().rename(columns={'index':'_idx'}).iterrows():
        player = str(row.get('Player', 'N/A'))[:26]
        pos = str(row.get('Pos', 'MF'))[:16]
        rating = safe_float(row.get('PlayerRating', 65.0), 65.0)
        print(f"{row['_idx']: <4d} {player: <28} {pos: <15} {rating:.1f}")

def parse_id_input(s):
    s = s.strip()
    if not s:
        return []
    ids = []
    for part in s.split(","):
        part = part.strip()
        if not part: continue
        if "-" in part:
            a,b = part.split("-",1)
            try:
                a = int(a); b = int(b)
                ids.extend(list(range(min(a,b), max(a,b)+1)))
            except:
                continue
        else:
            try:
                ids.append(int(part))
            except:
                continue
    return sorted(list(set(ids)))

def build_feature_row(
    home_team, away_team,
    df_home, df_away,
    home_start_ids, home_sub_ids,
    away_start_ids, away_sub_ids,
    df_matches_form, df_players
):
    # Ratingler
    h_team_rating, h_pos, h11, hbench = compute_team_rating_from_lineup(df_home, home_start_ids, home_sub_ids)
    a_team_rating, a_pos, a11, abench = compute_team_rating_from_lineup(df_away, away_start_ids, away_sub_ids)

    # Form & momentum
    home_form = compute_team_form_snapshot(df_matches_form, home_team)
    away_form = compute_team_form_snapshot(df_matches_form, away_team)

    # Takım değer/yaş (varsa)
    hv_feats = maybe_team_value_features(df_players, home_team) or {}
    av_feats = maybe_team_value_features(df_players, away_team) or {}

    # Feature set
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

        # Takım değeri / yaş (varsa, yoksa 0; 0 olanları LOG'DA YAZDIRMAYACAĞIZ)
        'home_current_value_eur': safe_float(hv_feats.get('current_value_eur', 0.0), 0.0),
        'away_current_value_eur': safe_float(av_feats.get('current_value_eur', 0.0), 0.0),
        'home_squad_avg_age': safe_float(hv_feats.get('squad_avg_age', 0.0), 0.0),
        'away_squad_avg_age': safe_float(av_feats.get('squad_avg_age', 0.0), 0.0),
        'home_value_change_pct': safe_float(hv_feats.get('value_change_pct', 0.0), 0.0),
        'away_value_change_pct': safe_float(av_feats.get('value_change_pct', 0.0), 0.0),

        # Log için
        'Home_BenchRating': safe_float(hbench, np.nan),
        'Away_BenchRating': safe_float(abench, np.nan),
        '_HomeXI': safe_float(h11, np.nan),
        '_AwayXI': safe_float(a11, np.nan),
    }
    return row

def topk_importances(model, feat_names, k=5):
    try:
        importances = model.named_steps['lgbm'].feature_importances_
        pairs = list(zip(feat_names, importances))
        pairs.sort(key=lambda x: x[1], reverse=True)
        return pairs[:k]
    except Exception:
        return []

def save_history_excel(payload, path="data/prediction_history.xlsx"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    stamp = datetime.now().strftime("Pred_%Y-%m-%d_%H-%M-%S")
    df = pd.DataFrame([payload])
    mode = 'a' if os.path.exists(path) else 'w'
    with pd.ExcelWriter(path, engine='openpyxl', mode=mode) as writer:
        df.to_excel(writer, index=False, sheet_name=stamp)
    print(f"✅ Tahmin '{stamp}' sayfasına kaydedildi: {path}")

# --------- Fuzzy name matching + match/get_last_matches helpers ----------
def build_normalized_team_map(team_dict):
    """Verilen team_dict (original_name -> df) için normalized_name -> original_name map oluştur."""
    norm_map = {}
    for orig in team_dict.keys():
        n = normalize_name(orig)
        if n:
            norm_map[n] = orig
    return norm_map

def match_team_name(candidate: str, norm_map: dict, cutoff=0.55):
    """
    Kullanıcı girişi ya da serbest bir takım ismini dataset'teki en yakın orijinal isme eşleştirir.
    Eşleşme yoksa None döner.
    """
    if not candidate:
        return None
    q = normalize_name(candidate)
    if not q:
        return None
    # doğrudan eşleşme
    if q in norm_map:
        return norm_map[q]
    # fuzzy
    keys = list(norm_map.keys())
    matches = difflib.get_close_matches(q, keys, n=1, cutoff=cutoff)
    if matches:
        return norm_map[matches[0]]
    return None

def get_last_matches_for_team(df_form, team_candidate, norm_map, n=5):
    """
    team_candidate: kullanıcı girişi veya orijinal isim
    norm_map: normalized_name -> original_name
    dönen: pandas.DataFrame (son n maç), veya boş DataFrame
    """
    matched = match_team_name(team_candidate, norm_map)
    if not matched:
        return pd.DataFrame()  # boş df -> kolay kontrol
    norm = normalize_name(matched)
    if '_HomeNorm' not in df_form.columns or '_AwayNorm' not in df_form.columns:
        # fallback: normalize on the fly
        df_form = df_form.copy()
        df_form['_HomeNorm'] = df_form['HomeTeam'].astype(str).apply(normalize_name)
        df_form['_AwayNorm'] = df_form['AwayTeam'].astype(str).apply(normalize_name)
    team_matches = df_form[(df_form['_HomeNorm'] == norm) | (df_form['_AwayNorm'] == norm)].copy()
    team_matches = team_matches.sort_values('Date').tail(n)
    return team_matches.reset_index(drop=True)

# --------- last5 pretty report (daha güvenli) ----------
def last5_report_pretty(df_form, team_candidate, norm_map, max_lines=5):
    """
    Daha okunaklı 'son 5 maç' raporu. Eğer veri yoksa boş dizi / uyarı verir.
    """
    tm = get_last_matches_for_team(df_form, team_candidate, norm_map, n=5)
    if tm.empty:
        return None  # veri yok
    # ters sırada: en yeni önce
    tm = tm.sort_values('Date', ascending=False).reset_index(drop=True)
    wins, draws, losses = 0, 0, 0
    lines = []
    for i, m in tm.iterrows():
        d = pd.to_datetime(m['Date']).strftime("%d.%m.%Y")
        hg = int(safe_float(m.get('score.fullTime.home', 0), 0))
        ag = int(safe_float(m.get('score.fullTime.away', 0), 0))
        home = str(m.get('HomeTeam', ''))
        away = str(m.get('AwayTeam', ''))
        # hedef takım nerede?
        norm_target = normalize_name(match_team_name(team_candidate, norm_map) or team_candidate)
        is_home = (normalize_name(home) == norm_target)
        opponent = away if is_home else home
        # sonuç
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

# ================== MAIN ==================
def main():
    # ----------------- Yüklemeler -----------------
    if not os.path.exists(MODEL_PATH):
        print(f"❌ Model bulunamadı: {MODEL_PATH}")
        sys.exit(1)
    if not os.path.exists(FEATURE_INFO_PATH):
        print(f"❌ Feature info bulunamadı: {FEATURE_INFO_PATH}")
        sys.exit(1)
    if not os.path.exists(DATA_PATH):
        print(f"❌ Maç verisi bulunamadı: {DATA_PATH}")
        sys.exit(1)
    if not os.path.exists(PLAYER_DATA_PATH):
        print(f"❌ Oyuncu verisi bulunamadı: {PLAYER_DATA_PATH}")
        sys.exit(1)

    model = joblib.load(MODEL_PATH)
    feat_info = joblib.load(FEATURE_INFO_PATH)

    # Modelin beklediği feature sırası
    if isinstance(feat_info, dict) and 'important_features' in feat_info and feat_info['important_features']:
        features_order = feat_info['important_features']
    else:
        features_order = SELECTED_FEATURES[:]

    # Verileri yükle
    df_players = load_player_data(PLAYER_DATA_PATH)
    team_dict = team_players_dict(df_players)

    df_matches = pd.read_excel(DATA_PATH)
    df_form = prepare_matches_for_form(df_matches)

    # df_form için normalizasyon kolonları ekle (böylece filter'larda rahat kullanırız)
    df_form = df_form.copy()
    df_form['HomeTeam'] = df_form['HomeTeam'].astype(str)
    df_form['AwayTeam'] = df_form['AwayTeam'].astype(str)
    df_form['_HomeNorm'] = df_form['HomeTeam'].apply(normalize_name)
    df_form['_AwayNorm'] = df_form['AwayTeam'].apply(normalize_name)

    # normalized map (normalize_name -> original_name)
    norm_map = build_normalized_team_map(team_dict)

    # ----------------- Takım seçimleri -----------------
    teams = print_team_list(team_dict)
    try:
        home_idx = int(input("\n🏠 Ev sahibi takım numarasını girin: ").strip())
        home_team = teams[home_idx - 1]
    except Exception:
        print("❌ Geçersiz seçim."); sys.exit(1)
    print(f"✅ Ev sahibi: {home_team}")

    try:
        away_idx = int(input("✈️ Deplasman takımı numarasını girin: ").strip())
        away_team = teams[away_idx - 1]
    except Exception:
        print("❌ Geçersiz seçim."); sys.exit(1)
    print(f"✅ Deplasman: {away_team}")

    if home_team == away_team:
        print("❌ Aynı takımlar seçilemez."); sys.exit(1)

    # ----------------- Kadroları göster / seç -----------------
    df_home = team_dict[home_team].copy().reset_index(drop=True)
    df_away = team_dict[away_team].copy().reset_index(drop=True)

    show_squad(home_team, df_home)
    inp = input("\n🔢 Başlangıç oyuncularını seçin (ID'leri virgülle, aralık için 3-7):\nSeçimleriniz: ").strip()
    home_start_ids = parse_id_input(inp)
    if not home_start_ids:
        print("⚠️ Hiç oyuncu seçilmedi, en iyi oyuncular otomatik seçilecek")
        home_start_ids = select_topn_by_rating(df_home, TOP_N_STARTERS)
    inp = input("\n🔢 Yedek oyuncularını seçin (ID'leri virgülle):\nSeçimleriniz: ").strip()
    home_sub_ids = parse_id_input(inp)
    if not home_sub_ids:
        print("⚠️ Hiç oyuncu seçilmedi, en iyi oyuncular otomatik seçilecek")
        all_idxs = df_home['PlayerRating'].dropna().sort_values(ascending=False).index.tolist()
        home_sub_ids = [i for i in all_idxs if i not in home_start_ids][:TOP_N_SUBS]

    show_squad(away_team, df_away)
    inp = input("\n🔢 Başlangıç oyuncularını seçin (ID'leri virgülle, aralık için 3-7):\nSeçimleriniz: ").strip()
    away_start_ids = parse_id_input(inp)
    if not away_start_ids:
        print("⚠️ Hiç oyuncu seçilmedi, en iyi oyuncular otomatik seçilecek")
        away_start_ids = select_topn_by_rating(df_away, TOP_N_STARTERS)
    inp = input("\n🔢 Yedek oyuncularını seçin (ID'leri virgülle):\nSeçimleriniz: ").strip()
    away_sub_ids = parse_id_input(inp)
    if not away_sub_ids:
        print("⚠️ Hiç oyuncu seçilmedi, en iyi oyuncular otomatik seçilecek")
        all_idxs = df_away['PlayerRating'].dropna().sort_values(ascending=False).index.tolist()
        away_sub_ids = [i for i in all_idxs if i not in away_start_ids][:TOP_N_SUBS]

    print("ℹ️ Ev sahibi kadrosu en iyi oyuncularla otomatik dolduruldu" if len(home_start_ids) < TOP_N_STARTERS else "")
    print("ℹ️ Deplasman kadrosu en iyi oyuncularla otomatik dolduruldu" if len(away_start_ids) < TOP_N_STARTERS else "")

    # ----------------- Feature oluştur -----------------
    print(f"\n🔮 Tahmin yapılıyor: {home_team} vs {away_team}")

    row = build_feature_row(
        home_team, away_team,
        df_home, df_away,
        home_start_ids, home_sub_ids,
        away_start_ids, away_sub_ids,
        df_form, df_players
    )

    # LOG — Ratingler
    if not np.isnan(row.get('_HomeXI', np.nan)) and not np.isnan(row.get('Home_BenchRating', np.nan)):
        print(f"⭐ Ev Sahibi Rating (11): {row['_HomeXI']:.2f}, Yedek Ortalaması: {row['Home_BenchRating']:.2f}")
    if not np.isnan(row.get('_AwayXI', np.nan)) and not np.isnan(row.get('Away_BenchRating', np.nan)):
        print(f"⭐ Deplasman Rating (11): {row['_AwayXI']:.2f}, Yedek Ortalaması: {row['Away_BenchRating']:.2f}")

    # Modelin beklediği sırada tek satırlık DF
    # Eksik feature varsa 0 ile doldur (eğitimde de böyleydi)
    feat_row = {f: row.get(f, 0) for f in features_order}
    X = pd.DataFrame([feat_row])[features_order].copy()

    # ----------------- Tahmin -----------------
    pred = model.predict(X)[0]
    probs = model.predict_proba(X)[0]
    labels = ['Draw', 'HomeWin', 'AwayWin']
    pred_label = labels[int(pred)]
    pred_prob = float(probs[int(pred)]) if 0 <= int(pred) < len(probs) else np.nan

    # ----------------- ÇIKTI -----------------
    print("\n" + "="*60)
    print("🎯 TAHMİN SONUÇLARI")
    print("="*60 + "\n")
    print("📊 Sonuç Olasılıkları (%):")
    print(f"   • Beraberlik: {probs[0]*100:.1f}%")
    print(f"   • Ev Sahibi Kazanır: {probs[1]*100:.1f}%")
    print(f"   • Deplasman Kazanır: {probs[2]*100:.1f}%")
    print(f"   🔮 Tahmin: {pred_label} ({pred_prob*100:.1f}% güven)")

    # Feature importances (top-5)
    tops = topk_importances(model, features_order, k=5)
    if tops:
        print("\n📈 En Önemli Özellikler:")
        for i,(n,v) in enumerate(tops,1):
            print(f"   {i}. {n}: {v:.4f}")

    # Takım istatistikleri (0 olanları yazma)
    print("\n👥 Takım İstatistikleri:")
    print(f"   ⭐ Ev Sahibi Rating: {row['Home_AvgRating']:.2f}")
    print(f"   ⭐ Deplasman Rating: {row['Away_AvgRating']:.2f}")
    print(f"   📈 Ev Sahibi Formu: {row['home_form']*100:.1f}%")
    print(f"   📈 Deplasman Formu: {row['away_form']*100:.1f}%")
    print(f"   ⚽ Ev Sahibi Momentum: {row['homeTeam_Momentum']}")
    print(f"   ⚽ Deplasman Momentum: {row['awayTeam_Momentum']}")

    # 0 olan finans/yaş gibi alanları loglama — sadece anlamlıysa yaz
    extra_logs = []
    if row.get('home_current_value_eur', 0) > 0:
        extra_logs.append(f"   💰 Ev Değeri (EUR): {row['home_current_value_eur']:.0f}")
    if row.get('away_current_value_eur', 0) > 0:
        extra_logs.append(f"   💰 Dep Değeri (EUR): {row['away_current_value_eur']:.0f}")
    if row.get('home_squad_avg_age', 0) > 0:
        extra_logs.append(f"   👶 Ev Yaş Ort.: {row['home_squad_avg_age']:.2f}")
    if row.get('away_squad_avg_age', 0) > 0:
        extra_logs.append(f"   👶 Dep Yaş Ort.: {row['away_squad_avg_age']:.2f}")
    if row.get('home_value_change_pct', 0) != 0:
        extra_logs.append(f"   📉 Ev Değer Değişim %: {row['home_value_change_pct']:.2f}")
    if row.get('away_value_change_pct', 0) != 0:
        extra_logs.append(f"   📉 Dep Değer Değişim %: {row['away_value_change_pct']:.2f}")
    if extra_logs:
        print("\n".join(extra_logs))

    # ----------------- Son 5 Maç Özeti -----------------
    print("\n📊 Takım Form Durumları:")
    print("========================================\n")

    # artık fuzzy matching ve boş durumlar güvenli şekilde ele alınıyor
    home_report = last5_report_pretty(df_form, home_team, norm_map, max_lines=5)
    if home_report is None:
        print(f"{home_team} Son 5 Maç:")
        print("   ⚠ Veri bulunamadı")
    else:
        print(f"{home_team} Son 5 Maç:")
        print(home_report)

    print("")  # boş satır
    away_report = last5_report_pretty(df_form, away_team, norm_map, max_lines=5)
    if away_report is None:
        print(f"{away_team} Son 5 Maç:")
        print("   ⚠ Veri bulunamadı")
    else:
        print(f"{away_team} Son 5 Maç:")
        print(away_report)

    # ----------------- Kayıt -----------------
    payload = {
        'timestamp': datetime.now().isoformat(timespec='seconds'),
        'home_team': home_team,
        'away_team': away_team,
        'prediction': pred_label,
        'prob_draw': probs[0],
        'prob_homewin': probs[1],
        'prob_awaywin': probs[2],
        'home_rating': row['Home_AvgRating'],
        'away_rating': row['Away_AvgRating'],
        'home_form': row['home_form'],
        'away_form': row['away_form'],
        'home_momentum': row['homeTeam_Momentum'],
        'away_momentum': row['awayTeam_Momentum'],
        'is_derby': row['IsDerby']
    }
    # 0 olmayan opsiyoneller
    if row.get('home_current_value_eur', 0) > 0: payload['home_value_eur'] = row['home_current_value_eur']
    if row.get('away_current_value_eur', 0) > 0: payload['away_value_eur'] = row['away_current_value_eur']
    if row.get('home_squad_avg_age', 0) > 0: payload['home_avg_age'] = row['home_squad_avg_age']
    if row.get('away_squad_avg_age', 0) > 0: payload['away_avg_age'] = row['away_squad_avg_age']
    save_history_excel(payload, "data/prediction_history.xlsx")

if __name__ == "__main__":
    main()
