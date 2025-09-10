import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from datetime import datetime

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from sklearn.base import BaseEstimator, TransformerMixin

# ========== KONFİGÜRASYON ==========
RANDOM_STATE = 42
TEST_SIZE = 0.2
VAL_SIZE = 0.2
N_JOBS = -1

DATA_PATH = "data/bundesliga_matches_2023_2025_final_fe_team_values_cleaned.xlsx"
PLAYER_DATA_PATH = "data/final_bundesliga_dataset_complete.xlsx"

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

# ========== ÖZEL TRANSFORMERLAR ==========
class FeatureSelector(BaseEstimator, TransformerMixin):
    """Önemli feature'ları seçmek için transformer"""
    def __init__(self, features_to_keep):
        self.features_to_keep = features_to_keep
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.features_to_keep]

# ========== VERİ HAZIRLAMA YAMASI ==========
def prepare_and_enrich_dataset(df_matches, df_players):
    """
    Eksik özellikleri otomatik olarak hesaplayarak dataseti zenginleştirir
    """
    print("🔧 Veri hazırlama ve zenginleştirme başlıyor...")
    
    # 1. Takım isimlerini standartlaştır
    if 'homeTeam.name' in df_matches.columns and 'HomeTeam' not in df_matches.columns:
        df_matches['HomeTeam'] = df_matches['homeTeam.name']
    if 'awayTeam.name' in df_matches.columns and 'AwayTeam' not in df_matches.columns:
        df_matches['AwayTeam'] = df_matches['awayTeam.name']
    
    # 2. Result_Numeric oluştur
    def get_result_numeric(row):
        home_goals = row.get('score.fullTime.home', 0)
        away_goals = row.get('score.fullTime.away', 0)
        
        if home_goals > away_goals:
            return 1  # Home win
        elif home_goals < away_goals:
            return 2  # Away win
        else:
            return 0  # Draw
    
    df_matches['Result_Numeric'] = df_matches.apply(get_result_numeric, axis=1)
    
    # 3. Tarih sütununu işle
    if 'utcDate' in df_matches.columns:
        df_matches['Date'] = pd.to_datetime(df_matches['utcDate'])
        df_matches = df_matches.sort_values('Date').reset_index(drop=True)
    
    # 4. IsDerby sütunu yoksa oluştur (basit mantık)
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
    
    # 5. Form ve momentum özelliklerini hesapla
    df_matches = calculate_form_features(df_matches)
    
    # 6. Takım ratinglerini hesapla
    print("🔁 Takım ratingleri hesaplanıyor...")
    df_matches = compute_ratings_for_matches(df_matches, df_players)
    
    # 7. Eksik özellikleri kontrol et ve gerekirse hesapla
    df_matches = calculate_missing_features(df_matches)
    
    print("✅ Veri zenginleştirme tamamlandı!")
    return df_matches

def calculate_form_features(df):
    """Form ve momentum özelliklerini hesaplar"""
    print("📊 Form ve momentum özellikleri hesaplanıyor...")
    
    # Takım listesi
    teams = pd.concat([df['HomeTeam'], df['AwayTeam']]).unique()
    
    # Yeni özellikleri başlat
    for col in ['home_form', 'away_form', 'homeTeam_GoalsScored_5', 'homeTeam_GoalsConceded_5',
                'awayTeam_GoalsScored_5', 'awayTeam_GoalsConceded_5', 'homeTeam_Momentum', 'awayTeam_Momentum']:
        if col not in df.columns:
            df[col] = 0.0
    
    # Her takım için form hesapla
    for team in teams:
        if team is None or pd.isna(team):
            continue
            
        team_matches = df[(df['HomeTeam'] == team) | (df['AwayTeam'] == team)].copy()
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
                    if m['HomeTeam'] == team:
                        home_goals = m.get('score.fullTime.home', 0)
                        away_goals = m.get('score.fullTime.away', 0)
                        goals_scored_5 += home_goals
                        goals_conceded_5 += away_goals
                        
                        if home_goals > away_goals:
                            points += 3  # Galibiyet
                        elif home_goals == away_goals:
                            points += 1  # Beraberlik
                    else:
                        home_goals = m.get('score.fullTime.home', 0)
                        away_goals = m.get('score.fullTime.away', 0)
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
            if team_matches.iloc[i]['HomeTeam'] == team:
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
    
    # Özellikleri ve varsayılan değerleri
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
        'Away_FW_Rating': 65.0
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

# ========== HELPERS ==========
def load_player_data(path=PLAYER_DATA_PATH):
    """Oyuncu verilerini yükler ve ratingleri hazırlar"""
    df = pd.read_excel(path)
    
    # Rating hesaplama - mevcut sütunlardan bir rating oluştur
    if 'Rating' in df.columns:
        df['PlayerRating'] = df['Rating']
    elif 'fbref__Goal_Contribution' in df.columns:
        # Örnek: Gol katkısı ve diğer istatistiklere dayalı basit bir rating
        df['PlayerRating'] = df['fbref__Goal_Contribution'] * 2 + df['fbref__Min'] / 90 * 0.5
    else:
        # Varsayılan rating
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
    
    return df

def pos_group(pos_str):
    if not isinstance(pos_str, str):
        return 'MF'
    p = pos_str.upper()
    if 'GK' in p or p == 'G': return 'GK'
    if p.startswith('D') or 'DF' in p or 'DEFENDER' in p: return 'DF'
    if p.startswith('M') or 'MF' in p or 'MIDFIELDER' in p: return 'MF'
    if p.startswith('F') or 'FW' in p or 'ST' in p or 'CF' in p or 'FORWARD' in p: return 'FW'
    return 'MF'

def team_players_dict(df_players):
    d = {}
    for team in df_players['Team'].unique():
        team_df = df_players[df_players['Team']==team].copy().reset_index(drop=True)
        if 'PlayerRating' not in team_df.columns:
            print(f"⚠️ Takım {team} içinde 'PlayerRating' sütunu yok!")
        d[team] = team_df
    return d

def avg_of_selected_players(df_team, idxs):
    if len(idxs)==0: return np.nan, {'GK':np.nan,'DF':np.nan,'MF':np.nan,'FW':np.nan}
    sel = df_team.loc[idxs]
    ratings = sel['PlayerRating'].dropna()
    overall = ratings.mean() if not ratings.empty else np.nan
    pos_means = {pos: sel[sel['Pos'].apply(pos_group)==pos]['PlayerRating'].dropna().mean() if 'PlayerRating' in sel else np.nan
                 for pos in ['GK','DF','MF','FW']}
    return overall, pos_means

def select_topn_by_rating(df_team, n):
    if 'PlayerRating' not in df_team.columns: return []
    return df_team['PlayerRating'].dropna().sort_values(ascending=False).index.tolist()[:n]

def compute_team_rating_from_lineup(df_team, starter_idxs, sub_idxs,
                                    starter_weight=STARTER_WEIGHT, sub_weight=SUB_WEIGHT):
    starter_mean, starter_pos = avg_of_selected_players(df_team, starter_idxs)
    sub_mean, sub_pos = avg_of_selected_players(df_team, sub_idxs)

    if np.isnan(starter_mean) and not np.isnan(sub_mean): team_rating=sub_mean
    elif np.isnan(sub_mean) and not np.isnan(starter_mean): team_rating=starter_mean
    elif np.isnan(starter_mean) and np.isnan(sub_mean): team_rating=np.nan
    else: team_rating=(starter_mean*starter_weight)+(sub_mean*sub_weight)

    pos_combined={}
    for pos in ['GK','DF','MF','FW']:
        s = starter_pos.get(pos,np.nan)
        b = sub_pos.get(pos,np.nan)
        if pd.isna(s) and not pd.isna(b): pos_combined[pos]=b
        elif pd.isna(b) and not pd.isna(s): pos_combined[pos]=s
        elif pd.isna(s) and pd.isna(b): pos_combined[pos]=np.nan
        else: pos_combined[pos]=(s*starter_weight)+(b*sub_weight)
    return team_rating, pos_combined

# ========== MATCH FEATURE ENGINEERING ==========
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

    name_to_idx={}
    for team, df_t in team_dict.items():
        mp = {n.lower().strip(): idx for idx,n in df_t['Player'].items() if isinstance(n,str)}
        name_to_idx[team]=mp

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

# ========== VERİ YÜKLEME VE PREPROCESSING ==========
def load_and_validate_data():
    print("\n📊 Veri yükleniyor...")
    
    # Maç verisini yükle
    df_matches = pd.read_excel(DATA_PATH)
    df_matches.columns = [col.strip().replace(' ', '_') for col in df_matches.columns]
    
    # Oyuncu verisini yükle
    df_players = load_player_data(PLAYER_DATA_PATH)
    
    # ✅ YENİ: Veriyi otomatik olarak hazırla ve zenginleştir
    df = prepare_and_enrich_dataset(df_matches, df_players)
    
    # Son kontroller
    for feat in SELECTED_FEATURES:
        if feat not in df.columns:
            print(f"⚠️ Kritik Uyarı: '{feat}' sütunu hala bulunamadı. 0 ile dolduruluyor.")
            df[feat] = 0

    numeric_cols = df[SELECTED_FEATURES].select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)

    print("✅ Veri hazırlığı tamamlandı")
    return df

# ========== ZAMAN BAZLI SPLIT ==========
def time_based_split(df, test_size=0.2, val_size=0.2):
    """Zaman bazlı train-validation-test split"""
    if 'Date' in df.columns:
        df_sorted = df.sort_values('Date').reset_index(drop=True)
    else:
        df_sorted = df.reset_index(drop=True)
        print("ℹ️ Date sütunu yok, orijinal sıra kullanılıyor")
    
    n = len(df_sorted)
    
    test_split_idx = int(n * (1 - test_size))
    val_split_idx = int(test_split_idx * (1 - val_size))
    
    train_df = df_sorted.iloc[:val_split_idx]
    val_df = df_sorted.iloc[val_split_idx:test_split_idx]
    test_df = df_sorted.iloc[test_split_idx:]
    
    print(f"📊 Split bilgisi: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df

# ========== FEATURE IMPORTANCE ANALIZI ==========
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

# ========== MODEL PIPELINE ==========
def create_lgbm_pipeline(important_features=None):
    """LightGBM pipeline oluşturma"""
    features_to_use = important_features if important_features else SELECTED_FEATURES
    
    preprocessor = ColumnTransformer([
        ('scaler', StandardScaler(), features_to_use)
    ], remainder='drop')
    
    lgbm_clf = lgb.LGBMClassifier(
        objective='multiclass', 
        num_class=3, 
        random_state=RANDOM_STATE, 
        n_jobs=N_JOBS, 
        verbosity=-1,
        n_estimators=1000,
        force_row_wise=True
    )
    
    return Pipeline([
        ('preprocessor', preprocessor), 
        ('lgbm', lgbm_clf)
    ])

# ========== MODEL EĞİTİMİ ==========
def train_lgbm_model():
    print("⚽ Bundesliga Tahmin Modeli Eğitimi (Geliştirilmiş Sürüm)")
    print("=" * 60)
    
    # Veriyi yükle
    df = load_and_validate_data()
    
    # Zaman bazlı split
    train_df, val_df, test_df = time_based_split(df, TEST_SIZE, VAL_SIZE)
    
    X_train = train_df[SELECTED_FEATURES].copy()
    y_train = train_df['Result_Numeric'].copy()
    
    X_val = val_df[SELECTED_FEATURES].copy()
    y_val = val_df['Result_Numeric'].copy()
    
    X_test = test_df[SELECTED_FEATURES].copy()
    y_test = test_df['Result_Numeric'].copy()
    
    # Class weights hesapla
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    
    # Sample weights
    sample_weights_train = np.array([class_weight_dict[yy] for yy in y_train])
    
    # Hiperparametre grid'i
    param_distributions = {
        'lgbm__learning_rate': [0.01, 0.05, 0.1],
        'lgbm__max_depth': [3, 5, 7],
        'lgbm__num_leaves': [20, 31, 50],
        'lgbm__min_child_samples': [15, 20, 30],
        'lgbm__reg_alpha': [0, 0.1, 0.5],
        'lgbm__reg_lambda': [0, 0.1, 0.5],
        'lgbm__subsample': [0.8, 0.9],
        'lgbm__colsample_bytree': [0.8, 0.9]
    }
    
    # TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=4)
    
    # İlk modeli eğit (feature importance için)
    print("\n🔎 İlk model eğitimi (feature importance analizi için)...")
    initial_model = create_lgbm_pipeline()
    initial_model.fit(X_train, y_train, lgbm__sample_weight=sample_weights_train)
    
    # Feature importance analizi
    important_features = analyze_feature_importance(initial_model, SELECTED_FEATURES)
    
    # RandomizedSearchCV ile hiperparametre optimizasyonu
    print("\n🎯 Hiperparametre Optimizasyonu (RandomizedSearchCV)...")
    model = create_lgbm_pipeline(important_features)
    
    random_search = RandomizedSearchCV(
        estimator=model, 
        param_distributions=param_distributions,
        n_iter=25,
        cv=tscv,
        scoring='f1_weighted',
        n_jobs=N_JOBS,
        verbose=1,
        random_state=RANDOM_STATE
    )
    
    random_search.fit(X_train, y_train, lgbm__sample_weight=sample_weights_train)
    
    best_params = random_search.best_params_
    print(f"\n🏆 En İyi Parametreler: {best_params}")
    print(f"🏆 En İyi CV Skoru: {random_search.best_score_:.4f}")
    
    # En iyi parametrelerle final modeli
    print("\n🚀 Final modeli eğitimi...")
    final_model = create_lgbm_pipeline(important_features)
    final_model.set_params(**best_params)

    # 1. ÖNCE PREPROCESSOR'U AYRI OLARAK FIT ET
    print("🔧 Preprocessor fit ediliyor...")
    final_model.named_steps['preprocessor'].fit(X_train)

    # 2. SONRA VALIDATION VERİSİNİ TRANSFORM ET
    print("🔧 Validation verisi transform ediliyor...")
    X_val_processed = final_model.named_steps['preprocessor'].transform(X_val)
    X_train_processed = final_model.named_steps['preprocessor'].transform(X_train)

    # 3. LGBM MODELİNİ DOĞRUDAN EĞİT (callback ile early stopping)
    print("🔧 LightGBM modeli eğitiliyor...")
    
    # Callback'ler
    from lightgbm import early_stopping, log_evaluation
    callbacks = [
        early_stopping(stopping_rounds=50),
        log_evaluation(period=100)
    ]
    
    # LightGBM modelini doğrudan fit et
    final_model.named_steps['lgbm'].fit(
        X_train_processed,
        y_train,
        eval_set=[(X_val_processed, y_val)],
        eval_metric='multi_logloss',
        callbacks=callbacks,
        sample_weight=sample_weights_train
    )

    # 4. PIPELINE'IN FITTED ÖZELLİĞİNİ AYARLA (ÖNEMLİ!)
    # Pipeline'ın fit edildiğini işaretlemek için
    final_model._is_fitted = True
    for step_name, step in final_model.named_steps.items():
        if hasattr(step, '_is_fitted'):
            step._is_fitted = True
    
    # Test seti performansı
    print("\n📊 Test Seti Performansı:")
    y_pred = final_model.predict(X_test)
    
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"F1 Score: {f1_score(y_test, y_pred, average='weighted'):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Draw', 'HomeWin', 'AwayWin']))
    
    # Modeli kaydet
    os.makedirs("models", exist_ok=True)
    model_path = "models/bundesliga_model_final.pkl"
    joblib.dump(final_model, model_path)
    
    # Feature listesini de kaydet
    feature_info = {
        'important_features': important_features,
        'all_features': SELECTED_FEATURES,
        'best_params': best_params
    }
    joblib.dump(feature_info, "models/feature_info.pkl")
    
    print(f"\n💾 Model kaydedildi: {model_path}")
    print("💾 Feature bilgileri kaydedildi: models/feature_info.pkl")
    
    # Cross-validation sonuçlarını analiz et
    print("\n📈 Cross-Validation Sonuçları:")
    cv_results = random_search.cv_results_
    for i in range(tscv.n_splits):
        fold_scores = [cv_results[f'split{i}_test_score'][j] for j in range(len(cv_results['params']))]
        print(f"Fold {i+1} skor aralığı: {np.min(fold_scores):.4f} - {np.max(fold_scores):.4f}")
    
    return final_model, important_features

# ========== MAIN ==========
def main():
    print("🏆 Bundesliga Tahmin Modeli - Geliştirilmiş Sürüm")
    print("=" * 50)
    print("✅ Time-based split")
    print("✅ Early stopping")
    print("✅ Feature importance analizi")
    print("✅ Regularization parametreleri")
    print("✅ RandomizedSearchCV optimizasyonu")
    print("✅ Result_Numeric kullanımı")
    print("✅ Otomatik veri zenginleştirme")
    print("=" * 50)
    
    os.makedirs("models", exist_ok=True)
    model, important_features = train_lgbm_model()
    
    print("\n🎉 Model eğitimi başarıyla tamamlandı!")
    print(f"📋 Kullanılan önemli feature'lar: {len(important_features)}/{len(SELECTED_FEATURES)}")

if __name__ == "__main__":
    main()