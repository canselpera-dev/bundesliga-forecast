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

DATA_PATH = "data/bundesliga_complete_dataset.xlsx"  # ✅ DEĞİŞTİRİLDİ
PLAYER_RATINGS_PATH = "data/player_ratings_v2_clean.xlsx"

TOP_N_STARTERS = 11
TOP_N_SUBS = 7
STARTER_WEIGHT = 0.7
SUB_WEIGHT = 0.3

SELECTED_FEATURES = [
    'Home_AvgRating', 'Away_AvgRating', 'Rating_Diff', 'Total_AvgRating',
    'Home_Form', 'Away_Form', 'Form_Diff', 'IsDerby',
    'homeTeam_GoalsScored_5', 'homeTeam_GoalsConceded_5',
    'awayTeam_GoalsScored_5', 'awayTeam_GoalsConceded_5',
    'homeTeam_Momentum', 'awayTeam_Momentum',
    'Home_GK_Rating', 'Home_DF_Rating', 'Home_MF_Rating', 'Home_FW_Rating',
    'Away_GK_Rating', 'Away_DF_Rating', 'Away_MF_Rating', 'Away_FW_Rating'
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

# ========== HELPERS ==========
def load_player_ratings(path=PLAYER_RATINGS_PATH):
    df = pd.read_excel(path)
    if 'PlayerRating' in df.columns:
        df['PlayerRating'] = df['PlayerRating'].astype(str).str.replace(',', '.')
        df['PlayerRating'] = pd.to_numeric(df['PlayerRating'], errors='coerce')
    df['Team'] = df['Team'].astype(str).str.strip()
    df['Pos'] = df['Pos'].astype(str).str.upper().str.strip()
    return df

def pos_group(pos_str):
    if not isinstance(pos_str, str):
        return 'MF'
    p = pos_str.upper()
    if 'GK' in p or p == 'G': return 'GK'
    if p.startswith('D') or 'DF' in p: return 'DF'
    if p.startswith('M') or 'MF' in p: return 'MF'
    if p.startswith('F') or 'FW' in p or 'ST' in p or 'CF' in p: return 'FW'
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
    
    for col in ['HomeTeam','AwayTeam','Season']:
        if col not in df_matches.columns and col.lower() in df_matches.columns:
            df_matches[col] = df_matches[col.lower()]

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

        def parse_lineup(cell):
            if pd.isna(cell): return []
            if isinstance(cell,str): return [p.strip() for p in cell.split(',') if p.strip()]
            if isinstance(cell,(list,tuple,np.ndarray)): return list(cell)
            return []

        for colname in ['Home_Lineup','HomeLineup','Home11','Home_Starters','home_lineup','home11']:
            if colname in df_matches.columns and not pd.isna(row.get(colname)):
                parsed=parse_lineup(row.get(colname))
                for nm in parsed:
                    if nm.lower() in name_to_idx.get(home,{}): home_starters.append(name_to_idx[home][nm.lower()])
                break

        for colname in ['Away_Lineup','AwayLineup','Away11','Away_Starters','away_lineup','away11']:
            if colname in df_matches.columns and not pd.isna(row.get(colname)):
                parsed=parse_lineup(row.get(colname))
                for nm in parsed:
                    if nm.lower() in name_to_idx.get(away,{}): away_starters.append(name_to_idx[away][nm.lower()])
                break

        if len(home_starters)<TOP_N_STARTERS: home_starters=select_topn_by_rating(df_home,TOP_N_STARTERS)
        if len(away_starters)<TOP_N_STARTERS: away_starters=select_topn_by_rating(df_away,TOP_N_STARTERS)

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
        df_matches[f'Home_{pos}_Rating'].fillna(df_players[df_players['Pos'].apply(pos_group)==pos]['PlayerRating'].mean(),inplace=True)
        df_matches[f'Away_{pos}_Rating'].fillna(df_players[df_players['Pos'].apply(pos_group)==pos]['PlayerRating'].mean(),inplace=True)

    return df_matches

# ========== VERİ YÜKLEME VE PREPROCESSING ==========
def load_and_validate_data():
    print("\n📊 Veri yükleniyor...")
    df = pd.read_excel(DATA_PATH)
    df.columns = [col.strip().replace(' ', '_') for col in df.columns]

    # ✅ DEĞİŞTİRİLDİ: Result_Numeric sütunu kontrolü
    if 'Result_Numeric' not in df.columns: 
        raise ValueError("⚠️ 'Result_Numeric' sütunu bulunamadı! Lütfen dataseti formatlayın.")
    
    # Tarih sütunu kontrolü ve sıralama
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df = df.sort_values('Date').reset_index(drop=True)
    else:
        print("⚠️ Date sütunu bulunamadı, index sırası kullanılacak")
    
    df['IsDerby'] = df['IsDerby'].astype(int) if 'IsDerby' in df.columns else 0

    df_players = load_player_ratings(PLAYER_RATINGS_PATH)

    need_compute = any(col not in df.columns for col in ['Home_AvgRating','Away_AvgRating','Total_AvgRating','Rating_Diff',
                                                      'Home_GK_Rating','Home_DF_Rating','Home_MF_Rating','Home_FW_Rating'])
    if need_compute:
        print("🔁 Maç bazlı ratingler ve pozisyon özellikleri hesaplanıyor (player_ratings dosyasından)...")
        df = compute_ratings_for_matches(df, df_players)

    for feat in SELECTED_FEATURES:
        if feat not in df.columns:
            print(f"⚠️ Uyarı: '{feat}' sütunu bulunamadı. 0 ile dolduruluyor.")
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
        n_estimators=1000,  # Early stopping için büyük değer
        force_row_wise=True  # Bellek optimizasyonu
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
    
    # ✅ DEĞİŞTİRİLDİ: Result_Numeric kullanılıyor
    X_train = train_df[SELECTED_FEATURES].copy()
    y_train = train_df['Result_Numeric'].copy()  # Direkt numeric değerler
    
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
    
    # Hiperparametre grid'i (overfitting'i azaltacak şekilde)
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
        n_iter=25,  # 25 rastgele kombinasyon
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
    
    # Early stopping ile eğitim
    final_model.fit(
        X_train, y_train,
        lgbm__eval_set=[(final_model.named_steps['preprocessor'].transform(X_val), y_val)],
        lgbm__early_stopping_rounds=50,
        lgbm__verbose=100,
        lgbm__sample_weight=sample_weights_train
    )
    
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
    print("=" * 50)
    
    os.makedirs("models", exist_ok=True)
    model, important_features = train_lgbm_model()
    
    print("\n🎉 Model eğitimi başarıyla tamamlandı!")
    print(f"📋 Kullanılan önemli feature'lar: {len(important_features)}/{len(SELECTED_FEATURES)}")

if __name__ == "__main__":
    main()