#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bundesliga Tahmin Modeli - REALISTIC BALANCE v10.1
Result_Numeric Hatası Düzeltmesi
"""

import os
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import lightgbm as lgb
from datetime import datetime, timedelta
from scipy import stats

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, RandomizedSearchCV, TimeSeriesSplit, cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score, balanced_accuracy_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel, RFE
import xgboost as xgb

# ========== REALISTIC BALANCE KONFİGÜRASYON ==========
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
N_JOBS = -1
MAX_FEATURES = 12

DATA_PATH = "data/bundesliga_matches_2023_2025_final_fe_team_values_cleaned.xlsx"
PLAYER_DATA_PATH = "data/final_bundesliga_dataset_complete.xlsx"

# PROBLEMLİ FEATURE'LARI TANIMLA - HEDEF DEĞİŞKENİ KALDIRMA!
PROBLEMATIC_FEATURES = [
    'id', 'score.fullTime.home', 'score.fullTime.away',
    'goals_difference', 'xg_difference', 'awayTeam.id', 'matchday',
    'home_goals', 'away_goals', 'home_xg', 'away_xg'
]

# TEMİZ ÖZELLİK LİSTESİ - Bundesliga gerçekleri
SELECTED_FEATURES = [
    # Temel Performans Metrikleri - EN KRİTİK
    'home_ppg_cumulative', 'away_ppg_cumulative',
    'home_gpg_cumulative', 'away_gpg_cumulative',
    'home_gapg_cumulative', 'away_gapg_cumulative',
    'home_form_5games', 'away_form_5games',
    
    # Power ve Form - ORTA KRİTİK
    'home_power_index', 'away_power_index', 
    'power_difference', 'form_difference',
    
    # H2H - ÖNEMLİ
    'h2h_win_ratio', 'h2h_goal_difference',
    
    # Value-based - DESTEK
    'value_difference', 'value_ratio',
    
    # Özel Durumlar
    'isDerby'
]

# ========== CLEAN FEATURE SELECTION ==========
def clean_feature_selection(X_train, y_train, X_val, X_test, max_features=MAX_FEATURES):
    """Temiz feature selection - problemli feature'ları kaldır (HEDEF DEĞİŞKENİ KORU)"""
    print(f"🧹 CLEAN Feature Selection (Max {max_features} özellik)...")
    
    # Problemli feature'ları kaldır - HEDEF DEĞİŞKENİ (y_train) KORU!
    X_train_clean = X_train.drop(columns=PROBLEMATIC_FEATURES, errors='ignore')
    X_val_clean = X_val.drop(columns=PROBLEMATIC_FEATURES, errors='ignore')
    X_test_clean = X_test.drop(columns=PROBLEMATIC_FEATURES, errors='ignore')
    
    print(f"🔍 Problemli {len(PROBLEMATIC_FEATURES)} feature kaldırıldı")
    print(f"🔢 Kalan sayısal sütun sayısı: {X_train_clean.select_dtypes(include=[np.number]).shape[1]}")
    
    # 3 farklı yöntemle feature importance hesapla
    feature_scores = {}
    
    # 1. RandomForest
    try:
        rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, max_depth=5)
        rf.fit(X_train_clean, y_train)
        for i, feature in enumerate(X_train_clean.columns):
            feature_scores[feature] = feature_scores.get(feature, 0) + rf.feature_importances_[i]
    except Exception as e:
        print(f"⚠️ RandomForest feature selection hatası: {e}")
    
    # 2. LightGBM
    try:
        lgb_model = lgb.LGBMClassifier(n_estimators=50, random_state=RANDOM_STATE, verbose=-1)
        lgb_model.fit(X_train_clean, y_train)
        for i, feature in enumerate(X_train_clean.columns):
            feature_scores[feature] = feature_scores.get(feature, 0) + lgb_model.feature_importances_[i]
    except Exception as e:
        print(f"⚠️ LightGBM feature selection hatası: {e}")
    
    # 3. Korelasyon bazlı
    try:
        for feature in X_train_clean.columns:
            correlation = abs(np.corrcoef(X_train_clean[feature], y_train)[0, 1])
            if not np.isnan(correlation):
                feature_scores[feature] = feature_scores.get(feature, 0) + correlation
    except Exception as e:
        print(f"⚠️ Korelasyon feature selection hatası: {e}")
    
    # En iyileri seç
    if not feature_scores:
        print("🚨 Tüm feature selection yöntemleri başarısız, tüm sayısal feature'ları kullanıyoruz...")
        selected_features = X_train_clean.columns[:max_features].tolist()
    else:
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [feat for feat, score in sorted_features[:max_features]]
    
    # Seçilen feature'ları kontrol et (problemli olanlar var mı?)
    problematic_found = [feat for feat in selected_features if feat in PROBLEMATIC_FEATURES]
    if problematic_found:
        print(f"🚨 UYARI: Seçilen feature'lar arasında problemli feature'lar bulundu: {problematic_found}")
        # Problemli feature'ları çıkar
        selected_features = [feat for feat in selected_features if feat not in PROBLEMATIC_FEATURES]
        # Yerlerine yeni feature'lar ekle
        backup_features = [feat for feat in X_train_clean.columns if feat not in selected_features and feat not in PROBLEMATIC_FEATURES]
        needed = max_features - len(selected_features)
        if needed > 0 and backup_features:
            selected_features.extend(backup_features[:needed])
    
    print(f"✅ Seçilen temiz özellikler: {selected_features}")
    
    X_train_selected = X_train_clean[selected_features]
    X_val_selected = X_val_clean[selected_features]
    X_test_selected = X_test_clean[selected_features]
    
    return X_train_selected, X_val_selected, X_test_selected, selected_features

# ========== REALISTIC CLASS WEIGHTS ==========
def compute_realistic_class_weights(y_train):
    """Bundesliga gerçeklerine uygun ve dengeli class weights"""
    class_counts = pd.Series(y_train).value_counts().sort_index()
    total_matches = len(y_train)
    
    # Bundesliga gerçek istatistikleri (daha gerçekçi)
    expected_distribution = [0.25, 0.45, 0.30]  # Draw, HomeWin, AwayWin
    
    # Daha dengeli weight hesaplama
    realistic_weights = []
    for i, count in enumerate(class_counts):
        expected_count = total_matches * expected_distribution[i]
        weight = expected_count / count if count > 0 else 1.0
        # Daha konservatif weighting (maksimum 1.5x)
        realistic_weights.append(min(weight, 1.5))
    
    class_weight_dict = dict(zip(class_counts.index, realistic_weights))
    print(f"⚖️ REALISTIC Class Weights: {class_weight_dict}")
    print(f"📊 Gerçek Dağılım: {dict(class_counts)}")
    print(f"🎯 Beklenen Dağılım: Draw: 25%, HomeWin: 45%, AwayWin: 30%")
    
    return class_weight_dict

# ========== REALISTIC MODEL PIPELINE ==========
def create_realistic_pipeline(selected_features, class_weight_dict=None):
    """Gerçekçi model pipeline - overfitting önleme"""
    
    preprocessor = ColumnTransformer([
        ('scaler', RobustScaler(), selected_features)
    ], remainder='drop')
    
    # DAHA GÜÇLÜ REGULARIZATION - Overfitting önleme
    lgbm_params = {
        'objective': 'multiclass',
        'num_class': 3,
        'random_state': RANDOM_STATE,
        'n_jobs': N_JOBS,
        'verbosity': -1,
        'n_estimators': 150,      # Daha az estimators
        'learning_rate': 0.05,
        'max_depth': 3,           # Daha sığ
        'num_leaves': 8,          # Daha az leaves
        'min_child_samples': 25,  # Daha fazla min child
        'subsample': 0.7,         # Daha az subsample
        'colsample_bytree': 0.7,  # Daha az feature
        'reg_alpha': 2.0,         # Daha güçlü L1
        'reg_lambda': 2.0,        # Daha güçlü L2
        'force_row_wise': True
    }
    
    # class_weight_dict'ı doğrudan LightGBM parametrelerine ekle
    if class_weight_dict:
        lgbm_params['class_weight'] = class_weight_dict
    
    lgbm_clf = lgb.LGBMClassifier(**lgbm_params)
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('lgbm', lgbm_clf)
    ])

# ========== BUNDESLIGA-SPECIFIC FEATURES ==========
class BundesligaFeatureEngineer(BaseEstimator, TransformerMixin):
    """Bundesliga'ya özel feature engineering - HEDEF DEĞİŞKENİ KORU!"""
    
    def __init__(self):
        self.feature_names = []
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # HEDEF DEĞİŞKENİ (Result_Numeric) KORU!
        target_column = None
        if 'Result_Numeric' in X.columns:
            target_column = X['Result_Numeric']
        
        # Problemli feature'ları kaldır (HEDEF DEĞİŞKENİ HARİÇ)
        columns_to_drop = [col for col in PROBLEMATIC_FEATURES if col in X.columns and col != 'Result_Numeric']
        X = X.drop(columns=columns_to_drop, errors='ignore')
        
        # SADECE SAYISAL SÜTUNLARI İŞLE
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        
        # 1. TEMEL FARKLAR - En kritik
        if all(col in numeric_columns for col in ['home_ppg_cumulative', 'away_ppg_cumulative']):
            X['ppg_difference'] = X['home_ppg_cumulative'] - X['away_ppg_cumulative']
        
        if all(col in numeric_columns for col in ['home_gpg_cumulative', 'away_gpg_cumulative']):
            X['gpg_difference'] = X['home_gpg_cumulative'] - X['away_gpg_cumulative']
            X['total_goals_expected'] = (X['home_gpg_cumulative'] + X['away_gpg_cumulative']) / 2
        
        if all(col in numeric_columns for col in ['home_form_5games', 'away_form_5games']):
            X['form_difference'] = X['home_form_5games'] - X['away_form_5games']
            X['form_similarity'] = 1 - abs(X['home_form_5games'] - X['away_form_5games'])
        
        # 2. GÜÇ METRİKLERİ
        if all(col in numeric_columns for col in ['home_power_index', 'away_power_index']):
            X['power_difference'] = X['home_power_index'] - X['away_power_index']
            X['strength_ratio'] = np.minimum(X['home_power_index'], X['away_power_index']) / (np.maximum(X['home_power_index'], X['away_power_index']) + 1e-8)
        
        # 3. EV SAHİBİ AVANTAJI - Bundesliga'da güçlü
        if all(col in numeric_columns for col in ['home_ppg_cumulative', 'home_form_5games']):
            X['home_advantage'] = X['home_ppg_cumulative'] * 0.6 + X['home_form_5games'] * 0.4
        
        # 4. DEPLASMAN RİSKİ
        if all(col in numeric_columns for col in ['away_gapg_cumulative', 'away_form_5games']):
            X['away_risk'] = X['away_gapg_cumulative'] * (1 - X['away_form_5games'])
        
        # 5. BERABERLİK POTANSİYELİ - GERÇEKÇİ
        if all(col in X.columns for col in ['form_similarity', 'power_difference', 'ppg_difference']):
            X['draw_potential'] = (
                X['form_similarity'] * 0.5 + 
                (1 - abs(X['power_difference'] / 3)) * 0.3 +
                (1 - abs(X['ppg_difference'] / 3)) * 0.2
            )
        
        # HEDEF DEĞİŞKENİ GERİ EKLE (eğer varsa)
        if target_column is not None and 'Result_Numeric' not in X.columns:
            X['Result_Numeric'] = target_column
        
        # SADECE SAYISAL SÜTUNLARI KEEP (HEDEF DEĞİŞKENİ KORU)
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        self.feature_names = X.columns.tolist()
        print(f"🔧 Feature engineering sonrası {len(self.feature_names)} özellik oluşturuldu")
        return X

# ========== MANUEL CUMULATIVE STATS ==========
def calculate_cumulative_stats(df_matches):
    """Maç verisinden takımların kümülatif istatistiklerini hesapla"""
    print("🔄 Manuel cumulative istatistikler hesaplanıyor...")
    
    df = df_matches.copy()
    
    # Tarihe göre sırala
    if 'Date' not in df.columns and 'utcDate' in df.columns:
        df['Date'] = pd.to_datetime(df['utcDate'], errors='coerce')
    
    if 'Date' in df.columns:
        df = df.sort_values('Date').reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    
    # Takım isimlerini standartlaştır
    if 'homeTeam.name' in df.columns and 'HomeTeam' not in df.columns:
        df['HomeTeam'] = df['homeTeam.name']
    if 'awayTeam.name' in df.columns and 'AwayTeam' not in df.columns:
        df['AwayTeam'] = df['awayTeam.name']
    
    team_stats = {}
    
    cumulative_features = [
        'home_ppg_cumulative', 'away_ppg_cumulative',
        'home_gpg_cumulative', 'away_gpg_cumulative', 
        'home_gapg_cumulative', 'away_gapg_cumulative',
        'home_form_5games', 'away_form_5games'
    ]
    
    for feature in cumulative_features:
        df[feature] = 0.0
    
    for idx, match in df.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        
        if home_team not in team_stats:
            team_stats[home_team] = {
                'points': 0, 'goals_for': 0, 'goals_against': 0, 'matches': 0,
                'recent_results': [],
                'goal_diff': 0
            }
        
        if away_team not in team_stats:
            team_stats[away_team] = {
                'points': 0, 'goals_for': 0, 'goals_against': 0, 'matches': 0,
                'recent_results': [],
                'goal_diff': 0
            }
        
        # BU MAÇ ÖNCESİ istatistikleri kaydet
        home_matches = max(team_stats[home_team]['matches'], 1)
        away_matches = max(team_stats[away_team]['matches'], 1)
        
        df.loc[idx, 'home_ppg_cumulative'] = team_stats[home_team]['points'] / home_matches
        df.loc[idx, 'away_ppg_cumulative'] = team_stats[away_team]['points'] / away_matches
        
        df.loc[idx, 'home_gpg_cumulative'] = team_stats[home_team]['goals_for'] / home_matches
        df.loc[idx, 'away_gpg_cumulative'] = team_stats[away_team]['goals_for'] / away_matches
        
        df.loc[idx, 'home_gapg_cumulative'] = team_stats[home_team]['goals_against'] / home_matches
        df.loc[idx, 'away_gapg_cumulative'] = team_stats[away_team]['goals_against'] / away_matches
        
        df.loc[idx, 'home_form_5games'] = calculate_form(team_stats[home_team]['recent_results'])
        df.loc[idx, 'away_form_5games'] = calculate_form(team_stats[away_team]['recent_results'])
        
        # 🔽 BU MAÇIN SONUCUNU İŞLE 🔽
        home_goals = match.get('score.fullTime.home', match.get('home_goals', 0))
        away_goals = match.get('score.fullTime.away', match.get('away_goals', 0))
        
        # Home team güncelleme
        team_stats[home_team]['goals_for'] += home_goals
        team_stats[home_team]['goals_against'] += away_goals
        team_stats[home_team]['matches'] += 1
        team_stats[home_team]['goal_diff'] += (home_goals - away_goals)
        
        # Away team güncelleme  
        team_stats[away_team]['goals_for'] += away_goals
        team_stats[away_team]['goals_against'] += home_goals
        team_stats[away_team]['matches'] += 1
        team_stats[away_team]['goal_diff'] += (away_goals - home_goals)
        
        # Puanları ve formu güncelle
        if home_goals > away_goals:
            team_stats[home_team]['points'] += 3
            team_stats[home_team]['recent_results'].append(1.0)
            team_stats[away_team]['recent_results'].append(0.0)
        elif away_goals > home_goals:
            team_stats[away_team]['points'] += 3
            team_stats[home_team]['recent_results'].append(0.0)
            team_stats[away_team]['recent_results'].append(1.0)
        else:
            team_stats[home_team]['points'] += 1
            team_stats[away_team]['points'] += 1
            team_stats[home_team]['recent_results'].append(0.5)
            team_stats[away_team]['recent_results'].append(0.5)
        
        # Recent results'u 5 maçla sınırla
        team_stats[home_team]['recent_results'] = team_stats[home_team]['recent_results'][-5:]
        team_stats[away_team]['recent_results'] = team_stats[away_team]['recent_results'][-5:]
    
    print(f"✅ Cumulative istatistikler hesaplandı: {len(team_stats)} takım")
    return df

def calculate_form(recent_results):
    """Son 5 maç formunu hesapla"""
    if not recent_results:
        return 0.5
    return sum(recent_results) / len(recent_results)

# ========== VERİ HAZIRLAMA ==========
def realistic_data_preparation(df_matches, df_players):
    """Gerçekçi veri hazırlama - HEDEF DEĞİŞKENİ KORU!"""
    print("🔧 REALISTIC veri hazırlama başlıyor...")
    
    df = df_matches.copy()
    
    # 1. Takım isimlerini standartlaştır
    if 'homeTeam.name' in df.columns and 'HomeTeam' not in df.columns:
        df['HomeTeam'] = df['homeTeam.name']
    if 'awayTeam.name' in df.columns and 'AwayTeam' not in df.columns:
        df['AwayTeam'] = df['awayTeam.name']
    
    # 2. Result_Numeric oluştur
    def safe_get_result(row):
        try:
            home_goals = row.get('score.fullTime.home', row.get('home_goals', 0))
            away_goals = row.get('score.fullTime.away', row.get('away_goals', 0))
            
            if pd.isna(home_goals) or pd.isna(away_goals):
                return 0
                
            if home_goals > away_goals:
                return 1
            elif home_goals < away_goals:
                return 2
            else:
                return 0
        except:
            return 0
    
    df['Result_Numeric'] = df.apply(safe_get_result, axis=1)
    
    # 3. Tarih işleme
    if 'utcDate' in df.columns:
        df['Date'] = pd.to_datetime(df['utcDate'], errors='coerce')
        df = df.sort_values('Date').reset_index(drop=True)
    
    # 4. MANUEL CUMULATIVE STATS HESAPLA
    df = calculate_cumulative_stats(df)
    
    # 5. Eksik değerleri doldur
    df = realistic_missing_value_imputation(df)
    
    # 6. GERÇEKÇİ Feature engineering
    df = realistic_feature_engineering(df)
    
    print("✅ REALISTIC veri hazırlama tamamlandı!")
    return df

def realistic_missing_value_imputation(df):
    """Gerçekçi eksik değer doldurma"""
    print("📊 Eksik değer analizi ve doldurma...")
    
    imputation_strategies = {
        'h2h_win_ratio': 0.5, 'h2h_goal_difference': 0,
        'home_form': 0.5, 'away_form': 0.5, 'form_difference': 0,
        'home_current_value_eur': 200000000, 'away_current_value_eur': 200000000,
        'home_goals': 1.5, 'away_goals': 1.5,
        'home_ppg_cumulative': 1.5, 'away_ppg_cumulative': 1.5,
        'home_gpg_cumulative': 1.5, 'away_gpg_cumulative': 1.5,
        'home_gapg_cumulative': 1.5, 'away_gapg_cumulative': 1.5,
        'home_form_5games': 0.5, 'away_form_5games': 0.5,
        'home_power_index': 0.5, 'away_power_index': 0.5,
        'power_difference': 0, 'ppg_difference': 0,
        'gpg_difference': 0, 'total_goals_expected': 2.8,
        'form_similarity': 0.5, 'strength_ratio': 1.0,
        'home_advantage': 0.5, 'away_risk': 0.5,
        'draw_potential': 0.3
    }
    
    for column, default_value in imputation_strategies.items():
        if column in df.columns:
            null_count = df[column].isnull().sum()
            if null_count > 0:
                df[column].fillna(default_value, inplace=True)
    
    return df

def realistic_feature_engineering(df):
    """Gerçekçi feature engineering - HEDEF DEĞİŞKENİ KORU!"""
    df = df.copy()
    
    feature_engineer = BundesligaFeatureEngineer()
    df = feature_engineer.fit_transform(df)
    
    return df

# ========== VERİ YÜKLEME ==========
def load_realistic_data():
    """Gerçekçi veri yükleme"""
    print("\n📊 REALISTIC veri yükleniyor...")
    
    try:
        df_matches = pd.read_excel(DATA_PATH)
        df_matches.columns = [col.strip().replace(' ', '_') for col in df_matches.columns]
        
        df_players = pd.read_excel(PLAYER_DATA_PATH)
        
        df = realistic_data_preparation(df_matches, df_players)
        
        # Eksik feature'ları doldur
        missing_features = []
        for feat in SELECTED_FEATURES:
            if feat not in df.columns:
                missing_features.append(feat)
                df[feat] = 0
        
        if missing_features:
            print(f"⚠️ Eksik özellikler dolduruldu: {len(missing_features)}")
        
        # Sadece sayısal sütunları doldur (Result_Numeric hariç)
        numeric_cols = [col for col in df.select_dtypes(include=np.number).columns if col != 'Result_Numeric']
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        # Result_Numeric kontrolü
        if 'Result_Numeric' not in df.columns:
            raise KeyError("❌ KRİTİK HATA: Result_Numeric sütunu kayboldu!")
        
        class_distribution = df['Result_Numeric'].value_counts().sort_index()
        print(f"📈 Sınıf Dağılımı: {dict(class_distribution)}")
        
        print("✅ REALISTIC veri hazırlığı tamamlandı")
        return df
        
    except Exception as e:
        print(f"❌ Veri yükleme hatası: {e}")
        raise

# ========== REALISTIC MODEL EĞİTİMİ ==========
def train_realistic_model():
    """GERÇEKÇİ MODEL EĞİTİMİ - HEDEF DEĞİŞKEN KORUMALI"""
    print("⚽ Bundesliga Tahmin Modeli - REALISTIC BALANCE v10.1")
    print("=" * 70)
    print("🎯 GERÇEKÇİ HEDEFLER: %55-65 Accuracy")
    print("🎯 Veri Sızıntısı Önleme") 
    print("🎯 Problemli Feature'lar Kaldırıldı")
    print("🎯 HEDEF DEĞİŞKEN KORUMALI")
    print("🎯 Bundesliga Pattern'lerine Uyum")
    print("=" * 70)
    
    # Veriyi yükle
    df = load_realistic_data()
    
    # Zaman bazlı split
    train_df, val_df, test_df = time_based_split(df, TEST_SIZE, VAL_SIZE)
    
    # Feature ve target'ları ayır - HEDEF DEĞİŞKENİ AYRI TUT!
    X_train = train_df.drop(columns=['Result_Numeric'], errors='ignore')
    y_train = train_df['Result_Numeric'].copy()
    
    X_val = val_df.drop(columns=['Result_Numeric'], errors='ignore')
    y_val = val_df['Result_Numeric'].copy()
    
    X_test = test_df.drop(columns=['Result_Numeric'], errors='ignore')
    y_test = test_df['Result_Numeric'].copy()
    
    print(f"🎯 Hedef değişken dağılımı - Eğitim: {y_train.value_counts().to_dict()}")
    
    # 1. GERÇEKÇİ feature engineering uygula
    print("🔧 Realistic feature engineering uygulanıyor...")
    feature_engineer = BundesligaFeatureEngineer()
    X_train = feature_engineer.fit_transform(X_train)
    X_val = feature_engineer.transform(X_val)
    X_test = feature_engineer.transform(X_test)
    
    # VERİ KONTROLÜ
    print(f"🔢 Eğitim verisi shape: {X_train.shape}")
    print(f"🔢 Sayısal sütun sayısı: {X_train.select_dtypes(include=[np.number]).shape[1]}")
    
    # 2. TEMİZ feature selection yap
    X_train_selected, X_val_selected, X_test_selected, important_features = clean_feature_selection(
        X_train, y_train, X_val, X_test, MAX_FEATURES
    )
    
    print(f"📊 Eğitim verisi: {X_train_selected.shape}")
    print(f"📊 Validation verisi: {X_val_selected.shape}")
    print(f"📊 Test verisi: {X_test_selected.shape}")
    
    # 3. Gerçekçi class weights hesapla
    class_weight_dict = compute_realistic_class_weights(y_train)
    
    # 4. GERÇEKÇİ pipeline oluştur
    model = create_realistic_pipeline(important_features, class_weight_dict)
    
    # 5. GERÇEKÇİ Hiperparametre optimizasyonu
    param_distributions = {
        'lgbm__learning_rate': [0.03, 0.05, 0.07],
        'lgbm__max_depth': [2, 3, 4],
        'lgbm__num_leaves': [5, 7, 9],
        'lgbm__min_child_samples': [20, 25, 30],
        'lgbm__subsample': [0.6, 0.7, 0.8],
        'lgbm__colsample_bytree': [0.6, 0.7, 0.8],
        'lgbm__reg_alpha': [1.0, 2.0, 3.0],
        'lgbm__reg_lambda': [1.0, 2.0, 3.0],
        'lgbm__n_estimators': [100, 150, 200]
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    print("\n🎯 REALISTIC Hiperparametre Optimizasyonu...")
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=20,
        cv=tscv,
        scoring='balanced_accuracy',
        n_jobs=N_JOBS,
        verbose=1,
        random_state=RANDOM_STATE,
        return_train_score=True
    )
    
    random_search.fit(X_train_selected, y_train)
    
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    
    print(f"\n🏆 En İyi Parametreler: {best_params}")
    print(f"🏆 En İyi CV Skoru: {best_score:.4f}")
    
    # 6. Final modeli EARLY STOPPING ile eğit
    print("\n🚀 Final model eğitimi (REALISTIC BALANCE ile)...")
    
    # Final modeli oluştur
    final_pipeline = create_realistic_pipeline(important_features, class_weight_dict)
    
    # Best parametreleri set et
    for param_name, param_value in best_params.items():
        final_pipeline.set_params(**{param_name: param_value})
    
    # Early stopping için pipeline'ı manuel eğit
    preprocessor = final_pipeline.named_steps['preprocessor']
    lgbm = final_pipeline.named_steps['lgbm']
    
    # Veriyi preprocessing et
    X_train_processed = preprocessor.fit_transform(X_train_selected)
    X_val_processed = preprocessor.transform(X_val_selected)
    
    # Early stopping ile eğit
    lgbm.set_params(
        n_estimators=300,
        early_stopping_rounds=50,  # Daha fazla early stopping
        verbose=20
    )
    
    lgbm.fit(
        X_train_processed, y_train,
        eval_set=[(X_val_processed, y_val)],
        eval_metric='multi_logloss',
        callbacks=[lgb.early_stopping(50), lgb.log_evaluation(20)]
    )
    
    # Pipeline'ı tekrar oluştur
    final_model = Pipeline([
        ('preprocessor', preprocessor),
        ('lgbm', lgbm)
    ])
    
    # 7. Model değerlendirme
    print("\n📊 Kapsamlı Model Değerlendirme:")
    evaluate_realistic_model(final_model, X_test_selected, y_test, X_train_selected, y_train)
    
    # 8. Modeli kaydet
    save_realistic_model(final_model, important_features, best_params)
    
    return final_model, important_features

def evaluate_realistic_model(model, X_test, y_test, X_train, y_train):
    """GERÇEKÇİ model değerlendirme"""
    
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    # Genel metrikler
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average='weighted')
    test_balanced_accuracy = balanced_accuracy_score(y_test, y_pred_test)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    
    accuracy_gap = train_accuracy - test_accuracy
    
    # Class-based metrikler
    class_report = classification_report(y_test, y_pred_test, output_dict=True)
    homewin_recall = class_report['1']['recall']
    awaywin_recall = class_report['2']['recall']
    draw_recall = class_report['0']['recall']
    
    print(f"📈 Test Accuracy: {test_accuracy:.4f}")
    print(f"📈 Test F1-Score: {test_f1:.4f}")
    print(f"📈 Balanced Accuracy: {test_balanced_accuracy:.4f}")
    print(f"🎯 HomeWin Recall: {homewin_recall:.4f}")
    print(f"🎯 AwayWin Recall: {awaywin_recall:.4f}")
    print(f"🎯 Draw Recall: {draw_recall:.4f}")
    print(f"🏋️ Train Accuracy: {train_accuracy:.4f}")
    print(f"📊 Accuracy Gap (Overfitting): {accuracy_gap:.4f}")
    
    # GERÇEKÇİ BAŞARI KRİTERLERİ
    targets_achieved = 0
    total_targets = 5
    
    # Accuracy - GERÇEKÇİ (Bundesliga için %55-65 makul)
    if test_accuracy >= 0.55:
        print("✅ HEDEF BAŞARILDI: Accuracy > 0.55")
        targets_achieved += 1
    elif test_accuracy >= 0.50:
        print("⚠️ KISMEN BAŞARILI: Accuracy > 0.50")
        targets_achieved += 0.5
    else:
        print(f"⚠️ HEDEF TUTMADI: Accuracy = {test_accuracy:.4f} (hedef: 0.55)")
    
    # HomeWin recall - Bundesliga'da yüksek olmalı
    if homewin_recall >= 0.55:
        print("✅ HEDEF BAŞARILDI: HomeWin recall > 0.55")
        targets_achieved += 1
    elif homewin_recall >= 0.50:
        print("⚠️ KISMEN BAŞARILI: HomeWin recall > 0.50")
        targets_achieved += 0.5
    else:
        print(f"⚠️ HEDEF TUTMADI: HomeWin recall = {homewin_recall:.4f} (hedef: 0.55)")
    
    # Draw recall - GERÇEKÇİ (Bundesliga'da ~%25)
    if 0.20 <= draw_recall <= 0.40:
        print("✅ HEDEF BAŞARILDI: Draw recall makul aralıkta (0.20-0.40)")
        targets_achieved += 1
    else:
        print(f"⚠️ HEDEF TUTMADI: Draw recall = {draw_recall:.4f} (makul aralık: 0.20-0.40)")
    
    # AwayWin recall - Makul
    if awaywin_recall >= 0.40:
        print("✅ HEDEF BAŞARILDI: AwayWin recall > 0.40")
        targets_achieved += 1
    elif awaywin_recall >= 0.35:
        print("⚠️ KISMEN BAŞARILI: AwayWin recall > 0.35")
        targets_achieved += 0.5
    else:
        print(f"⚠️ HEDEF TUTMADI: AwayWin recall = {awaywin_recall:.4f} (hedef: 0.40)")
    
    # Overfitting - MAKUL
    if accuracy_gap <= 0.08:
        print("✅ HEDEF BAŞARILDI: Overfitting gap < 0.08")
        targets_achieved += 1
    elif accuracy_gap <= 0.12:
        print("⚠️ KISMEN BAŞARILI: Overfitting gap < 0.12")
        targets_achieved += 0.5
    else:
        print(f"⚠️ HEDEF TUTMADI: Overfitting gap = {accuracy_gap:.4f} (hedef: 0.08)")
    
    print(f"🎯 Toplam Başarı: {targets_achieved:.1f}/5")
    
    # PERFORMANS ANALİZİ
    print(f"\n🏆 BUNDESLIGA PERFORMANS RAPORU:")
    print(f"📊 Tahmini Dağılım: Draw: {draw_recall:.1%}, HomeWin: {homewin_recall:.1%}, AwayWin: {awaywin_recall:.1%}")
    print(f"📈 Beklenen Dağılım: Draw: ~25%, HomeWin: ~45%, AwayWin: ~30%")
    
    if test_accuracy >= 0.65:
        print("🎉 MÜKEMMEL: Üst düzey accuracy!")
    elif test_accuracy >= 0.60:
        print("✅ ÇOK İYİ: Bundesliga için harika accuracy!")
    elif test_accuracy >= 0.55:
        print("✅ İYİ: Bundesliga için makul accuracy!")
    elif test_accuracy >= 0.50:
        print("⚠️ ORTA: Geliştirme gerekli!")
    else:
        print("🔴 ZAYIF: Temel problemi çöz!")
    
    # Veri sızıntısı kontrolü
    if test_accuracy > 0.85:
        print("🚨 DİKKAT: Yüksek accuracy - veri sızıntısı olabilir!")
    elif test_accuracy > 0.75:
        print("ℹ️ BİLGİ: İyi accuracy - model sağlıklı görünüyor")
    
    print("\n🎯 Detaylı Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=['Draw', 'HomeWin', 'AwayWin']))

def save_realistic_model(model, important_features, best_params):
    """GERÇEKÇİ model kaydetme"""
    os.makedirs("models", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/bundesliga_model_realistic_{timestamp}.pkl"
    joblib.dump(model, model_path)
    
    feature_info = {
        'important_features': important_features,
        'best_params': best_params,
        'timestamp': datetime.now().isoformat(),
        'model_version': 'realistic_balance_v10.1'
    }
    joblib.dump(feature_info, "models/feature_info_realistic.pkl")
    
    print(f"\n💾 Model kaydedildi: {model_path}")

# ========== YARDIMCI FONKSİYONLAR ==========
def time_based_split(df, test_size=0.15, val_size=0.15):
    """Zaman bazlı split"""
    if 'Date' in df.columns:
        df_sorted = df.sort_values('Date').reset_index(drop=True)
    else:
        df_sorted = df.reset_index(drop=True)
    
    n = len(df_sorted)
    test_split_idx = int(n * (1 - test_size))
    val_split_idx = int(test_split_idx * (1 - val_size))
    
    train_df = df_sorted.iloc[:val_split_idx]
    val_df = df_sorted.iloc[val_split_idx:test_split_idx]
    test_df = df_sorted.iloc[test_split_idx:]
    
    print(f"📊 Split bilgisi: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df

# ========== ANA FONKSİYON ==========
def main():
    print("🏆 Bundesliga Tahmin Modeli - REALISTIC BALANCE v10.1")
    print("=" * 60)
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    try:
        model, important_features = train_realistic_model()
        
        print("\n🎉 REALISTIC BALANCE MODEL eğitimi başarıyla tamamlandı!")
        print(f"📋 Kullanılan önemli feature'lar: {len(important_features)}")
        
        print("\n🏆 GERÇEKÇİ BUNDESLIGA HEDEFLERİ:")
        print("✅ %55+ accuracy hedefi")
        print("✅ HomeWin recall > %55 hedefi") 
        print("✅ Draw recall %20-40 aralığı")
        print("✅ AwayWin recall > %40 hedefi")
        print("✅ Overfitting gap < %8 hedefi")
        print("🚫 Veri sızıntısı önlendi")
        print("🎯 Bundesliga pattern'lerine uyum")
        
    except Exception as e:
        print(f"❌ Model eğitimi sırasında hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()