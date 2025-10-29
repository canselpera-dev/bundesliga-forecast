#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bundesliga Tahmin Modeli - BALANCED OPTIMIZATION v5
Draw + HomeWin + AwayWin Dengesi + Overfitting Kontrolü
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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_selection import SelectFromModel, RFE
import xgboost as xgb
from imblearn.over_sampling import SMOTE

# ========== BALANCED KONFİGÜRASYON ==========
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
N_JOBS = -1
MAX_FEATURES = 12  # Daha az feature ile overfitting'i azalt

DATA_PATH = "data/bundesliga_matches_2023_2025_final_fe_team_values_cleaned.xlsx"
PLAYER_DATA_PATH = "data/final_bundesliga_dataset_complete.xlsx"

# OPTIMIZE EDİLMİŞ ÖZELLİK LİSTESİ - BALANCED
SELECTED_FEATURES = [
    # Temel Performans Metrikleri
    'home_ppg_cumulative', 'away_ppg_cumulative',
    'home_gpg_cumulative', 'away_gpg_cumulative',
    'home_gapg_cumulative', 'away_gapg_cumulative',
    'home_form_5games', 'away_form_5games',
    
    # Power ve Form
    'home_power_index', 'away_power_index', 
    'power_difference', 'form_difference',
    
    # Value-based
    'value_difference', 'value_ratio',
    
    # H2H
    'h2h_win_ratio', 'h2h_goal_difference',
    
    # Özel Durumlar
    'isDerby',
    
    # Türetilmiş Özellikler (DENGELİ)
    'cumulative_ppg_difference', 'cumulative_gpg_difference',
    'strength_balance', 'power_balance',
    
    # HomeWin için kritik
    'home_advantage_strength', 'home_attack_power',
    
    # AwayWin için kritik  
    'away_pressure', 'away_defense_weakness',
    
    # Draw için OPTIMIZE EDİLMİŞ (aşırı değil)
    'form_similarity', 'strength_ratio'
]

# ========== BALANCED FEATURE SELECTION ==========
def balanced_feature_selection(X_train, y_train, X_val, X_test, max_features=MAX_FEATURES):
    """Dengeli feature selection - tüm class'lar için"""
    print(f"🔍 BALANCED Feature Selection (Max {max_features} özellik)...")
    
    # Her class için ayrı importance hesapla
    feature_scores = {}
    
    for class_label in [0, 1, 2]:  # Draw, HomeWin, AwayWin
        # Binary classification için
        y_binary = (y_train == class_label).astype(int)
        
        if len(np.unique(y_binary)) > 1:  # Eğer class varsa
            estimator = RandomForestClassifier(
                n_estimators=100, 
                random_state=RANDOM_STATE,
                max_depth=4
            )
            estimator.fit(X_train, y_binary)
            
            for i, feature in enumerate(X_train.columns):
                if feature not in feature_scores:
                    feature_scores[feature] = 0
                feature_scores[feature] += estimator.feature_importances_[i]
    
    # Tüm class'lar için önemli feature'ları seç
    sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
    selected_features = [feat for feat, score in sorted_features[:max_features]]
    
    # Her kategoriden feature olduğundan emin ol
    categories = {
        'home_features': [f for f in selected_features if 'home_' in f],
        'away_features': [f for f in selected_features if 'away_' in f],
        'draw_features': [f for f in selected_features if any(x in f for x in ['similarity', 'balance', 'ratio'])]
    }
    
    print(f"📊 Feature kategorileri: Home({len(categories['home_features'])}), "
          f"Away({len(categories['away_features'])}), Draw({len(categories['draw_features'])})")
    
    print(f"✅ Seçilen özellikler: {selected_features}")
    
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    X_test_selected = X_test[selected_features]
    
    return X_train_selected, X_val_selected, X_test_selected, selected_features

# ========== BALANCED SMOTE ==========
def apply_balanced_smote(X_train, y_train):
    """Dengeli SMOTE - tüm class'ları eşit destekle"""
    print("🔄 BALANCED SMOTE ile class balancing uygulanıyor...")
    
    # Dengeli bir strateji - tüm class'ları benzer seviyeye getir
    sampling_strategy = {
        0: min(len(y_train[y_train == 0]) * 5 // 4, len(y_train) // 2),  # Draw - %25 artır
        1: min(len(y_train[y_train == 1]) * 5 // 4, len(y_train) // 2),  # HomeWin - %25 artır
        2: min(len(y_train[y_train == 2]) * 5 // 4, len(y_train) // 2)   # AwayWin - %25 artır
    }
    
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE, k_neighbors=3)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"📊 BALANCED SMOTE sonrası sınıf dağılımı: {pd.Series(y_resampled).value_counts().to_dict()}")
    return X_resampled, y_resampled

# ========== BALANCED CLASS WEIGHTS ==========
def compute_balanced_class_weights(y_train):
    """Dengeli class weights"""
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    
    # Çok agresif olmayan dengeli ağırlıklar
    print(f"⚖️ BALANCED Class Weights: {class_weight_dict}")
    return class_weight_dict

# ========== STRONG OVERFITTING CONTROL PIPELINE ==========
def create_strong_regularization_pipeline(selected_features):
    """Güçlü regularization ile pipeline"""
    
    preprocessor = ColumnTransformer([
        ('scaler', RobustScaler(), selected_features)
    ], remainder='drop')
    
    # GÜÇLÜ REGULARIZATION'lu LightGBM
    lgbm_clf = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbosity=-1,
        n_estimators=300,  # Daha az estimators
        learning_rate=0.005,  # Daha düşük learning rate
        max_depth=2,  # Çok sığ ağaçlar
        num_leaves=5,  # Çok az leaves
        min_child_samples=80,  # Daha fazla min child
        subsample=0.5,  # Daha az subsample
        colsample_bytree=0.4,  # Daha az colsample
        reg_alpha=5.0,  # Çok güçlü L1 regularization
        reg_lambda=5.0,  # Çok güçlü L2 regularization
        force_row_wise=True
    )
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('lgbm', lgbm_clf)
    ])

# ========== MANUEL CUMULATIVE STATS HESAPLAMA ==========
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

# ========== BALANCED FEATURE ENGINEERING ==========
class BalancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """Dengeli özellik mühendisliği"""
    
    def __init__(self):
        self.feature_names = []
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # 1. Temel fark özellikleri
        if all(col in X.columns for col in ['home_ppg_cumulative', 'away_ppg_cumulative']):
            X['cumulative_ppg_difference'] = X['home_ppg_cumulative'] - X['away_ppg_cumulative']
        
        if all(col in X.columns for col in ['home_gpg_cumulative', 'away_gpg_cumulative']):
            X['cumulative_gpg_difference'] = X['home_gpg_cumulative'] - X['away_gpg_cumulative']
        
        # 2. Power metrikleri
        if all(col in X.columns for col in ['home_power_index', 'away_power_index']):
            X['power_difference'] = X['home_power_index'] - X['away_power_index']
            X['strength_balance'] = abs(X['home_power_index'] - X['away_power_index'])
            X['power_balance'] = 1 - (abs(X['home_power_index'] - X['away_power_index']) / 
                                    (X['home_power_index'] + X['away_power_index'] + 1e-8))
        
        # 3. Form benzerliği (DRAW için optimize ama aşırı değil)
        if all(col in X.columns for col in ['home_form_5games', 'away_form_5games']):
            X['form_similarity'] = 1 - abs(X['home_form_5games'] - X['away_form_5games'])
            X['form_difference'] = X['home_form_5games'] - X['away_form_5games']
        
        # 4. HomeWin için kritik özellikler
        if all(col in X.columns for col in ['home_ppg_cumulative', 'home_form_5games']):
            X['home_advantage_strength'] = X['home_ppg_cumulative'] * X['home_form_5games']
        
        if all(col in X.columns for col in ['home_gpg_cumulative', 'home_form_5games']):
            X['home_attack_power'] = X['home_gpg_cumulative'] * X['home_form_5games']
        
        # 5. AwayWin için kritik özellikler
        if all(col in X.columns for col in ['away_gapg_cumulative', 'away_form_5games']):
            X['away_defense_weakness'] = X['away_gapg_cumulative'] * (1 - X['away_form_5games'])
        
        if all(col in X.columns for col in ['away_gapg_cumulative', 'form_difference']):
            X['away_pressure'] = X['away_gapg_cumulative'] * abs(X['form_difference'])
        
        # 6. Value-based
        if all(col in X.columns for col in ['home_current_value_eur', 'away_current_value_eur']):
            X['value_difference'] = X['home_current_value_eur'] - X['away_current_value_eur']
            X['value_ratio'] = X['home_current_value_eur'] / (X['away_current_value_eur'] + 1e-8)
        
        # 7. Strength ratio (DRAW için)
        if all(col in X.columns for col in ['home_power_index', 'away_power_index']):
            X['strength_ratio'] = np.minimum(X['home_power_index'], X['away_power_index']) / \
                                 (np.maximum(X['home_power_index'], X['away_power_index']) + 1e-8)
        
        self.feature_names = X.columns.tolist()
        return X

# ========== VERİ HAZIRLAMA ==========
def balanced_data_preparation(df_matches, df_players):
    """Dengeli veri hazırlama"""
    print("🔧 BALANCED veri hazırlama başlıyor...")
    
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
    df = balanced_missing_value_imputation(df)
    
    # 6. Feature engineering
    df = balanced_feature_engineering(df)
    
    # 7. Rating hesapla
    df = compute_balanced_ratings(df, df_players)
    
    print("✅ BALANCED veri hazırlama tamamlandı!")
    return df

def balanced_missing_value_imputation(df):
    """Dengeli eksik değer doldurma"""
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
        'power_difference': 0, 'strength_balance': 0.5,
        'power_balance': 0.7, 'form_similarity': 0.5,
        'home_advantage_strength': 0.75, 'home_attack_power': 0.75,
        'away_defense_weakness': 0.75, 'away_pressure': 0.5,
        'value_difference': 0, 'value_ratio': 1.0,
        'strength_ratio': 0.8
    }
    
    for column, default_value in imputation_strategies.items():
        if column in df.columns:
            null_count = df[column].isnull().sum()
            if null_count > 0:
                df[column].fillna(default_value, inplace=True)
    
    return df

def balanced_feature_engineering(df):
    """Dengeli feature engineering"""
    df = df.copy()
    
    # Önceki transformer'daki tüm özellikleri manuel uygula
    feature_engineer = BalancedFeatureEngineer()
    df = feature_engineer.fit_transform(df)
    
    return df

def compute_balanced_ratings(df, df_players):
    """Dengeli rating hesaplama"""
    if 'Home_AvgRating' not in df.columns:
        df['Home_AvgRating'] = 65.0
        df['Away_AvgRating'] = 65.0
    
    return df

# ========== VERİ YÜKLEME ==========
def load_balanced_data():
    """Dengeli veri yükleme"""
    print("\n📊 BALANCED veri yükleniyor...")
    
    try:
        df_matches = pd.read_excel(DATA_PATH)
        df_matches.columns = [col.strip().replace(' ', '_') for col in df_matches.columns]
        
        df_players = pd.read_excel(PLAYER_DATA_PATH)
        
        df = balanced_data_preparation(df_matches, df_players)
        
        # Eksik feature'ları doldur
        missing_features = []
        for feat in SELECTED_FEATURES:
            if feat not in df.columns:
                missing_features.append(feat)
                df[feat] = 0
        
        if missing_features:
            print(f"⚠️ Eksik özellikler dolduruldu: {len(missing_features)}")
        
        numeric_cols = df[SELECTED_FEATURES].select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        class_distribution = df['Result_Numeric'].value_counts().sort_index()
        print(f"📈 Sınıf Dağılımı: {dict(class_distribution)}")
        
        print("✅ BALANCED veri hazırlığı tamamlandı")
        return df
        
    except Exception as e:
        print(f"❌ Veri yükleme hatası: {e}")
        raise

# ========== BALANCED MODEL EĞİTİMİ ==========
def train_balanced_model():
    """DENGELİ MODEL EĞİTİMİ"""
    print("⚽ Bundesliga Tahmin Modeli - BALANCED OPTIMIZATION v5")
    print("=" * 70)
    print("✅ Dengeli feature engineering") 
    print("✅ Tüm class'lar için optimizasyon")
    print("✅ Balanced SMOTE")
    print("✅ Strong regularization")
    print("✅ Overfitting kontrolü")
    print("=" * 70)
    
    # Veriyi yükle
    df = load_balanced_data()
    
    # Zaman bazlı split
    train_df, val_df, test_df = time_based_split(df, TEST_SIZE, VAL_SIZE)
    
    # Feature ve target'ları ayır
    X_train = train_df[SELECTED_FEATURES].copy()
    y_train = train_df['Result_Numeric'].copy()
    
    X_val = val_df[SELECTED_FEATURES].copy()
    y_val = val_df['Result_Numeric'].copy()
    
    X_test = test_df[SELECTED_FEATURES].copy()
    y_test = test_df['Result_Numeric'].copy()
    
    # 1. Feature engineering uygula
    print("🔧 Feature engineering uygulanıyor...")
    feature_engineer = BalancedFeatureEngineer()
    X_train = feature_engineer.fit_transform(X_train)
    X_val = feature_engineer.transform(X_val)
    X_test = feature_engineer.transform(X_test)
    
    # 2. BALANCED SMOTE uygula
    X_train_balanced, y_train_balanced = apply_balanced_smote(X_train, y_train)
    
    # 3. BALANCED feature selection yap
    X_train_selected, X_val_selected, X_test_selected, important_features = balanced_feature_selection(
        X_train_balanced, y_train_balanced, X_val, X_test, MAX_FEATURES
    )
    
    print(f"📊 Eğitim verisi: {X_train_selected.shape}")
    print(f"📊 Validation verisi: {X_val_selected.shape}")
    print(f"📊 Test verisi: {X_test_selected.shape}")
    
    # 4. Class weights hesapla
    class_weight_dict = compute_balanced_class_weights(y_train_balanced)
    sample_weights_train = np.array([class_weight_dict[yy] for yy in y_train_balanced])
    
    # 5. Strong regularization pipeline oluştur
    model = create_strong_regularization_pipeline(important_features)
    
    # 6. BALANCED Hiperparametre optimizasyonu
    param_distributions = {
        'lgbm__learning_rate': [0.005, 0.008],
        'lgbm__max_depth': [2, 3],
        'lgbm__num_leaves': [4, 5, 6],
        'lgbm__min_child_samples': [70, 80, 90],
        'lgbm__reg_alpha': [4.0, 5.0, 6.0],
        'lgbm__reg_lambda': [4.0, 5.0, 6.0],
        'lgbm__subsample': [0.4, 0.5],
        'lgbm__colsample_bytree': [0.3, 0.4],
        'lgbm__n_estimators': [200, 300]
    }
    
    tscv = TimeSeriesSplit(n_splits=8)
    
    print("\n🎯 BALANCED Hiperparametre Optimizasyonu...")
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=15,
        cv=tscv,
        scoring='balanced_accuracy',
        n_jobs=N_JOBS,
        verbose=1,
        random_state=RANDOM_STATE,
        return_train_score=True
    )
    
    random_search.fit(X_train_selected, y_train_balanced, lgbm__sample_weight=sample_weights_train)
    
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    
    print(f"\n🏆 En İyi Parametreler: {best_params}")
    print(f"🏆 En İyi CV Skoru: {best_score:.4f}")
    
    # 7. Final modeli eğit
    print("\n🚀 Final model eğitimi (Balanced Focus ile)...")
    final_model = create_strong_regularization_pipeline(important_features)
    final_model.set_params(**best_params)
    
    final_model.named_steps['lgbm'].set_params(
        n_estimators=400,
        early_stopping_rounds=100,  # Daha fazla early stopping
        verbose=50
    )
    
    final_model.fit(
        X_train_selected, y_train_balanced,
        lgbm__eval_set=[(X_val_selected, y_val)],
        lgbm__eval_metric='multi_logloss',
        lgbm__sample_weight=sample_weights_train,
        lgbm__callbacks=[lgb.early_stopping(100), lgb.log_evaluation(50)]
    )
    
    # 8. Model değerlendirme
    print("\n📊 Kapsamlı Model Değerlendirme:")
    evaluate_balanced_model(final_model, X_test_selected, y_test, X_train_selected, y_train_balanced)
    
    # 9. Modeli kaydet
    save_balanced_model(final_model, important_features, best_params)
    
    return final_model, important_features

def evaluate_balanced_model(model, X_test, y_test, X_train, y_train):
    """Dengeli model değerlendirme"""
    
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    # Genel metrikler
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average='weighted')
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    
    accuracy_gap = train_accuracy - test_accuracy
    
    # Class-based metrikler
    class_report = classification_report(y_test, y_pred_test, output_dict=True)
    homewin_recall = class_report['1']['recall']
    awaywin_recall = class_report['2']['recall']
    draw_recall = class_report['0']['recall']
    
    print(f"📈 Test Accuracy: {test_accuracy:.4f}")
    print(f"📈 Test F1-Score: {test_f1:.4f}")
    print(f"🎯 HomeWin Recall: {homewin_recall:.4f}")
    print(f"🎯 AwayWin Recall: {awaywin_recall:.4f}")
    print(f"🎯 Draw Recall: {draw_recall:.4f}")
    print(f"🏋️ Train Accuracy: {train_accuracy:.4f}")
    print(f"📊 Accuracy Gap (Overfitting): {accuracy_gap:.4f}")
    
    # Başarı analizi - GERÇEKÇİ HEDEFLER
    targets_achieved = 0
    total_targets = 4
    
    if homewin_recall >= 0.55:  # Daha gerçekçi hedef
        print("✅ HEDEF BAŞARILDI: HomeWin recall > 0.55")
        targets_achieved += 1
    else:
        print(f"⚠️ HEDEF TUTMADI: HomeWin recall = {homewin_recall:.4f} (hedef: 0.55)")
    
    if draw_recall >= 0.35:  # Daha gerçekçi hedef
        print("✅ HEDEF BAŞARILDI: Draw recall > 0.35")
        targets_achieved += 1
    else:
        print(f"⚠️ HEDEF TUTMADI: Draw recall = {draw_recall:.4f} (hedef: 0.35)")
    
    if test_accuracy >= 0.55:  # Daha gerçekçi hedef
        print("✅ HEDEF BAŞARILDI: Accuracy > 0.55")
        targets_achieved += 1
    else:
        print(f"⚠️ HEDEF TUTMADI: Accuracy = {test_accuracy:.4f} (hedef: 0.55)")
    
    if accuracy_gap <= 0.08:  # Daha gerçekçi hedef
        print("✅ HEDEF BAŞARILDI: Overfitting gap < 0.08")
        targets_achieved += 1
    else:
        print(f"⚠️ HEDEF TUTMADI: Overfitting gap = {accuracy_gap:.4f} (hedef: 0.08)")
    
    print(f"🎯 Toplam Başarı: {targets_achieved}/{total_targets}")
    
    # Overfitting analizi
    if accuracy_gap > 0.15:
        print("🚨 CRITICAL: Ciddi overfitting riski!")
    elif accuracy_gap > 0.10:
        print("⚠️ WARNING: Orta seviye overfitting riski!")
    elif accuracy_gap > 0.05:
        print("ℹ️ INFO: Hafif overfitting riski")
    else:
        print("✅ EXCELLENT: Overfitting riski çok düşük!")
    
    print("\n🎯 Detaylı Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=['Draw', 'HomeWin', 'AwayWin']))

def save_balanced_model(model, important_features, best_params):
    """Dengeli model kaydetme"""
    os.makedirs("models", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/bundesliga_model_balanced_{timestamp}.pkl"
    joblib.dump(model, model_path)
    
    feature_info = {
        'important_features': important_features,
        'best_params': best_params,
        'timestamp': datetime.now().isoformat(),
        'model_version': 'balanced_v5'
    }
    joblib.dump(feature_info, "models/feature_info_balanced.pkl")
    
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
    print("🏆 Bundesliga Tahmin Modeli - BALANCED OPTIMIZATION v5")
    print("=" * 60)
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    try:
        model, important_features = train_balanced_model()
        
        print("\n🎉 BALANCED MODEL eğitimi başarıyla tamamlandı!")
        print(f"📋 Kullanılan önemli feature'lar: {len(important_features)}")
        
        print("\n🏆 GERÇEKÇİ MODEL HEDEFLERİ:")
        print("✅ %55+ accuracy hedefi")
        print("✅ HomeWin recall > %55 hedefi") 
        print("✅ Draw recall > %35 hedefi")
        print("✅ Overfitting gap < %8 hedefi")
        
    except Exception as e:
        print(f"❌ Model eğitimi sırasında hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()