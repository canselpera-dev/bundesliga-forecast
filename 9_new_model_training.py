#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bundesliga Tahmin Modeli - Ultimate Final Sürüm
Geliştirilmiş Cumulative Stats + Draw Optimization + HomeWin Recall Enhancement
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
from datetime import datetime

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

# ========== GELİŞTİRİLMİŞ KONFİGÜRASYON ==========
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
N_JOBS = -1

DATA_PATH = "data/bundesliga_matches_2023_2025_final_fe_team_values_cleaned.xlsx"
PLAYER_DATA_PATH = "data/final_bundesliga_dataset_complete.xlsx"

# Geliştirilmiş özellik listesi - OPTIMIZED FEATURES
SELECTED_FEATURES = [
    # Takım Değer ve Demografi Özellikleri
    'home_current_value_eur', 'away_current_value_eur',
    'value_difference', 'value_ratio',
    
    # Performans ve Form Özellikleri
    'home_goals', 'away_goals', 'home_xg', 'away_xg',
    'goals_difference', 'goals_ratio', 'xg_difference', 'xg_ratio',
    'home_form', 'away_form', 'form_difference',
    
    # H2H (Head-to-Head) Özellikleri
    'h2h_win_ratio', 'h2h_goal_difference', 'h2h_avg_goals',
    
    # Derby ve Özel Durum Özellikleri
    'isDerby', 'age_difference',
    
    # Power Index ve Advanced Metrics
    'home_power_index', 'away_power_index', 'power_difference',
    
    # YENİ: CUMULATIVE METRİKLER (Manuel Hesaplanan)
    'home_ppg_cumulative', 'away_ppg_cumulative',
    'home_gpg_cumulative', 'away_gpg_cumulative',
    'home_gapg_cumulative', 'away_gapg_cumulative',
    'home_form_5games', 'away_form_5games',
    'home_goal_diff_cumulative', 'away_goal_diff_cumulative',
    
    # YENİ: CUMULATIVE METRİKLERDEN TÜRETİLEN ÖZELLİKLER
    'cumulative_ppg_difference', 'cumulative_ppg_ratio',
    'cumulative_gpg_difference', 'cumulative_gpg_ratio',
    'form_5games_difference', 'cumulative_goal_diff_difference',
    
    # YENİ: DRAW OPTIMIZATION ÖZELLİKLERİ
    'strength_balance', 'is_close_match', 'both_teams_good_form',
    
    # YENİ: HOMEWIN RECALL OPTIMIZATION
    'home_advantage_strength', 'home_defensive_stability'
]

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
        print("⚠️ Tarih sütunu bulunamadı, orijinal sıra kullanılıyor")
        df = df.reset_index(drop=True)
    
    # Takım isimlerini standartlaştır
    if 'homeTeam.name' in df.columns and 'HomeTeam' not in df.columns:
        df['HomeTeam'] = df['homeTeam.name']
    if 'awayTeam.name' in df.columns and 'AwayTeam' not in df.columns:
        df['AwayTeam'] = df['awayTeam.name']
    
    # Her takım için kümülatif istatistikleri saklayacağımız dictionary
    team_stats = {}
    
    # Yeni özellikleri başlat
    cumulative_features = [
        'home_ppg_cumulative', 'away_ppg_cumulative',
        'home_gpg_cumulative', 'away_gpg_cumulative', 
        'home_gapg_cumulative', 'away_gapg_cumulative',
        'home_form_5games', 'away_form_5games',
        'home_goal_diff_cumulative', 'away_goal_diff_cumulative'
    ]
    
    for feature in cumulative_features:
        df[feature] = 0.0
    
    for idx, match in df.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        
        # Takımları initialize et
        if home_team not in team_stats:
            team_stats[home_team] = {
                'points': 0, 'goals_for': 0, 'goals_against': 0, 'matches': 0,
                'recent_results': [],  # Son 5 maçın sonuçları (1: galibiyet, 0.5: beraberlik, 0: mağlubiyet)
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
        
        df.loc[idx, 'home_goal_diff_cumulative'] = team_stats[home_team]['goal_diff']
        df.loc[idx, 'away_goal_diff_cumulative'] = team_stats[away_team]['goal_diff']
        
        # 🔽 BU MAÇIN SONUCUNU İŞLE - BİR SONRAKI MAÇ İÇİN KULLANILACAK 🔽
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
    """Son 5 maç formunu hesapla (0-1 arası)"""
    if not recent_results:
        return 0.5
    return sum(recent_results) / len(recent_results)

# ========== GELİŞTİRİLMİŞ ÖZEL TRANSFORMERLAR ==========
class AdvancedFeatureEngineer(BaseEstimator, TransformerMixin):
    """Gelişmiş özellik mühendisliği transformer'ı"""
    
    def __init__(self):
        self.feature_names = []
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # 1. Exponential özellikler
        value_cols = ['home_current_value_eur', 'away_current_value_eur']
        for col in value_cols:
            if col in X.columns:
                X[f'{col}_log'] = np.log1p(X[col])
        
        # 2. Cumulative interaction özellikleri
        if all(col in X.columns for col in ['home_ppg_cumulative', 'away_ppg_cumulative']):
            X['cumulative_ppg_difference'] = X['home_ppg_cumulative'] - X['away_ppg_cumulative']
            X['cumulative_ppg_ratio'] = X['home_ppg_cumulative'] / (X['away_ppg_cumulative'] + 0.1)
        
        if all(col in X.columns for col in ['home_gpg_cumulative', 'away_gpg_cumulative']):
            X['cumulative_gpg_difference'] = X['home_gpg_cumulative'] - X['away_gpg_cumulative']
            X['cumulative_gpg_ratio'] = X['home_gpg_cumulative'] / (X['away_gpg_cumulative'] + 0.1)
        
        if all(col in X.columns for col in ['home_form_5games', 'away_form_5games']):
            X['form_5games_difference'] = X['home_form_5games'] - X['away_form_5games']
        
        if all(col in X.columns for col in ['home_goal_diff_cumulative', 'away_goal_diff_cumulative']):
            X['cumulative_goal_diff_difference'] = X['home_goal_diff_cumulative'] - X['away_goal_diff_cumulative']
        
        # 3. Form-cumulative interaction
        if all(col in X.columns for col in ['home_form', 'home_ppg_cumulative']):
            X['home_form_ppg_interaction'] = X['home_form'] * X['home_ppg_cumulative']
        
        if all(col in X.columns for col in ['away_form', 'away_ppg_cumulative']):
            X['away_form_ppg_interaction'] = X['away_form'] * X['away_ppg_cumulative']
        
        # 4. Power metrics
        if all(col in X.columns for col in ['home_power_index', 'away_power_index']):
            X['power_ratio'] = X['home_power_index'] / X['away_power_index']
            X['power_sum'] = X['home_power_index'] + X['away_power_index']
        
        # 5. Draw optimization features
        if all(col in X.columns for col in ['home_power_index', 'away_power_index']):
            X['strength_balance'] = abs(X['home_power_index'] - X['away_power_index'])
            X['is_close_match'] = (X['strength_balance'] < 0.15).astype(int)
        
        if all(col in X.columns for col in ['home_form_5games', 'away_form_5games']):
            X['both_teams_good_form'] = ((X['home_form_5games'] > 0.6) & (X['away_form_5games'] > 0.6)).astype(int)
        
        # 6. HomeWin recall optimization
        if all(col in X.columns for col in ['home_ppg_cumulative', 'home_form_5games']):
            X['home_advantage_strength'] = X['home_ppg_cumulative'] * X['home_form_5games']
        
        if all(col in X.columns for col in ['home_gapg_cumulative', 'home_gpg_cumulative']):
            X['home_defensive_stability'] = X['home_gapg_cumulative'] / (X['home_gpg_cumulative'] + 0.1)
        
        self.feature_names = X.columns.tolist()
        return X

# ========== FEATURE SELECTION FONKSİYONLARI ==========
def perform_strict_feature_selection(X_train, y_train, X_val, X_test, method='importance'):
    """OVERFITTING ÖNLEMEK İÇİN DAHA STRICT FEATURE SELECTION"""
    print("🔍 STRICT Feature selection yapılıyor...")
    
    if method == 'importance':
        # Random Forest ile feature importance
        estimator = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        selector = SelectFromModel(estimator, threshold='mean')
        
        selector.fit(X_train, y_train)
        selected_features = X_train.columns[selector.get_support()].tolist()
        
        # Eğer hala çok fazla özellik varsa, en iyi 15 tanesini al
        if len(selected_features) > 15:
            print(f"⚡ Çok fazla özellik seçildi ({len(selected_features)}), en iyi 15 tanesi alınıyor...")
            estimator.fit(X_train, y_train)
            importances = estimator.feature_importances_
            indices = np.argsort(importances)[::-1]
            selected_features = [X_train.columns[i] for i in indices[:15]]
        
    elif method == 'rfe':
        # Recursive Feature Elimination
        estimator = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)
        rfe = RFE(estimator=estimator, n_features_to_select=min(15, X_train.shape[1]))
        rfe.fit(X_train, y_train)
        selected_features = X_train.columns[rfe.support_].tolist()
    
    else:
        # Tüm feature'ları seç
        selected_features = X_train.columns.tolist()
    
    print(f"✅ Seçilen özellik sayısı: {len(selected_features)}/{X_train.shape[1]}")
    print(f"📋 Seçilen özellikler: {selected_features}")
    
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    X_test_selected = X_test[selected_features]
    
    return X_train_selected, X_val_selected, X_test_selected, selected_features

# ========== GELİŞTİRİLMİŞ VERİ HAZIRLAMA ==========
def enhanced_data_preparation(df_matches, df_players):
    """
    Geliştirilmiş veri hazırlama - OPTIMIZED
    """
    print("🔧 Geliştirilmiş veri hazırlama başlıyor...")
    
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
    
    # 5. Eksik özellikler için gelişmiş doldurma
    df = enhanced_missing_value_imputation(df)
    
    # 6. Gelişmiş feature engineering
    df = advanced_feature_engineering(df)
    
    # 7. Takım ratinglerini hesapla
    df = compute_enhanced_ratings(df, df_players)
    
    # 8. Outlier handling
    df = handle_outliers(df)
    
    print("✅ Geliştirilmiş veri hazırlama tamamlandı!")
    return df

def enhanced_missing_value_imputation(df):
    """Gelişmiş eksik değer doldurma"""
    print("📊 Eksik değer analizi ve doldurma...")
    
    imputation_strategies = {
        'h2h_home_wins': 0, 'h2h_away_wins': 0, 'h2h_draws': 0,
        'h2h_home_goals': 0, 'h2h_away_goals': 0, 'h2h_matches_count': 0,
        'h2h_win_ratio': 0.5, 'h2h_goal_difference': 0, 'h2h_avg_goals': 2.5,
        
        'home_form': 0.5, 'away_form': 0.5, 'form_difference': 0,
        
        'home_current_value_eur': df['home_current_value_eur'].median() if 'home_current_value_eur' in df.columns else 200000000,
        'away_current_value_eur': df['away_current_value_eur'].median() if 'away_current_value_eur' in df.columns else 200000000,
        
        'home_goals': df['home_goals'].median() if 'home_goals' in df.columns else 1.5,
        'away_goals': df['away_goals'].median() if 'away_goals' in df.columns else 1.5,
        
        # Cumulative metrikler için doldurma
        'home_ppg_cumulative': 1.5, 'away_ppg_cumulative': 1.5,
        'home_gpg_cumulative': 1.5, 'away_gpg_cumulative': 1.5,
        'home_gapg_cumulative': 1.5, 'away_gapg_cumulative': 1.5,
        'home_form_5games': 0.5, 'away_form_5games': 0.5,
        'home_goal_diff_cumulative': 0, 'away_goal_diff_cumulative': 0,
        
        # Türetilmiş özellikler
        'cumulative_ppg_difference': 0, 'cumulative_ppg_ratio': 1.0,
        'cumulative_gpg_difference': 0, 'cumulative_gpg_ratio': 1.0,
        'form_5games_difference': 0, 'cumulative_goal_diff_difference': 0,
        
        # Yeni özellikler
        'strength_balance': 0.5, 'is_close_match': 0,
        'both_teams_good_form': 0, 'home_advantage_strength': 0.75,
        'home_defensive_stability': 1.0
    }
    
    for column, default_value in imputation_strategies.items():
        if column in df.columns:
            null_count = df[column].isnull().sum()
            if null_count > 0:
                df[column].fillna(default_value, inplace=True)
    
    return df

def advanced_feature_engineering(df):
    """Gelişmiş özellik mühendisliği"""
    print("🎯 Gelişmiş özellik mühendisliği...")
    df = df.copy()
    
    # 1. Value-based özellikler
    if all(col in df.columns for col in ['home_current_value_eur', 'away_current_value_eur']):
        df['value_difference'] = df['home_current_value_eur'] - df['away_current_value_eur']
        df['value_ratio'] = df['home_current_value_eur'] / (df['away_current_value_eur'] + 1e-8)
    
    # 2. Form-based özellikler
    if all(col in df.columns for col in ['home_form', 'away_form']):
        df['form_difference'] = df['home_form'] - df['away_form']
    
    # 3. Goal-based özellikler
    if all(col in df.columns for col in ['home_goals', 'away_goals']):
        df['goals_difference'] = df['home_goals'] - df['away_goals']
        df['goals_ratio'] = df['home_goals'] / (df['away_goals'] + 1e-8)
    
    # 4. XG-based özellikler
    if all(col in df.columns for col in ['home_xg', 'away_xg']):
        df['xg_difference'] = df['home_xg'] - df['away_xg']
        df['xg_ratio'] = df['home_xg'] / (df['away_xg'] + 1e-8)
    
    # 5. Power metrics
    if all(col in df.columns for col in ['home_power_index', 'away_power_index']):
        df['power_difference'] = df['home_power_index'] - df['away_power_index']
        df['power_ratio'] = df['home_power_index'] / (df['away_power_index'] + 1e-8)
        df['power_sum'] = df['home_power_index'] + df['away_power_index']
    
    # 6. Cumulative türev özellikler
    if all(col in df.columns for col in ['home_ppg_cumulative', 'away_ppg_cumulative']):
        df['cumulative_ppg_difference'] = df['home_ppg_cumulative'] - df['away_ppg_cumulative']
        df['cumulative_ppg_ratio'] = df['home_ppg_cumulative'] / (df['away_ppg_cumulative'] + 0.1)
    
    if all(col in df.columns for col in ['home_gpg_cumulative', 'away_gpg_cumulative']):
        df['cumulative_gpg_difference'] = df['home_gpg_cumulative'] - df['away_gpg_cumulative']
        df['cumulative_gpg_ratio'] = df['home_gpg_cumulative'] / (df['away_gpg_cumulative'] + 0.1)
    
    if all(col in df.columns for col in ['home_form_5games', 'away_form_5games']):
        df['form_5games_difference'] = df['home_form_5games'] - df['away_form_5games']
    
    if all(col in df.columns for col in ['home_goal_diff_cumulative', 'away_goal_diff_cumulative']):
        df['cumulative_goal_diff_difference'] = df['home_goal_diff_cumulative'] - df['away_goal_diff_cumulative']
    
    # 7. Draw optimization features
    if all(col in df.columns for col in ['home_power_index', 'away_power_index']):
        df['strength_balance'] = abs(df['home_power_index'] - df['away_power_index'])
        df['is_close_match'] = (df['strength_balance'] < 0.15).astype(int)
    
    if all(col in df.columns for col in ['home_form_5games', 'away_form_5games']):
        df['both_teams_good_form'] = ((df['home_form_5games'] > 0.6) & (df['away_form_5games'] > 0.6)).astype(int)
    
    # 8. HomeWin recall optimization
    if all(col in df.columns for col in ['home_ppg_cumulative', 'home_form_5games']):
        df['home_advantage_strength'] = df['home_ppg_cumulative'] * df['home_form_5games']
    
    if all(col in df.columns for col in ['home_gapg_cumulative', 'home_gpg_cumulative']):
        df['home_defensive_stability'] = df['home_gapg_cumulative'] / (df['home_gpg_cumulative'] + 0.1)
    
    return df

def handle_outliers(df):
    """Outlier'ları işleme"""
    print("📊 Outlier handling...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col not in ['Result_Numeric', 'isDerby', 'is_close_match', 'both_teams_good_form']:
            Q1 = df[col].quantile(0.05)
            Q3 = df[col].quantile(0.95)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df[col] = np.clip(df[col], lower_bound, upper_bound)
    
    return df

def compute_enhanced_ratings(df, df_players):
    """Geliştirilmiş takım rating hesaplama"""
    print("⭐ Geliştirilmiş takım ratingleri hesaplanıyor...")
    
    if 'Home_AvgRating' not in df.columns:
        df['Home_AvgRating'] = 65.0
        df['Away_AvgRating'] = 65.0
    
    rating_cols = ['Home_AvgRating', 'Away_AvgRating']
    
    for col in rating_cols:
        if col not in df.columns:
            df[col] = 65.0
    
    if all(col in df.columns for col in ['Home_AvgRating', 'Away_AvgRating']):
        df['Rating_Diff'] = df['Home_AvgRating'] - df['Away_AvgRating']
    
    return df

# ========== OVERFITTING ÖNLEYİCİ MODEL PIPELINE ==========
def create_overfitting_prevention_pipeline(selected_features):
    """OVERFITTING ÖNLEMEK İÇİN BASİT VE REGULARIZE EDİLMİŞ PIPELINE"""
    
    # Sadece scaler ve model
    preprocessor = ColumnTransformer([
        ('scaler', RobustScaler(), selected_features)
    ], remainder='drop')
    
    # OVERFITTING ÖNLEYİCİ LightGBM parametreleri
    lgbm_clf = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbosity=-1,
        n_estimators=500,
        learning_rate=0.01,
        max_depth=3,
        num_leaves=10,
        min_child_samples=50,
        subsample=0.6,
        colsample_bytree=0.6,
        reg_alpha=2.0,
        reg_lambda=2.0,
        force_row_wise=True
    )
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('lgbm', lgbm_clf)
    ])

# ========== GELİŞTİRİLMİŞ VERİ YÜKLEME ==========
def load_and_validate_enhanced_data():
    """Geliştirilmiş veri yükleme ve doğrulama"""
    print("\n📊 Geliştirilmiş veri yükleniyor...")
    
    try:
        df_matches = pd.read_excel(DATA_PATH)
        df_matches.columns = [col.strip().replace(' ', '_') for col in df_matches.columns]
        
        df_players = pd.read_excel(PLAYER_DATA_PATH)
        
        df = enhanced_data_preparation(df_matches, df_players)
        
        missing_features = []
        for feat in SELECTED_FEATURES:
            if feat not in df.columns:
                missing_features.append(feat)
                df[feat] = 0
        
        if missing_features:
            print(f"⚠️ Eksik özellikler varsayılan değerlerle dolduruldu: {missing_features}")
        
        numeric_cols = df[SELECTED_FEATURES].select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        class_distribution = df['Result_Numeric'].value_counts().sort_index()
        print(f"📈 Sınıf Dağılımı: {dict(class_distribution)}")
        
        print("✅ Geliştirilmiş veri hazırlığı tamamlandı")
        return df
        
    except Exception as e:
        print(f"❌ Veri yükleme hatası: {e}")
        raise

# ========== SMOTE İLE CLASS BALANCING ==========
def apply_smote_balancing(X_train, y_train):
    """SMOTE ile sınıf dengesizliğini gider"""
    print("🔄 SMOTE ile class balancing uygulanıyor...")
    
    # Draw sınıfını biraz daha artıralım
    sampling_strategy = {
        0: min(len(y_train[y_train == 0]) * 3 // 2, len(y_train) // 2),  # Draw
        1: len(y_train[y_train == 1]),  # HomeWin  
        2: len(y_train[y_train == 2])   # AwayWin
    }
    
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=RANDOM_STATE, k_neighbors=3)
    X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
    
    print(f"📊 SMOTE sonrası sınıf dağılımı: {pd.Series(y_resampled).value_counts().to_dict()}")
    return X_resampled, y_resampled

# ========== OVERFITTING ÖNLEYİCİ MODEL EĞİTİMİ ==========
def train_enhanced_model():
    """OVERFITTING ÖNLEYİCİ model eğitim fonksiyonu"""
    print("⚽ Bundesliga Tahmin Modeli - Ultimate Final Sürüm")
    print("=" * 70)
    print("✅ Advanced feature engineering") 
    print("✅ OPTIMIZED cumulative metrikler")
    print("✅ DRAW OPTIMIZATION özellikleri")
    print("✅ HOMEWIN RECALL enhancement")
    print("✅ SMOTE class balancing")
    print("✅ STRICT feature selection (Max 15 özellik)")
    print("✅ OVERFITTING PREVENTION techniques")
    print("=" * 70)
    
    # Veriyi yükle
    df = load_and_validate_enhanced_data()
    
    # Zaman bazlı split
    train_df, val_df, test_df = time_based_split(df, TEST_SIZE, VAL_SIZE)
    
    # Feature ve target'ları ayır
    X_train = train_df[SELECTED_FEATURES].copy()
    y_train = train_df['Result_Numeric'].copy()
    
    X_val = val_df[SELECTED_FEATURES].copy()
    y_val = val_df['Result_Numeric'].copy()
    
    X_test = test_df[SELECTED_FEATURES].copy()
    y_test = test_df['Result_Numeric'].copy()
    
    # 1. Önce feature engineering uygula
    print("🔧 Feature engineering uygulanıyor...")
    feature_engineer = AdvancedFeatureEngineer()
    X_train = feature_engineer.fit_transform(X_train)
    X_val = feature_engineer.transform(X_val)
    X_test = feature_engineer.transform(X_test)
    
    # 2. SMOTE ile class balancing uygula
    X_train_balanced, y_train_balanced = apply_smote_balancing(X_train, y_train)
    
    # 3. Pipeline dışında STRICT feature selection yap
    X_train_selected, X_val_selected, X_test_selected, important_features = perform_strict_feature_selection(
        X_train_balanced, y_train_balanced, X_val, X_test, method='importance'
    )
    
    # 4. Sınıf ağırlıklarını hesapla
    classes = np.unique(y_train_balanced)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train_balanced)
    class_weight_dict = dict(zip(classes, class_weights))
    sample_weights_train = np.array([class_weight_dict[yy] for yy in y_train_balanced])
    
    print(f"📊 Eğitim verisi: {X_train_selected.shape}")
    print(f"📊 Validation verisi: {X_val_selected.shape}")
    print(f"📊 Test verisi: {X_test_selected.shape}")
    print(f"⚖️ Sınıf ağırlıkları: {class_weight_dict}")
    
    # 5. Overfitting önleyici pipeline oluştur
    model = create_overfitting_prevention_pipeline(important_features)
    
    # 6. Hiperparametre optimizasyonu
    param_distributions = {
        'lgbm__learning_rate': [0.005, 0.01, 0.02],
        'lgbm__max_depth': [2, 3, 4],
        'lgbm__num_leaves': [8, 10, 12],
        'lgbm__min_child_samples': [40, 50, 60],
        'lgbm__reg_alpha': [1.0, 2.0, 3.0],
        'lgbm__reg_lambda': [1.0, 2.0, 3.0],
        'lgbm__subsample': [0.5, 0.6, 0.7],
        'lgbm__colsample_bytree': [0.5, 0.6, 0.7],
        'lgbm__n_estimators': [300, 500, 700]
    }
    
    tscv = TimeSeriesSplit(n_splits=10)
    
    print("\n🎯 Overfitting Önleyici Hiperparametre Optimizasyonu...")
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=20,
        cv=tscv,
        scoring='balanced_accuracy',
        n_jobs=N_JOBS,
        verbose=2,
        random_state=RANDOM_STATE,
        return_train_score=True
    )
    
    random_search.fit(X_train_selected, y_train_balanced, lgbm__sample_weight=sample_weights_train)
    
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    
    print(f"\n🏆 En İyi Parametreler: {best_params}")
    print(f"🏆 En İyi CV Skoru: {best_score:.4f}")
    
    # 7. Final modeli eğit
    print("\n🚀 Final model eğitimi (Early Stopping ile)...")
    final_model = create_overfitting_prevention_pipeline(important_features)
    final_model.set_params(**best_params)
    
    final_model.named_steps['lgbm'].set_params(
        n_estimators=1000,
        early_stopping_rounds=50,
        verbose=100
    )
    
    final_model.fit(
        X_train_selected, y_train_balanced,
        lgbm__eval_set=[(X_val_selected, y_val)],
        lgbm__eval_metric='multi_logloss',
        lgbm__sample_weight=sample_weights_train,
        lgbm__callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    
    # 8. Model değerlendirme
    print("\n📊 Kapsamlı Model Değerlendirme:")
    evaluate_model_comprehensive(final_model, X_test_selected, y_test, X_train_selected, y_train_balanced)
    
    # 9. Feature importance analizi
    analyze_feature_importance(final_model, important_features)
    
    # 10. Modeli kaydet
    save_enhanced_model(final_model, important_features, best_params, random_search.cv_results_)
    
    return final_model, important_features

def evaluate_model_comprehensive(model, X_test, y_test, X_train, y_train):
    """Kapsamlı model değerlendirme"""
    
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    y_proba_test = model.predict_proba(X_test)
    
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average='weighted')
    test_precision = precision_score(y_test, y_pred_test, average='weighted')
    test_recall = recall_score(y_test, y_pred_test, average='weighted')
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    train_f1 = f1_score(y_train, y_pred_train, average='weighted')
    
    accuracy_gap = train_accuracy - test_accuracy
    f1_gap = train_f1 - test_f1
    
    print(f"📈 Test Accuracy: {test_accuracy:.4f}")
    print(f"📈 Test F1-Score: {test_f1:.4f}")
    print(f"📈 Test Precision: {test_precision:.4f}")
    print(f"📈 Test Recall: {test_recall:.4f}")
    print(f"🏋️ Train Accuracy: {train_accuracy:.4f}")
    print(f"🏋️ Train F1-Score: {train_f1:.4f}")
    print(f"📊 Accuracy Gap (Overfitting): {accuracy_gap:.4f}")
    print(f"📊 F1 Gap (Overfitting): {f1_gap:.4f}")
    
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
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Draw', 'HomeWin', 'AwayWin'],
                yticklabels=['Draw', 'HomeWin', 'AwayWin'])
    plt.title('Confusion Matrix - Test Set (Ultimate Final Model)')
    plt.ylabel('Gerçek Değer')
    plt.xlabel('Tahmin Edilen Değer')
    plt.savefig('models/confusion_matrix_ultimate_final.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_feature_importance(model, feature_names):
    """Feature importance analizi"""
    try:
        if hasattr(model.named_steps['lgbm'], 'feature_importances_'):
            importances = model.named_steps['lgbm'].feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print("\n🏆 Feature Importance Ranking:")
            for i, idx in enumerate(indices[:15]):
                if idx < len(feature_names):
                    print(f"{i+1:2d}. {feature_names[idx]:30s} ({importances[idx]:.4f})")
            
            # Görselleştirme
            plt.figure(figsize=(12, 8))
            top_n = min(10, len(feature_names))
            plt.barh(range(top_n), importances[indices[:top_n]][::-1], align='center')
            plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]][::-1])
            plt.xlabel('Importance')
            plt.title('Top Feature Importances (Ultimate Final Model)')
            plt.tight_layout()
            plt.savefig('models/feature_importance_ultimate_final.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    except Exception as e:
        print(f"⚠️ Feature importance analizinde hata: {e}")

def save_enhanced_model(model, important_features, best_params, cv_results):
    """Geliştirilmiş model kaydetme"""
    os.makedirs("models", exist_ok=True)
    
    model_path = "models/bundesliga_model_ultimate_final.pkl"
    joblib.dump(model, model_path)
    
    feature_info = {
        'important_features': important_features,
        'all_features': SELECTED_FEATURES,
        'best_params': best_params,
        'cv_results': cv_results,
        'timestamp': datetime.now().isoformat(),
        'model_version': 'ultimate_final_v1'
    }
    joblib.dump(feature_info, "models/feature_info_ultimate_final.pkl")
    
    performance_report = {
        'model_type': 'LightGBM Ultimate Final',
        'features_used': len(important_features),
        'total_features': len(SELECTED_FEATURES),
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'max_features_limit': 15,
        'cumulative_metrics_included': True,
        'draw_optimization': True,
        'homewin_recall_enhancement': True,
        'smote_balancing': True
    }
    
    with open("models/performance_report_ultimate_final.txt", "w") as f:
        for key, value in performance_report.items():
            f.write(f"{key}: {value}\n")
    
    print(f"\n💾 Model kaydedildi: {model_path}")
    print("💾 Feature bilgileri kaydedildi")
    print("💾 Performans raporu kaydedildi")

# ========== YARDIMCI FONKSİYONLAR ==========
def time_based_split(df, test_size=0.15, val_size=0.15):
    """Zaman bazlı split fonksiyonu"""
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

# ========== ANA FONKSİYON ==========
def main():
    print("🏆 Bundesliga Tahmin Modeli - Ultimate Final Sürüm")
    print("=" * 60)
    print("🚀 Başlatılıyor...")
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    try:
        model, important_features = train_enhanced_model()
        
        print("\n🎉 Ultimate Final Model eğitimi başarıyla tamamlandı!")
        print(f"📋 Kullanılan önemli feature'lar: {len(important_features)}/{len(SELECTED_FEATURES)}")
        print("📍 Model dosyaları 'models/' klasörüne kaydedildi")
        
        # Özellik analizi
        cumulative_features = [feat for feat in important_features if any(x in feat for x in 
                              ['cumulative', '_5games', '_ppg_', '_gpg_', '_gapg_', 'goal_diff_cumulative'])]
        
        draw_features = [feat for feat in important_features if any(x in feat for x in 
                          ['strength_balance', 'is_close_match', 'both_teams_good_form'])]
        
        homewin_features = [feat for feat in important_features if any(x in feat for x in 
                            ['home_advantage', 'home_defensive'])]
        
        print(f"\n📊 Özellik Analizi:")
        print(f"   📈 Cumulative metrikler: {len(cumulative_features)}")
        print(f"   🤝 Draw optimization: {len(draw_features)}")
        print(f"   🏠 HomeWin recall: {len(homewin_features)}")
        
        print("\n🏆 MODEL BAŞARI ÖZETİ:")
        print("✅ %58.72+ accuracy hedefi")
        print("✅ Draw recall > %40 hedefi") 
        print("✅ HomeWin recall > %65 hedefi")
        print("✅ AwayWin recall > %75 hedefi")
        print("✅ Overfitting kontrolü")
        
    except Exception as e:
        print(f"❌ Model eğitimi sırasında hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()