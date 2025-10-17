#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bundesliga Tahmin Modeli - ULTIMATE PRODUCTION SÜRÜM
Geliştirilmiş Data Drift Detection + Strict Feature Selection + Enhanced Regularization
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

# ========== GELİŞTİRİLMİŞ KONFİGÜRASYON ==========
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
N_JOBS = -1
MAX_FEATURES = 12  # Daha strict feature selection

DATA_PATH = "data/bundesliga_matches_2023_2025_final_fe_team_values_cleaned.xlsx"
PLAYER_DATA_PATH = "data/final_bundesliga_dataset_complete.xlsx"
OLD_MODEL_PATH = "models/bundesliga_model_ultimate_final.pkl"

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

# ========== DATA DRIFT DETECTION ==========
class DataDriftDetector:
    """Veri dağılımı değişimini tespit eden sınıf"""
    
    def __init__(self):
        self.drift_results = {}
        
    def detect_drift(self, old_data, new_data, feature_columns, alpha=0.05):
        """Eski ve yeni veri arasındaki dağılım farkını tespit et"""
        print("\n🔍 DATA DRIFT DETECTION ANALİZİ...")
        
        drift_detected = {}
        
        for feature in feature_columns:
            if feature in old_data.columns and feature in new_data.columns:
                # Eksik değerleri temizle
                old_vals = old_data[feature].dropna()
                new_vals = new_data[feature].dropna()
                
                if len(old_vals) > 10 and len(new_vals) > 10:
                    # Kolmogorov-Smirnov testi
                    stat, p_value = stats.ks_2samp(old_vals, new_vals)
                    
                    # Ortalama farkı
                    mean_diff = abs(old_vals.mean() - new_vals.mean())
                    std_ratio = old_vals.std() / (new_vals.std() + 1e-8)
                    
                    drift_detected[feature] = {
                        'p_value': p_value,
                        'drift_detected': p_value < alpha,
                        'mean_difference': mean_diff,
                        'std_ratio': std_ratio,
                        'ks_statistic': stat
                    }
        
        self.drift_results = drift_detected
        
        # Drift tespit edilen feature'ları raporla
        drifted_features = [feat for feat, result in drift_detected.items() 
                          if result['drift_detected']]
        
        print(f"📊 Drift analizi tamamlandı:")
        print(f"   ✅ Toplam feature: {len(drift_detected)}")
        print(f"   ⚠️  Drift tespit edilen: {len(drifted_features)}")
        
        if drifted_features:
            print(f"   🚨 Drift eden feature'lar: {drifted_features}")
            
            # Drift şiddetini analiz et
            for feat in drifted_features[:5]:  # İlk 5'i göster
                result = drift_detected[feat]
                print(f"      📈 {feat}: p-value={result['p_value']:.4f}, "
                      f"mean_diff={result['mean_difference']:.2f}")
        
        return drift_detected

# ========== GELİŞTİRİLMİŞ STRICT FEATURE SELECTION ==========
def enhanced_strict_feature_selection(X_train, y_train, X_val, X_test, max_features=MAX_FEATURES):
    """OVERFITTING ÖNLEMEK İÇİN DAHA STRICT FEATURE SELECTION"""
    print(f"🔍 STRICT Feature Selection (Max {max_features} özellik)...")
    
    # 1. Random Forest ile initial selection
    estimator = RandomForestClassifier(
        n_estimators=200, 
        random_state=RANDOM_STATE, 
        n_jobs=-1,
        max_depth=5
    )
    
    # Daha yüksek threshold ile selection
    selector = SelectFromModel(estimator, threshold="1.25*mean")
    selector.fit(X_train, y_train)
    
    selected_features = X_train.columns[selector.get_support()].tolist()
    
    # 2. Eğer hala çok fazla özellik varsa, en iyi N tanesini al
    if len(selected_features) > max_features:
        print(f"⚡ Çok fazla özellik seçildi ({len(selected_features)}), en iyi {max_features} tanesi alınıyor...")
        
        # Feature importance'ye göre sırala
        estimator.fit(X_train[selected_features], y_train)
        importances = estimator.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        # En iyi N feature'ı seç
        selected_features = [selected_features[i] for i in indices[:max_features]]
    
    # 3. Correlation analysis - yüksek korelasyonlu feature'ları çıkar
    selected_features = remove_highly_correlated_features(X_train[selected_features], selected_features)
    
    print(f"✅ Seçilen özellik sayısı: {len(selected_features)}/{X_train.shape[1]}")
    print(f"📋 Seçilen özellikler: {selected_features}")
    
    X_train_selected = X_train[selected_features]
    X_val_selected = X_val[selected_features]
    X_test_selected = X_test[selected_features]
    
    return X_train_selected, X_val_selected, X_test_selected, selected_features

def remove_highly_correlated_features(X, features, threshold=0.85):
    """Yüksek korelasyonlu feature'ları temizle"""
    if len(features) <= 2:
        return features
    
    correlation_matrix = X.corr().abs()
    upper_tri = correlation_matrix.where(np.triu(np.ones(correlation_matrix.shape), k=1).astype(bool))
    
    to_drop = []
    for column in upper_tri.columns:
        if column in features:
            high_corr_features = upper_tri[column][upper_tri[column] > threshold].index.tolist()
            for feat in high_corr_features:
                if feat in features and feat not in to_drop and column not in to_drop:
                    # İkisinden birini rastgele seç (daha sonra importance'ye göre geliştirilebilir)
                    to_drop.append(feat)
    
    # to_drop'daki feature'ları çıkar
    selected_features = [f for f in features if f not in to_drop]
    
    if to_drop:
        print(f"📊 Yüksek korelasyonlu {len(to_drop)} özellik çıkarıldı: {to_drop}")
    
    return selected_features

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
                if null_count > 0:
                    print(f"   ⚠️  {column}: {null_count} eksik değer dolduruldu")
    
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
            
            outliers_count = ((df[col] < lower_bound) | (df[col] > upper_bound)).sum()
            if outliers_count > 0:
                df[col] = np.clip(df[col], lower_bound, upper_bound)
                print(f"   📈 {col}: {outliers_count} outlier düzeltildi")
    
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
def create_enhanced_prevention_pipeline(selected_features):
    """GELİŞTİRİLMİŞ OVERFITTING ÖNLEYİCİ PIPELINE"""
    
    # Sadece scaler ve model
    preprocessor = ColumnTransformer([
        ('scaler', RobustScaler(), selected_features)
    ], remainder='drop')
    
    # GELİŞTİRİLMİŞ OVERFITTING ÖNLEYİCİ LightGBM parametreleri
    lgbm_clf = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS,
        verbosity=-1,
        n_estimators=400,
        learning_rate=0.008,
        max_depth=3,
        num_leaves=8,
        min_child_samples=80,
        subsample=0.5,
        colsample_bytree=0.5,
        reg_alpha=3.0,
        reg_lambda=3.0,
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

# ========== DATA DRIFT ANALYSIS ==========
def perform_data_drift_analysis(df):
    """Veri drift analizi yap"""
    print("\n🔍 DATA DRIFT ANALİZİ BAŞLATILIYOR...")
    
    # Eski ve yeni veriyi ayır (tarihe göre)
    if 'Date' in df.columns:
        cutoff_date = df['Date'].quantile(0.7)  # %70 eski, %30 yeni
        old_data = df[df['Date'] <= cutoff_date]
        new_data = df[df['Date'] > cutoff_date]
        
        print(f"📅 Veri split: Eski ({len(old_data)} maç) vs Yeni ({len(new_data)} maç)")
        
        # Data drift detection
        drift_detector = DataDriftDetector()
        drift_results = drift_detector.detect_drift(
            old_data, new_data, 
            [col for col in SELECTED_FEATURES if col in df.columns]
        )
        
        return drift_results
    else:
        print("⚠️ Tarih sütunu yok, drift analizi atlanıyor")
        return {}

# ========== INCREMENTAL LEARNING ==========
def incremental_learning_option(X_train, y_train, X_val, y_val, important_features):
    """Incremental learning seçeneği"""
    if os.path.exists(OLD_MODEL_PATH):
        print("\n🔄 INCREMENTAL LEARNING SEÇENEĞİ...")
        try:
            old_model = joblib.load(OLD_MODEL_PATH)
            print("✅ Eski model yüklendi, incremental learning uygulanıyor...")
            
            # Yeni verilerle fine-tuning
            old_model.named_steps['lgbm'].fit(
                X_train[important_features], y_train,
                init_model=old_model.named_steps['lgbm'],
                eval_set=[(X_val[important_features], y_val)],
                callbacks=[
                    lgb.early_stopping(30),
                    lgb.log_evaluation(50),
                    lgb.reset_parameter(learning_rate=[0.01] * 1000)
                ]
            )
            return old_model
        except Exception as e:
            print(f"⚠️ Incremental learning başarısız: {e}")
            print("🔁 Yeni model from scratch eğitilecek...")
            return None
    return None

# ========== GELİŞTİRİLMİŞ MODEL EĞİTİMİ ==========
def train_enhanced_production_model():
    """GELİŞTİRİLMİŞ PRODUCTION MODEL EĞİTİMİ"""
    print("⚽ Bundesliga Tahmin Modeli - ULTIMATE PRODUCTION SÜRÜM")
    print("=" * 70)
    print("✅ Advanced feature engineering") 
    print("✅ OPTIMIZED cumulative metrikler")
    print("✅ DRAW OPTIMIZATION özellikleri")
    print("✅ HOMEWIN RECALL enhancement")
    print("✅ SMOTE class balancing")
    print(f"✅ STRICT feature selection (Max {MAX_FEATURES} özellik)")
    print("✅ DATA DRIFT DETECTION")
    print("✅ ENHANCED REGULARIZATION")
    print("✅ INCREMENTAL LEARNING option")
    print("=" * 70)
    
    # Veriyi yükle
    df = load_and_validate_enhanced_data()
    
    # Data drift analizi yap
    drift_results = perform_data_drift_analysis(df)
    
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
    
    # 3. STRICT feature selection yap
    X_train_selected, X_val_selected, X_test_selected, important_features = enhanced_strict_feature_selection(
        X_train_balanced, y_train_balanced, X_val, X_test, MAX_FEATURES
    )
    
    # 4. Incremental learning seçeneğini dene
    incremental_model = incremental_learning_option(X_train_balanced, y_train_balanced, X_val, y_val, important_features)
    
    if incremental_model is not None:
        final_model = incremental_model
        print("✅ Incremental learning ile model güncellendi!")
    else:
        # 5. Sınıf ağırlıklarını hesapla
        classes = np.unique(y_train_balanced)
        class_weights = compute_class_weight('balanced', classes=classes, y=y_train_balanced)
        class_weight_dict = dict(zip(classes, class_weights))
        sample_weights_train = np.array([class_weight_dict[yy] for yy in y_train_balanced])
        
        print(f"📊 Eğitim verisi: {X_train_selected.shape}")
        print(f"📊 Validation verisi: {X_val_selected.shape}")
        print(f"📊 Test verisi: {X_test_selected.shape}")
        print(f"⚖️ Sınıf ağırlıkları: {class_weight_dict}")
        
        # 6. Overfitting önleyici pipeline oluştur
        model = create_enhanced_prevention_pipeline(important_features)
        
        # 7. GELİŞTİRİLMİŞ Hiperparametre optimizasyonu
        param_distributions = {
            'lgbm__learning_rate': [0.005, 0.008, 0.01],
            'lgbm__max_depth': [2, 3],
            'lgbm__num_leaves': [6, 8, 10],
            'lgbm__min_child_samples': [60, 80, 100],
            'lgbm__reg_alpha': [2.0, 3.0, 4.0],
            'lgbm__reg_lambda': [2.0, 3.0, 4.0],
            'lgbm__subsample': [0.4, 0.5, 0.6],
            'lgbm__colsample_bytree': [0.4, 0.5, 0.6],
            'lgbm__n_estimators': [300, 400, 500]
        }
        
        tscv = TimeSeriesSplit(n_splits=15)  # Daha fazla split
        
        print("\n🎯 GELİŞTİRİLMİŞ Hiperparametre Optimizasyonu...")
        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=25,
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
        
        # 8. Final modeli eğit
        print("\n🚀 Final model eğitimi (Enhanced Early Stopping ile)...")
        final_model = create_enhanced_prevention_pipeline(important_features)
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
    
    # 9. Model değerlendirme
    print("\n📊 Kapsamlı Model Değerlendirme:")
    evaluate_enhanced_model(final_model, X_test_selected, y_test, X_train_selected, y_train_balanced, drift_results)
    
    # 10. Feature importance analizi
    analyze_enhanced_feature_importance(final_model, important_features, drift_results)
    
    # 11. Modeli kaydet
    save_production_model(final_model, important_features, best_params if 'best_params' in locals() else {})
    
    return final_model, important_features

def evaluate_enhanced_model(model, X_test, y_test, X_train, y_train, drift_results):
    """Geliştirilmiş model değerlendirme"""
    
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
    
    # Data drift etkisi
    if drift_results:
        drifted_count = sum(1 for result in drift_results.values() if result['drift_detected'])
        if drifted_count > 5:
            print(f"⚠️ DATA DRIFT: {drifted_count} feature'da drift tespit edildi!")
    
    print("\n🎯 Detaylı Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=['Draw', 'HomeWin', 'AwayWin']))
    
    # Confusion matrix
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Draw', 'HomeWin', 'AwayWin'],
                yticklabels=['Draw', 'HomeWin', 'AwayWin'])
    plt.title('Confusion Matrix - Test Set (Production Model)')
    plt.ylabel('Gerçek Değer')
    plt.xlabel('Tahmin Edilen Değer')
    plt.savefig('models/confusion_matrix_production.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_enhanced_feature_importance(model, feature_names, drift_results):
    """Geliştirilmiş feature importance analizi"""
    try:
        if hasattr(model.named_steps['lgbm'], 'feature_importances_'):
            importances = model.named_steps['lgbm'].feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print("\n🏆 Enhanced Feature Importance Ranking:")
            for i, idx in enumerate(indices[:15]):
                if idx < len(feature_names):
                    feature_name = feature_names[idx]
                    importance_val = importances[idx]
                    
                    # Drift bilgisini ekle
                    drift_info = ""
                    if feature_name in drift_results and drift_results[feature_name]['drift_detected']:
                        drift_info = " 🚨DRIFT"
                    
                    print(f"{i+1:2d}. {feature_name:30s} ({importance_val:.4f}){drift_info}")
            
            # Görselleştirme
            plt.figure(figsize=(12, 8))
            top_n = min(10, len(feature_names))
            
            # Drift durumuna göre renk
            colors = []
            for i in indices[:top_n]:
                feature_name = feature_names[i]
                if feature_name in drift_results and drift_results[feature_name]['drift_detected']:
                    colors.append('red')  # Drift var
                else:
                    colors.append('blue')  # Drift yok
            
            plt.barh(range(top_n), importances[indices[:top_n]][::-1], 
                    align='center', color=colors[::-1])
            plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]][::-1])
            plt.xlabel('Importance')
            plt.title('Top Feature Importances (Production Model)\nRed: Data Drift Detected')
            plt.tight_layout()
            plt.savefig('models/feature_importance_production.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    except Exception as e:
        print(f"⚠️ Feature importance analizinde hata: {e}")

def save_production_model(model, important_features, best_params):
    """Production model kaydetme"""
    os.makedirs("models", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/bundesliga_model_production_{timestamp}.pkl"
    joblib.dump(model, model_path)
    
    # Latest model olarak da kaydet
    joblib.dump(model, "models/bundesliga_model_production_latest.pkl")
    
    feature_info = {
        'important_features': important_features,
        'all_features': SELECTED_FEATURES,
        'best_params': best_params,
        'timestamp': datetime.now().isoformat(),
        'model_version': 'production_v2',
        'max_features': MAX_FEATURES
    }
    joblib.dump(feature_info, "models/feature_info_production.pkl")
    
    performance_report = {
        'model_type': 'LightGBM Production',
        'features_used': len(important_features),
        'total_features': len(SELECTED_FEATURES),
        'max_features_limit': MAX_FEATURES,
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'cumulative_metrics_included': True,
        'draw_optimization': True,
        'homewin_recall_enhancement': True,
        'smote_balancing': True,
        'data_drift_detection': True,
        'enhanced_regularization': True
    }
    
    with open("models/performance_report_production.txt", "w") as f:
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
    print("🏆 Bundesliga Tahmin Modeli - ULTIMATE PRODUCTION SÜRÜM")
    print("=" * 60)
    print("🚀 Başlatılıyor...")
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    try:
        model, important_features = train_enhanced_production_model()
        
        print("\n🎉 PRODUCTION MODEL eğitimi başarıyla tamamlandı!")
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
        
        print("\n🏆 MODEL BAŞARI HEDEFLERİ:")
        print("✅ %61.47+ accuracy hedefi")
        print("✅ Draw recall > %40 hedefi") 
        print("✅ HomeWin recall > %65 hedefi")
        print("✅ AwayWin recall > %75 hedefi")
        print("✅ Overfitting gap < %5 hedefi")
        print("✅ Data drift monitoring")
        
    except Exception as e:
        print(f"❌ Model eğitimi sırasında hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()