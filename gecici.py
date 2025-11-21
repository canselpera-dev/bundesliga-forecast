#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bundesliga Tahmin Modeli - REALISTIC BALANCE v11.1
TAM HATA DÜZELTMELİ + YAŞ ORTALAMASI ENTEGRE
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
MAX_FEATURES = 15

DATA_PATH = "data/bundesliga_matches_2023_2025_final_fe_team_values_cleaned.xlsx"
PLAYER_DATA_PATH = "data/final_bundesliga_dataset_complete.xlsx"

# PROBLEMLİ FEATURE'LARI TANIMLA - HEDEF DEĞİŞKENİ KALDIRMA!
PROBLEMATIC_FEATURES = [
    'id', 'score.fullTime.home', 'score.fullTime.away',
    'goals_difference', 'xg_difference', 'awayTeam.id', 'matchday',
    'home_goals', 'away_goals', 'home_xg', 'away_xg'
]

# GELİŞTİRİLMİŞ ÖZELLİK LİSTESİ - YAŞ ORTALAMASI EKLENDİ
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
    
    # YENİ: YAŞ ORTALAMASI FEATURE'LARI
    'age_difference', 'total_experience', 'youth_advantage', 'experience_factor',
    
    # Özel Durumlar
    'isDerby', 'draw_potential'
]

# ========== YAŞ ORTALAMASI HESAPLAMA ==========
def calculate_team_avg_age(df_players):
    """Oyuncu verisinden takım yaş ortalamasını hesapla"""
    print("📊 Takım yaş ortalamaları hesaplanıyor...")
    
    # Age sütununu temizle ve sayısal yap
    df_players['Age'] = pd.to_numeric(df_players['Age'], errors='coerce')
    
    # NaN değerleri temizle
    df_players_clean = df_players.dropna(subset=['Age', 'Team'])
    
    # Takım bazında ortalama yaş hesapla
    team_avg_age = df_players_clean.groupby('Team')['Age'].agg(['mean', 'std', 'count']).reset_index()
    team_avg_age.rename(columns={'mean': 'Squad_Avg_Age', 'std': 'Age_Std', 'count': 'Player_Count'}, inplace=True)
    
    # Güvenilir olmayan verileri filtrele (en az 5 oyuncu)
    team_avg_age = team_avg_age[team_avg_age['Player_Count'] >= 5]
    
    print(f"✅ {len(team_avg_age)} takımın yaş ortalaması hesaplandı")
    
    # İstatistikleri göster
    if len(team_avg_age) > 0:
        print(f"📈 Yaş Dağılımı: Min={team_avg_age['Squad_Avg_Age'].min():.2f}, "
              f"Max={team_avg_age['Squad_Avg_Age'].max():.2f}, "
              f"Ort={team_avg_age['Squad_Avg_Age'].mean():.2f}")
    
    return team_avg_age[['Team', 'Squad_Avg_Age']]

def integrate_team_ages(df_matches, df_players):
    """Takım yaş ortalamalarını maç verisine entegre et"""
    print("🔄 Takım yaş ortalamaları entegre ediliyor...")
    
    df = df_matches.copy()
    
    # Takım yaş ortalamasını hesapla
    team_avg_age = calculate_team_avg_age(df_players)
    
    if team_avg_age.empty:
        print("⚠️ Takım yaş verisi bulunamadı, default değerler kullanılacak")
        df['home_squad_avg_age'] = 26.5
        df['away_squad_avg_age'] = 26.5
        return df
    
    # Takım isimlerini normalize et (eşleştirme için)
    team_avg_age['Team_Normalized'] = team_avg_age['Team'].str.lower().str.strip()
    
    # Maç verisinde takım isimlerini normalize et
    if 'HomeTeam' not in df.columns:
        if 'homeTeam.name' in df.columns:
            df['HomeTeam'] = df['homeTeam.name']
        else:
            print("⚠️ HomeTeam sütunu bulunamadı")
            df['HomeTeam'] = "Unknown"
    
    if 'AwayTeam' not in df.columns:
        if 'awayTeam.name' in df.columns:
            df['AwayTeam'] = df['awayTeam.name']
        else:
            print("⚠️ AwayTeam sütunu bulunamadı")
            df['AwayTeam'] = "Unknown"
    
    df['HomeTeam_Normalized'] = df['HomeTeam'].str.lower().str.strip()
    df['AwayTeam_Normalized'] = df['AwayTeam'].str.lower().str.strip()
    
    # Home team yaş ortalamasını entegre et
    df = df.merge(team_avg_age[['Team_Normalized', 'Squad_Avg_Age']], 
                 left_on='HomeTeam_Normalized', right_on='Team_Normalized', how='left')
    df.rename(columns={'Squad_Avg_Age': 'home_squad_avg_age'}, inplace=True)
    
    # Away team yaş ortalamasını entegre et
    df = df.merge(team_avg_age[['Team_Normalized', 'Squad_Avg_Age']], 
                 left_on='AwayTeam_Normalized', right_on='Team_Normalized', how='left')
    df.rename(columns={'Squad_Avg_Age': 'away_squad_avg_age'}, inplace=True)
    
    # Geçici sütunları temizle
    df.drop(columns=['Team_Normalized_x', 'Team_Normalized_y', 'HomeTeam_Normalized', 'AwayTeam_Normalized'], 
           inplace=True, errors='ignore')
    
    # Eksik değerleri Bundesliga ortalaması ile doldur
    bundesliga_avg_age = 26.5
    df['home_squad_avg_age'] = df['home_squad_avg_age'].fillna(bundesliga_avg_age)
    df['away_squad_avg_age'] = df['away_squad_avg_age'].fillna(bundesliga_avg_age)
    
    print(f"✅ Takım yaş ortalamaları entegre edildi: {len(team_avg_age)} takım")
    
    return df

# ========== BASİT EKSİK DEĞER DOLDURMA ==========
def simple_missing_value_imputation(df):
    """Basit ve güvenli eksik değer doldurma"""
    print("📊 Basit eksik değer doldurma...")
    
    # Tüm sayısal sütunları bul
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    # Her sayısal sütun için medyan ile doldur
    for col in numeric_cols:
        if col != 'Result_Numeric':  # Hedef değişkeni doldurma
            null_count = df[col].isnull().sum()
            if null_count > 0:
                # Medyan hesapla ve doldur
                median_val = df[col].median()
                if pd.isna(median_val):
                    median_val = 0
                df[col] = df[col].fillna(median_val)
                print(f"✅ {col}: {null_count} NaN değer {median_val:.2f} ile dolduruldu")
    
    return df

# ========== CLEAN FEATURE SELECTION ==========
def clean_feature_selection(X_train, y_train, X_val, X_test, max_features=MAX_FEATURES):
    """Geliştirilmiş feature selection - DRAW odaklı"""
    print(f"🧹 GELİŞTİRİLMİŞ Feature Selection (Max {max_features} özellik)...")
    
    # Problemli feature'ları kaldır
    X_train_clean = X_train.drop(columns=PROBLEMATIC_FEATURES, errors='ignore')
    X_val_clean = X_val.drop(columns=PROBLEMATIC_FEATURES, errors='ignore')
    X_test_clean = X_test.drop(columns=PROBLEMATIC_FEATURES, errors='ignore')
    
    print(f"🔍 Problemli {len(PROBLEMATIC_FEATURES)} feature kaldırıldı")
    print(f"🔢 Kalan sayısal sütun sayısı: {X_train_clean.select_dtypes(include=[np.number]).shape[1]}")
    
    # DRAW odaklı feature importance hesapla
    feature_scores = {}
    
    # 1. RandomForest - DRAW class'ına özel
    try:
        rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, max_depth=5,
                                   class_weight='balanced')
        rf.fit(X_train_clean, y_train)
        
        # Draw class'ına önem veren feature'ları bul
        for i, feature in enumerate(X_train_clean.columns):
            # Draw class'ı (0) için importance'ı ağırlıklı hesapla
            draw_importance = rf.feature_importances_[i] * (1 if feature in ['draw_potential', 'form_similarity', 'age_difference'] else 0.8)
            feature_scores[feature] = feature_scores.get(feature, 0) + draw_importance
    except Exception as e:
        print(f"⚠️ RandomForest feature selection hatası: {e}")
    
    # 2. LightGBM - Multi-class odaklı
    try:
        lgb_model = lgb.LGBMClassifier(n_estimators=50, random_state=RANDOM_STATE, verbose=-1,
                                      class_weight='balanced')
        lgb_model.fit(X_train_clean, y_train)
        for i, feature in enumerate(X_train_clean.columns):
            feature_scores[feature] = feature_scores.get(feature, 0) + lgb_model.feature_importances_[i]
    except Exception as e:
        print(f"⚠️ LightGBM feature selection hatası: {e}")
    
    # 3. Draw class'ı ile korelasyon
    try:
        y_draw = (y_train == 0).astype(int)  # Draw class'ı için binary target
        for feature in X_train_clean.columns:
            if X_train_clean[feature].dtype in [np.float64, np.int64]:
                try:
                    correlation = abs(np.corrcoef(X_train_clean[feature], y_draw)[0, 1])
                    if not np.isnan(correlation):
                        # Draw korelasyonuna ekstra ağırlık ver
                        feature_scores[feature] = feature_scores.get(feature, 0) + correlation * 1.5
                except:
                    continue
    except Exception as e:
        print(f"⚠️ Korelasyon feature selection hatası: {e}")
    
    # 4. Manuel önceliklendirme - DRAW için kritik feature'lar
    draw_priority_features = ['draw_potential', 'form_similarity', 'h2h_draws', 'age_difference', 
                             'power_difference', 'value_ratio', 'isDerby']
    for feature in draw_priority_features:
        if feature in X_train_clean.columns:
            feature_scores[feature] = feature_scores.get(feature, 0) + 0.5
    
    # En iyileri seç
    if not feature_scores:
        print("🚨 Tüm feature selection yöntemleri başarısız, tüm sayısal feature'ları kullanıyoruz...")
        selected_features = X_train_clean.columns[:max_features].tolist()
    else:
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [feat for feat, score in sorted_features[:max_features]]
    
    # Seçilen feature'ları kontrol et
    problematic_found = [feat for feat in selected_features if feat in PROBLEMATIC_FEATURES]
    if problematic_found:
        print(f"🚨 UYARI: Seçilen feature'lar arasında problemli feature'lar bulundu: {problematic_found}")
        selected_features = [feat for feat in selected_features if feat not in PROBLEMATIC_FEATURES]
    
    # Draw feature'larını önceliklendir
    draw_features = [feat for feat in selected_features if 'draw' in feat.lower() or 'similarity' in feat.lower()]
    if len(draw_features) < 2:  # En az 2 draw feature'u garantile
        additional_draw_features = [feat for feat in X_train_clean.columns 
                                  if any(keyword in feat.lower() for keyword in ['draw', 'similarity', 'difference'])
                                  and feat not in selected_features]
        if additional_draw_features:
            needed = 2 - len(draw_features)
            selected_features.extend(additional_draw_features[:needed])
    
    print(f"✅ Seçilen {len(selected_features)} özellik: {selected_features}")
    
    X_train_selected = X_train_clean[selected_features]
    X_val_selected = X_val_clean[selected_features]
    X_test_selected = X_test_clean[selected_features]
    
    return X_train_selected, X_val_selected, X_test_selected, selected_features

# ========== REALISTIC CLASS WEIGHTS ==========
def compute_realistic_class_weights(y_train):
    """DRAW odaklı class weights"""
    class_counts = pd.Series(y_train).value_counts().sort_index()
    total_matches = len(y_train)
    
    # Bundesliga gerçek istatistikleri - DRAW'ı güçlendir
    expected_distribution = [0.28, 0.42, 0.30]  # Draw: 28%, HomeWin: 42%, AwayWin: 30%
    
    # DRAW class'ına daha fazla ağırlık ver
    realistic_weights = []
    for i, count in enumerate(class_counts):
        expected_count = total_matches * expected_distribution[i]
        weight = expected_count / count if count > 0 else 1.0
        
        # Draw class'ına ekstra ağırlık (maksimum 2.0x)
        if i == 0:  # Draw class
            weight = min(weight * 1.3, 2.0)
        else:
            weight = min(weight, 1.5)
        
        realistic_weights.append(weight)
    
    class_weight_dict = dict(zip(class_counts.index, realistic_weights))
    print(f"⚖️ DRAW ODAKLI Class Weights: {class_weight_dict}")
    print(f"📊 Gerçek Dağılım: {dict(class_counts)}")
    print(f"🎯 Beklenen Dağılım: Draw: 28%, HomeWin: 42%, AwayWin: 30%")
    
    return class_weight_dict

# ========== REALISTIC MODEL PIPELINE ==========
def create_realistic_pipeline(selected_features, class_weight_dict=None):
    """Geliştirilmiş model pipeline - DRAW optimizasyonu"""
    
    preprocessor = ColumnTransformer([
        ('scaler', RobustScaler(), selected_features)
    ], remainder='drop')
    
    # DRAW odaklı hiperparametreler
    lgbm_params = {
        'objective': 'multiclass',
        'num_class': 3,
        'random_state': RANDOM_STATE,
        'n_jobs': N_JOBS,
        'verbosity': -1,
        'n_estimators': 200,      # Arttırıldı
        'learning_rate': 0.05,
        'max_depth': 4,           # Biraz daha derin
        'num_leaves': 12,         # Arttırıldı
        'min_child_samples': 15,  # Azaltıldı - daha hassas
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 1.0,         # Daha hafif regularization
        'reg_lambda': 1.0,
        'force_row_wise': True,
        'boosting_type': 'gbdt'
    }
    
    if class_weight_dict:
        lgbm_params['class_weight'] = class_weight_dict
    
    lgbm_clf = lgb.LGBMClassifier(**lgbm_params)
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('lgbm', lgbm_clf)
    ])

# ========== BASİT FEATURE ENGINEERING ==========
def simple_feature_engineering(df):
    """Basit ve güvenli feature engineering - HATA DÜZELTMELİ"""
    print("🔧 Basit feature engineering uygulanıyor...")
    
    df = df.copy()
    
    # 1. TEMEL FARKLAR
    if all(col in df.columns for col in ['home_ppg_cumulative', 'away_ppg_cumulative']):
        df['ppg_difference'] = df['home_ppg_cumulative'] - df['away_ppg_cumulative']
        df['ppg_similarity'] = 1 - abs(df['ppg_difference']) / 3  # DRAW için
    
    if all(col in df.columns for col in ['home_gpg_cumulative', 'away_gpg_cumulative']):
        df['gpg_difference'] = df['home_gpg_cumulative'] - df['away_gpg_cumulative']
        df['total_goals_expected'] = (df['home_gpg_cumulative'] + df['away_gpg_cumulative']) / 2
    
    if all(col in df.columns for col in ['home_form_5games', 'away_form_5games']):
        df['form_difference'] = df['home_form_5games'] - df['away_form_5games']
        df['form_similarity'] = 1 - abs(df['home_form_5games'] - df['away_form_5games'])  # DRAW için
    
    # 2. GÜÇ METRİKLERİ
    if all(col in df.columns for col in ['home_power_index', 'away_power_index']):
        df['power_difference'] = df['home_power_index'] - df['away_power_index']
        df['strength_ratio'] = np.minimum(df['home_power_index'], df['away_power_index']) / (np.maximum(df['home_power_index'], df['away_power_index']) + 1e-8)
        df['power_similarity'] = 1 - abs(df['power_difference']) / 2  # DRAW için
    
    # 3. YAŞ BAZLI ÖZELLİKLER - BASİT VE GÜVENLİ
    if all(col in df.columns for col in ['home_squad_avg_age', 'away_squad_avg_age']):
        # Basit çıkarma işlemi - indeks sorunu yok
        df['age_difference'] = df['home_squad_avg_age'].values - df['away_squad_avg_age'].values
        df['total_experience'] = (df['home_squad_avg_age'].values + df['away_squad_avg_age'].values) / 2
        df['youth_advantage'] = np.where(
            df['age_difference'] < 0, 
            abs(df['age_difference']),  # Genç takım avantajı
            0
        )
        # Tecrübe faktörü: Bundesliga ortalamasına göre
        df['experience_factor'] = (df['total_experience'] - 26.5) / 2
        # Yaş benzerliği - DRAW için
        df['age_similarity'] = 1 - abs(df['age_difference']) / 5
    
    # 4. EV SAHİBİ AVANTAJI
    if all(col in df.columns for col in ['home_ppg_cumulative', 'home_form_5games']):
        df['home_advantage'] = df['home_ppg_cumulative'] * 0.6 + df['home_form_5games'] * 0.4
    
    # 5. DEPLASMAN RİSKİ
    if all(col in df.columns for col in ['away_gapg_cumulative', 'away_form_5games']):
        df['away_risk'] = df['away_gapg_cumulative'] * (1 - df['away_form_5games'])
    
    # 6. GELİŞTİRİLMİŞ BERABERLİK POTANSİYELİ
    draw_components = []
    if 'form_similarity' in df.columns:
        draw_components.append(df['form_similarity'] * 0.3)
    if 'power_similarity' in df.columns:
        draw_components.append(df['power_similarity'] * 0.3)
    if 'ppg_similarity' in df.columns:
        draw_components.append(df['ppg_similarity'] * 0.2)
    if 'age_similarity' in df.columns:
        draw_components.append(df['age_similarity'] * 0.1)
    if 'isDerby' in df.columns:
        draw_components.append(df['isDerby'] * 0.1)  # Derby maçlarında daha az beraberlik
    
    if draw_components:
        # Tüm component'leri topla
        df['draw_potential'] = sum(draw_components)
    
    # H2H draw oranı
    if all(col in df.columns for col in ['h2h_draws', 'h2h_matches_count']):
        df['h2h_draw_ratio'] = df['h2h_draws'] / (df['h2h_matches_count'] + 1e-8)
    
    print(f"✅ Basit feature engineering tamamlandı: {len(df.columns)} özellik")
    
    return df

# ========== MANUEL CUMULATIVE STATS ==========
def calculate_cumulative_stats(df_matches):
    """Geliştirilmiş cumulative stats - DRAW pattern'leri"""
    print("🔄 Geliştirilmiş cumulative istatistikler hesaplanıyor...")
    
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
        'home_form_5games', 'away_form_5games',
        'home_draw_rate', 'away_draw_rate'  # YENİ: Beraberlik oranları
    ]
    
    for feature in cumulative_features:
        df[feature] = 0.0
    
    for idx, match in df.iterrows():
        home_team = match['HomeTeam']
        away_team = match['AwayTeam']
        
        if home_team not in team_stats:
            team_stats[home_team] = {
                'points': 0, 'goals_for': 0, 'goals_against': 0, 'matches': 0,
                'recent_results': [], 'draws': 0, 'goal_diff': 0
            }
        
        if away_team not in team_stats:
            team_stats[away_team] = {
                'points': 0, 'goals_for': 0, 'goals_against': 0, 'matches': 0,
                'recent_results': [], 'draws': 0, 'goal_diff': 0
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
        
        # YENİ: Beraberlik oranları
        df.loc[idx, 'home_draw_rate'] = team_stats[home_team]['draws'] / home_matches
        df.loc[idx, 'away_draw_rate'] = team_stats[away_team]['draws'] / away_matches
        
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
            team_stats[home_team]['draws'] += 1
            team_stats[away_team]['draws'] += 1
        
        # Recent results'u 5 maçla sınırla
        team_stats[home_team]['recent_results'] = team_stats[home_team]['recent_results'][-5:]
        team_stats[away_team]['recent_results'] = team_stats[away_team]['recent_results'][-5:]
    
    print(f"✅ Geliştirilmiş cumulative istatistikler hesaplandı: {len(team_stats)} takım")
    return df

def calculate_form(recent_results):
    """Son 5 maç formunu hesapla"""
    if not recent_results:
        return 0.5
    return sum(recent_results) / len(recent_results)

# ========== GELİŞTİRİLMİŞ VERİ HAZIRLAMA ==========
def realistic_data_preparation(df_matches, df_players):
    """Geliştirilmiş veri hazırlama - YAŞ ORTALAMASI ENTEGRE"""
    print("🔧 GELİŞTİRİLMİŞ veri hazırlama başlıyor...")
    
    df = df_matches.copy()
    
    # 1. Takım isimlerini standartlaştır
    if 'homeTeam.name' in df.columns and 'HomeTeam' not in df.columns:
        df['HomeTeam'] = df['homeTeam.name']
    if 'awayTeam.name' in df.columns and 'AwayTeam' not in df.columns:
        df['AwayTeam'] = df['awayTeam.name']
    
    # 2. YENİ: Takım yaş ortalamalarını entegre et
    df = integrate_team_ages(df, df_players)
    
    # 3. Result_Numeric oluştur
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
    
    # 4. Tarih işleme
    if 'utcDate' in df.columns:
        df['Date'] = pd.to_datetime(df['utcDate'], errors='coerce')
        df = df.sort_values('Date').reset_index(drop=True)
    
    # 5. GELİŞTİRİLMİŞ CUMULATIVE STATS
    df = calculate_cumulative_stats(df)
    
    # 6. Eksik değerleri doldur
    df = simple_missing_value_imputation(df)
    
    # 7. BASİT Feature engineering
    df = simple_feature_engineering(df)
    
    print("✅ GELİŞTİRİLMİŞ veri hazırlama tamamlandı!")
    return df

# ========== VERİ YÜKLEME ==========
def load_realistic_data():
    """Geliştirilmiş veri yükleme"""
    print("\n📊 GELİŞTİRİLMİŞ veri yükleniyor...")
    
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
        
        # Yaş feature'larını kontrol et
        age_features = [col for col in df.columns if 'age' in col.lower()]
        print(f"📊 Yaş Feature'ları: {age_features}")
        
        print("✅ GELİŞTİRİLMİŞ veri hazırlığı tamamlandı")
        return df
        
    except Exception as e:
        print(f"❌ Veri yükleme hatası: {e}")
        raise

# ========== GELİŞTİRİLMİŞ MODEL EĞİTİMİ ==========
def train_realistic_model():
    """GELİŞTİRİLMİŞ MODEL EĞİTİMİ - YAŞ ORTALAMASI + DRAW OPTIMIZATION"""
    print("⚽ Bundesliga Tahmin Modeli - REALISTIC BALANCE v11.1")
    print("=" * 70)
    print("🎯 GELİŞTİRİLMİŞ HEDEFLER: %60+ Accuracy, Draw Recall > %20")
    print("🎯 YAŞ ORTALAMASI Feature'ları Entegre") 
    print("🎯 DRAW Tahmini Optimizasyonu")
    print("🎯 Mevcut Başarı Korunacak")
    print("=" * 70)
    
    # Veriyi yükle
    df = load_realistic_data()
    
    # Zaman bazlı split
    train_df, val_df, test_df = time_based_split(df, TEST_SIZE, VAL_SIZE)
    
    # Feature ve target'ları ayır
    X_train = train_df.drop(columns=['Result_Numeric'], errors='ignore')
    y_train = train_df['Result_Numeric'].copy()
    
    X_val = val_df.drop(columns=['Result_Numeric'], errors='ignore')
    y_val = val_df['Result_Numeric'].copy()
    
    X_test = test_df.drop(columns=['Result_Numeric'], errors='ignore')
    y_test = test_df['Result_Numeric'].copy()
    
    print(f"🎯 Hedef değişken dağılımı - Eğitim: {y_train.value_counts().to_dict()}")
    
    # 1. BASİT feature engineering uygula
    print("🔧 Basit feature engineering uygulanıyor...")
    X_train = simple_feature_engineering(X_train)
    X_val = simple_feature_engineering(X_val)
    X_test = simple_feature_engineering(X_test)
    
    # VERİ KONTROLÜ
    print(f"🔢 Eğitim verisi shape: {X_train.shape}")
    print(f"🔢 Sayısal sütun sayısı: {X_train.select_dtypes(include=[np.number]).shape[1]}")
    
    # 2. GELİŞTİRİLMİŞ feature selection yap
    X_train_selected, X_val_selected, X_test_selected, important_features = clean_feature_selection(
        X_train, y_train, X_val, X_test, MAX_FEATURES
    )
    
    print(f"📊 Eğitim verisi: {X_train_selected.shape}")
    print(f"📊 Validation verisi: {X_val_selected.shape}")
    print(f"📊 Test verisi: {X_test_selected.shape}")
    
    # 3. DRAW odaklı class weights hesapla
    class_weight_dict = compute_realistic_class_weights(y_train)
    
    # 4. GELİŞTİRİLMİŞ pipeline oluştur
    model = create_realistic_pipeline(important_features, class_weight_dict)
    
    # 5. GELİŞTİRİLMİŞ Hiperparametre optimizasyonu
    param_distributions = {
        'lgbm__learning_rate': [0.03, 0.05, 0.07],
        'lgbm__max_depth': [3, 4, 5],
        'lgbm__num_leaves': [8, 12, 16],
        'lgbm__min_child_samples': [10, 15, 20],
        'lgbm__subsample': [0.7, 0.8, 0.9],
        'lgbm__colsample_bytree': [0.7, 0.8, 0.9],
        'lgbm__reg_alpha': [0.5, 1.0, 2.0],
        'lgbm__reg_lambda': [0.5, 1.0, 2.0],
        'lgbm__n_estimators': [150, 200, 250]
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    print("\n🎯 GELİŞTİRİLMİŞ Hiperparametre Optimizasyonu...")
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=25,
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
    print("\n🚀 Geliştirilmiş final model eğitimi...")
    
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
        n_estimators=400,
        early_stopping_rounds=50,
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
    
    # 7. Geliştirilmiş model değerlendirme
    print("\n📊 Kapsamlı Model Değerlendirme:")
    evaluate_improved_model(final_model, X_test_selected, y_test, X_train_selected, y_train)
    
    # 8. Modeli kaydet
    save_improved_model(final_model, important_features, best_params)
    
    return final_model, important_features

def evaluate_improved_model(model, X_test, y_test, X_train, y_train):
    """GELİŞTİRİLMİŞ model değerlendirme"""
    
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
    
    # Precision değerleri
    homewin_precision = class_report['1']['precision']
    awaywin_precision = class_report['2']['precision']
    draw_precision = class_report['0']['precision']
    
    print(f"📈 Test Accuracy: {test_accuracy:.4f}")
    print(f"📈 Test F1-Score: {test_f1:.4f}")
    print(f"📈 Balanced Accuracy: {test_balanced_accuracy:.4f}")
    print(f"🎯 HomeWin Recall: {homewin_recall:.4f} (Precision: {homewin_precision:.4f})")
    print(f"🎯 AwayWin Recall: {awaywin_recall:.4f} (Precision: {awaywin_precision:.4f})")
    print(f"🎯 Draw Recall: {draw_recall:.4f} (Precision: {draw_precision:.4f})")
    print(f"🏋️ Train Accuracy: {train_accuracy:.4f}")
    print(f"📊 Accuracy Gap (Overfitting): {accuracy_gap:.4f}")
    
    # GELİŞTİRİLMİŞ BAŞARI KRİTERLERİ
    targets_achieved = 0
    total_targets = 6
    
    # Accuracy - GELİŞTİRİLMİŞ
    if test_accuracy >= 0.60:
        print("✅ HEDEF BAŞARILDI: Accuracy > 0.60")
        targets_achieved += 1
    elif test_accuracy >= 0.55:
        print("⚠️ KISMEN BAŞARILI: Accuracy > 0.55")
        targets_achieved += 0.5
    else:
        print(f"⚠️ HEDEF TUTMADI: Accuracy = {test_accuracy:.4f} (hedef: 0.60)")
    
    # HomeWin recall - Koru
    if homewin_recall >= 0.75:
        print("✅ HEDEF BAŞARILDI: HomeWin recall > 0.75")
        targets_achieved += 1
    elif homewin_recall >= 0.70:
        print("⚠️ KISMEN BAŞARILI: HomeWin recall > 0.70")
        targets_achieved += 0.5
    else:
        print(f"⚠️ HEDEF TUTMADI: HomeWin recall = {homewin_recall:.4f} (hedef: 0.75)")
    
    # Draw recall - GELİŞTİRİLMİŞ (Ana hedef)
    if draw_recall >= 0.20:
        print("✅ HEDEF BAŞARILDI: Draw recall > 0.20")
        targets_achieved += 1.5
    elif draw_recall >= 0.15:
        print("⚠️ KISMEN BAŞARILI: Draw recall > 0.15")
        targets_achieved += 1
    elif draw_recall >= 0.10:
        print("⚠️ DÜŞÜK: Draw recall > 0.10")
        targets_achieved += 0.5
    else:
        print(f"⚠️ HEDEF TUTMADI: Draw recall = {draw_recall:.4f} (hedef: 0.20)")
    
    # AwayWin recall - Koru
    if awaywin_recall >= 0.70:
        print("✅ HEDEF BAŞARILDI: AwayWin recall > 0.70")
        targets_achieved += 1
    elif awaywin_recall >= 0.65:
        print("⚠️ KISMEN BAŞARILI: AwayWin recall > 0.65")
        targets_achieved += 0.5
    else:
        print(f"⚠️ HEDEF TUTMADI: AwayWin recall = {awaywin_recall:.4f} (hedef: 0.70)")
    
    # Overfitting - MAKUL
    if accuracy_gap <= 0.08:
        print("✅ HEDEF BAŞARILDI: Overfitting gap < 0.08")
        targets_achieved += 1
    elif accuracy_gap <= 0.12:
        print("⚠️ KISMEN BAŞARILI: Overfitting gap < 0.12")
        targets_achieved += 0.5
    else:
        print(f"⚠️ HEDEF TUTMADI: Overfitting gap = {accuracy_gap:.4f} (hedef: 0.08)")
    
    # Balanced Accuracy
    if test_balanced_accuracy >= 0.55:
        print("✅ HEDEF BAŞARILDI: Balanced Accuracy > 0.55")
        targets_achieved += 0.5
    else:
        print(f"⚠️ HEDEF TUTMADI: Balanced Accuracy = {test_balanced_accuracy:.4f} (hedef: 0.55)")
    
    print(f"🎯 Toplam Başarı: {targets_achieved:.1f}/6")
    
    # PERFORMANS ANALİZİ
    print(f"\n🏆 GELİŞTİRİLMİŞ BUNDESLIGA PERFORMANS RAPORU:")
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
    
    # Draw improvement analysis
    if draw_recall > 0.15:
        print("🎯 DRAW TAHMİNİ: İyileşme sağlandı!")
    elif draw_recall > 0.05:
        print("🎯 DRAW TAHMİNİ: Küçük iyileşme, daha fazla çalışma gerekli")
    else:
        print("🎯 DRAW TAHMİNİ: Ciddi problem devam ediyor")
    
    print("\n🎯 Detaylı Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=['Draw', 'HomeWin', 'AwayWin']))

def save_improved_model(model, important_features, best_params):
    """GELİŞTİRİLMİŞ model kaydetme"""
    os.makedirs("models", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/bundesliga_model_improved_{timestamp}.pkl"
    joblib.dump(model, model_path)
    
    feature_info = {
        'important_features': important_features,
        'best_params': best_params,
        'timestamp': datetime.now().isoformat(),
        'model_version': 'improved_v11.1',
        'features_count': len(important_features),
        'draw_optimized': True,
        'age_features_integrated': True
    }
    joblib.dump(feature_info, "models/feature_info_improved.pkl")
    
    print(f"\n💾 Geliştirilmiş model kaydedildi: {model_path}")

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
    print("🏆 Bundesliga Tahmin Modeli - GELİŞTİRİLMİŞ v11.1")
    print("=" * 60)
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    try:
        model, important_features = train_realistic_model()
        
        print("\n🎉 GELİŞTİRİLMİŞ MODEL eğitimi başarıyla tamamlandı!")
        print(f"📋 Kullanılan önemli feature'lar: {len(important_features)}")
        
        # Yaş feature'larını göster
        age_features = [f for f in important_features if 'age' in f]
        draw_features = [f for f in important_features if 'draw' in f or 'similarity' in f]
        
        print(f"🔍 Yaş Feature'ları: {age_features}")
        print(f"🔍 Draw Feature'ları: {draw_features}")
        
        print("\n🏆 GELİŞTİRİLMİŞ BUNDESLIGA HEDEFLERİ:")
        print("✅ %60+ accuracy hedefi")
        print("✅ HomeWin recall > %75 hedefi") 
        print("✅ Draw recall > %20 hedefi (ANA HEDEF)")
        print("✅ AwayWin recall > %70 hedefi")
        print("✅ Overfitting gap < %8 hedefi")
        print("✅ Balanced Accuracy > %55 hedefi")
        print("🚀 YAŞ ORTALAMASI feature'ları entegre edildi")
        print("🎯 DRAW tahmini optimize edildi")
        
    except Exception as e:
        print(f"❌ Model eğitimi sırasında hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()