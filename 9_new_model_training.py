#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bundesliga Tahmin Modeli - ULTIMATE BALANCE v12.1
Overfitting Çözümü + Gelişmiş Regularization
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

# ========== ULTIMATE BALANCE KONFİGÜRASYON ==========
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
N_JOBS = -1
MAX_FEATURES = 18

# GÜNCEL VERİ YOLLARI
DATA_PATH = "data/bundesliga_matches_2023_2025_final_fe_team_values_cleaned.pkl"
PLAYER_DATA_PATH = "data/final_bundesliga_dataset_complete.xlsx"

# PROBLEMLİ FEATURE'LARI TANIMLA
PROBLEMATIC_FEATURES = [
    'id', 'score.fullTime.home', 'score.fullTime.away',
    'goals_difference', 'xg_difference', 'awayTeam.id', 'matchday',
    'home_goals', 'away_goals', 'home_xg', 'away_xg', 'homeTeam.id',
    'home_injury_count', 'away_injury_count', 'home_last5_form_points', 
    'away_last5_form_points', 'performance_ratio', 'home_goals', 'away_goals'
]

# ========== AGGRESSIVE CLASS WEIGHTS ==========
def compute_aggressive_class_weights(y_train):
    """AGRESİF class weights - DRAW odaklı"""
    class_counts = pd.Series(y_train).value_counts().sort_index()
    total_matches = len(y_train)
    
    # ÇOK AGRESİF dağılım - draw'ı önceliklendir
    expected_distribution = [0.40, 0.35, 0.25]  # Draw: 40%, HomeWin: 35%, AwayWin: 25%
    
    aggressive_weights = []
    for i, count in enumerate(class_counts):
        expected_count = total_matches * expected_distribution[i]
        weight = expected_count / count if count > 0 else 1.0
        
        # Draw için ÇOK AGRESİF weighting
        if i == 0:  # Draw sınıfı
            weight = min(weight, 4.0)  # Draw'a 4x weight
        elif i == 1:  # HomeWin
            weight = min(weight, 1.2)
        else:  # AwayWin
            weight = min(weight, 1.2)
    
    class_weight_dict = dict(zip(class_counts.index, aggressive_weights))
    print(f"⚖️ AGGRESSIVE Class Weights: {class_weight_dict}")
    print(f"📊 Gerçek Dağılım: {dict(class_counts)}")
    print(f"🎯 Beklenen Dağılım: Draw: 40%, HomeWin: 35%, AwayWin: 25%")
    
    return class_weight_dict

# ========== OYUNCU VERİLERİNİ İŞLEME ==========
def load_and_process_player_data():
    """Oyuncu verilerini yükle ve takım yaş ortalamalarını hesapla"""
    print("📊 Oyuncu verileri yükleniyor ve yaş ortalamaları hesaplanıyor...")
    
    try:
        df_players = pd.read_excel(PLAYER_DATA_PATH)
        df_players.columns = [col.strip() for col in df_players.columns]
        print(f"📋 Oyuncu verisi sütunları: {list(df_players.columns)}")
        
        team_col = None
        age_col = None
        
        for col in df_players.columns:
            if 'team' in col.lower() or 'takım' in col.lower():
                team_col = col
            if 'age' in col.lower() or 'yaş' in col.lower():
                age_col = col
        
        if team_col is None or age_col is None:
            print("❌ Takım veya yaş sütunu bulunamadı!")
            return None
        
        print(f"✅ Takım sütunu: {team_col}, Yaş sütunu: {age_col}")
        
        df_players[age_col] = pd.to_numeric(df_players[age_col], errors='coerce')
        team_age_avg = df_players.groupby(team_col)[age_col].mean().reset_index()
        team_age_avg.columns = ['Team', 'Squad_Age_Avg']
        team_age_avg['Team_Normalized'] = team_age_avg['Team'].apply(normalize_team_name)
        
        print(f"📊 {len(team_age_avg)} takımın yaş ortalaması hesaplandı:")
        for _, row in team_age_avg.iterrows():
            print(f"   {row['Team']:25} → {row['Squad_Age_Avg']:.2f} yaş")
        
        return team_age_avg
        
    except Exception as e:
        print(f"❌ Oyuncu verileri yüklenirken hata: {e}")
        return None

def normalize_team_name(team_name):
    """Takım isimlerini normalize et"""
    if pd.isna(team_name):
        return None
    
    team_name = str(team_name).lower().strip()
    
    team_name = (team_name.replace('fc ', '')
                .replace('1. ', '')
                .replace('borussia ', '')
                .replace('sv ', '')
                .replace('tsg ', '')
                .replace('sc ', '')
                .replace('vfl ', '')
                .replace('fsv ', ''))
    
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
        'darmstadt': 'darmstadt 98'
    }
    
    for key, value in mapping.items():
        if key in team_name:
            return value
    
    return team_name

# ========== GÜNCEL VERİ YÜKLEME ==========
def load_integrated_data():
    """Entegre edilmiş güncel veriyi yükle"""
    print("🔄 Entegre edilmiş güncel veri yükleniyor...")
    
    try:
        if DATA_PATH.endswith('.pkl'):
            df_matches = pd.read_pickle(DATA_PATH)
        else:
            df_matches = pd.read_excel(DATA_PATH)
        
        print(f"✅ Ana veri seti yüklendi: {df_matches.shape}")
        
        team_age_data = load_and_process_player_data()
        
        if team_age_data is not None:
            df_matches['home_norm'] = df_matches['homeTeam.name'].apply(normalize_team_name)
            df_matches['away_norm'] = df_matches['awayTeam.name'].apply(normalize_team_name)
            
            age_mapping = team_age_data.set_index('Team_Normalized')['Squad_Age_Avg'].to_dict()
            
            df_matches['home_squad_avg_age'] = df_matches['home_norm'].map(age_mapping)
            df_matches['away_squad_avg_age'] = df_matches['away_norm'].map(age_mapping)
            
            print("✅ Oyuncu yaş ortalamaları entegre edildi")
        else:
            print("⚠️ Oyuncu verileri yüklenemedi, yaş ortalamaları mevcut veriden kullanılacak")
            if 'home_squad_avg_age' not in df_matches.columns:
                df_matches['home_squad_avg_age'] = 26.0
            if 'away_squad_avg_age' not in df_matches.columns:
                df_matches['away_squad_avg_age'] = 26.0
        
        df_matches['age_difference'] = df_matches['home_squad_avg_age'] - df_matches['away_squad_avg_age']
        
        print(f"📊 Yaş ortalaması istatistikleri:")
        print(f"   Ev sahibi: {df_matches['home_squad_avg_age'].mean():.2f} ± {df_matches['home_squad_avg_age'].std():.2f}")
        print(f"   Deplasman: {df_matches['away_squad_avg_age'].mean():.2f} ± {df_matches['away_squad_avg_age'].std():.2f}")
        print(f"   Yaş farkı: {df_matches['age_difference'].mean():.2f} ± {df_matches['age_difference'].std():.2f}")
        
        return df_matches
        
    except Exception as e:
        print(f"❌ Entegre veri yüklenirken hata: {e}")
        raise

# ========== BERABERLİK ODAKLI FEATURE ENGINEERING ==========
class UltimateBundesligaFeatureEngineer(BaseEstimator, TransformerMixin):
    """ULTIMATE feature engineering - BERABERLİK ODAKLI"""
    
    def __init__(self):
        self.feature_names = []
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        target_column = None
        if 'Result_Numeric' in X.columns:
            target_column = X['Result_Numeric']
        
        columns_to_drop = [col for col in PROBLEMATIC_FEATURES if col in X.columns and col != 'Result_Numeric']
        X = X.drop(columns=columns_to_drop, errors='ignore')
        
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        
        # 1. TEMEL FARKLAR
        if all(col in numeric_columns for col in ['home_ppg_cumulative', 'away_ppg_cumulative']):
            X['ppg_difference'] = X['home_ppg_cumulative'] - X['away_ppg_cumulative']
            X['ppg_similarity'] = 1 - (abs(X['ppg_difference']) / 3.0)  # BERABERLİK İÇİN
        
        if all(col in numeric_columns for col in ['home_gpg_cumulative', 'away_gpg_cumulative']):
            X['gpg_difference'] = X['home_gpg_cumulative'] - X['away_gpg_cumulative']
            X['total_goals_expected'] = (X['home_gpg_cumulative'] + X['away_gpg_cumulative']) / 2
            X['offensive_parity'] = 1 - (abs(X['gpg_difference']) / 3.0)  # BERABERLİK İÇİN
        
        # 2. GÜÇ METRİKLERİ
        if all(col in numeric_columns for col in ['home_power_index', 'away_power_index']):
            X['power_difference'] = X['home_power_index'] - X['away_power_index']
            X['strength_ratio'] = np.minimum(X['home_power_index'], X['away_power_index']) / (np.maximum(X['home_power_index'], X['away_power_index']) + 1e-8)
            X['power_similarity'] = 1 - (abs(X['power_difference']) / 2.0)  # BERABERLİK İÇİN
        
        # 3. YAŞ BAZLI ÖZELLİKLER
        if all(col in numeric_columns for col in ['home_squad_avg_age', 'away_squad_avg_age']):
            X['age_difference'] = X['home_squad_avg_age'] - X['away_squad_avg_age']
            X['age_similarity'] = 1 - (abs(X['age_difference']) / 5.0)  # BERABERLİK İÇİN
            X['experience_factor'] = (X['home_squad_avg_age'] * 0.6 + X['away_squad_avg_age'] * 0.4) / 25.0
        
        # 4. VALUE BAZLI ÖZELLİKLER
        if all(col in numeric_columns for col in ['home_current_value_eur', 'away_current_value_eur']):
            X['value_difference'] = X['home_current_value_eur'] - X['away_current_value_eur']
            X['value_ratio'] = X['home_current_value_eur'] / (X['away_current_value_eur'] + 1e-8)
            X['value_similarity'] = 1 - (abs(np.log1p(X['home_current_value_eur']) - np.log1p(X['away_current_value_eur'])) / 5.0)  # BERABERLİK İÇİN
        
        # 5. DEFANSİF DENGE - BERABERLİK İÇİN KRİTİK
        if all(col in numeric_columns for col in ['home_gapg_cumulative', 'away_gapg_cumulative']):
            X['defensive_strength'] = (X['home_gapg_cumulative'] + X['away_gapg_cumulative']) / 2
            X['defensive_parity'] = 1 - (abs(X['home_gapg_cumulative'] - X['away_gapg_cumulative']) / 2.0)  # BERABERLİK İÇİN
        
        # 6. FORM VE PERFORMANS
        if all(col in numeric_columns for col in ['home_form_5games', 'away_form_5games']):
            X['form_difference'] = X['home_form_5games'] - X['away_form_5games']
            X['form_similarity'] = 1 - (abs(X['form_difference']) / 2.0)  # BERABERLİK İÇİN
        
        # 7. H2H ÖZELLİKLERİ
        if all(col in numeric_columns for col in ['h2h_win_ratio', 'h2h_goal_difference']):
            X['h2h_dominance'] = X['h2h_win_ratio'] * 0.7 + (X['h2h_goal_difference'] / 10) * 0.3
            X['h2h_competitiveness'] = 1 - abs(X['h2h_win_ratio'] - 0.5)  # BERABERLİK İÇİN
        
        # 8. BERABERLİK POTANSİYELİ İNDEKSİ - YENİ VE KRİTİK
        draw_components = []
        if 'power_similarity' in X.columns:
            draw_components.append('power_similarity')
        if 'form_similarity' in X.columns:
            draw_components.append('form_similarity')
        if 'defensive_parity' in X.columns:
            draw_components.append('defensive_parity')
        if 'offensive_parity' in X.columns:
            draw_components.append('offensive_parity')
        if 'value_similarity' in X.columns:
            draw_components.append('value_similarity')
        if 'age_similarity' in X.columns:
            draw_components.append('age_similarity')
        if 'ppg_similarity' in X.columns:
            draw_components.append('ppg_similarity')
        
        if len(draw_components) >= 3:
            X['draw_potential_index'] = X[draw_components].mean(axis=1)
            print(f"✅ Beraberlik potansiyeli indeksi oluşturuldu ({len(draw_components)} component)")
        
        # 9. MAÇ DENGESİZLİK İNDEKSİ
        imbalance_components = []
        if 'power_difference' in X.columns:
            imbalance_components.append('power_difference')
        if 'form_difference' in X.columns:
            imbalance_components.append('form_difference')
        if 'value_difference' in X.columns:
            imbalance_components.append('value_difference')
        
        if len(imbalance_components) >= 2:
            X['match_imbalance_index'] = X[imbalance_components].std(axis=1)
            X['match_balance_index'] = 1 - X['match_imbalance_index']  # BERABERLİK İÇİN
        
        # HEDEF DEĞİŞKENİ GERİ EKLE
        if target_column is not None and 'Result_Numeric' not in X.columns:
            X['Result_Numeric'] = target_column
        
        numeric_cols = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_cols]
        
        self.feature_names = X.columns.tolist()
        print(f"🔧 ULTIMATE feature engineering sonrası {len(self.feature_names)} özellik oluşturuldu")
        print(f"🎯 Beraberlik odaklı {len(draw_components)} özellik eklendi")
        
        return X

# ========== THRESHOLD TUNING FONKSİYONLARI ==========
def predict_with_draw_threshold(model, X, draw_threshold=0.3):
    """Beraberlik için threshold tuning"""
    y_pred_proba = model.predict_proba(X)
    
    y_pred_custom = []
    for proba in y_pred_proba:
        draw_prob = proba[0]
        home_prob = proba[1]
        away_prob = proba[2]
        
        # Beraberlik olasılığı threshold'u geçerse ve diğerlerinden çok düşük değilse
        if draw_prob >= draw_threshold and draw_prob >= home_prob * 0.7 and draw_prob >= away_prob * 0.7:
            y_pred_custom.append(0)  # Draw
        else:
            # Normal prediction
            y_pred_custom.append(np.argmax(proba))
    
    return np.array(y_pred_custom)

def find_optimal_draw_threshold(model, X_val, y_val):
    """Validation set üzerinde optimal draw threshold'u bul"""
    print("🎯 Optimal draw threshold aranıyor...")
    
    best_threshold = 0.25
    best_score = 0
    best_draw_recall = 0
    
    thresholds = np.arange(0.15, 0.45, 0.05)
    
    for threshold in thresholds:
        y_val_pred = predict_with_draw_threshold(model, X_val, threshold)
        
        # Balanced accuracy ve draw recall kombinasyonu
        balanced_acc = balanced_accuracy_score(y_val, y_val_pred)
        draw_recall = recall_score(y_val, y_val_pred, labels=[0], average='macro', zero_division=0)
        
        # Draw recall'a daha fazla ağırlık ver
        score = 0.4 * balanced_acc + 0.6 * draw_recall
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
            best_draw_recall = draw_recall
    
    print(f"🎯 Optimal Draw Threshold: {best_threshold:.2f}")
    print(f"🎯 En iyi draw recall: {best_draw_recall:.4f}")
    print(f"🎯 En iyi skor: {best_score:.4f}")
    
    return best_threshold

# ========== CLEAN FEATURE SELECTION ==========
def clean_feature_selection(X_train, y_train, X_val, X_test, max_features=MAX_FEATURES):
    """Temiz feature selection - BERABERLİK feature'larını önceliklendir"""
    print(f"🧹 CLEAN Feature Selection (Max {max_features} özellik)...")
    
    X_train_clean = X_train.drop(columns=PROBLEMATIC_FEATURES, errors='ignore')
    X_val_clean = X_val.drop(columns=PROBLEMATIC_FEATURES, errors='ignore')
    X_test_clean = X_test.drop(columns=PROBLEMATIC_FEATURES, errors='ignore')
    
    print(f"🔍 Problemli {len(PROBLEMATIC_FEATURES)} feature kaldırıldı")
    print(f"🔢 Kalan sayısal sütun sayısı: {X_train_clean.select_dtypes(include=[np.number]).shape[1]}")
    
    # BERABERLİK feature'larını önceliklendir
    feature_scores = {}
    draw_keywords = ['similarity', 'parity', 'balance', 'draw', 'potential']
    
    # 1. RandomForest
    try:
        rf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, max_depth=5)
        rf.fit(X_train_clean, y_train)
        for i, feature in enumerate(X_train_clean.columns):
            base_score = rf.feature_importances_[i]
            
            # Beraberlik feature'larına bonus puan
            bonus = 1.0
            for keyword in draw_keywords:
                if keyword in feature.lower():
                    bonus += 0.5
                    break
            
            feature_scores[feature] = feature_scores.get(feature, 0) + base_score * bonus
    except Exception as e:
        print(f"⚠️ RandomForest feature selection hatası: {e}")
    
    # 2. LightGBM
    try:
        lgb_model = lgb.LGBMClassifier(n_estimators=50, random_state=RANDOM_STATE, verbose=-1)
        lgb_model.fit(X_train_clean, y_train)
        for i, feature in enumerate(X_train_clean.columns):
            base_score = lgb_model.feature_importances_[i]
            
            bonus = 1.0
            for keyword in draw_keywords:
                if keyword in feature.lower():
                    bonus += 0.5
                    break
            
            feature_scores[feature] = feature_scores.get(feature, 0) + base_score * bonus
    except Exception as e:
        print(f"⚠️ LightGBM feature selection hatası: {e}")
    
    # 3. Korelasyon bazlı
    try:
        for feature in X_train_clean.columns:
            if feature in X_train_clean.select_dtypes(include=[np.number]).columns:
                correlation = abs(np.corrcoef(X_train_clean[feature], y_train)[0, 1])
                if not np.isnan(correlation):
                    
                    bonus = 1.0
                    for keyword in draw_keywords:
                        if keyword in feature.lower():
                            bonus += 0.3
                            break
                    
                    feature_scores[feature] = feature_scores.get(feature, 0) + correlation * bonus
    except Exception as e:
        print(f"⚠️ Korelasyon feature selection hatası: {e}")
    
    # En iyileri seç
    if not feature_scores:
        print("🚨 Tüm feature selection yöntemleri başarısız, tüm sayısal feature'ları kullanıyoruz...")
        selected_features = X_train_clean.columns[:max_features].tolist()
    else:
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [feat for feat, score in sorted_features[:max_features]]
    
    # Beraberlik feature'larını kontrol et
    draw_features = [feat for feat in selected_features if any(keyword in feat.lower() for keyword in draw_keywords)]
    print(f"✅ Seçilen {len(draw_features)} beraberlik odaklı özellik:")
    for feat in draw_features:
        print(f"   📍 {feat}")
    
    problematic_found = [feat for feat in selected_features if feat in PROBLEMATIC_FEATURES]
    if problematic_found:
        print(f"🚨 UYARI: Seçilen feature'lar arasında problemli feature'lar bulundu: {problematic_found}")
        selected_features = [feat for feat in selected_features if feat not in PROBLEMATIC_FEATURES]
        backup_features = [feat for feat in X_train_clean.columns if feat not in selected_features and feat not in PROBLEMATIC_FEATURES]
        needed = max_features - len(selected_features)
        if needed > 0 and backup_features:
            selected_features.extend(backup_features[:needed])
    
    print(f"✅ Seçilen temiz özellikler ({len(selected_features)} adet)")
    
    X_train_selected = X_train_clean[selected_features]
    X_val_selected = X_val_clean[selected_features]
    X_test_selected = X_test_clean[selected_features]
    
    return X_train_selected, X_val_selected, X_test_selected, selected_features

# ========== FEATURE SELECTION STABILTY CHECK ==========
def check_feature_selection_stability(X_train, y_train, selected_features, n_iterations=3):
    """Feature selection stabilitesini kontrol et"""
    print("🔍 Feature selection stabilitesi kontrol ediliyor...")
    
    try:
        from sklearn.utils import resample
        
        feature_ranks = []
        for i in range(n_iterations):
            X_sample, y_sample = resample(X_train, y_train, random_state=i, stratify=y_train)
            
            # Basit bir LGBM ile özellik önemleri hesapla
            temp_model = lgb.LGBMClassifier(n_estimators=50, max_depth=3, random_state=42, verbose=-1)
            temp_model.fit(X_sample, y_sample)
            
            # Özellikleri önemlerine göre sırala
            importances = temp_model.feature_importances_
            sorted_indices = np.argsort(importances)[::-1]
            feature_ranks.append(sorted_indices)
        
        # Sıralamaların benzerliğini kontrol et
        top_features_per_run = []
        for rank in feature_ranks:
            top_features = [X_train.columns[idx] for idx in rank[:10]]  # İlk 10 özelliği al
            top_features_per_run.append(set(top_features))
        
        # Tüm çalıştırmalarda ortak olan özellikler
        common_top_features = set.intersection(*top_features_per_run)
        
        print(f"✅ Tutarlı şekilde seçilen önemli özellik sayısı: {len(common_top_features)}")
        
        # Seçilen özelliklerle karşılaştır
        selected_set = set(selected_features)
        overlap = selected_set.intersection(common_top_features)
        
        print(f"📊 Seçilen özelliklerin {len(overlap)} tanesi stabil özellikler listesinde")
        
        if len(overlap) >= len(selected_features) * 0.6:  # %60 overlap
            print("🎉 Feature selection stabilitesi: MÜKEMMEL")
        elif len(overlap) >= len(selected_features) * 0.4:  # %40 overlap
            print("✅ Feature selection stabilitesi: İYİ")
        else:
            print("⚠️ Feature selection stabilitesi: DÜŞÜK - Özellik seçimi tutarsız")
            
        return len(overlap)
        
    except Exception as e:
        print(f"⏭️ Stabilite kontrolü atlandı: {e}")
        return 0

# ========== ULTIMATE MODEL PIPELINE - OVERFITTING ÇÖZÜMLÜ ==========
def create_ultimate_pipeline(selected_features, class_weight_dict=None):
    """ULTIMATE model pipeline - OVERFITTING ÇÖZÜMLÜ ve BERABERLİK odaklı"""
    
    preprocessor = ColumnTransformer([
        ('scaler', RobustScaler(), selected_features)
    ], remainder='drop')
    
    # GÜNCELLENMİŞ PARAMETRELER: Daha basit ve daha fazla düzenlileştirmeli
    lgbm_params = {
        'objective': 'multiclass',
        'num_class': 3,
        'random_state': RANDOM_STATE,
        'n_jobs': N_JOBS,
        'verbosity': -1,
        'n_estimators': 150,       # Azaltıldı: Ağaç sayısını sınırla
        'learning_rate': 0.05,     # Düşük öğrenme oranı
        'max_depth': 3,            # Azaltıldı: Daha sığ ağaçlar
        'num_leaves': 8,           # ÖNEMLİ: Büyük ölçüde azaltıldı. num_leaves < 2^(max_depth) olmalı.
        'min_child_samples': 30,   # Arttırıldı: Bir yaprakta gereken minimum veri sayısı
        'subsample': 0.7,          # Azaltıldı: Her ağaç için kullanılan verinin %70'i
        'colsample_bytree': 0.7,   # Azaltıldı: Her ağaç için kullanılan feature'ların %70'i
        'reg_alpha': 3.0,          # Arttırıldı: L1 düzenlileştirme (Ağaç başına ceza)
        'reg_lambda': 3.0,         # Arttırıldı: L2 düzenlileştirme (Yaprak ağırlıklarına ceza)
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

# ========== MANUEL CUMULATIVE STATS ==========
def calculate_cumulative_stats(df_matches):
    """Maç verisinden takımların kümülatif istatistiklerini hesapla"""
    print("🔄 Gelişmiş cumulative istatistikler hesaplanıyor...")
    
    df = df_matches.copy()
    
    if 'Date' not in df.columns and 'utcDate' in df.columns:
        df['Date'] = pd.to_datetime(df['utcDate'], errors='coerce')
    
    if 'Date' in df.columns:
        df = df.sort_values('Date').reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    
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
        
        home_goals = match.get('score.fullTime.home', match.get('home_goals', 0))
        away_goals = match.get('score.fullTime.away', match.get('away_goals', 0))
        
        team_stats[home_team]['goals_for'] += home_goals
        team_stats[home_team]['goals_against'] += away_goals
        team_stats[home_team]['matches'] += 1
        team_stats[home_team]['goal_diff'] += (home_goals - away_goals)
        
        team_stats[away_team]['goals_for'] += away_goals
        team_stats[away_team]['goals_against'] += home_goals
        team_stats[away_team]['matches'] += 1
        team_stats[away_team]['goal_diff'] += (away_goals - home_goals)
        
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
        
        team_stats[home_team]['recent_results'] = team_stats[home_team]['recent_results'][-5:]
        team_stats[away_team]['recent_results'] = team_stats[away_team]['recent_results'][-5:]
    
    print(f"✅ Gelişmiş cumulative istatistikler hesaplandı: {len(team_stats)} takım")
    return df

def calculate_form(recent_results):
    """Son 5 maç formunu hesapla"""
    if not recent_results:
        return 0.5
    return sum(recent_results) / len(recent_results)

# ========== ULTIMATE VERİ HAZIRLAMA ==========
def ultimate_data_preparation(df_matches):
    """ULTIMATE veri hazırlama - BERABERLİK ODAKLI"""
    print("🔧 ULTIMATE veri hazırlama başlıyor...")
    
    df = df_matches.copy()
    
    if 'homeTeam.name' in df.columns and 'HomeTeam' not in df.columns:
        df['HomeTeam'] = df['homeTeam.name']
    if 'awayTeam.name' in df.columns and 'AwayTeam' not in df.columns:
        df['AwayTeam'] = df['awayTeam.name']
    
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
    
    if 'utcDate' in df.columns:
        df['Date'] = pd.to_datetime(df['utcDate'], errors='coerce')
        df = df.sort_values('Date').reset_index(drop=True)
    
    df = calculate_cumulative_stats(df)
    df = ultimate_missing_value_imputation(df)
    df = ultimate_feature_engineering(df)
    
    print("✅ ULTIMATE veri hazırlama tamamlandı!")
    return df

def ultimate_missing_value_imputation(df):
    """ULTIMATE eksik değer doldurma"""
    print("📊 ULTIMATE eksik değer analizi ve doldurma...")
    
    imputation_strategies = {
        'h2h_win_ratio': 0.5, 'h2h_goal_difference': 0, 'h2h_avg_goals': 2.5,
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
        'home_squad_avg_age': 26.0, 'away_squad_avg_age': 26.0,
        'age_difference': 0, 'value_difference': 0,
        'value_ratio': 1.0, 'draw_potential_index': 0.3,
        'power_similarity': 0.5, 'form_similarity': 0.5,
        'defensive_parity': 0.5, 'offensive_parity': 0.5,
        'value_similarity': 0.5, 'age_similarity': 0.5,
        'ppg_similarity': 0.5, 'match_balance_index': 0.5
    }
    
    for column, default_value in imputation_strategies.items():
        if column in df.columns:
            null_count = df[column].isnull().sum()
            if null_count > 0:
                df[column].fillna(default_value, inplace=True)
                if null_count > 10:
                    print(f"   📝 {column}: {null_count} NaN değer {default_value} ile dolduruldu")
    
    return df

def ultimate_feature_engineering(df):
    """ULTIMATE feature engineering"""
    df = df.copy()
    
    feature_engineer = UltimateBundesligaFeatureEngineer()
    df = feature_engineer.fit_transform(df)
    
    return df

# ========== ULTIMATE MODEL EĞİTİMİ - OVERFITTING ÇÖZÜMLÜ ==========
def train_ultimate_model():
    """ULTIMATE MODEL EĞİTİMİ - OVERFITTING ÇÖZÜMLÜ ve BERABERLİK OPTİMİZASYONLU"""
    print("⚽ Bundesliga Tahmin Modeli - ULTIMATE BALANCE v12.1")
    print("=" * 70)
    print("🎯 ULTIMATE HEDEFLER: %60+ Accuracy + %25+ Draw Recall")
    print("🎯 OVERFITTING ÇÖZÜMÜ: Basit Model + Güçlü Regularization") 
    print("🎯 Beraberlik Odaklı Feature Engineering")
    print("🎯 Aggressive Class Weighting")
    print("🎯 Threshold Tuning")
    print("=" * 70)
    
    df = load_integrated_data()
    df = ultimate_data_preparation(df)
    
    train_df, val_df, test_df = time_based_split(df, TEST_SIZE, VAL_SIZE)
    
    X_train = train_df.drop(columns=['Result_Numeric'], errors='ignore')
    y_train = train_df['Result_Numeric'].copy()
    
    X_val = val_df.drop(columns=['Result_Numeric'], errors='ignore')
    y_val = val_df['Result_Numeric'].copy()
    
    X_test = test_df.drop(columns=['Result_Numeric'], errors='ignore')
    y_test = test_df['Result_Numeric'].copy()
    
    print(f"🎯 Hedef değişken dağılımı - Eğitim: {y_train.value_counts().to_dict()}")
    
    feature_engineer = UltimateBundesligaFeatureEngineer()
    X_train = feature_engineer.fit_transform(X_train)
    X_val = feature_engineer.transform(X_val)
    X_test = feature_engineer.transform(X_test)
    
    print(f"🔢 Eğitim verisi shape: {X_train.shape}")
    print(f"🔢 Sayısal sütun sayısı: {X_train.select_dtypes(include=[np.number]).shape[1]}")
    
    X_train_selected, X_val_selected, X_test_selected, important_features = clean_feature_selection(
        X_train, y_train, X_val, X_test, MAX_FEATURES
    )
    
    # Feature selection stabilite kontrolü
    check_feature_selection_stability(X_train, y_train, important_features)
    
    print(f"📊 Eğitim verisi: {X_train_selected.shape}")
    print(f"📊 Validation verisi: {X_val_selected.shape}")
    print(f"📊 Test verisi: {X_test_selected.shape}")
    
    class_weight_dict = compute_aggressive_class_weights(y_train)
    model = create_ultimate_pipeline(important_features, class_weight_dict)
    
    # GÜNCELLENMİŞ Hiperparametre Arama Uzayı
    param_distributions = {
        'lgbm__learning_rate': [0.03, 0.05],  # Daha yüksek değerler kaldırıldı
        'lgbm__max_depth': [3, 4],            # 5 kaldırıldı
        'lgbm__num_leaves': [7, 8, 10],       # Çok daha düşük değerler
        'lgbm__min_child_samples': [25, 30, 35], # Daha yüksek değerler
        'lgbm__subsample': [0.7, 0.75],
        'lgbm__colsample_bytree': [0.7, 0.75],
        'lgbm__reg_alpha': [2.0, 3.0, 4.0],   # Daha güçlü düzenlileştirme
        'lgbm__reg_lambda': [2.0, 3.0, 4.0],  # Daha güçlü düzenlileştirme
        'lgbm__n_estimators': [100, 125, 150] # Daha küçük değerler
    }
    
    tscv = TimeSeriesSplit(n_splits=5)
    
    print("\n🎯 ULTIMATE Hiperparametre Optimizasyonu (Overfitting Önleyici)...")
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=20,  # Daha az iterasyon
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
    
    # Overfitting kontrolü - train ve test skorlarını karşılaştır
    train_scores = random_search.cv_results_['mean_train_score']
    test_scores = random_search.cv_results_['mean_test_score']
    avg_gap = np.mean([train - test for train, test in zip(train_scores, test_scores)])
    print(f"📊 CV Overfitting Gap Ortalaması: {avg_gap:.4f}")
    
    print("\n🚀 Final model eğitimi (OVERFITTING ÇÖZÜMLÜ)...")
    
    final_pipeline = create_ultimate_pipeline(important_features, class_weight_dict)
    
    for param_name, param_value in best_params.items():
        final_pipeline.set_params(**{param_name: param_value})
    
    preprocessor = final_pipeline.named_steps['preprocessor']
    lgbm = final_pipeline.named_steps['lgbm']
    
    X_train_processed = preprocessor.fit_transform(X_train_selected)
    X_val_processed = preprocessor.transform(X_val_selected)
    
    # Daha agresif early stopping
    lgbm.set_params(
        n_estimators=200,  # Daha az ağaç
        early_stopping_rounds=30,  # Daha agresif early stopping
        verbose=20
    )
    
    lgbm.fit(
        X_train_processed, y_train,
        eval_set=[(X_val_processed, y_val)],
        eval_metric='multi_logloss',
        callbacks=[lgb.early_stopping(30), lgb.log_evaluation(20)]
    )
    
    final_model = Pipeline([
        ('preprocessor', preprocessor),
        ('lgbm', lgbm)
    ])
    
    optimal_threshold = find_optimal_draw_threshold(final_model, X_val_selected, y_val)
    
    print("\n📊 ULTIMATE Model Değerlendirme (Overfitting Çözümlü):")
    evaluate_ultimate_model(final_model, X_test_selected, y_test, X_train_selected, y_train, optimal_threshold)
    
    save_ultimate_model(final_model, important_features, best_params, optimal_threshold)
    
    return final_model, important_features, optimal_threshold

def evaluate_ultimate_model(model, X_test, y_test, X_train, y_train, draw_threshold=0.25):
    """ULTIMATE model değerlendirme"""
    
    y_pred_test = predict_with_draw_threshold(model, X_test, draw_threshold)
    y_pred_train = predict_with_draw_threshold(model, X_train, draw_threshold)
    
    test_accuracy = accuracy_score(y_test, y_pred_test)
    test_f1 = f1_score(y_test, y_pred_test, average='weighted')
    test_balanced_accuracy = balanced_accuracy_score(y_test, y_pred_test)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    accuracy_gap = train_accuracy - test_accuracy
    
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
    
    # OVERFITTING ÇÖZÜMÜ BAŞARI METRİKLERİ
    print(f"\n🎯 OVERFITTING ÇÖZÜMÜ PERFORMANSI:")
    
    targets_achieved = 0
    total_targets = 5
    
    if test_accuracy >= 0.60:  # Hedef %60+ accuracy
        print("✅ HEDEF BAŞARILDI: Accuracy > 0.60")
        targets_achieved += 1
    elif test_accuracy >= 0.55:
        print("⚠️ KISMEN BAŞARILI: Accuracy > 0.55")
        targets_achieved += 0.5
    else:
        print(f"⚠️ HEDEF TUTMADI: Accuracy = {test_accuracy:.4f} (hedef: 0.60)")
    
    if homewin_recall >= 0.60:
        print("✅ HEDEF BAŞARILDI: HomeWin recall > 0.60")
        targets_achieved += 1
    elif homewin_recall >= 0.55:
        print("⚠️ KISMEN BAŞARILI: HomeWin recall > 0.55")
        targets_achieved += 0.5
    else:
        print(f"⚠️ HEDEF TUTMADI: HomeWin recall = {homewin_recall:.4f} (hedef: 0.60)")
    
    if draw_recall >= 0.25:
        print("✅ HEDEF BAŞARILDI: Draw recall > 0.25")
        targets_achieved += 1
    elif draw_recall >= 0.20:
        print("⚠️ KISMEN BAŞARILI: Draw recall > 0.20")
        targets_achieved += 0.5
    else:
        print(f"⚠️ HEDEF TUTMADI: Draw recall = {draw_recall:.4f} (hedef: 0.25)")
    
    if awaywin_recall >= 0.50:
        print("✅ HEDEF BAŞARILDI: AwayWin recall > 0.50")
        targets_achieved += 1
    elif awaywin_recall >= 0.45:
        print("⚠️ KISMEN BAŞARILI: AwayWin recall > 0.45")
        targets_achieved += 0.5
    else:
        print(f"⚠️ HEDEF TUTMADI: AwayWin recall = {awaywin_recall:.4f} (hedef: 0.50)")
    
    # OVERFITTING GAP HEDEFİ (Ana hedef)
    if accuracy_gap <= 0.10:  # Daha gerçekçi hedef
        print("✅ HEDEF BAŞARILDI: Overfitting gap < 0.10")
        targets_achieved += 1
    elif accuracy_gap <= 0.15:
        print("⚠️ KISMEN BAŞARILI: Overfitting gap < 0.15")
        targets_achieved += 0.5
    else:
        print(f"⚠️ HEDEF TUTMADI: Overfitting gap = {accuracy_gap:.4f} (hedef: 0.10)")
    
    print(f"🎯 Toplam Başarı: {targets_achieved:.1f}/5")
    
    print(f"\n🏆 ULTIMATE BUNDESLIGA PERFORMANS RAPORU:")
    print(f"📊 Tahmini Dağılım: Draw: {draw_recall:.1%}, HomeWin: {homewin_recall:.1%}, AwayWin: {awaywin_recall:.1%}")
    print(f"📈 Beklenen Dağılım: Draw: ~25%, HomeWin: ~45%, AwayWin: ~30%")
    
    if test_accuracy >= 0.65:
        print("🎉 MÜKEMMEL: Üst düzey accuracy!")
    elif test_accuracy >= 0.60:
        print("✅ ÇOK İYİ: Ultimate model ile harika accuracy!")
    elif test_accuracy >= 0.55:
        print("✅ İYİ: Bundesliga için makul accuracy!")
    elif test_accuracy >= 0.50:
        print("⚠️ ORTA: Geliştirme gerekli!")
    else:
        print("🔴 ZAYIF: Temel problemi çöz!")
    
    if accuracy_gap <= 0.15:
        print("🎉 OVERFITTING ÇÖZÜLDÜ: Model genelleme performansı iyi!")
    elif accuracy_gap <= 0.20:
        print("✅ OVERFITTING AZALTILDI: Model genellemesi kabul edilebilir!")
    else:
        print("🔴 OVERFITTING YÜKSEK: Model hala aşırı uyum sağlıyor!")
    
    if draw_recall >= 0.25:
        print("🎉 BERABERLİK TAHMİNİ BAŞARILI: Model artık draw'ları tahmin edebiliyor!")
    else:
        print("🔴 BERABERLİK TAHMİNİ ZAYIF: Draw'lar hala problemli")
    
    print("\n🎯 Detaylı Classification Report:")
    print(classification_report(y_test, y_pred_test, target_names=['Draw', 'HomeWin', 'AwayWin']))

def save_ultimate_model(model, important_features, best_params, optimal_threshold):
    """ULTIMATE model kaydetme"""
    os.makedirs("models", exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = f"models/bundesliga_model_ultimate_v12.1_{timestamp}.pkl"
    joblib.dump(model, model_path)
    
    feature_info = {
        'important_features': important_features,
        'best_params': best_params,
        'optimal_threshold': optimal_threshold,
        'timestamp': datetime.now().isoformat(),
        'model_version': 'ultimate_balance_v12.1_overfitting_fixed'
    }
    joblib.dump(feature_info, "models/feature_info_ultimate_v12.1.pkl")
    
    print(f"\n💾 Model kaydedildi: {model_path}")
    print(f"📋 Özellik bilgisi kaydedildi: models/feature_info_ultimate_v12.1.pkl")
    print(f"🎯 Optimal Draw Threshold: {optimal_threshold}")

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
    print("🏆 Bundesliga Tahmin Modeli - ULTIMATE BALANCE v12.1")
    print("=" * 60)
    print("🎯 OVERFITTING ÇÖZÜMLÜ + %60+ ACCURACY HEDEFLİ")
    print("=" * 60)
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    try:
        model, important_features, optimal_threshold = train_ultimate_model()
        
        print("\n🎉 ULTIMATE BALANCE v12.1 MODEL eğitimi başarıyla tamamlandı!")
        print(f"📋 Kullanılan önemli feature'lar: {len(important_features)}")
        print(f"🎯 Optimal Draw Threshold: {optimal_threshold}")
        
        print("\n🏆 ULTIMATE BUNDESLIGA HEDEFLERİ (v12.1):")
        print("✅ %60+ accuracy hedefi (Overfitting çözümlü)")
        print("✅ %25+ Draw recall hedefi") 
        print("✅ HomeWin recall > %60 hedefi")
        print("✅ AwayWin recall > %50 hedefi")
        print("✅ Overfitting gap < %10 hedefi")
        print("✅ Beraberlik odaklı feature engineering")
        print("✅ Aggressive class weighting")
        print("✅ Threshold tuning")
        print("✅ Feature selection stabilite kontrolü")
        
    except Exception as e:
        print(f"❌ Model eğitimi sırasında hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()