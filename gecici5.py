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
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# ========== GELİŞTİRİLMİŞ KONFİGÜRASYON ==========
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
N_JOBS = -1

DATA_PATH = "data/bundesliga_matches_2023_2025_final_fe_team_values_cleaned.xlsx"
PLAYER_DATA_PATH = "data/final_bundesliga_dataset_complete.xlsx"

# Geliştirilmiş özellik listesi
SELECTED_FEATURES = [
    # Takım Değer ve Demografi Özellikleri
    'home_current_value_eur', 'away_current_value_eur',
    'home_previous_value_eur', 'away_previous_value_eur',
    'home_value_change_pct', 'away_value_change_pct',
    'home_squad_avg_age', 'away_squad_avg_age',
    'home_absolute_change', 'away_absolute_change',
    'home_log_current_value', 'away_log_current_value',
    'value_difference', 'value_ratio',
    
    # Performans ve Form Özellikleri
    'home_goals', 'away_goals', 'home_xg', 'away_xg',
    'goals_difference', 'goals_ratio', 'xg_difference', 'xg_ratio',
    'home_form', 'away_form', 'form_difference',
    'home_last5_form_points', 'away_last5_form_points',
    
    # H2H (Head-to-Head) Özellikleri
    'h2h_home_wins', 'h2h_away_wins', 'h2h_draws',
    'h2h_home_goals', 'h2h_away_goals', 'h2h_matches_count',
    'h2h_win_ratio', 'h2h_goal_difference', 'h2h_avg_goals',
    
    # Derby ve Özel Durum Özellikleri
    'isDerby', 'age_difference', 'injury_difference',
    
    # Power Index ve Advanced Metrics
    'home_power_index', 'away_power_index', 'power_difference',
    'performance_ratio'
]

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
                X[f'{col}_sqrt'] = np.sqrt(X[col])
        
        # 2. Interaction özellikleri - OVERFITTING'E NEDEN OLANLARI KALDIRDIK
        # form_value_interaction_home ve form_value_interaction_away overfitting'e neden oluyor
        # Bu yüzden bu karmaşık interaction'ları kaldırıyoruz
        
        # 3. Rolling ortalamalar
        if 'matchday' in X.columns:
            X['matchday_sin'] = np.sin(2 * np.pi * X['matchday'] / 34)
            X['matchday_cos'] = np.cos(2 * np.pi * X['matchday'] / 34)
        
        # 4. Kategori bazlı özellikler
        if 'h2h_win_ratio' in X.columns:
            X['h2h_dominant'] = (X['h2h_win_ratio'] > 0.6).astype(int)
            X['h2h_balanced'] = ((X['h2h_win_ratio'] >= 0.4) & (X['h2h_win_ratio'] <= 0.6)).astype(int)
        
        # 5. Gelişmiş power metrics
        if all(col in X.columns for col in ['home_power_index', 'away_power_index']):
            X['power_ratio'] = X['home_power_index'] / X['away_power_index']
            X['power_sum'] = X['home_power_index'] + X['away_power_index']
        
        self.feature_names = X.columns.tolist()
        return X

# ========== FEATURE SELECTION FONKSİYONLARI ==========
def perform_strict_feature_selection(X_train, y_train, X_val, X_test, method='importance'):
    """OVERFITTING ÖNLEMEK İÇİN DAHA STRICT FEATURE SELECTION"""
    print("🔍 STRICT Feature selection yapılıyor...")
    
    if method == 'importance':
        # Random Forest ile feature importance - DAHA AGRESIF THRESHOLD
        estimator = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1)
        selector = SelectFromModel(estimator, threshold='mean')  # 'mean' daha agresif
        
        selector.fit(X_train, y_train)
        selected_features = X_train.columns[selector.get_support()].tolist()
        
        # Eğer hala çok fazla özellik varsa, en iyi 10-15 tanesini al
        if len(selected_features) > 15:
            print(f"⚡ Çok fazla özellik seçildi ({len(selected_features)}), en iyi 15 tanesi alınıyor...")
            estimator.fit(X_train, y_train)
            importances = estimator.feature_importances_
            indices = np.argsort(importances)[::-1]
            selected_features = [X_train.columns[i] for i in indices[:15]]
        
    elif method == 'rfe':
        # Recursive Feature Elimination - DAHA AZ ÖZELLİK
        estimator = RandomForestClassifier(n_estimators=50, random_state=RANDOM_STATE)
        rfe = RFE(estimator=estimator, n_features_to_select=min(15, X_train.shape[1]))  # Max 15 özellik
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
    Geliştirilmiş veri hazırlama ve zenginleştirme fonksiyonu
    """
    print("🔧 Geliştirilmiş veri hazırlama başlıyor...")
    
    # 1. Temel veri temizliği
    df = df_matches.copy()
    
    # 2. Takım isimlerini standartlaştır
    if 'homeTeam.name' in df.columns and 'HomeTeam' not in df.columns:
        df['HomeTeam'] = df['homeTeam.name']
    if 'awayTeam.name' in df.columns and 'AwayTeam' not in df.columns:
        df['AwayTeam'] = df['awayTeam.name']
    
    # 3. Result_Numeric oluştur
    def safe_get_result(row):
        try:
            home_goals = row.get('score.fullTime.home', row.get('score.fullTime.home', 0))
            away_goals = row.get('score.fullTime.away', row.get('score.fullTime.away', 0))
            
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
        'home_last5_form_points': 0, 'away_last5_form_points': 0,
        
        'home_current_value_eur': df['home_current_value_eur'].median() if 'home_current_value_eur' in df.columns else 200000000,
        'away_current_value_eur': df['away_current_value_eur'].median() if 'away_current_value_eur' in df.columns else 200000000,
        
        'home_goals': df['home_goals'].median() if 'home_goals' in df.columns else 1.5,
        'away_goals': df['away_goals'].median() if 'away_goals' in df.columns else 1.5,
    }
    
    for column, default_value in imputation_strategies.items():
        if column in df.columns:
            null_count = df[column].isnull().sum()
            if null_count > 0:
                if callable(default_value):
                    df[column].fillna(default_value(df), inplace=True)
                else:
                    df[column].fillna(default_value, inplace=True)
    
    return df

def simplified_feature_engineering(df):
    """OVERFITTING'I ÖNLEMEK İÇİN SADELEŞTİRİLMİŞ FEATURE ENGINEERING"""
    print("🎯 Sadeleştirilmiş özellik mühendisliği...")
    df = df.copy()
    
    # SADECE EN TEMEL VE ANLAMLI ÖZELLİKLER
    # 1. Value-based özellikler
    if all(col in df.columns for col in ['home_current_value_eur', 'away_current_value_eur']):
        df['value_difference'] = df['home_current_value_eur'] - df['away_current_value_eur']
        df['value_ratio'] = df['home_current_value_eur'] / (df['away_current_value_eur'] + 1e-8)
    
    # 2. Form-based özellikler
    if all(col in df.columns for col in ['home_form', 'away_form']):
        df['form_difference'] = df['home_form'] - df['away_form']
        df['form_sum'] = df['home_form'] + df['away_form']
    
    # 3. Goal-based özellikler
    if all(col in df.columns for col in ['home_goals', 'away_goals']):
        df['goals_difference'] = df['home_goals'] - df['away_goals']
        df['total_goals'] = df['home_goals'] + df['away_goals']
    
    # KARMAŞIK INTERACTION FEATURE'LARI ÇIKARDIK
    # form_value_interaction gibi overfitting'e neden olan feature'lar kaldırıldı
    
    return df

def advanced_feature_engineering(df):
    """Gelişmiş özellik mühendisliği - Sadeleştirilmiş versiyon"""
    print("🎯 Gelişmiş özellik mühendisliği...")
    
    # Önce temel özellikleri oluştur
    df = simplified_feature_engineering(df)
    
    # 1. Polynomial özellikler (sınırlı sayıda)
    if all(col in df.columns for col in ['home_form', 'away_form']):
        df['form_product'] = df['home_form'] * df['away_form']
    
    # 2. Ratio-based özellikler
    if all(col in df.columns for col in ['home_current_value_eur', 'away_current_value_eur']):
        df['value_ratio_log'] = np.log1p(df['home_current_value_eur']) - np.log1p(df['away_current_value_eur'])
    
    # 3. Momentum-based özellikler
    if 'home_last5_form_points' in df.columns and 'away_last5_form_points' in df.columns:
        df['momentum_difference'] = df['home_last5_form_points'] - df['away_last5_form_points']
    
    # 4. H2H dominance özellikleri
    if 'h2h_win_ratio' in df.columns:
        df['h2h_dominance'] = (df['h2h_win_ratio'] - 0.5) * df['h2h_matches_count'].clip(upper=10)  # Max 10 maç
    
    # 5. Power metrics
    if all(col in df.columns for col in ['home_power_index', 'away_power_index']):
        df['relative_power'] = df['home_power_index'] / (df['away_power_index'] + 1e-8)
        df['power_advantage'] = df['home_power_index'] - df['away_power_index']
    
    return df

def handle_outliers(df):
    """Outlier'ları işleme"""
    print("📊 Outlier handling...")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    
    for col in numeric_cols:
        if col not in ['Result_Numeric', 'isDerby']:
            Q1 = df[col].quantile(0.05)
            Q3 = df[col].quantile(0.95)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            df[col] = np.clip(df[col], lower_bound, upper_bound)
    
    return df

# ========== GELİŞTİRİLMİŞ RATING HESAPLAMA ==========
def compute_enhanced_ratings(df, df_players):
    """Geliştirilmiş takım rating hesaplama"""
    print("⭐ Geliştirilmiş takım ratingleri hesaplanıyor...")
    
    if 'Home_AvgRating' not in df.columns:
        df['Home_AvgRating'] = 65.0
        df['Away_AvgRating'] = 65.0
    
    rating_cols = ['Home_AvgRating', 'Away_AvgRating', 'Home_GK_Rating', 'Home_DF_Rating', 
                   'Home_MF_Rating', 'Home_FW_Rating', 'Away_GK_Rating', 'Away_DF_Rating', 
                   'Away_MF_Rating', 'Away_FW_Rating']
    
    for col in rating_cols:
        if col not in df.columns:
            if 'AvgRating' in col:
                df[col] = 65.0
            else:
                df[col] = 65.0
    
    if all(col in df.columns for col in ['Home_AvgRating', 'Away_AvgRating']):
        df['Rating_Diff'] = df['Home_AvgRating'] - df['Away_AvgRating']
        df['Total_AvgRating'] = df['Home_AvgRating'] + df['Away_AvgRating']
    
    return df

# ========== OVERFITTING ÖNLEYİCİ MODEL PIPELINE ==========
def create_overfitting_prevention_pipeline(selected_features):
    """OVERFITTING ÖNLEMEK İÇİN BASİT VE REGULARIZE EDİLMİŞ PIPELINE"""
    
    # Sadece scaler ve model - feature selection pipeline dışında
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
        n_estimators=500,  # ⬅️ ÇOK AZALTILDI (2000 → 500)
        learning_rate=0.01,  # ⬅️ DÜŞÜRÜLDÜ (0.05 → 0.01)
        max_depth=3,  # ⬅️ DERİNLİK AZALTILDI (6 → 3)
        num_leaves=10,  # ⬅️ ÇOK AZALTILDI (31 → 10)
        min_child_samples=50,  # ⬅️ ARTIRILDI (20 → 50)
        subsample=0.6,  # ⬅️ AZALTILDI
        colsample_bytree=0.6,  # ⬅️ AZALTILDI
        reg_alpha=2.0,  # ⬅️ REGULARIZATION ARTIRILDI (0.1 → 2.0)
        reg_lambda=2.0,  # ⬅️ REGULARIZATION ARTIRILDI (0.1 → 2.0)
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

# ========== OVERFITTING ÖNLEYİCİ MODEL EĞİTİMİ ==========
def train_enhanced_model():
    """OVERFITTING ÖNLEYİCİ model eğitim fonksiyonu"""
    print("⚽ Bundesliga Tahmin Modeli - Overfitting Önleyici Sürüm")
    print("=" * 70)
    print("✅ Advanced feature engineering")
    print("✅ STRICT feature selection (Max 15 özellik)") 
    print("✅ Robust outlier handling")
    print("✅ ENHANCED regularization")
    print("✅ Advanced cross-validation")
    print("✅ Class balancing techniques")
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
    
    # 2. Pipeline dışında STRICT feature selection yap
    X_train_selected, X_val_selected, X_test_selected, important_features = perform_strict_feature_selection(
        X_train, y_train, X_val, X_test, method='importance'
    )
    
    # 3. Sınıf ağırlıklarını hesapla
    classes = np.unique(y_train)
    class_weights = compute_class_weight('balanced', classes=classes, y=y_train)
    class_weight_dict = dict(zip(classes, class_weights))
    sample_weights_train = np.array([class_weight_dict[yy] for yy in y_train])
    
    print(f"📊 Eğitim verisi: {X_train_selected.shape}")
    print(f"📊 Validation verisi: {X_val_selected.shape}")
    print(f"📊 Test verisi: {X_test_selected.shape}")
    print(f"⚖️ Sınıf ağırlıkları: {class_weight_dict}")
    
    # 4. Overfitting önleyici pipeline oluştur
    model = create_overfitting_prevention_pipeline(important_features)
    
    # 5. OVERFITTING ÖNLEYİCİ hiperparametre optimizasyonu
    param_distributions = {
        'lgbm__learning_rate': [0.005, 0.01, 0.02],  # ⬅️ Daha düşük
        'lgbm__max_depth': [2, 3, 4],  # ⬅️ Daha sığ
        'lgbm__num_leaves': [8, 10, 12],  # ⬅️ Çok daha az
        'lgbm__min_child_samples': [40, 50, 60],  # ⬅️ Daha büyük
        'lgbm__reg_alpha': [1.0, 2.0, 3.0],  # ⬅️ Daha güçlü regularization
        'lgbm__reg_lambda': [1.0, 2.0, 3.0],
        'lgbm__subsample': [0.5, 0.6, 0.7],
        'lgbm__colsample_bytree': [0.5, 0.6, 0.7],
        'lgbm__n_estimators': [300, 500, 700]  # ⬅️ Çok daha az
    }
    
    # Daha fazla fold ve daha katı split
    tscv = TimeSeriesSplit(n_splits=10)  # ⬅️ Fold sayısını artır
    
    print("\n🎯 Overfitting Önleyici Hiperparametre Optimizasyonu...")
    random_search = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_distributions,
        n_iter=20,  # ⬅️ Daha az iterasyon
        cv=tscv,
        scoring='balanced_accuracy',  # ⬅️ Daha dengeli bir metrik
        n_jobs=N_JOBS,
        verbose=2,
        random_state=RANDOM_STATE,
        return_train_score=True
    )
    
    # Optimizasyonu gerçekleştir (seçilmiş özelliklerle)
    random_search.fit(X_train_selected, y_train, lgbm__sample_weight=sample_weights_train)
    
    # En iyi parametreler ve skor
    best_params = random_search.best_params_
    best_score = random_search.best_score_
    
    print(f"\n🏆 En İyi Parametreler: {best_params}")
    print(f"🏆 En İyi CV Skoru: {best_score:.4f}")
    
    # 6. Final modeli eğit (EARLY STOPPING ile)
    print("\n🚀 Final model eğitimi (Early Stopping ile)...")
    final_model = create_overfitting_prevention_pipeline(important_features)
    final_model.set_params(**best_params)
    
    # Early stopping için parametreleri ayarla
    final_model.named_steps['lgbm'].set_params(
        n_estimators=1000,  # Early stopping için yeterince büyük
        early_stopping_rounds=50,
        verbose=100
    )
    
    # Tüm veriyi birleştir (train + val)
    X_combined = pd.concat([X_train_selected, X_val_selected])
    y_combined = pd.concat([y_train, y_val])
    sample_weights_combined = np.array([class_weight_dict[yy] for yy in y_combined])
    
    # Final modeli EARLY STOPPING ile eğit
    final_model.fit(
        X_train_selected, y_train,
        lgbm__eval_set=[(X_val_selected, y_val)],
        lgbm__eval_metric='multi_logloss',
        lgbm__sample_weight=sample_weights_train,
        lgbm__callbacks=[lgb.early_stopping(50), lgb.log_evaluation(100)]
    )
    
    # 7. Model değerlendirme
    print("\n📊 Kapsamlı Model Değerlendirme:")
    evaluate_model_comprehensive(final_model, X_test_selected, y_test, X_train_selected, y_train)
    
    # 8. Feature importance analizi
    analyze_feature_importance(final_model, important_features)
    
    # 9. Modeli kaydet
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
    plt.title('Confusion Matrix - Test Set (Overfitting Önleyici)')
    plt.ylabel('Gerçek Değer')
    plt.xlabel('Tahmin Edilen Değer')
    plt.savefig('models/confusion_matrix_overfitting_fixed.png', dpi=300, bbox_inches='tight')
    plt.close()

def analyze_feature_importance(model, feature_names):
    """Feature importance analizi"""
    try:
        if hasattr(model.named_steps['lgbm'], 'feature_importances_'):
            importances = model.named_steps['lgbm'].feature_importances_
            indices = np.argsort(importances)[::-1]
            
            print("\n🏆 Feature Importance Ranking:")
            for i, idx in enumerate(indices[:15]):  # Sadece top 15
                if idx < len(feature_names):
                    print(f"{i+1:2d}. {feature_names[idx]:30s} ({importances[idx]:.4f})")
            
            # Görselleştirme
            plt.figure(figsize=(12, 8))
            top_n = min(10, len(feature_names))  # Sadece top 10
            plt.barh(range(top_n), importances[indices[:top_n]][::-1], align='center')
            plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]][::-1])
            plt.xlabel('Importance')
            plt.title('Top Feature Importances (Overfitting Önleyici)')
            plt.tight_layout()
            plt.savefig('models/feature_importance_overfitting_fixed.png', dpi=300, bbox_inches='tight')
            plt.close()
    
    except Exception as e:
        print(f"⚠️ Feature importance analizinde hata: {e}")

def save_enhanced_model(model, important_features, best_params, cv_results):
    """Geliştirilmiş model kaydetme"""
    os.makedirs("models", exist_ok=True)
    
    model_path = "models/bundesliga_model_overfitting_fixed.pkl"
    joblib.dump(model, model_path)
    
    feature_info = {
        'important_features': important_features,
        'all_features': SELECTED_FEATURES,
        'best_params': best_params,
        'cv_results': cv_results,
        'timestamp': datetime.now().isoformat(),
        'model_version': 'overfitting_prevention_v1'
    }
    joblib.dump(feature_info, "models/feature_info_overfitting_fixed.pkl")
    
    performance_report = {
        'model_type': 'LightGBM Overfitting Prevention',
        'features_used': len(important_features),
        'total_features': len(SELECTED_FEATURES),
        'training_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'max_features_limit': 15
    }
    
    with open("models/performance_report_overfitting_fixed.txt", "w") as f:
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
    print("🏆 Bundesliga Tahmin Modeli - Overfitting Önleyici Sürüm")
    print("=" * 60)
    print("🚀 Başlatılıyor...")
    
    os.makedirs("models", exist_ok=True)
    os.makedirs("data", exist_ok=True)
    
    try:
        model, important_features = train_enhanced_model()
        
        print("\n🎉 Overfitting önleyici model eğitimi başarıyla tamamlandı!")
        print(f"📋 Kullanılan önemli feature'lar: {len(important_features)}/{len(SELECTED_FEATURES)}")
        print("📍 Model dosyaları 'models/' klasörüne kaydedildi")
        
        # Overfitting durumunu değerlendir
        print("\n📊 Overfitting Önleme Özeti:")
        print("✅ Model karmaşıklığı azaltıldı")
        print("✅ Feature sayısı sınırlandı (max 15)")
        print("✅ Regularization artırıldı")
        print("✅ Early stopping eklendi")
        print("✅ Cross-validation artırıldı")
        
    except Exception as e:
        print(f"❌ Model eğitimi sırasında hata: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()