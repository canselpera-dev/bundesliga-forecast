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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
import xgboost as xgb

# ========== OPTİMİZE KONFİGÜRASYON ==========
RANDOM_STATE = 42
TEST_SIZE = 0.15
VAL_SIZE = 0.15
N_JOBS = 2  # Daha az CPU kullanımı

DATA_PATH = "data/bundesliga_matches_2023_2025_final_fe_team_values_cleaned.xlsx"
PLAYER_DATA_PATH = "data/final_bundesliga_dataset_complete.xlsx"

# OPTİMİZE ÖZELLİK LİSTESİ (Sadece en önemli 15 özellik)
OPTIMIZED_FEATURES = [
    'value_difference', 'form_difference', 'power_difference',
    'home_form', 'away_form', 'goals_ratio', 'xg_ratio',
    'h2h_avg_goals', 'value_ratio', 'age_difference',
    'home_power_index', 'away_power_index', 'isDerby',
    'home_goals', 'away_goals'
]

# ========== HIZLI FEATURE ENGINEERING ==========
class FastFeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        self.feature_names = []
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # Sadece en kritik özellikler
        if all(col in X.columns for col in ['home_current_value_eur', 'away_current_value_eur']):
            X['value_ratio'] = X['home_current_value_eur'] / (X['away_current_value_eur'] + 1e-8)
            X['value_difference'] = X['home_current_value_eur'] - X['away_current_value_eur']
        
        if all(col in X.columns for col in ['home_form', 'away_form']):
            X['form_difference'] = X['home_form'] - X['away_form']
        
        if all(col in X.columns for col in ['home_goals', 'away_goals']):
            X['goals_ratio'] = X['home_goals'] / (X['away_goals'] + 1e-8)
        
        return X

# ========== HIZLI FEATURE SELECTION ==========
def fast_feature_selection(X_train, y_train, X_val, X_test, max_features=10):
    print("⚡ Hızlı feature selection yapılıyor...")
    
    # Basit correlation-based selection
    correlation_with_target = X_train.corrwith(y_train).abs().sort_values(ascending=False)
    selected_features = correlation_with_target.head(max_features).index.tolist()
    
    print(f"✅ Seçilen özellik sayısı: {len(selected_features)}")
    print(f"📋 Özellikler: {selected_features}")
    
    return (X_train[selected_features], X_val[selected_features], 
            X_test[selected_features], selected_features)

# ========== HIZLI MODEL PIPELINE ==========
def create_fast_pipeline(selected_features):
    """Hızlı ve optimize pipeline - RANDOM SEARCH YOK"""
    
    preprocessor = ColumnTransformer([
        ('scaler', RobustScaler(), selected_features)
    ], remainder='drop')
    
    # ⚡ OPTİMİZE LIGHTGBM PARAMETRELERİ (Sabit - optimizasyon yok)
    lgbm_clf = lgb.LGBMClassifier(
        objective='multiclass',
        num_class=3,
        random_state=RANDOM_STATE,
        n_jobs=1,  # Tek çekirdek
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        num_leaves=20,
        min_child_samples=25,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.5,
        reg_lambda=0.5,
        verbosity=-1,
        force_row_wise=True
    )
    
    return Pipeline([
        ('preprocessor', preprocessor),
        ('lgbm', lgbm_clf)
    ])

# ========== BASİT VERİ YÜKLEME ==========
def load_data_fast():
    print("📊 Hızlı veri yükleniyor...")
    
    try:
        df_matches = pd.read_excel(DATA_PATH)
        
        # Basit veri hazırlama
        df = df_matches.copy()
        
        # Result_Numeric oluştur
        def get_result(row):
            try:
                home_goals = row.get('score.fullTime.home', 0) or row.get('FTHG', 0) or 0
                away_goals = row.get('score.fullTime.away', 0) or row.get('FTAG', 0) or 0
                
                if home_goals > away_goals:
                    return 1
                elif home_goals < away_goals:
                    return 2
                else:
                    return 0
            except:
                return 0
        
        df['Result_Numeric'] = df.apply(get_result, axis=1)
        
        # Eksik özellikleri doldur
        for feat in OPTIMIZED_FEATURES:
            if feat not in df.columns:
                df[feat] = 0
        
        numeric_cols = df[OPTIMIZED_FEATURES].select_dtypes(include=np.number).columns
        df[numeric_cols] = df[numeric_cols].fillna(0)
        
        print(f"✅ Veri hazır: {len(df)} satır, {len(OPTIMIZED_FEATURES)} özellik")
        return df
        
    except Exception as e:
        print(f"❌ Veri yükleme hatası: {e}")
        raise

# ========== HIZLI ZAMAN SPLIT ==========
def fast_time_split(df, test_size=0.15, val_size=0.15):
    n = len(df)
    test_split_idx = int(n * (1 - test_size))
    val_split_idx = int(test_split_idx * (1 - val_size))
    
    train_df = df.iloc[:val_split_idx]
    val_df = df.iloc[val_split_idx:test_split_idx]
    test_df = df.iloc[test_split_idx:]
    
    print(f"📊 Hızlı split: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    return train_df, val_df, test_df

# ========== HIZLI MODEL DEĞERLENDİRME ==========
def fast_model_evaluation(model, X_train, y_train, X_test, y_test):
    print("\n📊 HIZLI DEĞERLENDİRME:")
    
    # Tahminler
    y_pred_test = model.predict(X_test)
    y_pred_train = model.predict(X_train)
    
    test_accuracy = accuracy_score(y_test, y_pred_test)
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_f1 = f1_score(y_test, y_pred_test, average='weighted')
    
    accuracy_gap = train_accuracy - test_accuracy
    
    print(f"✅ Test Accuracy:  {test_accuracy:.4f}")
    print(f"🏋️ Train Accuracy: {train_accuracy:.4f}")
    print(f"📈 Test F1-Score:  {test_f1:.4f}")
    print(f"📊 Accuracy Gap:   {accuracy_gap:.4f}")
    
    # Overfitting durumu
    if accuracy_gap < 0.02:
        print("🎉 EXCELLENT: Minimal overfitting!")
    elif accuracy_gap < 0.05:
        print("✅ GOOD: Low overfitting risk")
    else:
        print("⚠️  WARNING: Moderate overfitting risk")
    
    print("\n🎯 DETAYLI SINIF ANALİZİ:")
    print(classification_report(y_test, y_pred_test, target_names=['Draw', 'HomeWin', 'AwayWin']))
    
    # Basit confusion matrix
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix - Test Set')
    plt.ylabel('Gerçek Değer')
    plt.xlabel('Tahmin Edilen Değer')
    plt.savefig('models/fast_confusion_matrix.png', dpi=200, bbox_inches='tight')
    plt.close()
    
    return {
        'test_accuracy': test_accuracy,
        'test_f1': test_f1,
        'overfitting_gap': accuracy_gap
    }

# ========== OPTİMİZE MODEL EĞİTİMİ ==========
def train_fast_model():
    print("⚡ HIZLI MODEL EĞİTİMİ BAŞLATILIYOR...")
    print("="*50)
    print("✅ Optimize feature set (15 özellik)")
    print("✅ Hızlı LightGBM (300 estimators)") 
    print("✅ Hiperparametre optimizasyonu YOK")
    print("✅ Hızlı convergence")
    print("="*50)
    
    start_time = datetime.now()
    
    # Veriyi yükle
    df = load_data_fast()
    
    # Veriyi böl
    train_df, val_df, test_df = fast_time_split(df, TEST_SIZE, VAL_SIZE)
    
    # Feature ve target'ları ayır
    X_train = train_df[OPTIMIZED_FEATURES].copy()
    y_train = train_df['Result_Numeric'].copy()
    
    X_val = val_df[OPTIMIZED_FEATURES].copy()
    y_val = val_df['Result_Numeric'].copy()
    
    X_test = test_df[OPTIMIZED_FEATURES].copy()
    y_test = test_df['Result_Numeric'].copy()
    
    # 1. Hızlı Feature Engineering
    print("🔧 Hızlı feature engineering...")
    feature_engineer = FastFeatureEngineer()
    X_train = feature_engineer.fit_transform(X_train)
    X_val = feature_engineer.transform(X_val)
    X_test = feature_engineer.transform(X_test)
    
    # 2. Hızlı Feature Selection
    X_train_selected, X_val_selected, X_test_selected, important_features = fast_feature_selection(
        X_train, y_train, X_val, X_test, max_features=10
    )
    
    # 3. Hızlı Model Pipeline (RANDOM SEARCH YOK)
    print("🚀 Model eğitimi başlıyor (hiperparametre optimizasyonu YOK)...")
    model = create_fast_pipeline(important_features)
    
    # 4. DOĞRUDAN EĞİTİM (hiç optimizasyon yok)
    model.fit(X_train_selected, y_train)
    
    # 5. Hızlı Değerlendirme
    results = fast_model_evaluation(model, X_train_selected, y_train, X_test_selected, y_test)
    
    # 6. Hızlı Kaydetme
    os.makedirs("models", exist_ok=True)
    
    model_path = "models/fast_bundesliga_model.pkl"
    joblib.dump(model, model_path)
    
    metadata = {
        'important_features': important_features,
        'test_accuracy': results['test_accuracy'],
        'test_f1': results['test_f1'],
        'training_time': str(datetime.now() - start_time),
        'model_version': 'fast_no_optimization_v1'
    }
    
    joblib.dump(metadata, "models/fast_model_metadata.pkl")
    
    end_time = datetime.now()
    training_duration = (end_time - start_time).total_seconds() / 60
    
    print(f"\n⏱️  EĞİTİM SÜRESİ: {training_duration:.1f} dakika")
    print(f"💾 Model kaydedildi: {model_path}")
    
    return model, important_features, results['test_accuracy']

# ========== ANA FONKSİYON ==========
if __name__ == "__main__":
    print("🏆 Bundesliga Hızlı Model Eğitimi")
    print("="*40)
    
    try:
        model, features, accuracy = train_fast_model()
        
        print(f"\n🎉 HIZLI EĞİTİM TAMAMLANDI!")
        print(f"📊 Test Accuracy: {accuracy:.4f}")
        print(f"📋 Özellik sayısı: {len(features)}")
        print(f"📍 Model: models/fast_bundesliga_model.pkl")
        
    except Exception as e:
        print(f"❌ Eğitim hatası: {e}")
        import traceback
        traceback.print_exc()