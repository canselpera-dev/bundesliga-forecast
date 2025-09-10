import pandas as pd
import numpy as np
import os

def diagnose_nan_problem():
    """NaN değerlerin kaynağını bul"""
    print("🔍 NaN Problemi Diagnostik Analizi")
    print("=" * 50)
    
    df = pd.read_excel("data/bundesliga_complete_dataset.xlsx")
    
    print(f"📊 Toplam NaN sayısı: {df.isna().sum().sum()}")
    print(f"📈 Toplam hücre sayısı: {df.shape[0] * df.shape[1]}")
    print(f"📉 NaN oranı: {df.isna().sum().sum()/(df.shape[0] * df.shape[1])*100:.1f}%")
    
    # Hangi sütunlarda NaN var?
    print("\n📋 NaN Dağılımı (Sütun Bazında):")
    nan_by_column = df.isna().sum()
    nan_columns = nan_by_column[nan_by_column > 0]
    
    for col, count in nan_columns.items():
        print(f"   ❌ {col}: {count} NaN (%{count/len(df)*100:.1f})")
    
    # En problemli 10 sütun
    print(f"\n🎯 En Problemli 10 Sütun:")
    worst_columns = nan_by_column.sort_values(ascending=False).head(10)
    for col, count in worst_columns.items():
        print(f"   ⚠️  {col}: {count} NaN")
    
    return df, worst_columns

def fix_nan_problems():
    """NaN problemlerini düzelt"""
    print("\n🛠️  NaN Problemleri Düzeltiliyor...")
    print("=" * 50)
    
    df, worst_columns = diagnose_nan_problem()
    
    # 1. Öncelikle kritik sütunları temizle
    critical_columns = ['Result_Numeric', 'Home_AvgRating', 'Away_AvgRating', 
                       'Home_Form', 'Away_Form', 'Rating_Diff']
    
    print("\n🧹 Kritik Sütunlar Temizleniyor...")
    for col in critical_columns:
        if col in df.columns:
            before = df[col].isna().sum()
            if df[col].dtype in ['int64', 'float64']:
                df[col] = df[col].fillna(0)
            else:
                # Eğer categorical veya string ise
                df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'Unknown')
            after = df[col].isna().sum()
            print(f"   ✅ {col}: {before} → {after} NaN")
    
    # 2. Position rating'leri temizle
    print("\n⭐ Position Rating'ler Temizleniyor...")
    position_columns = ['Home_GK_Rating', 'Home_DF_Rating', 'Home_MF_Rating', 'Home_FW_Rating',
                       'Away_GK_Rating', 'Away_DF_Rating', 'Away_MF_Rating', 'Away_FW_Rating']
    
    for col in position_columns:
        if col in df.columns:
            before = df[col].isna().sum()
            df[col] = df[col].fillna(65.0)  # Ortalama rating değeri
            after = df[col].isna().sum()
            print(f"   ✅ {col}: {before} → {after} NaN")
    
    # 3. Numeric sütunları 0 ile doldur
    print("\n🔢 Numeric Sütunlar Temizleniyor...")
    numeric_cols = df.select_dtypes(include=np.number).columns
    df[numeric_cols] = df[numeric_cols].fillna(0)
    
    # 4. String sütunları temizle
    print("\n🔤 String Sütunlar Temizleniyor...")
    string_cols = df.select_dtypes(include='object').columns
    df[string_cols] = df[string_cols].fillna('Unknown')
    
    # 5. Son kontrol
    final_nan = df.isna().sum().sum()
    print(f"\n🎉 SON DURUM: {final_nan} NaN değer kaldı")
    
    # Kaydet
    output_path = "data/bundesliga_CLEAN_dataset.xlsx"
    df.to_excel(output_path, index=False)
    print(f"💾 Temiz dataset kaydedildi: {output_path}")
    
    return df

def create_smart_merge_pipeline():
    """Akıllı birleştirme pipeline'ı"""
    print("🤖 Akıllı Birleştirme Pipeline'ı")
    print("=" * 50)
    
    # Sadece gerçekten gerekli dosyaları seç
    essential_files = [
        "data/bundesliga_matches_2023_2025_updated.xlsx",  # Temel maç verisi
        "data/bundesliga_features_complete.xlsx",          # Form ve momentum
        "data/player_ratings_v2_clean.xlsx"                # Rating verisi
    ]
    
    print("📁 Kullanılacak Temel Dosyalar:")
    for file in essential_files:
        if os.path.exists(file):
            print(f"   ✅ {os.path.basename(file)}")
        else:
            print(f"   ❌ {os.path.basename(file)} (BULUNAMADI!)")
            return None
    
    # Ana dataframe'i yükle
    print("\n📥 Ana dataframe yükleniyor...")
    main_df = pd.read_excel("data/bundesliga_matches_2023_2025_updated.xlsx")
    
    # Diğer dosyaları merge et
    for file in essential_files[1:]:
        try:
            extra_df = pd.read_excel(file)
            print(f"\n🔗 {os.path.basename(file)} birleştiriliyor...")
            
            # Ortak sütunları bul
            common_cols = list(set(main_df.columns) & set(extra_df.columns))
            if common_cols:
                print(f"   Ortak sütunlar: {common_cols}")
                main_df = pd.merge(main_df, extra_df, on=common_cols, how='left')
                print(f"   ✅ Başarıyla birleştirildi. Yeni shape: {main_df.shape}")
            else:
                print("   ⚠️ Ortak sütun yok, concat yapılıyor...")
                main_df = pd.concat([main_df, extra_df], axis=1)
                
        except Exception as e:
            print(f"   ❌ Birleştirme hatası: {e}")
    
    return main_df

# ========== MAIN ==========
if __name__ == "__main__":
    print("🏆 Bundesliga NaN Temizleme ve Birleştirme")
    print("=" * 50)
    
    # Seçenek 1: Mevcut dataset'i temizle
    print("1. Mevcut dataset'i temizle (Hızlı)")
    print("2. Yeniden akıllı birleştirme (Tavsiye)")
    
    choice = input("\n🔄 Seçiminiz (1 veya 2): ")
    
    if choice == "1":
        # Mevcut dataset'i temizle
        clean_df = fix_nan_problems()
        print("\n🎉 TEMİZLİK TAMAMLANDI! Artık eğitime hazır.")
        
    elif choice == "2":
        # Yeniden akıllı birleştirme
        merged_df = create_smart_merge_pipeline()
        if merged_df is not None:
            # NaN'leri temizle
            merged_df = merged_df.fillna(0)
            # Kaydet
            merged_df.to_excel("data/bundesliga_SMART_MERGED.xlsx", index=False)
            print("\n🎉 AKILLI BİRLEŞTİRME TAMAMLANDI!")
            print(f"📊 Final shape: {merged_df.shape}")
            print(f"📉 NaN sayısı: {merged_df.isna().sum().sum()}")
            
    else:
        print("❌ Geçersiz seçim!")