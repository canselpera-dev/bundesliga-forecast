import pandas as pd
import numpy as np
import os
from datetime import datetime

def convert_result_format():
    """
    Bundesliga verisindeki Result sütununu kodun beklediği formata dönüştürür
    ve yeni bir dataset oluşturur.
    """
    
    # Dosya yolları
    input_file = "data/bundesliga_matches_2023_2025_updated.xlsx"
    output_file = "data/bundesliga_complete_dataset.xlsx"
    
    print("📊 Veri dosyası okunuyor...")
    
    try:
        # Veriyi yükle
        df = pd.read_excel(input_file)
        print(f"✅ Dosya yüklendi: {input_file}")
        print(f"📋 Shape: {df.shape}")
        
        # Sütun isimlerini standartlaştır
        df.columns = [col.strip().replace(' ', '_') for col in df.columns]
        
        # Result sütunu kontrolü - küçük/büyük harf duyarlılığını kaldır
        result_col = None
        for col in df.columns:
            if col.lower() == 'result':
                result_col = col
                break
        
        if not result_col:
            raise ValueError("❌ 'Result' sütunu bulunamadı! Mevcut sütunlar: " + str(df.columns.tolist()))
        
        print(f"🔍 Result sütunu unique değerleri: {df[result_col].unique()}")
        
        # Date sütunu kontrolü - utcDate'den Date'e çevir
        date_col = None
        for col in df.columns:
            if col.lower() in ['utcdate', 'date']:
                date_col = col
                break
        
        if date_col and date_col != 'Date':
            df['Date'] = df[date_col]  # utcDate'i Date olarak yeniden adlandır
            print(f"✅ {date_col} sütunu Date olarak yeniden adlandırıldı")
        
        # Result sütununu kodun beklediği formata dönüştür
        print("🔄 Result sütunu dönüştürülüyor...")
        
        # Result değerlerini kontrol et ve dönüştür
        result_mapping = {
            -1: 'AwayWin',
            0: 'Draw',
            1: 'HomeWin',
            '-1': 'AwayWin',
            '0': 'Draw', 
            '1': 'HomeWin',
            'A': 'AwayWin',
            'D': 'Draw',
            'H': 'HomeWin',
            'Away': 'AwayWin',
            'Draw': 'Draw',
            'Home': 'HomeWin',
            'AwayWin': 'AwayWin',
            'HomeWin': 'HomeWin'
        }
        
        # Tüm değerleri stringe çevir ve mapping uygula
        df['Result_Formatted'] = df[result_col].astype(str).str.strip().map(result_mapping)
        
        # Mapping edilemeyen değerleri kontrol et
        missing_values = df[df['Result_Formatted'].isna()]
        if len(missing_values) > 0:
            print(f"⚠️ Mapping edilemeyen {len(missing_values)} değer bulundu:")
            print(missing_values[result_col].unique())
            # NaN değerleri orijinal değerlerle doldur
            df['Result_Formatted'] = df['Result_Formatted'].fillna(df[result_col].astype(str))
        
        # Yeni sütunları ekle
        df['Result_Numeric'] = df['Result_Formatted'].map({
            'Draw': 0,
            'HomeWin': 1,
            'AwayWin': 2
        })
        
        # NaN değerleri kontrol et ve düzelt
        if df['Result_Numeric'].isna().any():
            print(f"⚠️ {df['Result_Numeric'].isna().sum()} adet NaN değer düzeltiliyor...")
            # NaN değerleri ortalama veya mod ile doldur
            df['Result_Numeric'] = df['Result_Numeric'].fillna(df['Result_Numeric'].mode()[0] if not df['Result_Numeric'].mode().empty else 0)
            df['Result_Formatted'] = df['Result_Formatted'].fillna('Draw')
        
        # Timestamp ekle (güncelleme takibi için)
        df['Last_Updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Güncellenmiş veriyi kaydet
        print("💾 Yeni dataset kaydediliyor...")
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            df.to_excel(writer, sheet_name='Matches', index=False)
            # Mapping bilgilerini de kaydet
            mapping_info = pd.DataFrame({
                'Original_Value': [-1, 0, 1, '-1', '0', '1', 'A', 'D', 'H', 'Away', 'Draw', 'Home'],
                'Mapped_Value': ['AwayWin', 'Draw', 'HomeWin', 'AwayWin', 'Draw', 'HomeWin', 'AwayWin', 'Draw', 'HomeWin', 'AwayWin', 'Draw', 'HomeWin'],
                'Numeric_Value': [2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]
            })
            mapping_info.to_excel(writer, sheet_name='Mapping_Info', index=False)
        
        print(f"✅ Dataset kaydedildi: {output_file}")
        print(f"📊 Toplam maç sayısı: {len(df)}")
        print(f"🎯 Result dağılımı:")
        print(df['Result_Formatted'].value_counts())
        print(f"🔢 Numeric dağılım:")
        print(df['Result_Numeric'].value_counts())
        
        return df
        
    except Exception as e:
        print(f"❌ Hata oluştu: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def update_dataset_with_new_data(new_data_file=None):
    """
    Yeni veri geldiğinde dataseti günceller
    """
    print("🔄 Dataset güncelleme işlemi başlatılıyor...")
    
    # Çıktı dosyası zaten varsa önce onu yükle
    output_file = "data/bundesliga_complete_dataset.xlsx"
    
    if os.path.exists(output_file):
        try:
            existing_df = pd.read_excel(output_file)
            print(f"📁 Mevcut dataset yüklendi: {len(existing_df)} maç")
        except:
            existing_df = pd.DataFrame()
            print("ℹ️ Mevcut dataset bulunamadı, yeni oluşturulacak")
    else:
        existing_df = pd.DataFrame()
    
    # Yeni veri dosyası veya güncellenmiş dosya
    if new_data_file and os.path.exists(new_data_file):
        input_file = new_data_file
        print(f"📥 Yeni veri dosyası kullanılacak: {new_data_file}")
    else:
        input_file = "data/bundesliga_matches_2023_2025_updated.xlsx"
        print("📥 Varsayılan güncellenmiş dosya kullanılacak")
    
    # Yeni veriyi yükle ve formatla
    try:
        new_df = pd.read_excel(input_file)
        new_df.columns = [col.strip().replace(' ', '_') for col in new_df.columns]
        
        # Result sütununu bul (case-insensitive)
        result_col = None
        for col in new_df.columns:
            if col.lower() == 'result':
                result_col = col
                break
        
        if not result_col:
            raise ValueError("❌ Result sütunu bulunamadı!")
        
        # Date sütunu kontrolü
        date_col = None
        for col in new_df.columns:
            if col.lower() in ['utcdate', 'date']:
                date_col = col
                break
        
        if date_col and date_col != 'Date':
            new_df['Date'] = new_df[date_col]
        
        # Result formatını dönüştür
        result_mapping = {
            -1: 'AwayWin', 0: 'Draw', 1: 'HomeWin',
            '-1': 'AwayWin', '0': 'Draw', '1': 'HomeWin',
            'A': 'AwayWin', 'D': 'Draw', 'H': 'HomeWin',
            'Away': 'AwayWin', 'Draw': 'Draw', 'Home': 'HomeWin'
        }
        
        new_df['Result_Formatted'] = new_df[result_col].astype(str).str.strip().map(result_mapping)
        new_df['Result_Numeric'] = new_df['Result_Formatted'].map({'Draw': 0, 'HomeWin': 1, 'AwayWin': 2})
        new_df['Last_Updated'] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # NaN değerleri düzelt
        new_df['Result_Numeric'] = new_df['Result_Numeric'].fillna(new_df['Result_Numeric'].mode()[0] if not new_df['Result_Numeric'].mode().empty else 0)
        new_df['Result_Formatted'] = new_df['Result_Formatted'].fillna('Draw')
        
        # Benzersiz identifier olarak maç ID veya tarih+takımlar kullan
        if 'Match_ID' in new_df.columns:
            id_col = 'Match_ID'
        elif all(col in new_df.columns for col in ['Date', 'HomeTeam', 'AwayTeam']):
            new_df['Match_Identifier'] = new_df['Date'].astype(str) + '_' + new_df['HomeTeam'] + '_' + new_df['AwayTeam']
            id_col = 'Match_Identifier'
        else:
            id_col = None
        
        # Mevcut veriyle birleştir (duplicate'leri önle)
        if not existing_df.empty and id_col:
            if id_col in existing_df.columns and id_col in new_df.columns:
                # Sadece yeni veya güncellenmiş maçları ekle
                existing_ids = set(existing_df[id_col].astype(str))
                new_ids = set(new_df[id_col].astype(str))
                
                # Yeni maçları bul
                new_matches = new_df[~new_df[id_col].astype(str).isin(existing_ids)]
                
                if len(new_matches) > 0:
                    print(f"🆕 {len(new_matches)} yeni maç eklenecek")
                    updated_df = pd.concat([existing_df, new_matches], ignore_index=True)
                else:
                    print("ℹ️ Yeni maç bulunamadı")
                    updated_df = existing_df
            else:
                updated_df = pd.concat([existing_df, new_df], ignore_index=True)
        else:
            updated_df = new_df
        
        # Kaydet
        with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
            updated_df.to_excel(writer, sheet_name='Matches', index=False)
            mapping_info = pd.DataFrame({
                'Original_Value': [-1, 0, 1, '-1', '0', '1', 'A', 'D', 'H', 'Away', 'Draw', 'Home'],
                'Mapped_Value': ['AwayWin', 'Draw', 'HomeWin', 'AwayWin', 'Draw', 'HomeWin', 'AwayWin', 'Draw', 'HomeWin', 'AwayWin', 'Draw', 'HomeWin'],
                'Numeric_Value': [2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1]
            })
            mapping_info.to_excel(writer, sheet_name='Mapping_Info', index=False)
        
        print(f"✅ Dataset güncellendi: {output_file}")
        print(f"📊 Toplam maç sayısı: {len(updated_df)}")
        print(f"🎯 Güncel Result dağılımı:")
        print(updated_df['Result_Formatted'].value_counts())
        
        return updated_df
        
    except Exception as e:
        print(f"❌ Güncelleme hatası: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

# ========== MAIN ==========
if __name__ == "__main__":
    print("🏆 Bundesliga Dataset Format Dönüştürücü")
    print("=" * 50)
    
    # İlk dönüşümü yap
    df = convert_result_format()
    
    if df is not None:
        print("✅ Dönüşüm başarılı!")
        
        print("\n🔄 Otomatik güncelleme testi...")
        # Kendi kendini güncelleme testi
        updated_df = update_dataset_with_new_data()
        
    print("\n🎉 İşlem tamamlandı!")