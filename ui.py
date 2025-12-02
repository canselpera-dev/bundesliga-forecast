import os
import pandas as pd
import json
import warnings
warnings.filterwarnings('ignore')

# Klasör yolu
klasor_yolu = r"C:\Users\canse\OneDrive\Masaüstü\Bundesliga Forecast\data"

def dosya_icerigini_incele(dosya_yolu):
    """Dosya türüne göre başlıkları/sütunları oku"""
    dosya_adi = os.path.basename(dosya_yolu)
    dosya_uzanti = os.path.splitext(dosya_adi)[1].lower()
    
    try:
        if dosya_uzanti == '.csv':
            # CSV dosyası için
            try:
                # Encoding problemlerini önlemek için farklı encoding'ler dene
                for encoding in ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']:
                    try:
                        df = pd.read_csv(dosya_yolu, nrows=5, encoding=encoding)
                        sütunlar = df.columns.tolist()
                        print(f"  Dosya: {dosya_adi}")
                        print(f"  Format: CSV")
                        print(f"  Sütun Sayısı: {len(sütunlar)}")
                        print(f"  Sütunlar: {sütunlar}")
                        print(f"  Örnek Veri:")
                        print(df.head(3).to_string())
                        print("-" * 80)
                        return
                    except UnicodeDecodeError:
                        continue
                print(f"  Dosya: {dosya_adi}")
                print(f"  HATA: Encoding hatası - dosya okunamadı")
                print("-" * 80)
            except Exception as e:
                print(f"  Dosya: {dosya_adi}")
                print(f"  HATA: {str(e)}")
                print("-" * 80)
                
        elif dosya_uzanti in ['.xlsx', '.xls']:
            # Excel dosyası için
            try:
                # Excel dosyasındaki tüm sheet'leri kontrol et
                excel_file = pd.ExcelFile(dosya_yolu)
                sheet_names = excel_file.sheet_names
                
                print(f"  Dosya: {dosya_adi}")
                print(f"  Format: Excel")
                print(f"  Sheet Sayısı: {len(sheet_names)}")
                print(f"  Sheet Adları: {sheet_names}")
                
                for i, sheet in enumerate(sheet_names[:3]):  # İlk 3 sheet'i göster
                    df = excel_file.parse(sheet, nrows=5)
                    sütunlar = df.columns.tolist()
                    print(f"  Sheet {i+1}: {sheet}")
                    print(f"    Sütun Sayısı: {len(sütunlar)}")
                    print(f"    Sütunlar: {sütunlar}")
                    if len(sheet_names) > 1 and i < 2:  # İlk 2 sheet için örnek veri göster
                        print(f"    Örnek Veri:")
                        print(df.head(3).to_string())
                print("-" * 80)
            except Exception as e:
                print(f"  Dosya: {dosya_adi}")
                print(f"  HATA: {str(e)}")
                print("-" * 80)
                
        elif dosya_uzanti == '.json':
            # JSON dosyası için
            try:
                with open(dosya_yolu, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                print(f"  Dosya: {dosya_adi}")
                print(f"  Format: JSON")
                
                # JSON'ın yapısını analiz et
                if isinstance(data, list) and len(data) > 0:
                    # Liste içinde sözlük varsa
                    ilk_eleman = data[0]
                    if isinstance(ilk_eleman, dict):
                        sütunlar = list(ilk_eleman.keys())
                        print(f"  Kayıt Sayısı: {len(data)}")
                        print(f"  Sütunlar: {sütunlar}")
                        print(f"  Örnek Veri (ilk kayıt):")
                        print(json.dumps(ilk_eleman, indent=2, ensure_ascii=False))
                elif isinstance(data, dict):
                    # Tek bir sözlük varsa
                    sütunlar = list(data.keys())
                    print(f"  Yapı: Tek sözlük")
                    print(f"  Anahtarlar: {sütunlar}")
                    print(f"  Örnek Veri:")
                    print(json.dumps({k: data[k] for k in list(data.keys())[:5]}, indent=2, ensure_ascii=False))
                else:
                    print(f"  Yapı: {type(data).__name__}")
                    print(f"  İçerik: {str(data)[:100]}...")
                print("-" * 80)
            except Exception as e:
                print(f"  Dosya: {dosya_adi}")
                print(f"  HATA: {str(e)}")
                print("-" * 80)
                
        else:
            # Diğer dosya türleri
            print(f"  Dosya: {dosya_adi}")
            print(f"  Format: {dosya_uzanti} (desteklenmiyor)")
            print("-" * 80)
            
    except Exception as e:
        print(f"  Dosya: {dosya_adi}")
        print(f"  Genel HATA: {str(e)}")
        print("-" * 80)

def ana_fonksiyon():
    print("=" * 80)
    print("DATA KLASÖRÜ İÇERİK ANALİZİ")
    print("=" * 80)
    print(f"Klasör Yolu: {klasor_yolu}")
    print("=" * 80)
    
    try:
        # Klasördeki tüm dosyaları al
        dosyalar = os.listdir(klasor_yolu)
        
        if not dosyalar:
            print("Klasör boş!")
            return
        
        # Sadece dosyaları filtrele
        dosya_listesi = []
        for dosya in dosyalar:
            dosya_yolu = os.path.join(klasor_yolu, dosya)
            if os.path.isfile(dosya_yolu):
                dosya_listesi.append(dosya_yolu)
        
        print(f"Toplam {len(dosya_listesi)} dosya bulundu.\n")
        
        # Her dosyayı analiz et
        for i, dosya_yolu in enumerate(dosya_listesi, 1):
            print(f"\n[{i}/{len(dosya_listesi)}] Analiz Ediliyor...")
            dosya_icerigini_incele(dosya_yolu)
            
    except FileNotFoundError:
        print(f"Hata: Belirtilen klasör bulunamadı: {klasor_yolu}")
    except PermissionError:
        print(f"Hata: Klasöre erişim izniniz yok: {klasor_yolu}")
    except Exception as e:
        print(f"Beklenmeyen bir hata oluştu: {e}")

# Daha özet bir versiyon (sadece başlıklar için)
def ozet_analiz():
    print("=" * 80)
    print("DATA KLASÖRÜ - ÖZET BAŞLIK ANALİZİ")
    print("=" * 80)
    
    try:
        dosyalar = os.listdir(klasor_yolu)
        dosya_listesi = []
        
        for dosya in dosyalar:
            dosya_yolu = os.path.join(klasor_yolu, dosya)
            if os.path.isfile(dosya_yolu):
                dosya_listesi.append(dosya_yolu)
        
        for dosya_yolu in dosya_listesi:
            dosya_adi = os.path.basename(dosya_yolu)
            dosya_uzanti = os.path.splitext(dosya_adi)[1].lower()
            
            print(f"\n📁 {dosya_adi}")
            
            try:
                if dosya_uzanti == '.csv':
                    df = pd.read_csv(dosya_yolu, nrows=1)
                    sütunlar = df.columns.tolist()
                    print(f"  📊 Sütunlar ({len(sütunlar)}):")
                    for sutun in sütunlar[:10]:  # İlk 10 sütunu göster
                        print(f"     • {sutun}")
                    if len(sütunlar) > 10:
                        print(f"     ... ve {len(sütunlar)-10} sütun daha")
                        
                elif dosya_uzanti in ['.xlsx', '.xls']:
                    excel_file = pd.ExcelFile(dosya_yolu)
                    sheet_names = excel_file.sheet_names
                    print(f"  📑 Sheet'ler ({len(sheet_names)}): {', '.join(sheet_names)}")
                    
                elif dosya_uzanti == '.json':
                    with open(dosya_yolu, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if isinstance(data, list) and len(data) > 0:
                        if isinstance(data[0], dict):
                            sütunlar = list(data[0].keys())
                            print(f"  📋 Anahtarlar ({len(sütunlar)}):")
                            for sutun in sütunlar[:10]:
                                print(f"     • {sutun}")
                            if len(sütunlar) > 10:
                                print(f"     ... ve {len(sütunlar)-10} anahtar daha")
                                
            except Exception as e:
                print(f"  ❗ Hata: {str(e)[:50]}...")
            
            print("-" * 60)
            
    except Exception as e:
        print(f"Hata: {e}")

if __name__ == "__main__":
    # Hangisini kullanmak istiyorsanız onu seçin:
    
    # 1. Detaylı analiz için:
    print("DETAYLI ANALİZ:")
    ana_fonksiyon()
    
    print("\n" + "=" * 80 + "\n")
    
    # 2. Özet analiz için:
    print("ÖZET ANALİZ:")
    ozet_analiz()