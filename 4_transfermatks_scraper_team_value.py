import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import unicodedata
import re
from fuzzywuzzy import process, fuzz
import warnings
warnings.filterwarnings('ignore')

# ------------------------------
# 1. VERİ YÜKLEME ve ÖN İŞLEME
# ------------------------------
def load_and_preprocess_data():
    """Verileri yükler ve ön işlem yapar"""
    # Dosya yolları
    squads_path = "data/bundesliga_squads_hybrid.xlsx"
    fbref_path = "data/fbref_team_stats_all_seasons.csv"
    
    # Verileri yükle
    print("📂 Veriler yükleniyor...")
    squads_df = pd.read_excel(squads_path)
    fbref_df = pd.read_csv(fbref_path)
    
    print(f"Squads veri boyutu: {squads_df.shape}")
    print(f"FBref veri boyutu: {fbref_df.shape}")
    
    # Sütun isimlerini temizle
    squads_df.columns = squads_df.columns.str.strip()
    fbref_df.columns = fbref_df.columns.str.strip()
    
    return squads_df, fbref_df

# ------------------------------
# 2. TAM MANUEL TAKIM MAPPING
# ------------------------------
def setup_team_mappings():
    """Takım eşleştirme mapping'lerini kurar"""
    # FBref takım isimlerinden Hybrid takım isimlerine TAM mapping
    COMPLETE_TEAM_MAPPING = {
        # Bayern Munich -> FC Bayern München
        'Bayern Munich': 'FC Bayern München',
        
        # Leverkusen -> Bayer 04 Leverkusen
        'Leverkusen': 'Bayer 04 Leverkusen',
        
        # Dortmund -> Borussia Dortmund
        'Dortmund': 'Borussia Dortmund',
        
        # RB Leipzig -> RB Leipzig
        'RB Leipzig': 'RB Leipzig',
        
        # Stuttgart -> VfB Stuttgart
        'Stuttgart': 'VfB Stuttgart',
        
        # Eint Frankfurt -> Eintracht Frankfurt
        'Eint Frankfurt': 'Eintracht Frankfurt',
        
        # Hoffenheim -> TSG 1899 Hoffenheim
        'Hoffenheim': 'TSG 1899 Hoffenheim',
        
        # Freiburg -> SC Freiburg
        'Freiburg': 'SC Freiburg',
        
        # Heidenheim -> 1. FC Heidenheim 1846
        'Heidenheim': '1. FC Heidenheim 1846',
        
        # Werder Bremen -> SV Werder Bremen
        'Werder Bremen': 'SV Werder Bremen',
        
        # Gladbach -> Borussia Mönchengladbach
        'Gladbach': 'Borussia Mönchengladbach',
        
        # Wolfsburg -> VfL Wolfsburg
        'Wolfsburg': 'VfL Wolfsburg',
        
        # Augsburg -> FC Augsburg
        'Augsburg': 'FC Augsburg',
        
        # Union Berlin -> 1. FC Union Berlin
        'Union Berlin': '1. FC Union Berlin',
        
        # Bochum -> VfL Bochum
        'Bochum': 'VfL Bochum',
        
        # Köln -> 1. FC Köln
        'Köln': '1. FC Köln',
        
        # Mainz 05 -> 1. FSV Mainz 05
        'Mainz 05': '1. FSV Mainz 05',
        
        # Darmstadt 98 -> SV Darmstadt 98
        'Darmstadt 98': 'SV Darmstadt 98',
        
        # St. Pauli -> FC St. Pauli
        'St. Pauli': 'FC St. Pauli',
        
        # Holstein Kiel -> Holstein Kiel
        'Holstein Kiel': 'Holstein Kiel',
        
        # Hamburger SV -> Hamburger SV
        'Hamburger SV': 'Hamburger SV'
    }
    
    # Reverse mapping (Hybrid -> FBref)
    HYBRID_TO_FBREF_MAPPING = {v: k for k, v in COMPLETE_TEAM_MAPPING.items()}
    
    return COMPLETE_TEAM_MAPPING, HYBRID_TO_FBREF_MAPPING

# ------------------------------
# 3. OYUNCU İSİM STANDARDİZASYONU
# ------------------------------
def normalize_player_name(name):
    """Oyuncu isimlerini standardize eder"""
    if pd.isna(name):
        return name
    
    name = str(name).lower().strip()
    name = unicodedata.normalize('NFKD', name).encode('ascii', 'ignore').decode('utf-8')
    name = re.sub(r'\(.*?\)', '', name)
    name = re.sub(r'[^\w\s]', '', name)
    name = re.sub(r'\s+', ' ', name).strip()
    
    return name

# ------------------------------
# 4. TAKIM İSİM STANDARDİZASYONU
# ------------------------------
def normalize_team_name(name):
    """Takım isimlerini standardize eder"""
    if pd.isna(name):
        return name
    
    name = str(name).strip()
    return name

# ------------------------------
# 5. ANA İŞLEM FONKSİYONU
# ------------------------------
def main():
    # Verileri yükle
    squads_df, fbref_df = load_and_preprocess_data()
    
    # Mapping'leri kur
    COMPLETE_TEAM_MAPPING, HYBRID_TO_FBREF_MAPPING = setup_team_mappings()
    
    print("🔄 Takım isimleri manuel eşleştiriliyor...")
    
    # FBref verisine Hybrid takım isimlerini ekle
    fbref_df['Hybrid_Team'] = fbref_df['Team'].map(COMPLETE_TEAM_MAPPING)
    
    # Eşleşmeyen takımları kontrol et
    unmatched_fbref = fbref_df[fbref_df['Hybrid_Team'].isna()]['Team'].unique()
    if len(unmatched_fbref) > 0:
        print(f"⚠️ Eşleşmeyen FBref takımları: {unmatched_fbref}")
    
    # Squad takımlarını kaldır
    fbref_df = fbref_df[fbref_df['Team'] != 'Squad']
    
    print("🔄 Oyuncu isimleri standardize ediliyor...")
    squads_df['Player_std'] = squads_df['Player'].apply(normalize_player_name)
    fbref_df['Player_std'] = fbref_df['Player'].apply(normalize_player_name)
    
    # Takım isimlerini standardize et
    squads_df['Team_std'] = squads_df['Team'].apply(normalize_team_name)
    fbref_df['Team_std'] = fbref_df['Hybrid_Team'].apply(normalize_team_name)
    
    # FBref sayısal sütunları dönüştür
    numeric_cols = ['MP', 'Gls', 'Ast', 'Min', 'Sh', 'SoT', 'Cmp', 'Att']
    for col in numeric_cols:
        if col in fbref_df.columns:
            fbref_df[col] = pd.to_numeric(fbref_df[col], errors='coerce')
    
    print("🔍 Doğrudan merge işlemi yapılıyor...")
    
    # Doğrudan birleştirme
    merged_df = pd.merge(
        squads_df, 
        fbref_df,
        how='left',
        left_on=['Player_std', 'Team_std'],
        right_on=['Player_std', 'Team_std'],
        suffixes=('_hybrid', '_fbref')
    )
    
    print("🔍 Aggressive fuzzy matching yapılıyor...")
    
    # Eşleşmemiş oyuncuları bul
    unmatched_mask = merged_df['MP'].isna()
    unmatched_count = unmatched_mask.sum()
    print(f" {unmatched_count} eşleşmemiş oyuncu bulundu")
    
    if unmatched_count > 0:
        unmatched_df = merged_df[unmatched_mask].copy()
        
        for idx, row in unmatched_df.iterrows():
            # Güvenli şekilde takım bilgisini al
            hybrid_team = row.get('Team_std') or row.get('Team_hybrid') or row.get('Team')
            if pd.isna(hybrid_team):
                print(f" ⚠️ Takım bilgisi bulunamadı, index: {idx}")
                continue
                
            hybrid_player = row['Player_std']
            
            # Takımın FBref'teki karşılığını bul
            fbref_team_name = None
            for team_key in HYBRID_TO_FBREF_MAPPING:
                if team_key in hybrid_team or hybrid_team in team_key:
                    fbref_team_name = HYBRID_TO_FBREF_MAPPING[team_key]
                    break
                    
            if not fbref_team_name:
                # Fuzzy matching ile takım eşleştirme
                fbref_teams = list(HYBRID_TO_FBREF_MAPPING.values())
                best_team_match, team_score = process.extractOne(hybrid_team, fbref_teams, scorer=fuzz.token_sort_ratio)
                if team_score >= 75:
                    fbref_team_name = best_team_match
                    
            if fbref_team_name:
                # Takımdaki tüm FBref oyuncularını getir
                team_players = fbref_df[fbref_df['Team'] == fbref_team_name]
                
                if not team_players.empty:
                    player_names = team_players['Player_std'].tolist()
                    
                    # Fuzzy matching
                    best_match, score = process.extractOne(hybrid_player, player_names, scorer=fuzz.token_sort_ratio)
                    
                    if score >= 75:
                        matched_data = team_players[team_players['Player_std'] == best_match].iloc[0]
                        
                        # Tüm FBref sütunlarını doldur
                        for col in fbref_df.columns:
                            if col not in ['Player_std', 'Team', 'Hybrid_Team', 'Team_std']:
                                merged_df.at[idx, col] = matched_data[col]
                        
                        print(f" ✅ {hybrid_player} -> {best_match} ({score}%)")
    
    print("🔄 Takım bazlı NaN değerler dolduruluyor...")
    
    # Sayısal sütunları tekrar kontrol et ve dönüştür
    numeric_cols = ['MP', 'Gls', 'Ast', 'Min', 'Sh', 'SoT', 'Cmp', 'Att']
    for col in numeric_cols:
        if col in merged_df.columns:
            merged_df[col] = pd.to_numeric(merged_df[col], errors='coerce')
    
    for team in merged_df['Team_std'].unique():
        if pd.isna(team):
            continue
            
        team_mask = merged_df['Team_std'] == team
        
        for col in numeric_cols:
            if col in merged_df.columns:
                # Takımın ortalamasını al
                team_mean = merged_df.loc[team_mask, col].mean()
                
                if not pd.isna(team_mean):
                    # NaN değerleri takım ortalaması ile doldur
                    merged_df.loc[team_mask & merged_df[col].isna(), col] = team_mean
    
    # Hala NaN varsa, lig geneli ortalama ile doldur
    for col in numeric_cols:
        if col in merged_df.columns and merged_df[col].isna().any():
            lig_mean = merged_df[col].mean()
            if not pd.isna(lig_mean):
                merged_df[col] = merged_df[col].fillna(lig_mean)
            else:
                merged_df[col] = merged_df[col].fillna(0)
    
    print("📊 Rating hesaplanıyor...")
    
    # Rating hesaplama
    rating_metrics = ['MP', 'Gls', 'Ast', 'Sh', 'SoT']
    available_metrics = [col for col in rating_metrics if col in merged_df.columns]
    
    if available_metrics:
        for col in available_metrics:
            col_max = merged_df[col].max()
            if col_max > 0:
                merged_df[f'{col}_norm'] = merged_df[col] / col_max * 100
            else:
                merged_df[f'{col}_norm'] = 0
        
        # Ağırlıklı rating
        weights = [0.25, 0.30, 0.20, 0.15, 0.10]  # MP, Gls, Ast, Sh, SoT
        merged_df['Rating'] = 0
        
        for i, col in enumerate(available_metrics):
            if i < len(weights):
                merged_df['Rating'] += weights[i] * merged_df[f'{col}_norm']
        
        # 0-100 arasına scale et
        scaler = MinMaxScaler(feature_range=(0, 100))
        merged_df['Rating'] = scaler.fit_transform(merged_df[['Rating']])
    
    print("🔍 Son kontroller yapılıyor...")
    
    total_players = len(merged_df)
    matched_players = merged_df[merged_df['MP'] > 0].shape[0]
    match_rate = (matched_players / total_players) * 100
    
    print(f"📊 Eşleştirme İstatistikleri:")
    print(f" Toplam oyuncu: {total_players}")
    print(f" Eşleşen oyuncu: {matched_players}")
    print(f" Eşleşme oranı: {match_rate:.1f}%")
    
    # Takım bazlı istatistikler
    team_stats = []
    for team in merged_df['Team_std'].unique():
        if pd.isna(team):
            continue
            
        team_data = merged_df[merged_df['Team_std'] == team]
        total = len(team_data)
        matched = (team_data['MP'] > 0).sum()
        rate = (matched / total) * 100 if total > 0 else 0
        
        team_stats.append({'Team': team, 'Total': total, 'Matched': matched, 'Rate': rate})
    
    team_stats_df = pd.DataFrame(team_stats).sort_values('Rate', ascending=False)
    
    print("\n🏟️ Takım Bazlı Eşleşme Oranları:")
    for _, row in team_stats_df.iterrows():
        print(f" {row['Team']:25s}: {row['Rate']:5.1f}% ({row['Matched']:2d}/{row['Total']:2d})")
    
    # Gereksiz sütunları temizle
    cols_to_drop = [col for col in merged_df.columns if col.endswith(('_hybrid', '_fbref')) or col in ['Hybrid_Team']]
    cols_to_drop = [col for col in cols_to_drop if col in merged_df.columns]
    merged_df = merged_df.drop(columns=cols_to_drop, errors='ignore')
    
    # Kaydet
    output_path = "data/final_bundesliga_dataset_complete.xlsx"
    merged_df.to_excel(output_path, index=False)
    
    print(f"\n✅ İşlem tamamlandı!")
    print(f"💾 Kaydedildi: {output_path}")
    print(f"📊 Final veri boyutu: {merged_df.shape}")
    
    # En yüksek ratingli 10 oyuncu - Sütun isimlerini kontrol ederek
    if 'Rating' in merged_df.columns:
        print("\n🏆 En yüksek ratingli 10 oyuncu:")
        
        # Mevcut sütunları kontrol et
        available_cols = []
        for col in ['Player', 'Team_hybrid', 'Team', 'Squad', 'Rating', 'Gls', 'Ast', 'MP']:
            if col in merged_df.columns:
                available_cols.append(col)
        
        if available_cols:
            top_players = merged_df.nlargest(10, 'Rating')[available_cols]
            print(top_players.to_string(index=False))
        else:
            print("⚠️ Görüntülenecek sütun bulunamadı")

if __name__ == "__main__":
    main()