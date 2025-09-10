import pandas as pd
import numpy as np

# Dosya yolları
PLAYER_RATINGS_PATH = 'data/calculate_player_rating.xlsx'
MATCHES_PATH = 'data/bundesliga_matches_2023_2025.xlsx'
OUTPUT_PATH = 'data/matches_with_team_ratings.xlsx'

def compute_team_ratings(df_matches, df_players):
    """
    Takım bazlı ortalama ratingleri hesaplar
    """
    df_matches = df_matches.copy()
    df_players = df_players.copy()
    
    # PlayerRating sayıya çevir ve takımları strip yap
    df_players['PlayerRating'] = pd.to_numeric(df_players['PlayerRating'], errors='coerce')
    df_players['Team'] = df_players['Team'].astype(str).str.strip()
    
    df_matches['homeTeam.name'] = df_matches['homeTeam.name'].astype(str).str.strip()
    df_matches['awayTeam.name'] = df_matches['awayTeam.name'].astype(str).str.strip()
    
    # Yeni sütunlar
    for col in ['Home_AvgRating', 'Away_AvgRating', 'Total_AvgRating', 'Rating_Diff']:
        df_matches[col] = np.nan

    # Her maç için hesapla
    for idx, row in df_matches.iterrows():
        home_team = row['homeTeam.name']
        away_team = row['awayTeam.name']

        home_ratings = df_players[df_players['Team']==home_team]['PlayerRating'].dropna()
        away_ratings = df_players[df_players['Team']==away_team]['PlayerRating'].dropna()

        home_avg = home_ratings.mean() if not home_ratings.empty else np.nan
        away_avg = away_ratings.mean() if not away_ratings.empty else np.nan
        total_avg = np.nanmean([home_avg, away_avg])
        diff = home_avg - away_avg if not np.isnan(home_avg) and not np.isnan(away_avg) else np.nan

        df_matches.at[idx, 'Home_AvgRating'] = home_avg
        df_matches.at[idx, 'Away_AvgRating'] = away_avg
        df_matches.at[idx, 'Total_AvgRating'] = total_avg
        df_matches.at[idx, 'Rating_Diff'] = diff

    return df_matches

def fill_missing_ratings(df_matches):
    """
    Eksik değerleri doldurur (basit: sütun ortalamasıyla)
    """
    for col in ['Home_AvgRating', 'Away_AvgRating', 'Total_AvgRating', 'Rating_Diff']:
        df_matches[col] = df_matches[col].fillna(df_matches[col].mean())
    return df_matches

if __name__ == "__main__":
    # Maç ve oyuncu datasını yükle
    df_matches = pd.read_excel(MATCHES_PATH)
    df_players = pd.read_excel(PLAYER_RATINGS_PATH)

    # Takım bazlı ratingleri hesapla
    df_matches = compute_team_ratings(df_matches, df_players)

    # Eksik değerleri doldur
    df_matches = fill_missing_ratings(df_matches)

    # Yeni dataset kaydet
    df_matches.to_excel(OUTPUT_PATH, index=False)
    print(f"✅ Takım bazlı ratingler hesaplandı ve kaydedildi: {OUTPUT_PATH}")
