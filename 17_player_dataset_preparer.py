import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Dosya yolları
FEATURES_PATH = 'data/bundesliga_features_24_25.xlsx'
PLAYER_RATINGS_PATH = 'data/calculate_player_rating.xlsx'
OUTPUT_PATH = 'data/bundesliga_features_complete.xlsx'

# Oyuncu datasını yükle
df_players = pd.read_excel(PLAYER_RATINGS_PATH)
df_players['PlayerRating'] = pd.to_numeric(df_players['PlayerRating'], errors='coerce')
df_players['Team'] = df_players['Team'].astype(str).str.strip()
df_players['Pos'] = df_players['Pos'].astype(str).str.strip()

# Maç datasını yükle
df_matches = pd.read_excel(FEATURES_PATH)
df_matches['homeTeam.name'] = df_matches['homeTeam.name'].astype(str).str.strip()
df_matches['awayTeam.name'] = df_matches['awayTeam.name'].astype(str).str.strip()

# Eksik sütunları ekle
rating_cols = ['Home_AvgRating','Away_AvgRating','Total_AvgRating','Rating_Diff']
position_cols = ['Home_GK_Rating','Home_DF_Rating','Home_MF_Rating','Home_FW_Rating',
                 'Away_GK_Rating','Away_DF_Rating','Away_MF_Rating','Away_FW_Rating']
form_cols = ['Home_Form','Away_Form','Form_Diff','Home_Momentum','Away_Momentum']
goals_cols = ['Home_GoalsScored_5','Home_GoalsConceded_5','Away_GoalsScored_5','Away_GoalsConceded_5']

for col in rating_cols + position_cols + form_cols + goals_cols + ['IsDerby']:
    df_matches[col] = np.nan

# Manuel derby listesi (örnek)
derby_teams = [
    ('FC Bayern München','TSG 1899 Hoffenheim'),
    ('SV Werder Bremen','Hamburger SV'),
    # ... ekleyebilirsin
]

# Helper fonksiyon: lineer regresyon eğimi
def calc_trend(values):
    y = np.array(values).reshape(-1,1)
    x = np.arange(len(values)).reshape(-1,1)
    if len(values) < 2:
        return 0
    model = LinearRegression().fit(x, y)
    return float(model.coef_[0])

for idx, row in df_matches.iterrows():
    home_team = row['homeTeam.name']
    away_team = row['awayTeam.name']

    # Takım bazlı rating hesaplaması
    home_ratings = df_players[df_players['Team']==home_team]['PlayerRating'].dropna()
    away_ratings = df_players[df_players['Team']==away_team]['PlayerRating'].dropna()

    df_matches.at[idx,'Home_AvgRating'] = home_ratings.mean() if not home_ratings.empty else np.nan
    df_matches.at[idx,'Away_AvgRating'] = away_ratings.mean() if not away_ratings.empty else np.nan

    if not home_ratings.empty and not away_ratings.empty:
        df_matches.at[idx,'Total_AvgRating'] = (home_ratings.mean() + away_ratings.mean()) / 2
        df_matches.at[idx,'Rating_Diff'] = home_ratings.mean() - away_ratings.mean()

    # Pozisyon bazlı ratingler (GK/DF/MF/FW)
    for pos in ['GK','DF','MF','FW']:
        home_pos_ratings = df_players[(df_players['Team']==home_team) & (df_players['Pos']==pos)]['PlayerRating'].dropna()
        away_pos_ratings = df_players[(df_players['Team']==away_team) & (df_players['Pos']==pos)]['PlayerRating'].dropna()
        df_matches.at[idx,f'Home_{pos}_Rating'] = home_pos_ratings.mean() if not home_pos_ratings.empty else np.nan
        df_matches.at[idx,f'Away_{pos}_Rating'] = away_pos_ratings.mean() if not away_pos_ratings.empty else np.nan

    # Form ve momentum (son 5 maç)
    for side in ['Home','Away']:
        wins = df_matches.at[idx, f'{side.lower()}_team_last5_wins'] if f'{side.lower()}_team_last5_wins' in df_matches.columns else np.nan
        draws = df_matches.at[idx, f'{side.lower()}_team_last5_draws'] if f'{side.lower()}_team_last5_draws' in df_matches.columns else np.nan
        losses = df_matches.at[idx, f'{side.lower()}_team_last5_losses'] if f'{side.lower()}_team_last5_losses' in df_matches.columns else np.nan

        if pd.notna(wins) and pd.notna(draws) and pd.notna(losses):
            form_points = np.array([3]*int(wins) + [1]*int(draws) + [0]*int(losses))
            if len(form_points)>0:
                df_matches.at[idx,f'{side}_Form'] = form_points.mean()
                df_matches.at[idx,f'{side}_Momentum'] = calc_trend(form_points)

        # Goals scored/conceded
        df_matches.at[idx,f'{side}_GoalsScored_5'] = row.get(f'{side.lower()}_team_last5_avg_goals_scored', np.nan)
        df_matches.at[idx,f'{side}_GoalsConceded_5'] = row.get(f'{side.lower()}_team_last5_avg_goals_conceded', np.nan)


    # Form farkı
    if not np.isnan(df_matches.at[idx,'Home_Form']) and not np.isnan(df_matches.at[idx,'Away_Form']):
        df_matches.at[idx,'Form_Diff'] = df_matches.at[idx,'Home_Form'] - df_matches.at[idx,'Away_Form']

    # IsDerby
    df_matches.at[idx,'IsDerby'] = int((home_team,away_team) in derby_teams or (away_team,home_team) in derby_teams)

# Yeni dataset kaydet
df_matches.to_excel(OUTPUT_PATH, index=False)
print(f"✅ Tüm eksik sütunlar hesaplandı ve kaydedildi: {OUTPUT_PATH}")
