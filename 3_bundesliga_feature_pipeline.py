# -*- coding: utf-8 -*-
import pandas as pd
import os

# 📂 Çıktı klasörü
output_dir = "data"
os.makedirs(output_dir, exist_ok=True)

# 1️⃣ Mevcut Bundesliga maç verisi
matches_path = os.path.join(output_dir, "bundesliga_matches_2023_2025.pkl")
df_matches = pd.read_pickle(matches_path)

# 2️⃣ 2024-25 güncel takımlar
current_bundesliga_teams = [
    "FC Bayern München", "Bayer 04 Leverkusen", "Eintracht Frankfurt", "Borussia Dortmund",
    "SC Freiburg", "1. FSV Mainz 05", "RB Leipzig", "SV Werder Bremen", "VfB Stuttgart",
    "Borussia Mönchengladbach", "VfL Wolfsburg", "FC Augsburg", "1. FC Union Berlin",
    "FC St. Pauli", "TSG 1899 Hoffenheim", "1. FC Heidenheim 1846", "1. FC Köln", "Hamburger SV"
]

# 3️⃣ Takım isim mapping - tüm takımlar
team_name_mapping = {
    "Sport-Club Freiburg": "SC Freiburg",
    "TSG Hoffenheim": "TSG 1899 Hoffenheim",
    "FC St. Pauli 1910": "FC St. Pauli",
    "1. FC Heidenheim 1846": "1. FC Heidenheim 1846",
    "FC Köln": "1. FC Köln",
    "Hamburg": "Hamburger SV",
    "Bayern München": "FC Bayern München",
    "Bayer Leverkusen": "Bayer 04 Leverkusen",
    "Borussia M'gladbach": "Borussia Mönchengladbach",
    "Mainz 05": "1. FSV Mainz 05",
    "VfB Stuttgart": "VfB Stuttgart",
    "VfL Wolfsburg": "VfL Wolfsburg",
    "Eintracht Frankfurt": "Eintracht Frankfurt",
    "Borussia Dortmund": "Borussia Dortmund",
    "RB Leipzig": "RB Leipzig",
    "SV Werder Bremen": "SV Werder Bremen",
    "FC Augsburg": "FC Augsburg",
    "Union Berlin": "1. FC Union Berlin"
}

# 4️⃣ Düşen takımlar
drop_teams = ["Holstein Kiel", "Fortuna Düsseldorf"]

# 5️⃣ Takım isimlerini mapping ile düzelt
df_matches["homeTeam.name"] = df_matches["homeTeam.name"].replace(team_name_mapping)
df_matches["awayTeam.name"] = df_matches["awayTeam.name"].replace(team_name_mapping)

# 6️⃣ Yeni çıkan takımların 2. lig maçları
df_new = pd.read_excel(os.path.join(output_dir, "bundesliga2_new_teams_2024_25.xlsx"))
df_new['HomeTeam'] = df_new['HomeTeam'].replace(team_name_mapping)
df_new['AwayTeam'] = df_new['AwayTeam'].replace(team_name_mapping)

# 7️⃣ ID ve matchday eksiklerini kontrol et ve doldur
start_id = df_matches['id'].max() + 1
start_home_id = df_matches['homeTeam.id'].max() + 1
start_away_id = df_matches['awayTeam.id'].max() + 1001

df_new_mapped = pd.DataFrame({
    "id": range(start_id, start_id + len(df_new)),
    "utcDate": pd.to_datetime(df_new['Date'], dayfirst=True).dt.strftime('%Y-%m-%dT%H:%M:%SZ'),
    "matchday": df_new.get('Matchday', [0]*len(df_new)),
    "homeTeam.id": range(start_home_id, start_home_id + len(df_new)),
    "homeTeam.name": df_new['HomeTeam'],
    "awayTeam.id": range(start_away_id, start_away_id + len(df_new)),
    "awayTeam.name": df_new['AwayTeam'],
    "score.fullTime.home": df_new['FTHG'],
    "score.fullTime.away": df_new['FTAG'],
    "result": df_new['FTR']
})

# 8️⃣ Birleştir
df_final = pd.concat([df_matches, df_new_mapped], ignore_index=True)

# 9️⃣ Eksik değer kontrolü ve id / team id sıfırlaması
df_final['homeTeam.id'] = df_final['homeTeam.id'].fillna(-1).astype(int)
df_final['awayTeam.id'] = df_final['awayTeam.id'].fillna(-1).astype(int)
df_final['id'] = df_final['id'].fillna(-1).astype(int)
df_final['matchday'] = df_final['matchday'].fillna(0).astype(int)

# 🔹 Feature Engineering: Temel istatistikler
def calc_team_stats(df):
    teams = df['homeTeam.name'].unique()
    stats = []
    for team in teams:
        home = df[df['homeTeam.name']==team]
        away = df[df['awayTeam.name']==team]
        total_games = len(home) + len(away)
        wins = len(home[home['result']=="HomeWin"]) + len(away[away['result']=="AwayWin"])
        draws = len(home[home['result']=="Draw"]) + len(away[away['result']=="Draw"])
        losses = total_games - wins - draws
        goals_for = home['score.fullTime.home'].sum() + away['score.fullTime.away'].sum()
        goals_against = home['score.fullTime.away'].sum() + away['score.fullTime.home'].sum()
        stats.append({
            'team': team,
            'total_games': total_games,
            'wins': wins,
            'draws': draws,
            'losses': losses,
            'goals_for': goals_for,
            'goals_against': goals_against
        })
    return pd.DataFrame(stats)

df_team_stats = calc_team_stats(df_final)

# 🔹 Son 5 maç formu (ev/deplasman ayrımı)
for team in current_bundesliga_teams:
    team_matches = df_final[(df_final['homeTeam.name']==team) | (df_final['awayTeam.name']==team)].sort_values('utcDate')
    home_points = []
    away_points = []
    for idx, row in team_matches.iterrows():
        if row['homeTeam.name']==team:
            pt_home = 3 if row['result']=='HomeWin' else 1 if row['result']=='Draw' else 0
            pt_away = 0
        else:
            pt_home = 0
            pt_away = 3 if row['result']=='AwayWin' else 1 if row['result']=='Draw' else 0
        home_points.append(pt_home)
        away_points.append(pt_away)
    last5_home = [sum(home_points[max(0,i-5):i]) for i in range(1,len(home_points)+1)]
    last5_away = [sum(away_points[max(0,i-5):i]) for i in range(1,len(away_points)+1)]
    df_final.loc[team_matches.index, 'home_form'] = last5_home
    df_final.loc[team_matches.index, 'away_form'] = last5_away

# 🔹 Gelecek tahminler için filtre
df_future = df_final[
    df_final["homeTeam.name"].isin(current_bundesliga_teams) &
    df_final["awayTeam.name"].isin(current_bundesliga_teams)
]

# 🔹 Features ve target oluşturma
team_stats_dict = df_team_stats.set_index('team').to_dict('index')

def get_features(row):
    home_team = row['homeTeam.name']
    away_team = row['awayTeam.name']
    
    home_stats = team_stats_dict.get(home_team, {})
    away_stats = team_stats_dict.get(away_team, {})
    
    return pd.Series({
        'home_total_games': home_stats.get('total_games', 0),
        'home_wins': home_stats.get('wins', 0),
        'home_draws': home_stats.get('draws', 0),
        'home_losses': home_stats.get('losses', 0),
        'home_goals_for': home_stats.get('goals_for', 0),
        'home_goals_against': home_stats.get('goals_against', 0),
        'away_total_games': away_stats.get('total_games', 0),
        'away_wins': away_stats.get('wins', 0),
        'away_draws': away_stats.get('draws', 0),
        'away_losses': away_stats.get('losses', 0),
        'away_goals_for': away_stats.get('goals_for', 0),
        'away_goals_against': away_stats.get('goals_against', 0),
        'home_form': row.get('home_form', 0),
        'away_form': row.get('away_form', 0)
    })

df_features = df_final.apply(get_features, axis=1)
df_target = df_final['result']

# 🔹 Kaydet
df_final.to_pickle(os.path.join(output_dir, "bundesliga_matches_2023_2025_final_fe.pkl"))
df_final.to_csv(os.path.join(output_dir, "bundesliga_matches_2023_2025_final_fe.csv"), index=False, encoding="utf-8-sig")
df_final.to_excel(os.path.join(output_dir, "bundesliga_matches_2023_2025_final_fe.xlsx"), index=False, engine="openpyxl")

print("[✓] Final dataset (feature engineering + ML-ready) kaydedildi")
print("Toplam maç sayısı:", len(df_final))
print("Gelecek tahminler için filtrelenmiş maç sayısı:", len(df_future))
print("Örnek Features (X) ve Target (y):")
print(df_features.head(3))
print(df_target.head(3))
