import pandas as pd

# Bundesliga 2 2024-25 sezonu CSV URL
url = "https://www.football-data.co.uk/mmz4281/2425/D2.csv"

# CSV’yi oku
df_bundesliga2 = pd.read_csv(url)

# İlk 5 satırı kontrol et
print(df_bundesliga2.head())

new_teams_csv = ["FC Koln", "Hamburg"]  # CSV’deki isimler
df_new_teams = df_bundesliga2[
    (df_bundesliga2['HomeTeam'].isin(new_teams_csv)) |
    (df_bundesliga2['AwayTeam'].isin(new_teams_csv))
].copy()
name_mapping = {
    "FC Koln": "1. FC Köln",
    "Hamburg": "Hamburger SV"
}

df_new_teams['HomeTeam'] = df_new_teams['HomeTeam'].map(name_mapping).fillna(df_new_teams['HomeTeam'])
df_new_teams['AwayTeam'] = df_new_teams['AwayTeam'].map(name_mapping).fillna(df_new_teams['AwayTeam'])
df_new_teams.to_pickle("data/bundesliga2_new_teams_2024_25.pkl")
df_new_teams.to_csv("data/bundesliga2_new_teams_2024_25.csv", index=False)
df_new_teams.to_excel("data/bundesliga2_new_teams_2024_25.xlsx", index=False)
print(f"[✓] Yeni takımlar kaydedildi: {len(df_new_teams)} maç")
print("Örnek satır:")
print(df_new_teams.head())


