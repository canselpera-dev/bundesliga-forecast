import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

# Dosya yolları
FEATURES_PATH = 'data/bundesliga_features_24_25.xlsx'
OUTPUT_PATH = 'data/bundesliga_features_complete.xlsx'

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

for col in rating_cols + position_cols + form_cols + goals_cols + ['IsDerby','group']:
    df_matches[col] = np.nan

# Manuel derby listesi
derby_teams = [
    ('FC Bayern München','TSG 1899 Hoffenheim'),
    ('SV Werder Bremen','Hamburger SV'),
    ('Borussia Dortmund','FC Schalke 04'),
    ('1. FC Köln','Borussia Mönchengladbach'),
    # İhtiyaç halinde tüm derbiler buraya eklenebilir
]

# Helper fonksiyon: lineer regresyon eğimi
def calc_trend(values):
    x = np.arange(len(values)).reshape(-1,1)
    y = np.array(values).reshape(-1,1)
    if len(values) < 2:
        return 0
    model = LinearRegression().fit(x, y)
    return float(model.coef_[0][0])

# Helper fonksiyon: takım bazlı rating hesaplama
def calc_team_rating(team_name, df_matches, last_n=5):
    df_team_home = df_matches[df_matches['homeTeam.name'] == team_name].tail(last_n)
    df_team_away = df_matches[df_matches['awayTeam.name'] == team_name].tail(last_n)
    
    points_home = df_team_home['score.fullTime.home'].sub(df_team_home['score.fullTime.away'])
    points_home = points_home.apply(lambda x: 3 if x>0 else (1 if x==0 else 0))
    
    points_away = df_team_away['score.fullTime.away'].sub(df_team_away['score.fullTime.home'])
    points_away = points_away.apply(lambda x: 3 if x>0 else (1 if x==0 else 0))
    
    all_points = pd.concat([points_home, points_away])
    avg_rating = all_points.mean() if len(all_points) > 0 else np.nan
    
    gk_rating = avg_rating * 0.2
    df_rating = avg_rating * 0.25
    mf_rating = avg_rating * 0.25
    fw_rating = avg_rating * 0.3
    
    return avg_rating, gk_rating, df_rating, mf_rating, fw_rating

# Helper fonksiyon: son 5 maç verisi skor istatistiklerinden türet
def get_last5_stats(team_name):
    matches = df_matches[
        (df_matches['homeTeam.name'] == team_name) | 
        (df_matches['awayTeam.name'] == team_name)
    ].sort_values('utcDate')  # tarih sırasına göre sırala
    last5 = matches.tail(5)
    wins = draws = losses = 0
    goals_scored = goals_conceded = []

    for _, row in last5.iterrows():
        if row['homeTeam.name'] == team_name:
            gf = row['score.fullTime.home']
            ga = row['score.fullTime.away']
        else:
            gf = row['score.fullTime.away']
            ga = row['score.fullTime.home']

        goals_scored.append(gf)
        goals_conceded.append(ga)

        if gf > ga:
            wins += 1
        elif gf == ga:
            draws += 1
        else:
            losses += 1

    avg_goals_scored = np.mean(goals_scored) if goals_scored else np.nan
    avg_goals_conceded = np.mean(goals_conceded) if goals_conceded else np.nan
    return wins, draws, losses, avg_goals_scored, avg_goals_conceded

# Her maç için hesaplamalar
for idx, row in df_matches.iterrows():
    home_team = row['homeTeam.name']
    away_team = row['awayTeam.name']

    # Home/Away Rating ve pozisyon bazlı rating
    home_avg, home_gk, home_df, home_mf, home_fw = calc_team_rating(home_team, df_matches)
    away_avg, away_gk, away_df, away_mf, away_fw = calc_team_rating(away_team, df_matches)
    
    df_matches.at[idx,'Home_AvgRating'] = home_avg
    df_matches.at[idx,'Away_AvgRating'] = away_avg
    df_matches.at[idx,'Total_AvgRating'] = np.nanmean([home_avg, away_avg])
    df_matches.at[idx,'Rating_Diff'] = home_avg - away_avg if not np.isnan(home_avg) and not np.isnan(away_avg) else np.nan
    
    df_matches.at[idx,'Home_GK_Rating'] = home_gk
    df_matches.at[idx,'Home_DF_Rating'] = home_df
    df_matches.at[idx,'Home_MF_Rating'] = home_mf
    df_matches.at[idx,'Home_FW_Rating'] = home_fw
    df_matches.at[idx,'Away_GK_Rating'] = away_gk
    df_matches.at[idx,'Away_DF_Rating'] = away_df
    df_matches.at[idx,'Away_MF_Rating'] = away_mf
    df_matches.at[idx,'Away_FW_Rating'] = away_fw

    # Form ve momentum (son 5 maç)
    for side, team in [('Home', home_team), ('Away', away_team)]:
        wins, draws, losses, avg_gf, avg_ga = get_last5_stats(team)
        form_points = np.array([3]*wins + [1]*draws + [0]*losses)
        df_matches.at[idx,f'{side}_Form'] = form_points.mean() if len(form_points) > 0 else np.nan
        df_matches.at[idx,f'{side}_Momentum'] = calc_trend(form_points) if len(form_points) > 0 else 0
        df_matches.at[idx,f'{side}_GoalsScored_5'] = avg_gf
        df_matches.at[idx,f'{side}_GoalsConceded_5'] = avg_ga

    # Form farkı
    if not np.isnan(df_matches.at[idx,'Home_Form']) and not np.isnan(df_matches.at[idx,'Away_Form']):
        df_matches.at[idx,'Form_Diff'] = df_matches.at[idx,'Home_Form'] - df_matches.at[idx,'Away_Form']

    # IsDerby
    df_matches.at[idx,'IsDerby'] = int((home_team,away_team) in derby_teams or (away_team,home_team) in derby_teams)

# Group sütunu: maç tarihine göre hafta numarası (sıradaki maç numarası)
df_matches = df_matches.sort_values('utcDate').reset_index(drop=True)
df_matches['group'] = df_matches.groupby(['homeTeam.name','awayTeam.name']).cumcount() + 1

# Eksik değer loglama
missing_info = df_matches.isna().sum()
print("\n=== Eksik Sütunlar ===")
for col, count in missing_info.items():
    if count > 0:
        print(f"{col}: {count} eksik değer")
missing_rows = df_matches.isna().any(axis=1).sum()
print(f"\n=== Eksik Değer İçeren Satırlar ===")
print(f"Toplam eksik değer içeren satır sayısı: {missing_rows}")

# Kritik eksik değer analizi
critical_feature = missing_info.idxmax()
print(f"\n⚠️ En kritik eksik veri: {critical_feature} ({missing_info.max()} eksik değer)")

# Yeni dataset kaydet
df_matches.to_excel(OUTPUT_PATH, index=False)
print(f"\n✅ Tüm hesaplamalar tamamlandı ve kaydedildi: {OUTPUT_PATH}")
