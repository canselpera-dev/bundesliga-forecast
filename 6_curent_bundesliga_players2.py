# bundesliga_squad_scraper_hybrid.py

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import random
import os
from fake_useragent import UserAgent
import urllib3
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import re

# SSL uyarılarını kapat
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

ua = UserAgent()

# Bundesliga takımları (2025 sezonu için)
teams_urls = {
    # Bayern Munich
    "fc bayern munchen": "https://www.transfermarkt.com/fc-bayern-munchen/startseite/verein/27",
    
    # Bayer Leverkusen
    "bayer 04 leverkusen": "https://www.transfermarkt.com/bayer-04-leverkusen/startseite/verein/15",
    
    # Borussia Dortmund
    "borussia dortmund": "https://www.transfermarkt.com/borussia-dortmund/startseite/verein/16",
    
    # RB Leipzig
    "rb leipzig": "https://www.transfermarkt.com/rb-leipzig/startseite/verein/23826",
    
    # VfB Stuttgart
    "vfb stuttgart": "https://www.transfermarkt.com/vfb-stuttgart/startseite/verein/79",
    
    # Eintracht Frankfurt
    "eintracht frankfurt": "https://www.transfermarkt.com/eintracht-frankfurt/startseite/verein/24",
    
    # TSG Hoffenheim
    "tsg 1899 hoffenheim": "https://www.transfermarkt.com/tsg-1899-hoffenheim/startseite/verein/533",
    
    # SC Freiburg
    "sc freiburg": "https://www.transfermarkt.com/sc-freiburg/startseite/verein/60",
    
    # 1. FC Heidenheim 1846
    "1. fc heidenheim 1846": "https://www.transfermarkt.com/1-fc-heidenheim/startseite/verein/2036",
    
    # SV Werder Bremen
    "sv werder bremen": "https://www.transfermarkt.com/sv-werder-bremen/startseite/verein/86",
    
    # Borussia Mönchengladbach
    "borussia monchengladbach": "https://www.transfermarkt.com/borussia-monchengladbach/startseite/verein/18",
    
    # VfL Wolfsburg
    "vfl wolfsburg": "https://www.transfermarkt.com/vfl-wolfsburg/startseite/verein/82",
    
    # FC Augsburg
    "fc augsburg": "https://www.transfermarkt.com/fc-augsburg/startseite/verein/167",
    
    # 1. FC Union Berlin
    "1. fc union berlin": "https://www.transfermarkt.com/1-fc-union-berlin/startseite/verein/89",
    
    # VfL Bochum
    "vfl bochum": "https://www.transfermarkt.com/vfl-bochum/startseite/verein/80",
    
    # 1. FC Köln
    "1. fc koln": "https://www.transfermarkt.com/1-fc-koln/startseite/verein/3",
    
    # Mainz 05
    "1. fsv mainz 05": "https://www.transfermarkt.com/1-fsv-mainz-05/startseite/verein/39",
    
    # Holstein Kiel
    "holstein kiel": "https://www.transfermarkt.com/holstein-kiel/startseite/verein/1121",
    
    # FC St. Pauli
    "fc st. pauli": "https://www.transfermarkt.com/fc-st-pauli/startseite/verein/560",
    
    # Hamburger SV
    "hamburger sv": "https://www.transfermarkt.com/hamburger-sv/startseite/verein/3"
}


DATA_PATH = f"data/bundesliga_squads_hybrid.xlsx"

def create_session():
    """Retry mekanizmalı session"""
    session = requests.Session()
    retry_strategy = Retry(
        total=3,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"]
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

def parse_market_value(value_str):
    """Piyasa değerini sayısal değere çevirir"""
    if not value_str or value_str == '-':
        return None
    
    value_str = value_str.replace('€', '').replace(' ', '')
    
    if 'm' in value_str:
        return float(value_str.replace('m', '').replace(',', '.')) * 1_000_000
    elif 'Th.' in value_str:
        return float(value_str.replace('Th.', '').replace(',', '.')) * 1_000
    elif 'k' in value_str:
        return float(value_str.replace('k', '').replace(',', '.')) * 1_000
    else:
        try:
            return float(value_str)
        except:
            return None

def parse_age(birth_date_str):
    """Yaşı doğrudan ya da doğum tarihi + yaş formatından hesapla"""
    if not birth_date_str:
        return None
    birth_date_str = birth_date_str.strip()
    if birth_date_str.isdigit():
        return int(birth_date_str)
    age_match = re.search(r'\((\d+)\)', birth_date_str)
    if age_match:
        return int(age_match.group(1))
    return None

def extract_all_data_hybrid(row):
    """HIBRIT veri çekme"""
    cells = row.find_all('td')
    player_data = {
        'Player': None,
        'Position': None,
        'Number': None,
        'Age': None,
        'Birth_Date': None,
        'Nationality': None,
        'Market_Value': None,
        'Market_Value_Text': None
    }
    
    # Index bazlı
    if len(cells) >= 6:
        if cells[0].text.strip().isdigit():
            player_data['Number'] = cells[0].text.strip()
        player_data['Player'] = cells[3].text.strip() if len(cells) > 3 else None
        player_data['Position'] = cells[4].text.strip() if len(cells) > 4 else None
        birth_text = cells[5].text.strip() if len(cells) > 5 else None
        if birth_text:
            player_data['Birth_Date'] = birth_text
            player_data['Age'] = parse_age(birth_text)

    # Class bazlı fallback
    if not player_data['Player']:
        name_tag = row.find("a", {"class": "spielprofil_tooltip"})
        if name_tag and name_tag.text.strip():
            player_data['Player'] = name_tag.text.strip()
    if not player_data['Position']:
        position_tag = row.find("td", {"class": "posrela"})
        if position_tag:
            text = position_tag.text.strip()
            lines = text.split('\n')
            player_data['Position'] = lines[-1].strip() if len(lines) > 1 else text
    if not player_data['Number']:
        number_tag = row.find("div", {"class": "rn_nummer"})
        if number_tag and number_tag.text.strip().isdigit():
            player_data['Number'] = number_tag.text.strip()
    flags = row.find_all('img', {'class': 'flaggenrahmen'})
    player_data['Nationality'] = ', '.join([flag['title'] for flag in flags if 'title' in flag.attrs]) or None
    market_value_tag = row.find("td", {"class": "rechts"})
    if market_value_tag:
        mv_text = market_value_tag.text.strip()
        player_data['Market_Value_Text'] = mv_text
        player_data['Market_Value'] = parse_market_value(mv_text)
    if player_data['Age'] is None and player_data['Birth_Date']:
        player_data['Age'] = parse_age(player_data['Birth_Date'])
    return player_data

def fetch_team_squad_hybrid(team_name, team_url, season=2025, retries=5):
    """Takım kadrosu çekme (retry ile)"""
    url = f"{team_url}/saison_id/{season}"
    session = create_session()
    for attempt in range(1, retries+1):
        try:
            headers = {"User-Agent": ua.random}
            time.sleep(random.uniform(7, 15))  # daha uzun bekleme
            response = session.get(url, headers=headers, timeout=30, verify=False)
            if response.status_code != 200:
                print(f"⚠ {team_name}: HTTP {response.status_code}, deneme {attempt}/{retries}")
                continue
            soup = BeautifulSoup(response.text, "html.parser")
            table = soup.find("table", {"class": "items"})
            if not table:
                print(f"⚠ {team_name}: tablo bulunamadı, deneme {attempt}/{retries}")
                continue
            rows = table.find_all("tr", {"class": ["odd", "even"]})
            players = []
            for row in rows:
                pdata = extract_all_data_hybrid(row)
                link_tag = row.find("a", {"class": "spielprofil_tooltip"})
                player_link = "https://www.transfermarkt.com" + link_tag['href'] if link_tag and 'href' in link_tag.attrs else None
                pdata.update({
                    "Team": team_name,
                    "Player_Link": player_link,
                    "Season": f"{season}/{season+1}"
                })
                players.append(pdata)
            return pd.DataFrame(players)
        except Exception as e:
            print(f"⚠ {team_name}: hata {e}, deneme {attempt}/{retries}")
    return pd.DataFrame()

def main():
    os.makedirs("data", exist_ok=True)
    all_players = []
    for team_name, team_url in teams_urls.items():
        print(f"\n⏳ {team_name} kadrosu çekiliyor...")
        df_team = fetch_team_squad_hybrid(team_name, team_url)
        if not df_team.empty:
            print(f"✅ {team_name}: {len(df_team)} oyuncu çekildi")
            all_players.append(df_team)
        else:
            print(f"❌ {team_name}: veri çekilemedi")
    if all_players:
        df_all = pd.concat(all_players, ignore_index=True)
        df_all.to_excel(DATA_PATH, index=False)
        print(f"\n🎉 TÜM TAKIMLAR KAYDEDİLDİ → {DATA_PATH}")

if __name__ == "__main__":
    main()
