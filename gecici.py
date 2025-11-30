#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bundesliga_card_pipeline_api.py (RATE-LIMIT SAFE)
Football-Data API tabanlı Bundesliga kart ve disiplin pipeline
- 429 Too Many Requests Hatası Almayan Sürüm
- Otomatik rate-limit handling
- Her istek arası bekleme + 429 gelirse ekstra bekleme
"""

import requests
import pandas as pd
import numpy as np
import time
import logging
from typing import Dict, List

# ---------- CONFIG ----------
API_KEY = "745aa92502704e82902bdd4cd5df40e4"
BASE_URL = "https://api.football-data.org/v4"
COMPETITION = "BL1"
SEASON = 2025

OUTPUT_PATH = r"C:\Users\canse\OneDrive\Masaüstü\bundesliga_card_enriched.xlsx"

REQUEST_TIMEOUT = 15

# RATE-LIMIT AYARLARI
RATE_LIMIT_SLEEP = 7
RETRY_SLEEP_429 = 20
MAX_RETRY_429 = 5

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

HEADERS = {"X-Auth-Token": API_KEY}

# ---------- RATE-LIMIT SAFE API GET ----------
def api_get(endpoint: str, params=None) -> Dict:
    url = f"{BASE_URL}/{endpoint}"

    for attempt in range(MAX_RETRY_429):
        try:
            r = requests.get(url, headers=HEADERS, params=params, timeout=REQUEST_TIMEOUT)

            if r.status_code == 200:
                time.sleep(RATE_LIMIT_SLEEP)
                return r.json()

            if r.status_code == 429:
                logging.warning(f"429 TOO MANY REQUESTS → {RETRY_SLEEP_429} saniye bekleniyor...")
                time.sleep(RETRY_SLEEP_429)
                continue

            logging.warning(f"API GET {url} failed: {r.status_code}")
            return {}

        except Exception as e:
            logging.error(f"API GET {url} exception: {e}")
            time.sleep(5)
            continue

    logging.error(f"API GET {url} → MAX RETRY AŞILDI")
    return {}

# ---------- MATCH LIST ----------
def fetch_matches(season: int = SEASON) -> List[Dict]:
    endpoint = f"competitions/{COMPETITION}/matches"
    params = {"season": season}
    data = api_get(endpoint, params=params)
    matches = data.get("matches", [])
    logging.info(f"Fetched {len(matches)} matches")
    return matches

# ---------- MATCH EVENTS ----------
def fetch_match_events(match_id: int) -> Dict:
    endpoint = f"matches/{match_id}"
    data = api_get(endpoint)
    return data

# ---------- FEATURE ENGINEERING ----------
def extract_player_events(match_data: Dict) -> List[Dict]:
    events = []

    if "match" not in match_data:
        return events

    match = match_data["match"]
    match_id = match.get("id")

    for team_key in ["homeTeam", "awayTeam"]:
        team = match.get(team_key, {})
        team_id = team.get("id")
        team_name = team.get("name")

        for ev in team.get("events", []):
            if ev.get("type") in ["YELLOW_CARD", "RED_CARD", "YELLOW_RED"]:
                events.append({
                    "match_id": match_id,
                    "team_id": team_id,
                    "team": team_name,
                    "player_id": ev.get("player", {}).get("id"),
                    "player": ev.get("player", {}).get("name"),
                    "card_type": ev.get("type"),
                    "minute": ev.get("minute"),
                })
    return events

# ---------- PIPELINE ----------
def run_pipeline():
    logging.info("Bundesliga kart-disiplin pipeline başladı...")

    matches = fetch_matches()
    all_events = []

    for m in matches:
        match_id = m.get("id")
        if not match_id:
            continue

        logging.info(f"Maç çekiliyor → ID: {match_id}")
        match_data = fetch_match_events(match_id)

        events = extract_player_events(match_data)
        all_events.extend(events)

    df = pd.DataFrame(all_events)

    if df.empty:
        logging.warning("Hiç event bulunamadı. API cevap vermemiş olabilir.")
        return

    df["yellow"] = (df["card_type"] == "YELLOW_CARD").astype(int)
    df["red"] = (df["card_type"] == "RED_CARD").astype(int)
    df["yellow_red"] = (df["card_type"] == "YELLOW_RED").astype(int)

    summary = df.groupby(["player_id", "player", "team"]).agg({
        "yellow": "sum",
        "red": "sum",
        "yellow_red": "sum",
    }).reset_index()

    logging.info("Excel kaydediliyor...")
    summary.to_excel(OUTPUT_PATH, index=False)

    logging.info(f"Tamamlandı → {OUTPUT_PATH}")

# ---------- RUN ----------
if __name__ == "__main__":
    run_pipeline() 
