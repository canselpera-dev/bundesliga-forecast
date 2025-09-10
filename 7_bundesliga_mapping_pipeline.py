#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bundesliga Mapping & Merge — Enhanced Player Matching Version
=============================================================

Geliştirmeler:
1. Çoklu fuzzy scoring yöntemleri (ratio, partial_ratio, token_sort_ratio, token_set_ratio)
2. Gelişmiş oyuncu normalizasyonu (soyad odaklı)
3. Soyad fallback mekanizması
4. Detaylı player mapping raporu
5. Manuel inceleme için öneriler
"""

import os
import json
import pandas as pd
import unicodedata
import re
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# fuzzy helper: tercih rapidfuzz, yoksa fuzzywuzzy
try:
    from rapidfuzz import process, fuzz  # type: ignore
    def extract_one(query, choices, scorer=None):
        if not choices:
            return None, 0.0
        res = process.extractOne(query, choices, scorer=scorer or fuzz.WRatio)
        if res is None:
            return None, 0.0
        cand, score, _ = res
        return cand, float(score)
    
    def best_player_match(query, candidates):
        """Çoklu fuzzy yöntemleri ile en iyi eşleşmeyi bul"""
        if not candidates:
            return None, 0.0, None
        
        methods = [
            (fuzz.ratio, "ratio"),
            (fuzz.partial_ratio, "partial_ratio"),
            (fuzz.token_sort_ratio, "token_sort_ratio"),
            (fuzz.token_set_ratio, "token_set_ratio")
        ]
        
        best_score = 0
        best_candidate = None
        best_method = None
        
        for scorer, method_name in methods:
            try:
                result = process.extractOne(query, candidates, scorer=scorer)
                if result:
                    cand, score, _ = result
                    if score > best_score:
                        best_score = score
                        best_candidate = cand
                        best_method = method_name
            except:
                continue
        
        return best_candidate, best_score, best_method
    
    FUZZY_LIB = 'rapidfuzz'
except Exception:
    try:
        from fuzzywuzzy import process, fuzz  # type: ignore
        def extract_one(query, choices, scorer=None):
            if not choices:
                return None, 0.0
            res = process.extractOne(query, choices, scorer=scorer or fuzz.WRatio)
            if res is None:
                return None, 0.0
            cand, score = res
            return cand, float(score)
        
        def best_player_match(query, candidates):
            """Çoklu fuzzy yöntemleri ile en iyi eşleşmeyi bul"""
            if not candidates:
                return None, 0.0, None
            
            methods = [
                (fuzz.ratio, "ratio"),
                (fuzz.partial_ratio, "partial_ratio"),
                (fuzz.token_sort_ratio, "token_sort_ratio"),
                (fuzz.token_set_ratio, "token_set_ratio")
            ]
            
            best_score = 0
            best_candidate = None
            best_method = None
            
            for scorer, method_name in methods:
                try:
                    result = process.extractOne(query, candidates, scorer=scorer)
                    if result:
                        cand, score = result
                        if score > best_score:
                            best_score = score
                            best_candidate = cand
                            best_method = method_name
                except:
                    continue
            
            return best_candidate, best_score, best_method
        
        FUZZY_LIB = 'fuzzywuzzy'
    except Exception:
        raise ImportError('rapidfuzz veya fuzzywuzzy kitaplığından biri gerekli')

# --- AYARLAR ---
SQUADS_PATH = "data/bundesliga_squads_hybrid.xlsx"
FBREF_PATH = "data/fbref_team_stats_all_seasons.csv"
OUTPUT_DIR = "logs"
os.makedirs(OUTPUT_DIR, exist_ok=True)
STAMP = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_PATH = os.path.join(OUTPUT_DIR, f'mapping_audit_{STAMP}.txt')
TEAM_SUGGEST_CSV = os.path.join(OUTPUT_DIR, 'team_mapping_suggestions.csv')
MATCHED_TEAMS_CSV = os.path.join(OUTPUT_DIR, 'matched_teams.csv')
UNMATCHED_SQUADS_CSV = os.path.join(OUTPUT_DIR, 'unmatched_squads_teams.csv')
UNMATCHED_FBREF_CSV = os.path.join(OUTPUT_DIR, 'unmatched_fbref_teams.csv')
PLAYER_REPORT_CSV = os.path.join(OUTPUT_DIR, 'player_match_report.csv')
PLAYER_MAPPING_SUGGEST_CSV = os.path.join(OUTPUT_DIR, 'player_mapping_suggestions.csv')

# Eşikler (adaptif kullanacağız)
TEAM_FUZZY_HIGH = 90.0
TEAM_FUZZY_MED  = 80.0
TEAM_FUZZY_LOW  = 70.0
PLAYER_FUZZY_HIGH = 90.0
PLAYER_FUZZY_MED  = 80.0
PLAYER_FUZZY_LOW  = 70.0
PLAYER_SURNAME_ONLY_THRESHOLD = 75.0  # Soyad-only eşleşme için eşik

# Manuel mapping (FBref -> Squads) ORJİNAL halleriyle verildi; kod altında normalize edilecek
COMPLETE_TEAM_MAPPING_RAW = {
    # Bayern Munich
    "fc bayern münchen": "Bayern Munich",
    "fc bayern munchen": "Bayern Munich",
    "bayern münchen": "Bayern Munich",
    "bayern munchen": "Bayern Munich",
    "fc bayern": "Bayern Munich",
    "bayern": "Bayern Munich",
    "bayern münih": "Bayern Munich",
    "bayern munih": "Bayern Munich",
    "bayern munich": "Bayern Munich",
    
    # Borussia Dortmund
    "borussia dortmund": "Borussia Dortmund",
    "bvb dortmund": "Borussia Dortmund",
    "bvb": "Borussia Dortmund",
    "dortmund": "Borussia Dortmund",
    
    # RB Leipzig
    "rb leipzig": "RB Leipzig",
    "rasenballsport leipzig": "RB Leipzig",
    "rasenball leipzig": "RB Leipzig",
    "leipzig": "RB Leipzig",
    
    # Bayer Leverkusen
    "bayer 04 leverkusen": "Bayer Leverkusen",
    "bayer leverkusen": "Bayer Leverkusen",
    "leverkusen": "Bayer Leverkusen",
    
    # VfB Stuttgart
    "vfb stuttgart": "VfB Stuttgart",
    "stuttgart": "VfB Stuttgart",
    
    # Eintracht Frankfurt
    "eintracht frankfurt": "Eintracht Frankfurt",
    "frankfurt": "Eintracht Frankfurt",
    
    # TSG Hoffenheim
    "1899 hoffenheim": "Hoffenheim",
    "tsg 1899 hoffenheim": "Hoffenheim",
    "tsg hoffenheim": "Hoffenheim",
    "hoffenheim": "Hoffenheim",
    
    # SC Freiburg
    "sc freiburg": "SC Freiburg",
    "freiburg": "SC Freiburg",
    
    # Werder Bremen
    "sv werder bremen": "Werder Bremen",
    "werder bremen": "Werder Bremen",
    "bremen": "Werder Bremen",
    
    # Borussia Mönchengladbach
    "borussia mönchengladbach": "Borussia M'gladbach",
    "bor. mönchengladbach": "Borussia M'gladbach",
    "borussia monchengladbach": "Borussia M'gladbach",
    "borussia m'gladbach": "Borussia M'gladbach",
    "mönchengladbach": "Borussia M'gladbach",
    "monchengladbach": "Borussia M'gladbach",
    "gladbach": "Borussia M'gladbach",
    
    # VfL Wolfsburg
    "vfl wolfsburg": "VfL Wolfsburg",
    "wolfsburg": "VfL Wolfsburg",
    
    # FC Augsburg
    "fc augsburg": "FC Augsburg",
    "augsburg": "FC Augsburg",
    
    # 1. FC Heidenheim
    "1. fc heidenheim 1846": "Heidenheim",
    "heidenheim 1846": "Heidenheim",
    "heidenheim": "Heidenheim",
    
    # 1. FC Union Berlin
    "1. fc union berlin": "Union Berlin",
    "union berlin": "Union Berlin",
    
    # Mainz 05
    "1. fsv mainz 05": "Mainz 05",
    "mainz 05": "Mainz 05",
    "mainz": "Mainz 05",
    
    # VfL Bochum
    "vfl bochum 1848": "Bochum",
    "vfl bochum": "Bochum",
    "bochum": "Bochum",
    
    # 1. FC Köln
    "fc köln": "1. FC Köln",
    "1. fc koln": "1. FC Köln",
    "1. fc köln": "1. FC Köln",
    "1.fc köln": "1. FC Köln",
    "1.fc koln": "1. FC Köln",
    "fc koln": "1. FC Köln",
    "1. fc cologne": "1. FC Köln",
    "cologne": "1. FC Köln",
    "koeln": "1. FC Köln",
    "köln": "1. FC Köln",
    
    # Darmstadt 98
    "sv darmstadt 98": "Darmstadt",
    "darmstadt 98": "Darmstadt",
    "darmstadt": "Darmstadt",
    
    # FC St. Pauli
    "fc st. pauli": "FC St. Pauli",
    "st. pauli": "FC St. Pauli",
    "sankt pauli": "FC St. Pauli",
    
    # Holstein Kiel
    "holstein kiel": "Holstein Kiel",
    "kiel": "Holstein Kiel",
    
    # Hamburger SV
    "hamburger sv": "Hamburger SV",
    "hamburg": "Hamburger SV",
    
    # Diğer / eski Bundesliga takımları ve alternatif yazımlar
    "schalke 04": "Schalke 04",
    "hannover 96": "Hannover 96",
    "hannover": "Hannover 96",
    "energie cottbus": "Energie Cottbus",
    "cottbus": "Energie Cottbus",
    "fc nürnberg": "1. FC Nürnberg",
    "nürnberg": "1. FC Nürnberg",
    "nuremberg": "1. FC Nürnberg",
    "1. fc nürnberg": "1. FC Nürnberg",
    "1. fc nuremberg": "1. FC Nürnberg",
    "fc nuremberg": "1. FC Nürnberg",
    "greuther fürth": "Greuther Fürth",
    "greuther furth": "Greuther Fürth",
    "fürth": "Greuther Fürth",
    "furth": "Greuther Fürth",
    "arminia bielefeld": "Arminia Bielefeld",
    "bielefeld": "Arminia Bielefeld",
    "vfl osnabrück": "VfL Osnabrück",
    "osnabrück": "VfL Osnabrück",
    "osnabruck": "VfL Osnabrück",
    "vfl osnabruck": "VfL Osnabrück",
    "fc würzburger kickers": "FC Würzburger Kickers",
    "würzburger kickers": "FC Würzburger Kickers",
    "wurzburger kickers": "FC Würzburger Kickers",
    "fc wurzburger kickers": "FC Würzburger Kickers",
    "würzburg": "FC Würzburger Kickers",
    "wurzburg": "FC Würzburger Kickers",
    "fc ingolstadt 04": "FC Ingolstadt 04",
    "ingolstadt": "FC Ingolstadt 04",
    "fc ingolstadt": "FC Ingolstadt 04",
    "fc erzgebirge aue": "FC Erzgebirge Aue",
    "erzgebirge aue": "FC Erzgebirge Aue",
    "aue": "FC Erzgebirge Aue",
}


# Özel oyuncu eşleştirme için manuel mapping
MANUAL_PLAYER_MAPPING = {
    'marcandre ter stegen': 'm ter stegen',
    'josip stanisic': 'josip stanišić',
    'eric maxim choupo moting': 'choupo moting',
    'jakub kaminski': 'jakub kamiński',
    'alejandro grimaldo': 'alex grimaldo',       # eklendi
    'grant-leon ranos': 'grant ranos'           # eklendi
}



# --- Yardımcı Fonksiyonlar ---
def normalize_text(s: Optional[str]) -> str:
    """Genel amaçlı normalize fonksiyonu: hem takım hem oyuncu için kullan."""
    if pd.isna(s):
        return ""
    s = str(s)
    s = s.strip().lower()
    # Normalize unicode
    s = unicodedata.normalize('NFKD', s)
    s = ''.join(ch for ch in s if not unicodedata.combining(ch))
    # Parantez içeriğini kaldır (ör. (loan))
    s = re.sub(r"\(.*?\)", "", s)
    # Noktalama ve özel karakterleri temizle, ama rakam ve harf bırak
    s = re.sub(r"[^a-z0-9\s]", "", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def normalize_player_name(name: Optional[str]) -> str:
    """Oyuncu isimleri için gelişmiş normalizasyon"""
    if pd.isna(name):
        return ""
    
    name = str(name).strip().lower()
    
    # Unicode normalizasyonu
    name = unicodedata.normalize("NFKD", name)
    name = ''.join([c for c in name if not unicodedata.combining(c)])
    
    # Özel karakterleri kaldır
    name = re.sub(r'[^a-z\s]', '', name)
    
    # Fazla boşlukları temizle
    name = re.sub(r'\s+', ' ', name).strip()
    
    # Kısaltmaları standartlaştır
    name = re.sub(r'\bmg\b', '', name)  # "mg" gibi ekleri kaldır
    
    return name

def extract_surname(full_name: str) -> str:
    """Soyadı çıkar (son kelimeyi al)"""
    parts = full_name.split()
    if len(parts) > 0:
        return parts[-1]
    return full_name

def detect_columns(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    cols = list(df.columns)
    col_map = {c.lower(): c for c in cols}
    for cand in candidates:
        if cand.lower() in col_map:
            return col_map[cand.lower()]
    # Fallback: isim içinde anahtar kelime arama
    for c in cols:
        cl = c.lower()
        if any(k in cl for k in ['team','club','squad']):
            return c
    return None

# Kısa candidate listeleri
TEAM_CANDIDATES = ['team','club','squad','team_name','club_name']
PLAYER_CANDIDATES = ['player','player_name','name']

# --- Veri yükleme ---
def read_squads(path: str) -> pd.DataFrame:
    # Excel'de birden fazla sheet varsa hepsini birleştir
    xls = pd.ExcelFile(path)
    frames = []
    for s in xls.sheet_names:
        try:
            df = pd.read_excel(path, sheet_name=s)
            df['__source_sheet'] = s
            frames.append(df)
        except Exception:
            continue
    if not frames:
        raise FileNotFoundError(f"Squads dosyasi okunamadi: {path}")
    return pd.concat(frames, ignore_index=True)

def load_data(squads_path: str, fbref_path: str):
    print('📂 Veriler yükleniyor...')
    squads = read_squads(squads_path)
    try:
        fbref = pd.read_csv(fbref_path)
    except UnicodeDecodeError:
        fbref = pd.read_csv(fbref_path, encoding='latin-1')
    print(f' - Squads shape: {squads.shape}')
    print(f' - FBref  shape: {fbref.shape}')
    # Trim column names
    squads.columns = squads.columns.str.strip()
    fbref.columns = fbref.columns.str.strip()
    return squads, fbref

# --- Main entegre pipeline ---
def integrated_pipeline():
    squads_df, fbref_df = load_data(SQUADS_PATH, FBREF_PATH)

    # Kolon tespiti
    team_col_squads = detect_columns(squads_df, TEAM_CANDIDATES)
    team_col_fbref  = detect_columns(fbref_df, TEAM_CANDIDATES)
    player_col_squads = detect_columns(squads_df, PLAYER_CANDIDATES)
    player_col_fbref  = detect_columns(fbref_df, PLAYER_CANDIDATES)

    if not team_col_squads or not team_col_fbref:
        raise ValueError('Takim kolonu tespit edilemedi.')

    print(f'🔎 Bulunan kolonlar -> Squads team: {team_col_squads}, FBref team: {team_col_fbref}')
    print(f'🔎 Bulunan oyuncu kolonlari -> Squads player: {player_col_squads}, FBref player: {player_col_fbref}')

    # Normalize team names & player names
    squads_df['team_norm'] = squads_df[team_col_squads].apply(normalize_text)
    fbref_df['team_norm']  = fbref_df[team_col_fbref].apply(normalize_text)

    if player_col_squads:
        squads_df['player_norm'] = squads_df[player_col_squads].apply(normalize_player_name)
        squads_df['player_surname'] = squads_df['player_norm'].apply(extract_surname)
    else:
        squads_df['player_norm'] = ''
        squads_df['player_surname'] = ''

    if player_col_fbref:
        fbref_df['player_norm'] = fbref_df[player_col_fbref].apply(normalize_player_name)
        fbref_df['player_surname'] = fbref_df['player_norm'].apply(extract_surname)
    else:
        fbref_df['player_norm'] = ''
        fbref_df['player_surname'] = ''

    # Normalize manual mapping dictionary
    complete_map = {}
    for fbk, sqv in COMPLETE_TEAM_MAPPING_RAW.items():
        k = normalize_text(fbk)
        v = normalize_text(sqv)
        complete_map[k] = v
    # Reverse mapping squads->fbref (normalized)
    reverse_map = {v: k for k, v in complete_map.items()}

    # Oyuncu manuel mapping'i normalize et
    manual_player_map = {}
    for fbk, sqv in MANUAL_PLAYER_MAPPING.items():
        k = normalize_player_name(fbk)
        v = normalize_player_name(sqv)
        manual_player_map[k] = v

    # Unique team lists
    squads_teams = sorted(set(squads_df['team_norm'].dropna().unique()))
    fbref_teams  = sorted(set(fbref_df['team_norm'].dropna().unique()))

    # Exact matches
    exact = sorted(set(squads_teams) & set(fbref_teams))

    # Build mapping using exact + manual + fuzzy
    mapping = {}
    suggestions = []

    # 1) exact matches -> map to same
    for t in exact:
        mapping[t] = t

    # 2) manual map (normalized keys)
    for fb_norm, sq_norm in complete_map.items():
        # If fb_norm exists in fbref_teams and sq_norm in squads_teams -> use
        if fb_norm in fbref_teams and sq_norm in squads_teams:
            mapping[sq_norm] = fb_norm
        else:
            # still add suggestion for review
            suggestions.append({'squad_norm': sq_norm, 'fbref_norm': fb_norm, 'reason': 'manual_map'})

    # 3) fuzzy for squads teams not yet mapped
    unmapped_squads = [s for s in squads_teams if s not in mapping]
    for s in unmapped_squads:
        cand, score = extract_one(s, fbref_teams, scorer=None)
        if cand is None:
            suggestions.append({'squad_norm': s, 'fbref_norm': '', 'score': 0})
            continue
        score = float(score)
        accepted = False
        if score >= TEAM_FUZZY_HIGH:
            accepted = True
        elif score >= TEAM_FUZZY_MED:
            # ekstra kontrol: ilk token eşleşiyorsa kabul et
            if s.split()[0] == cand.split()[0]:
                accepted = True
        elif score >= TEAM_FUZZY_LOW:
            # düşük eşik: yalnızca öneri, kabul değil
            accepted = False
        if accepted:
            mapping[s] = cand
        suggestions.append({'squad_norm': s, 'fbref_norm': cand, 'score': round(score,2), 'accepted': accepted})

    # Save suggestions & matched teams
    sug_df = pd.DataFrame(suggestions).fillna('')
    sug_df.to_csv(TEAM_SUGGEST_CSV, index=False)

    matched_pairs = [{'squad_norm': k, 'fbref_norm': v} for k, v in mapping.items()]
    pd.DataFrame(matched_pairs).to_csv(MATCHED_TEAMS_CSV, index=False)

    unmatched_squads = sorted([s for s in squads_teams if s not in mapping])
    unmatched_fbref  = sorted([f for f in fbref_teams if f not in set(mapping.values())])
    pd.DataFrame({'team_norm': unmatched_squads}).to_csv(UNMATCHED_SQUADS_CSV, index=False)
    pd.DataFrame({'team_norm': unmatched_fbref}).to_csv(UNMATCHED_FBREF_CSV, index=False)

    # --- Merge oyuncu bazlı ---
    # İlk olarak squads_df üzerinde takım bazlı fbref team adı ekle (normalized)
    squads_df['fbref_team_norm'] = squads_df['team_norm'].map(mapping)

    # Prepare a lookup dict: fbref team -> list of player_norms and full rows
    fb_team_players = {}
    for team in fbref_teams:
        team_rows = fbref_df[fbref_df['team_norm'] == team]
        fb_team_players[team] = {
            'player_norms': team_rows['player_norm'].dropna().unique().tolist(),
            'player_surnames': team_rows['player_surname'].dropna().unique().tolist(),
            'rows': team_rows
        }

    # copy fbref columns to merged frame
    merged = squads_df.copy()
    fbref_cols = [c for c in fbref_df.columns if c not in [player_col_fbref, 'player_norm', 'player_surname']]
    for c in fbref_cols:
        merged[f'fbref__{c}'] = pd.NA

    # Player mapping raporu için
    player_mapping_report = []

    # First pass: exact player_norm match
    if player_col_fbref:
        for idx, row in merged.iterrows():
            team_norm = row['fbref_team_norm']
            if not team_norm:
                continue
            player_norm = row['player_norm'] if 'player_norm' in row and pd.notna(row['player_norm']) else ''
            if not player_norm:
                continue
            
            # Manuel mapping kontrolü
            if player_norm in manual_player_map:
                mapped_norm = manual_player_map[player_norm]
                fb_players = fb_team_players.get(team_norm, {}).get('player_norms', [])
                if mapped_norm in fb_players:
                    fr = fb_team_players[team_norm]['rows']
                    fr_match = fr[fr['player_norm'] == mapped_norm].iloc[0]
                    for c in fbref_df.columns:
                        merged.at[idx, f'fbref__{c}'] = fr_match.get(c)
                    player_mapping_report.append({
                        'squad_player': row.get(player_col_squads, ''),
                        'squad_player_norm': player_norm,
                        'fbref_player': fr_match.get(player_col_fbref, ''),
                        'fbref_player_norm': mapped_norm,
                        'score': 100.0,
                        'method': 'manual_mapping',
                        'status': 'matched'
                    })
                    continue
            
            fb_players = fb_team_players.get(team_norm, {}).get('player_norms', [])
            if player_norm in fb_players:
                # get first matching fbref row
                fr = fb_team_players[team_norm]['rows']
                fr_match = fr[fr['player_norm'] == player_norm].iloc[0]
                for c in fbref_df.columns:
                    merged.at[idx, f'fbref__{c}'] = fr_match.get(c)
                player_mapping_report.append({
                    'squad_player': row.get(player_col_squads, ''),
                    'squad_player_norm': player_norm,
                    'fbref_player': fr_match.get(player_col_fbref, ''),
                    'fbref_player_norm': player_norm,
                    'score': 100.0,
                    'method': 'exact_match',
                    'status': 'matched'
                })

    # Second pass: fuzzy matching for still-unmatched
    still_unmatched = merged[merged[f'fbref__{fbref_df.columns[0]}'].isna()].index.tolist() if len(fbref_df.columns)>0 else []
    
    for idx in still_unmatched:
        row = merged.loc[idx]
        team_norm = row['fbref_team_norm']
        player_norm = row['player_norm'] if 'player_norm' in row and pd.notna(row['player_norm']) else ''
        player_surname = row['player_surname'] if 'player_surname' in row and pd.notna(row['player_surname']) else ''
        
        if not team_norm or not player_norm:
            continue
        
        fb_players = fb_team_players.get(team_norm, {}).get('player_norms', [])
        if not fb_players:
            continue
        
        # Çoklu fuzzy yöntemleri ile eşleştirme
        cand, score, method = best_player_match(player_norm, fb_players)
        
        if cand and score >= PLAYER_FUZZY_HIGH:
            # Yüksek skorlu eşleşme - kabul et
            fr = fb_team_players[team_norm]['rows']
            fr_match = fr[fr['player_norm'] == cand].iloc[0]
            for c in fbref_df.columns:
                merged.at[idx, f'fbref__{c}'] = fr_match.get(c)
            
            player_mapping_report.append({
                'squad_player': row.get(player_col_squads, ''),
                'squad_player_norm': player_norm,
                'fbref_player': fr_match.get(player_col_fbref, ''),
                'fbref_player_norm': cand,
                'score': round(score, 2),
                'method': f'fuzzy_{method}',
                'status': 'matched'
            })
            continue
        
        # Soyad-only eşleştirme (düşük skorlu durumlarda)
        if score < PLAYER_SURNAME_ONLY_THRESHOLD and player_surname:
            fb_surnames = fb_team_players.get(team_norm, {}).get('player_surnames', [])
            if player_surname in fb_surnames:
                fr = fb_team_players[team_norm]['rows']
                fr_match = fr[fr['player_surname'] == player_surname].iloc[0]
                for c in fbref_df.columns:
                    merged.at[idx, f'fbref__{c}'] = fr_match.get(c)
                
                player_mapping_report.append({
                    'squad_player': row.get(player_col_squads, ''),
                    'squad_player_norm': player_norm,
                    'fbref_player': fr_match.get(player_col_fbref, ''),
                    'fbref_player_norm': fr_match.get('player_norm', ''),
                    'score': 100.0,  # Soyad eşleşmesi tam kabul
                    'method': 'surname_match',
                    'status': 'matched'
                })
                continue
        
        # Manuel inceleme için öneri
        if cand and score >= PLAYER_FUZZY_MED:
            player_mapping_report.append({
                'squad_player': row.get(player_col_squads, ''),
                'squad_player_norm': player_norm,
                'fbref_player': '',  # Bulunamadı
                'fbref_player_norm': cand,
                'score': round(score, 2),
                'method': f'fuzzy_{method}',
                'status': 'manual_review'
            })
        else:
            player_mapping_report.append({
                'squad_player': row.get(player_col_squads, ''),
                'squad_player_norm': player_norm,
                'fbref_player': '',  # Bulunamadı
                'fbref_player_norm': '',
                'score': round(score, 2) if cand else 0.0,
                'method': 'no_match' if not cand else f'fuzzy_{method}',
                'status': 'unmatched'
            })

    # Player mapping raporunu kaydet
    player_report_df = pd.DataFrame(player_mapping_report)
    player_report_df.to_csv(PLAYER_MAPPING_SUGGEST_CSV, index=False)

    # Count matched players
    fbref_primary_col = fbref_df.columns[0] if len(fbref_df.columns)>0 else None
    if fbref_primary_col is not None:
        merged['matched'] = merged[f'fbref__' + fbref_primary_col].notna()
        total_players = len(merged)
        matched_players = merged['matched'].sum()
        match_pct = round(100.0 * matched_players / max(total_players,1),2)
    else:
        total_players = len(merged)
        matched_players = 0
        match_pct = 0.0

    # Player report per team
    player_report = []
    for squad_team_norm in sorted(set(merged['team_norm'].dropna().unique())):
        team_rows = merged[merged['team_norm'] == squad_team_norm]
        total = len(team_rows)
        matched = team_rows['matched'].sum() if 'matched' in team_rows else 0
        pct = round(100.0 * matched / max(total,1),2)
        player_report.append({'team_norm': squad_team_norm, 'total_players': total, 'matched_players': int(matched), 'match_pct': pct})
    pr_df = pd.DataFrame(player_report).sort_values('match_pct', ascending=False)
    pr_df.to_csv(PLAYER_REPORT_CSV, index=False)

    # Fill numeric columns from fbref (if any) and impute by team mean / league mean like original
    # Detect numeric columns in fbref
    numeric_candidates = ['MP','Gls','Ast','Min','Sh','SoT','Cmp','Att']
    numeric_cols_present = [c for c in numeric_candidates if c in fbref_df.columns]
    for col in numeric_cols_present:
        merged[f'fbref__{col}'] = pd.to_numeric(merged[f'fbref__{col}'], errors='coerce')

    # Team-level imputation
    for col in numeric_cols_present:
        for team in merged['team_norm'].dropna().unique():
            mask = merged['team_norm'] == team
            team_mean = merged.loc[mask, f'fbref__{col}'].mean()
            if pd.notna(team_mean):
                merged.loc[mask & merged[f'fbref__{col}'].isna(), f'fbref__{col}'] = team_mean
    # League mean
    for col in numeric_cols_present:
        if merged[f'fbref__{col}'].isna().any():
            lg_mean = merged[f'fbref__{col}'].mean()
            if pd.notna(lg_mean):
                merged[f'fbref__{col}'] = merged[f'fbref__{col}'].fillna(lg_mean)
            else:
                merged[f'fbref__{col}'] = merged[f'fbref__{col}'].fillna(0)

    # Rating calculation (like original)
    from sklearn.preprocessing import MinMaxScaler
    rating_metrics = [c for c in ['MP','Gls','Ast','Sh','SoT'] if c in numeric_cols_present]
    if rating_metrics:
        for c in rating_metrics:
            merged[f'{c}_norm'] = merged[f'fbref__{c}'] / max(merged[f'fbref__{c}'].max(), 1) * 100
        weights = [0.25, 0.30, 0.20, 0.15, 0.10]
        merged['Rating_raw'] = 0.0
        for i, c in enumerate(rating_metrics):
            w = weights[i] if i < len(weights) else 0
            merged['Rating_raw'] += w * merged[f'{c}_norm']
        scaler = MinMaxScaler(feature_range=(0,100))
        merged['Rating'] = scaler.fit_transform(merged[['Rating_raw']])

    # Save final merged dataset
    out_path = 'data/final_bundesliga_dataset_complete.xlsx'
    merged.to_excel(out_path, index=False)

    # Write human readable log
    lines = []
    lines.append('=== Bundesliga Mapping & Merge Report ===')
    lines.append(f'Timestamp: {STAMP}')
    lines.append('')
    lines.append('[Summary]')
    lines.append(f'- Fuzzy library used: {FUZZY_LIB}')
    lines.append(f'- Squads unique teams: {len(squads_teams)}')
    lines.append(f'- FBref unique teams: {len(fbref_teams)}')
    lines.append(f'- Exact team matches: {len(exact)}')
    lines.append(f'- Mapped teams (after manual+fuzzy): {len(mapping)}')
    lines.append(f'- Unmatched squads teams: {len(unmatched_squads)}')
    lines.append(f'- Unmatched fbref teams: {len(unmatched_fbref)}')
    lines.append(f'- Total players (squads rows): {total_players}')
    lines.append(f'- Matched players: {int(matched_players)} ({match_pct} %)')
    lines.append('')
    
    # Player matching istatistikleri
    if len(player_mapping_report) > 0:
        report_df = pd.DataFrame(player_mapping_report)
        matched_count = len(report_df[report_df['status'] == 'matched'])
        manual_review_count = len(report_df[report_df['status'] == 'manual_review'])
        unmatched_count = len(report_df[report_df['status'] == 'unmatched'])
        
        lines.append('[Player Matching Details]')
        lines.append(f'- Matched players: {matched_count}')
        lines.append(f'- Needs manual review: {manual_review_count}')
        lines.append(f'- Unmatched players: {unmatched_count}')
        lines.append(f'- Match methods:')
        method_counts = report_df['method'].value_counts()
        for method, count in method_counts.items():
            lines.append(f'  * {method}: {count}')
    
    lines.append('')
    lines.append('[Files written]')
    lines.append(f'- Matched teams: {MATCHED_TEAMS_CSV}')
    lines.append(f'- Team suggestions: {TEAM_SUGGEST_CSV}')
    lines.append(f'- Unmatched squads: {UNMATCHED_SQUADS_CSV}')
    lines.append(f'- Unmatched fbref: {UNMATCHED_FBREF_CSV}')
    lines.append(f'- Player report: {PLAYER_REPORT_CSV}')
    lines.append(f'- Player mapping suggestions: {PLAYER_MAPPING_SUGGEST_CSV}')
    lines.append(f'- Final merged dataset: {out_path}')

    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print('\n'.join(lines))
    print('\nDone. Outputs saved to logs/ and data/')

if __name__ == '__main__':
    integrated_pipeline()