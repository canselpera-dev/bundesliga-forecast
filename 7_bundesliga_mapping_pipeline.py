#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bundesliga Mapping & Merge — Enhanced Player Matching Version with Dynamic Ratings
==================================================================================

Geliştirmeler:
1. Dinamik rating sistemi - veri güncellendikçe otomatik yeniden hesaplama
2. Pozisyon bazlı ağırlıklandırma
3. Form ve süreklilik analizi
4. Gelişmiş metrik normalizasyonu
5. Mevcut pipeline ile tam uyumluluk
"""

import os
import json
import pandas as pd
import unicodedata
import re
import numpy as np
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Any
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression

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
RATING_CONFIG_PATH = os.path.join(OUTPUT_DIR, 'rating_config.json')

# Çıktı dosyası - MEVCUT PIPELINE İLE UYUMLU
OUTPUT_FILE = "data/final_bundesliga_dataset_complete.xlsx"

# Eşikler (DÜŞÜRÜLMÜŞ DEĞERLER)
TEAM_FUZZY_HIGH = 75.0
TEAM_FUZZY_MED  = 65.0
TEAM_FUZZY_LOW  = 55.0
PLAYER_FUZZY_HIGH = 85.0
PLAYER_FUZZY_MED  = 75.0
PLAYER_FUZZY_LOW  = 65.0
PLAYER_SURNAME_ONLY_THRESHOLD = 70.0

# Rating Konfigürasyonu
DEFAULT_RATING_CONFIG = {
    'base_metrics': {
        'MP': {'weight': 0.10, 'normalization': 'max', 'position_weights': {'GK': 0.15, 'DEF': 0.12, 'MID': 0.08, 'FWD': 0.05}},
        'Gls': {'weight': 0.20, 'normalization': 'max', 'position_weights': {'GK': 0.05, 'DEF': 0.10, 'MID': 0.15, 'FWD': 0.25}},
        'Ast': {'weight': 0.15, 'normalization': 'max', 'position_weights': {'GK': 0.05, 'DEF': 0.08, 'MID': 0.18, 'FWD': 0.15}},
        'Sh': {'weight': 0.08, 'normalization': 'max', 'position_weights': {'GK': 0.02, 'DEF': 0.05, 'MID': 0.08, 'FWD': 0.12}},
        'SoT': {'weight': 0.07, 'normalization': 'max', 'position_weights': {'GK': 0.02, 'DEF': 0.04, 'MID': 0.06, 'FWD': 0.10}},
        'Cmp': {'weight': 0.06, 'normalization': 'percentile', 'position_weights': {'GK': 0.10, 'DEF': 0.08, 'MID': 0.07, 'FWD': 0.03}},
        'PrgC': {'weight': 0.08, 'normalization': 'percentile', 'position_weights': {'GK': 0.05, 'DEF': 0.10, 'MID': 0.09, 'FWD': 0.06}},
        'PrgP': {'weight': 0.08, 'normalization': 'percentile', 'position_weights': {'GK': 0.03, 'DEF': 0.06, 'MID': 0.10, 'FWD': 0.08}},
        'PrgR': {'weight': 0.06, 'normalization': 'percentile', 'position_weights': {'GK': 0.02, 'DEF': 0.05, 'MID': 0.07, 'FWD': 0.08}},
        'Min': {'weight': 0.05, 'normalization': 'max', 'position_weights': {'GK': 0.08, 'DEF': 0.06, 'MID': 0.05, 'FWD': 0.04}},
        'G+A': {'weight': 0.07, 'normalization': 'max', 'position_weights': {'GK': 0.03, 'DEF': 0.06, 'MID': 0.08, 'FWD': 0.12}}
    },
    'advanced_metrics': {
        'efficiency_metrics': ['Gls/Sh', 'Ast/90', 'G+A/90'],
        'consistency_bonus': 0.1,
        'form_weight': 0.3,
        'age_weight': 0.05,
        'market_value_weight': 0.05
    },
    'position_detection': {
        'GK_keywords': ['goalkeeper', 'keeper', 'gk'],
        'DEF_keywords': ['defender', 'defence', 'back', 'cb', 'rb', 'lb'],
        'MID_keywords': ['midfielder', 'midfield', 'cm', 'cam', 'cdm'],
        'FWD_keywords': ['forward', 'striker', 'attacker', 'winger', 'fw', 'cf']
    },
    'scaling': {
        'min_rating': 0,
        'max_rating': 100,
        'use_percentile': True,
        'percentile_range': (10, 90)
    }
}

# Manuel mapping (FBref -> Squads) GENİŞLETİLMİŞ VERSİYON
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
    "borussia dortmund": "Borussia Dortmund",
    
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
    "eint frankfurt": "Eintracht Frankfurt",
    
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
    "b mgladbach": "Borussia M'gladbach",
    "borussia mg": "Borussia M'gladbach",
    
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
    "fc heidenheim": "Heidenheim",
    
    # 1. FC Union Berlin
    "1. fc union berlin": "Union Berlin",
    "union berlin": "Union Berlin",
    "union ber": "Union Berlin",
    
    # Mainz 05
    "1. fsv mainz 05": "Mainz 05",
    "mainz 05": "Mainz 05",
    "mainz": "Mainz 05",
    "fsv mainz": "Mainz 05",
    
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
    "fc koeln": "1. FC Köln",
    
    # Darmstadt 98
    "sv darmstadt 98": "Darmstadt",
    "darmstadt 98": "Darmstadt",
    "darmstadt": "Darmstadt",
    
    # FC St. Pauli
    "fc st. pauli": "FC St. Pauli",
    "st. pauli": "FC St. Pauli",
    "sankt pauli": "FC St. Pauli",
    "st pauli": "FC St. Pauli",
    
    # Holstein Kiel
    "holstein kiel": "Holstein Kiel",
    "kiel": "Holstein Kiel",
    
    # Hamburger SV
    "hamburger sv": "Hamburger SV",
    "hamburg": "Hamburger SV",
    "hamburger": "Hamburger SV",
    
    # Schalke 04
    "schalke 04": "Schalke 04",
    "schalke": "Schalke 04",
    
    # Hannover 96
    "hannover 96": "Hannover 96",
    "hannover": "Hannover 96",
}

# Özel oyuncu eşleştirme için genişletilmiş manuel mapping
MANUAL_PLAYER_MAPPING = {
    'marcandre ter stegen': 'm ter stegen',
    'josip stanisic': 'josip stanišić',
    'eric maxim choupo moting': 'choupo moting',
    'jakub kaminski': 'jakub kamiński',
    'alejandro grimaldo': 'alex grimaldo',
    'grant-leon ranos': 'grant ranos',
    'niklas sule': 'niklas süle',
    'thomas muller': 'thomas müller',
    'leroy sane': 'leroy sané',
    'manuel neuer': 'manuel neuer',
    'matthijs de ligt': 'matthijs de ligt',
    'sadio mane': 'sadio mané',
    'yann sommer': 'yann sommer',
    'peter gulacsi': 'péter gulácsi',
    'lukas hradecky': 'lukáš hrádecký',
    'andre silva': 'andré silva',
    'pavel kaderabek': 'pavel kadeřábek',
    'christian gunter': 'christian günter',
    'niclas fullkrug': 'niclas füllkrug',
    'florian neuhaus': 'florian neuhaus',
    'rami bensebaini': 'rami bensebaini',
    'frederik ronnow': 'frederik rønnow',
    'aaron martin': 'aaron martín',
    'tom koubek': 'tomáš koubek',
    'rafal gikiewicz': 'rafał gikiewicz',
    'salih ozcan': 'salih özcan',
    'angelino': 'ángelino',
    'tomas hrozensky': 'tomáš hrozenský',
    'mario vuskovic': 'mario vušković',
}

# --- Dinamik Rating Sistemi ---
class DynamicRatingSystem:
    def __init__(self, config=None):
        self.config = config or DEFAULT_RATING_CONFIG
        self.rating_history = {}
        self.league_averages = {}
        self.position_stats = {}
        
    def save_config(self, path=None):
        """Rating konfigürasyonunu kaydet"""
        path = path or RATING_CONFIG_PATH
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.config, f, indent=2, ensure_ascii=False)
    
    def load_config(self, path=None):
        """Rating konfigürasyonunu yükle"""
        path = path or RATING_CONFIG_PATH
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.config = json.load(f)
        except FileNotFoundError:
            self.config = DEFAULT_RATING_CONFIG
            self.save_config(path)
    
    def detect_position(self, position_str, player_name=None):
        """Oyuncu pozisyonunu tespit et"""
        if pd.isna(position_str) or position_str == '':
            return 'UNK'
        
        position_str = str(position_str).lower()
        
        # Pozisyon anahtar kelimelerine göre tespit
        for pos, keywords in self.config['position_detection'].items():
            if any(keyword in position_str for keyword in keywords):
                return pos.replace('_keywords', '')
        
        # Fallback: pozisyon kısaltmaları
        if any(abbr in position_str for abbr in ['gk', 'goalie']):
            return 'GK'
        elif any(abbr in position_str for abbr in ['cb', 'rb', 'lb', 'wb', 'def']):
            return 'DEF'
        elif any(abbr in position_str for abbr in ['cm', 'cam', 'cdm', 'lm', 'rm', 'mid']):
            return 'MID'
        elif any(abbr in position_str for abbr in ['fw', 'cf', 'st', 'lw', 'rw', 'att']):
            return 'FWD'
        
        return 'UNK'
    
    def calculate_league_averages(self, df):
        """Lig ortalamalarını hesapla"""
        # Sadece sayısal sütunları seç
        numeric_cols = [f'fbref__{metric}' for metric in self.config['base_metrics'].keys() 
                       if f'fbref__{metric}' in df.columns]
        
        for col in numeric_cols:
            # Sütunu sayısala dönüştür
            df[col] = pd.to_numeric(df[col], errors='coerce')
            self.league_averages[col] = df[col].mean()
        
        # Pozisyon bazlı istatistikler
        for position in ['GK', 'DEF', 'MID', 'FWD']:
            pos_mask = df['position_detected'] == position
            if pos_mask.any():
                self.position_stats[position] = {}
                for col in numeric_cols:
                    self.position_stats[position][col] = {
                        'mean': df.loc[pos_mask, col].mean(),
                        'std': df.loc[pos_mask, col].std(),
                        'max': df.loc[pos_mask, col].max()
                    }
    
    def normalize_metric(self, values, method='max', position=None):
        """Metrikleri normalize et"""
        if method == 'max':
            max_val = values.max()
            return values / max_val * 100 if max_val > 0 else values * 0
        elif method == 'percentile':
            if self.config['scaling']['use_percentile']:
                low, high = self.config['scaling']['percentile_range']
                low_p = np.percentile(values, low)
                high_p = np.percentile(values, high)
                if high_p > low_p:
                    normalized = (values - low_p) / (high_p - low_p) * 100
                    return np.clip(normalized, 0, 100)
            return (values - values.min()) / (values.max() - values.min()) * 100 if values.max() > values.min() else values * 0
        elif method == 'zscore' and position and position in self.position_stats:
            # Pozisyon bazlı z-score normalizasyonu
            pos_stats = self.position_stats[position]
            col_name = values.name.replace('_norm', '')
            if col_name in pos_stats:
                mean = pos_stats[col_name]['mean']
                std = pos_stats[col_name]['std']
                if std > 0:
                    z_scores = (values - mean) / std
                    return np.clip(50 + z_scores * 10, 0, 100)
        return values
    
    def calculate_advanced_metrics(self, df):
        """Gelişmiş metrikleri hesapla"""
        # Gol + Asist
        if 'fbref__Gls' in df.columns and 'fbref__Ast' in df.columns:
            df['G+A'] = df['fbref__Gls'] + df['fbref__Ast']
        
        # Verimlilik metrikleri
        if 'fbref__Gls' in df.columns and 'fbref__Sh' in df.columns:
            df['Gls/Sh'] = df['fbref__Gls'] / df['fbref__Sh']
            df['Gls/Sh'] = df['Gls/Sh'].fillna(0)
        
        if 'fbref__Ast' in df.columns and 'fbref__Min' in df.columns:
            df['Ast/90'] = (df['fbref__Ast'] / df['fbref__Min']) * 90
            df['Ast/90'] = df['Ast/90'].fillna(0)
        
        if 'G+A' in df.columns and 'fbref__Min' in df.columns:
            df['G+A/90'] = (df['G+A'] / df['fbref__Min']) * 90
            df['G+A/90'] = df['G+A/90'].fillna(0)
        
        return df
    
    def calculate_player_rating(self, df):
        """Tüm oyuncular için dinamik rating hesapla"""
        # Pozisyonları tespit et
        df['position_detected'] = df['Position'].apply(self.detect_position)
        
        # Gelişmiş metrikleri hesapla
        df = self.calculate_advanced_metrics(df)
        
        # FBref sütunlarını sayısal forma dönüştür
        for metric in self.config['base_metrics'].keys():
            col_name = f'fbref__{metric}'
            if col_name in df.columns:
                df[col_name] = pd.to_numeric(df[col_name], errors='coerce')
        
        # Lig ortalamalarını güncelle
        self.calculate_league_averages(df)
        
        # Rating hesaplama
        df['Rating_raw'] = 0.0
        total_weight = 0
        
        for metric, config in self.config['base_metrics'].items():
            col_name = f'fbref__{metric}'
            if col_name not in df.columns:
                continue
            
            # Metrik için normalize sütun oluştur
            norm_col = f'{metric}_norm'
            
            for position in ['GK', 'DEF', 'MID', 'FWD', 'UNK']:
                pos_mask = df['position_detected'] == position
                if pos_mask.any():
                    # Pozisyon bazlı ağırlık
                    pos_weight = config['position_weights'].get(position, config['weight'])
                    
                    # Normalizasyon
                    df.loc[pos_mask, norm_col] = self.normalize_metric(
                        df.loc[pos_mask, col_name], 
                        config['normalization'],
                        position
                    )
                    
                    # Rating'e katkı
                    df.loc[pos_mask, 'Rating_raw'] += pos_weight * df.loc[pos_mask, norm_col]
                    total_weight += pos_weight
        
        # Gelişmiş metrikler için ek puanlar
        advanced_config = self.config['advanced_metrics']
        
        # Form puanı (son 5 maç performansı)
        if 'fbref__MP' in df.columns:
            recent_form = df['fbref__MP'] / df['fbref__MP'].max() * 100
            df['Rating_raw'] += advanced_config['form_weight'] * recent_form
            total_weight += advanced_config['form_weight']
        
        # Yaş faktörü
        if 'Age' in df.columns:
            # 24-28 yaş arası prime dönem bonusu
            age_bonus = np.where(
                (df['Age'] >= 24) & (df['Age'] <= 28),
                10,  # Prime bonus
                np.where(df['Age'] < 21, 5, 0)  # Genç yetenek bonusu
            )
            df['Rating_raw'] += advanced_config['age_weight'] * age_bonus
            total_weight += advanced_config['age_weight']
        
        # Piyasa değeri faktörü
        if 'Market_Value' in df.columns:
            market_value_norm = self.normalize_metric(df['Market_Value'], 'percentile')
            df['Rating_raw'] += advanced_config['market_value_weight'] * market_value_norm
            total_weight += advanced_config['market_value_weight']
        
        # Süreklilik bonusu
        df['Rating_raw'] += advanced_config['consistency_bonus'] * 100
        total_weight += advanced_config['consistency_bonus']
        
        # Final rating hesaplama
        if total_weight > 0:
            df['Rating_raw'] = df['Rating_raw'] / total_weight
        
        # Min-Max scaling
        scaler = MinMaxScaler(feature_range=(
            self.config['scaling']['min_rating'],
            self.config['scaling']['max_rating']
        ))
        df['Rating'] = scaler.fit_transform(df[['Rating_raw']])
        
        # Rating geçmişini güncelle
        self.update_rating_history(df)
        
        return df
    
    def update_rating_history(self, df):
        """Rating geçmişini güncelle"""
        current_date = datetime.now()
        
        for idx, row in df.iterrows():
            player_key = f"{row['player_norm']}_{row['fbref_team_norm']}"
            
            if player_key not in self.rating_history:
                self.rating_history[player_key] = []
            
            self.rating_history[player_key].append({
                'date': current_date,
                'rating': row['Rating'],
                'position': row['position_detected'],
                'team': row['Team']
            })
            
            # Eski kayıtları temizle (30 günden eski)
            self.rating_history[player_key] = [
                record for record in self.rating_history[player_key]
                if current_date - record['date'] <= timedelta(days=30)
            ]
    
    def get_rating_trend(self, player_key, window=5):
        """Oyuncunun rating trendini hesapla"""
        if player_key not in self.rating_history or len(self.rating_history[player_key]) < 2:
            return 0
        
        ratings = [record['rating'] for record in self.rating_history[player_key]]
        dates = [record['date'] for record in self.rating_history[player_key]]
        
        # Son 'window' sayıda rating üzerinden trend hesapla
        recent_ratings = ratings[-window:]
        
        if len(recent_ratings) < 2:
            return 0
        
        # Basit lineer regresyon ile trend
        x = np.arange(len(recent_ratings)).reshape(-1, 1)
        y = np.array(recent_ratings)
        
        model = LinearRegression()
        model.fit(x, y)
        
        return model.coef_[0]  # Trend katsayısı

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
    """Oyuncu isimleri için gelişmiş normalizasyon (özel karakter dönüşümü ile)"""
    if pd.isna(name):
        return ""
    
    name = str(name).strip().lower()
    
    # Özel karakter dönüşümleri
    special_chars = {
        'ä': 'ae', 'ö': 'oe', 'ü': 'ue', 'ß': 'ss',
        'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
        'à': 'a', 'è': 'e', 'ì': 'i', 'ò': 'o', 'ù': 'u',
        'â': 'a', 'ê': 'e', 'î': 'i', 'ô': 'o', 'û': 'u',
        'ç': 'c', 'ş': 's', 'ğ': 'g', 'ı': 'i',
        'š': 's', 'č': 'c', 'ž': 'z', 'ć': 'c',
        'ñ': 'n', 'ø': 'o', 'å': 'a', 'æ': 'ae'
    }
    
    for char, replacement in special_chars.items():
        name = name.replace(char, replacement)
    
    # Unicode normalizasyonu
    name = unicodedata.normalize("NFKD", name)
    name = ''.join([c for c in name if not unicodedata.combining(c)])
    
    # Parantez içeriğini kaldır (ör. (loan))
    name = re.sub(r"\(.*?\)", "", name)
    
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

def token_based_team_match(query, choices, min_score=0.5):
    """Token bazlı takım eşleştirme - fuzzy matching'e alternatif"""
    if not choices:
        return None, 0.0
    
    query_tokens = set(query.split())
    best_match = None
    best_score = 0
    
    for choice in choices:
        choice_tokens = set(choice.split())
        common_tokens = query_tokens & choice_tokens
        
        if not common_tokens:
            continue
            
        score = len(common_tokens) / max(len(query_tokens), len(choice_tokens))
        
        if score > best_score:
            best_score = score
            best_match = choice
    
    if best_score >= min_score:
        return best_match, best_score * 100
    return None, 0.0

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

# --- Güncelleme Mekanizmaları ---
def update_ratings_with_new_data(existing_file, new_fbref_data, rating_system):
    """
    Mevcut dataset'e yeni FBref verilerini ekleyerek ratingleri günceller
    """
    # Mevcut veriyi oku
    existing_df = pd.read_excel(existing_file)
    
    # Yeni FBref verisini işle
    new_fbref_df = pd.read_csv(new_fbref_data)
    
    # Oyuncu eşleştirme (mevcut mantığı kullan)
    updated_df = merge_new_fbref_data(existing_df, new_fbref_df)
    
    # Ratingleri yeniden hesapla
    updated_df = rating_system.calculate_player_rating(updated_df)
    
    return updated_df

def merge_new_fbref_data(existing_df, new_fbref_df):
    """
    Yeni FBref verilerini mevcut dataset ile birleştir
    """
    # FBref sütunlarını güncelle
    fbref_cols = [col for col in new_fbref_df.columns if col not in ['Player', 'Team', 'Season', 'player_norm', 'team_norm']]
    
    for index, row in existing_df.iterrows():
        player_name = row['player_norm']
        team_name = row['fbref_team_norm']
        
        # Yeni veride eşleşen oyuncuyu bul
        match = new_fbref_df[
            (new_fbref_df['player_norm'] == player_name) & 
            (new_fbref_df['team_norm'] == team_name)
        ]
        
        if not match.empty:
            # FBref verilerini güncelle
            for col in fbref_cols:
                fbref_col_name = f'fbref__{col}'
                if fbref_col_name in existing_df.columns:
                    existing_df.at[index, fbref_col_name] = match[col].iloc[0]
    
    return existing_df

# --- Main entegre pipeline ---
def integrated_pipeline(update_mode=False):
    """
    Ana pipeline fonksiyonu
    
    Args:
        update_mode (bool): True ise sadece rating güncellemesi yapar
    """
    # Rating sistemini başlat
    rating_system = DynamicRatingSystem()
    rating_system.load_config()
    
    if update_mode:
        print("🔄 Update modu: Sadece rating güncellemesi yapılıyor...")
        return update_dataset_with_new_fbref_data(rating_system)
    
    print("🚀 Yeni pipeline çalıştırılıyor...")
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
        # Önce fuzzy matching dene
        cand, score = extract_one(s, fbref_teams, scorer=None)
        
        # Eğer fuzzy matching başarısız olursa, token bazlı matching dene
        if cand is None or score < TEAM_FUZZY_LOW:
            cand_token, score_token = token_based_team_match(s, fbref_teams, min_score=0.3)
            if cand_token and score_token > (score or 0):
                cand, score = cand_token, score_token
        
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
    # Detect numeric columns in fbref - FIXED: fbref__ önekli sütunları kullan
    numeric_candidates = ['MP','Gls','Ast','Min','Sh','SoT','Cmp','Att','PrgC','PrgP','PrgR']
    numeric_cols_present = [f'fbref__{c}' for c in numeric_candidates if f'fbref__{c}' in merged.columns]
    
    for col in numeric_cols_present:
        merged[col] = pd.to_numeric(merged[col], errors='coerce')

    # Team-level imputation - FIXED: fbref__ önekli sütunları kullan
    for col in numeric_cols_present:
        for team in merged['team_norm'].dropna().unique():
            mask = merged['team_norm'] == team
            team_mean = merged.loc[mask, col].mean()
            if pd.notna(team_mean):
                merged.loc[mask & merged[col].isna(), col] = team_mean
    
    # League mean - FIXED: fbref__ önekli sütunları kullan
    for col in numeric_cols_present:
        if merged[col].isna().any():
            lg_mean = merged[col].mean()
            if pd.notna(lg_mean):
                merged[col] = merged[col].fillna(lg_mean)
            else:
                merged[col] = merged[col].fillna(0)

    # --- DİNAMİK RATING HESAPLAMA ---
    print("🎯 Dinamik rating hesaplanıyor...")
    merged = rating_system.calculate_player_rating(merged)
    
    # Rating trendlerini hesapla
    print("📈 Rating trendleri hesaplanıyor...")
    merged['Rating_Trend'] = 0.0
    for idx, row in merged.iterrows():
        player_key = f"{row['player_norm']}_{row['fbref_team_norm']}"
        trend = rating_system.get_rating_trend(player_key)
        merged.at[idx, 'Rating_Trend'] = trend

    # Save final merged dataset - MEVCUT DOSYA ADIYLA
    merged.to_excel(OUTPUT_FILE, index=False)
    
    # Rating konfigürasyonunu kaydet
    rating_system.save_config()

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
    
    # Rating istatistikleri
    lines.append('[Rating Statistics]')
    lines.append(f'- Average Rating: {merged["Rating"].mean():.2f}')
    lines.append(f'- Max Rating: {merged["Rating"].max():.2f}')
    lines.append(f'- Min Rating: {merged["Rating"].min():.2f}')
    lines.append(f'- Rating Std Dev: {merged["Rating"].std():.2f}')
    
    # Pozisyon bazlı rating ortalamaları
    lines.append('\n[Position-based Averages]')
    for position in ['GK', 'DEF', 'MID', 'FWD']:
        pos_avg = merged[merged['position_detected'] == position]['Rating'].mean()
        pos_count = len(merged[merged['position_detected'] == position])
        lines.append(f'- {position}: {pos_avg:.2f} ({pos_count} players)')
    
    # Player matching istatistikleri
    if len(player_mapping_report) > 0:
        report_df = pd.DataFrame(player_mapping_report)
        matched_count = len(report_df[report_df['status'] == 'matched'])
        manual_review_count = len(report_df[report_df['status'] == 'manual_review'])
        unmatched_count = len(report_df[report_df['status'] == 'unmatched'])
        
        lines.append('\n[Player Matching Details]')
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
    lines.append(f'- Rating config: {RATING_CONFIG_PATH}')
    lines.append(f'- Final merged dataset: {OUTPUT_FILE}')

    with open(LOG_PATH, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print('\n'.join(lines))
    print('\n✅ Done. Outputs saved to logs/ and data/')
    
    return merged, rating_system

# --- Güncelleme Fonksiyonu ---
def update_dataset_with_new_fbref_data(rating_system):
    """Yeni FBref verileri ile dataset'i güncelle"""
    print("🔄 Dataset güncelleniyor...")
    
    # Yeni verilerle güncelle
    existing_file = OUTPUT_FILE
    new_fbref_data = FBREF_PATH
    
    if os.path.exists(existing_file):
        updated_df = update_ratings_with_new_data(existing_file, new_fbref_data, rating_system)
        
        # Güncellenmiş dataset'i kaydet - AYNI DOSYA ADIYLA
        updated_df.to_excel(OUTPUT_FILE, index=False)
        
        print(f"✅ Dataset güncellendi: {OUTPUT_FILE}")
        return updated_df
    else:
        print("❌ Mevcut dataset bulunamadı. Yeni bir pipeline çalıştırın.")
        return None

if __name__ == '__main__':
    # Komut satırı argümanlarını kontrol et
    update_mode = False
    if len(sys.argv) > 1 and sys.argv[1] == '--update':
        update_mode = True
    
    if update_mode:
        # Sadece rating güncellemesi yap
        rating_system = DynamicRatingSystem()
        rating_system.load_config()
        update_dataset_with_new_fbref_data(rating_system)
    else:
        # Tam pipeline çalıştır
        merged_data, rating_system = integrated_pipeline(update_mode=False)