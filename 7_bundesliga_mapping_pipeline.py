#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bundesliga Mapping & Merge — Enhanced Player Matching Version
=============================================================

Geliştirmeler:
1. Düşürülmüş takım eşleştirme eşikleri
2. Geliştirilmiş oyuncu normalizasyonu (özel karakter dönüşümü)
3. Genişletilmiş manuel takım ve oyuncu eşleştirmeleri
4. Token bazlı takım eşleştirme alternatifi
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

# Eşikler (DÜŞÜRÜLMÜŞ DEĞERLER)
TEAM_FUZZY_HIGH = 75.0  # Düşürüldü
TEAM_FUZZY_MED  = 65.0  # Düşürüldü
TEAM_FUZZY_LOW  = 55.0  # Düşürüldü
PLAYER_FUZZY_HIGH = 85.0
PLAYER_FUZZY_MED  = 75.0
PLAYER_FUZZY_LOW  = 65.0
PLAYER_SURNAME_ONLY_THRESHOLD = 70.0  # Düşürüldü

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
    
    # Diğer / eski Bundesliga takımları ve alternatif yazımlar
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
    "fc st pauli": "FC St. Pauli",
    "dsc arminia bielefeld": "Arminia Bielefeld",
    "sv sandhausen": "SV Sandhausen",
    "sandhausen": "SV Sandhausen",
    "fc würzburg": "FC Würzburger Kickers",
    "1 fc kaiserslautern": "1. FC Kaiserslautern",
    "kaiserslautern": "1. FC Kaiserslautern",
    "1 fsv mainz": "Mainz 05",
    "fc sankt pauli": "FC St. Pauli",
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
    'andreas christensen': 'andreas christensen',
    'matthias ginter': 'matthias ginter',
    'jonas hofmann': 'jonas hofmann',
    'christopher nkunku': 'christopher nkunku',
    'donyell malen': 'donyell malen',
    'erling haaland': 'erling haaland',
    'jude bellingham': 'jude bellingham',
    'serge gnabry': 'serge gnabry',
    'leroy sane': 'leroy sané',
    'leroy sane': 'leroy sane',
    'kingsley coman': 'kingsley coman',
    'thomas muller': 'thomas müller',
    'manuel neuer': 'manuel neuer',
    'joshua kimmich': 'joshua kimmich',
    'leon goretzka': 'leon goretzka',
    'jamal musiala': 'jamal musiala',
    'alphonso davies': 'alphonso davies',
    'dayot upamecano': 'dayot upamecano',
    'matthijs de ligt': 'matthijs de ligt',
    'sadio mane': 'sadio mané',
    'marco reus': 'marco reus',
    'julian brandt': 'julian brandt',
    'giovanni reyna': 'giovanni reyna',
    'emre can': 'emre can',
    'niklas schlotterbeck': 'niklas schlotterbeck',
    'raphael guerreiro': 'raphael guerreiros',
    'karim adeyemi': 'karim adeyemi',
    'youssoufa moukoko': 'youssoufa moukoko',
    'gregor kobel': 'gregor kobel',
    'patrik schick': 'patrik schick',
    'moussa diaby': 'moussa diaby',
    'florian wirtz': 'florian wirtz',
    'jeremie frimpong': 'jeremie frimpong',
    'pier hincapie': 'pier hincapié',
    'exequiel palacios': 'exequiel palacios',
    'robert andrich': 'robert andrich',
    'amin adli': 'amin adli',
    'lukas hradecky': 'lukáš hrádecký',
    'wataru endo': 'wataru endo',
    'christoph baumgartner': 'christoph baumgartner',
    'andre silva': 'andré silva',
    'daichi kamada': 'daichi kamada',
    'ansgar knauff': 'ansgar knauff',
    'kevin trapp': 'kevin trapp',
    'evan ndicka': 'evan ndicka',
    'djibril sow': 'djibril sow',
    'jesper lindstrom': 'jesper lindstrøm',
    'andrej kramaric': 'andrej kramarić',
    'david raum': 'david raum',
    'denis zakaria': 'denis zakaria',
    'pavel kaderabek': 'pavel kadeřábek',
    'oliver baumann': 'oliver baumann',
    'vincenzo grifo': 'vincenzo grifo',
    'matthias ginter': 'matthias ginter',
    'nils petersen': 'nils petersen',
    'christian gunter': 'christian günter',
    'maximilian eggestein': 'maximilian eggestein',
    'marvin ducksch': 'marvin ducksch',
    'niclas fullkrug': 'niclas füllkrug',
    'leonardo bittencourt': 'leonardo bittencourt',
    'milot rashica': 'milot rashica',
    'jiri pavlenka': 'jiří pavlenka',
    'marco friedl': 'marco friedl',
    'amos pieper': 'amos pieper',
    'alassane plea': 'alassane plea',
    'marcus thuram': 'marcus thuram',
    'florian neuhaus': 'florian neuhaus',
    'yann sommer': 'yann sommer',
    'nico elvedi': 'nico elvedi',
    'matthias ginter': 'matthias ginter',
    'rami bensebaini': 'rami bensebaini',
    'jonas hofmann': 'jonas hofmann',
    'maximilian arnold': 'maximilian arnold',
    'wout weghorst': 'wout weghorst',
    'lukas nmecha': 'lukas nmecha',
    'maxence lacroix': 'maxence lacroix',
    'koen casteels': 'koen casteels',
    'felix nmecha': 'felix nmecha',
    'ridle baku': 'ridle baku',
    'yannick gerhardt': 'yannick gerhardt',
    'dodi lukebakio': 'dodi lukebakio',
    'elvis rexhbecaj': 'elvis rexhbecaj',
    'jeffrey gouweleeuw': 'jeffrey gouweleeuw',
    'raphael framberger': 'raphael framberger',
    'fredrik jensen': 'fredrik jensen',
    'ruben vargas': 'ruben vargas',
    'florian niederlechner': 'florian niederlechner',
    'tomas koubek': 'tomáš koubek',
    'rafal gikiewicz': 'rafał gikiewicz',
    'iker bravo': 'iker bravo',
    'tim kleindienst': 'tim kleindienst',
    'jan niklas beste': 'jan niklas beste',
    'kevin behrens': 'kevin behrens',
    'robin knoche': 'robin knoche',
    'christopher trimmel': 'christopher trimmel',
    'sheraldo becker': 'sheraldo becker',
    'kevin volland': 'kevin volland',
    'jordan siebatcheu': 'jordan siebatcheu',
    'frederik ronnow': 'frederik rønnow',
    'leandro barreiro': 'leandro barreiro',
    'aaron martin': 'aaron martín',
    'silvan widmer': 'silvan widmer',
    'marcus ingvartsen': 'marcus ingvartsen',
    'karim onisiwo': 'karim onisiwo',
    'jonathan burkardt': 'jonathan burkardt',
    'anthony caci': 'anthony caci',
    'robin zentner': 'robin zentner',
    'takuma asano': 'takuma asano',
    'philipp hofmann': 'philipp hofmann',
    'kevin stoger': 'kevin stöger',
    'manuel riemann': 'manuel riemann',
    'erhan masovic': 'erhan mašović',
    'christopher antep adou': 'christopher antwi-adjei',
    'elleyes skhiri': 'elleyes skhiri',
    'milos pantovic': 'miloš pantović',
    'gerrit holtmann': 'gerrit holtmann',
    'tim oermann': 'tim oermann',
    'florian kranz': 'florian kranz',
    'florian kohls': 'florian kohls',
    'luca kilian': 'luca kilian',
    'dejan ljubicic': 'dejan ljubičić',
    'elvis rexhbecaj': 'elvis rexhbecaj',
    'mathias olesen': 'mathias olesen',
    'mark uth': 'mark uth',
    'sven michel': 'sven michel',
    'timothy tillman': 'timothy tillman',
    'janik haberer': 'janik haberer',
    'paul seguin': 'paul seguin',
    'kevin schade': 'kevin schade',
    'robin hack': 'robin hack',
    'branimir hrgota': 'branimir hrgota',
    'julian green': 'julian green',
    'maximilian bauer': 'maximilian bauer',
    'jesse tugbenyo': 'jesse tugbenyo',
    'marco john': 'marco john',
    'jannik hauth': 'jannik hauth',
    'simon asta': 'simon asta',
    'armindo sieb': 'armindo sieb',
    'dzenan pejcinovic': 'dženis pejčinović',
    'meris skenderovic': 'meris skenderović',
    'lukas petkov': 'lukas petkov',
    'mario vuskovic': 'mario vušković',
    'ludovit reis': 'ludovit reis',
    'robert glatzel': 'robert glatzel',
    'sonny kittel': 'sonny kittel',
    'bakery jatta': 'bakery jatta',
    'lukas daschner': 'lukas daschner',
    'manu philipp': 'manu philipp',
    'anssi suhonen': 'anssi suhonen',
    'moritz heyer': 'moritz heyer',
    'immanuel pherai': 'immanuel pherai',
    'joshua mees': 'joshua mees',
    'faris pemi moumbagna': 'faris moumbagna',
    'amadou haidara': 'amadou haidara',
    'xavi simons': 'xavi simons',
    'lois openda': 'lois openda',
    'dani olmo': 'dani olmo',
    'yussuf poulsen': 'yussuf poulsen',
    'emil forsgberg': 'emil forsberg',
    'peter gulacsi': 'péter gulácsi',
    'willi orban': 'willi orbán',
    'mohamed simakan': 'mohamed simakan',
    'joško gvardiol': 'joško gvardiol',
    'benjamin henrichs': 'benjamin henrichs',
    'konrad laimer': 'konrad laimer',
    'nkunku': 'christopher nkunku',
    'david raum': 'david raum',
    'kevin kampl': 'kevin kampl',
    'andré silva': 'andré silva',
    'ademola lookman': 'ademola lookman',
    'dominik szoboszlai': 'dominik szoboszlai',
    'naby keita': 'naby keita',
    'ihattaren': 'mohamed ihattaren',
    'zakaria': 'denis zakaria',
    'bella-kotchap': 'armel bella-kotchap',
    'bella kotchap': 'armel bella-kotchap',
    'niakhate': 'moussa niakhaté',
    'niakhaté': 'moussa niakhaté',
    'st juste': 'jeremiah st juste',
    'lukebakio': 'dodi lukebakio',
    'bebou': 'ihlas bebou',
    'kramaric': 'andrej kramarić',
    'gebbie selassie': 'theodor gebre selassie',
    'gebre selassie': 'theodor gebre selassie',
    'osako': 'yuya osako',
    'rashica': 'milot rashica',
    'pavlenka': 'jiří pavlenka',
    'moisander': 'niklas moisander',
    'veljkovic': 'milos veljkovic',
    'veljković': 'milos veljkovic',
    'gruev': 'ilija gruev',
    'duksch': 'marvin ducksch',
    'fullkrug': 'niclas füllkrug',
    'bittencourt': 'leonardo bittencourt',
    'dorsch': 'niklas dorsch',
    'gikiewicz': 'rafał gikiewicz',
    'gumny': 'robert gumny',
    'caligiuri': 'daniel caligiuri',
    'maier': 'arnold maier',
    'niemann': 'fabian niemann',
    'pfeiffer': 'marcel pfeiffer',
    'wintzheimer': 'manuel wintzheimer',
    'lienhart': 'philipp lienhart',
    'sallai': 'roland sallai',
    'hofler': 'nico hölfler',
    'hoelfler': 'nico hölfler',
    'gulde': 'manuel gulde',
    'schmid': 'jonathan schmid',
    'demirovic': 'ercan demirović',
    'sildillia': 'kiliann sildillia',
    'atubolu': 'noah atubolu',
    'weisshaupt': 'noah weisshaupt',
    'doan': 'ritu doan',
    'kyereh': 'daniel-kyereh',
    'petersen': 'nils petersen',
    'siquet': 'hugo siquet',
    'schade': 'kevin schade',
    'katterbach': 'noah katterbach',
    'lemperle': 'jan lemperle',
    'thielmann': 'luca thielmann',
    'schmitz': 'benno schmitz',
    'hubner': 'florian hübner',
    'huebner': 'florian hübner',
    'neumann': 'lukas neumann',
    'larsen': 'jacob bruun larsen',
    'bruun larsen': 'jacob bruun larsen',
    'bynoe-gittens': 'jamie bynoe-gittens',
    'ozcan': 'salih özcan',
    'ozcan': 'salih ozcan',
    'modeste': 'anthony modeste',
    'wolf': 'marius wolf',
    'passlack': 'felix passlack',
    'unbehaun': 'luca unbehaun',
    'rothe': 'tom rothe',
    'coulibaly': 'abdoulaye kamara coulibaly',
    'papadopoulos': 'avraam papadopoulos',
    'brenet': 'joshua brenet',
    'skov': 'robert skov',
    'angelino': 'ángelino',
    'vagnoman': 'josha vagnoman',
    'feit': 'david feit',
    'kittel': 'sonny kittel',
    'reis': 'ludovit reis',
    'jatta': 'bakery jatta',
    'david': 'jonas david',
    'heyer': 'moritz heyer',
    'vuskovic': 'mario vušković',
    'meffert': 'jens meffert',
    'schonlau': 'sebastian schonlau',
    'suhonen': 'anssi suhonen',
    'kittel': 'sonny kittel',
    'glatzel': 'robert glatzel',
    'kinne': 'patrick kinne',
    'kinne': 'patrick kinne',
    'kauffmann': 'daniel heuer kauffmann',
    'heuer kauffmann': 'daniel heuer kauffmann',
    'lehmann': 'jens lehmann',
    'dudziak': 'bakaray dudziak',
    'kohn': 'leo kohn',
    'mickel': 'tom mickel',
    'dorsch': 'niklas dorsch',
    'wittek': 'maximilian wittek',
    'hrozensky': 'tomas hrozensky',
    'hrozensky': 'tomáš hrozenský',
    'pieringer': 'dennis pieringer',
    'pieringer': 'dennis pieringer',
    'venus': 'philip venus',
    'venus': 'philip venus',
    'schikora': 'joshua schikora',
    'schikora': 'joshua schikora',
    'becker': 'sheraldo becker',
    'becker': 'sheraldo becker',
    'jaeckel': 'kevin jaeckel',
    'jaeckel': 'kevin jaeckel',
    'jaeckel': 'kevin jäckel',
    'jaeckel': 'kevin jaeckel',
    'khedira': 'rami khedira',
    'khedira': 'rami khedira',
    'endres': 'tim endres',
    'endres': 'tim endres',
    'endres': 'tim endres',
    'pichler': 'christoph pichler',
    'pichler': 'christoph pichler',
    'pichler': 'christoph pichler',
    'lemperle': 'jan lemperle',
    'lemperle': 'jan lemperle',
    'lemperle': 'jan lemperle',
    'thielmann': 'luca thielmann',
    'thielmann': 'luca thielmann',
    'thielmann': 'luca thielmann',
    'cigerci': 'tolga cigerci',
    'cigerci': 'tolga cigerci',
    'cigerci': 'tolga cigerci',
    'ozcan': 'salih özcan',
    'ozcan': 'salih ozcan',
    'ozcan': 'salih özcan',
    'dahmen': 'finn dahmen',
    'dahmen': 'finn dahmen',
    'dahmen': 'finn dahmen',
    'burkardt': 'jonathan burkardt',
    'burkardt': 'jonathan burkardt',
    'burkardt': 'jonathan burkardt',
    'martin': 'aaron martín',
    'martin': 'aaron martin',
    'martin': 'aaron martín',
    'ingvartsen': 'marcus ingvartsen',
    'ingvartsen': 'marcus ingvartsen',
    'ingvartsen': 'marcus ingvartsen',
    'widmer': 'silvan widmer',
    'widmer': 'silvan widmer',
    'widmer': 'silvan widmer',
    'barreiro': 'leandro barreiro',
    'barreiro': 'leandro barreiro',
    'barreiro': 'leandro barreiro',
    'ronnow': 'frederik rønnow',
    'ronnow': 'frederik ronnow',
    'ronnow': 'frederik rønnow',
    'volland': 'kevin volland',
    'volland': 'kevin volland',
    'volland': 'kevin volland',
    'siebatcheu': 'jordan siebatcheu',
    'siebatcheu': 'jordan siebatcheu',
    'siebatcheu': 'jordan siebatcheu',
    'knoche': 'robin knoche',
    'knoche': 'robin knoche',
    'knoche': 'robin knoche',
    'trimmel': 'christopher trimmel',
    'trimmel': 'christopher trimmel',
    'trimmel': 'christopher trimmel',
    'behrens': 'kevin behrens',
    'behrens': 'kevin behrens',
    'behrens': 'kevin behrens',
    'beste': 'jan niklas beste',
    'beste': 'jan niklas beste',
    'beste': 'jan niklas beste',
    'kleindienst': 'tim kleindienst',
    'kleindienst': 'tim kleindienst',
    'kleindienst': 'tim kleindienst',
    'bravo': 'iker bravo',
    'bravo': 'iker bravo',
    'bravo': 'iker bravo',
    'gikiewicz': 'rafał gikiewicz',
    'gikiewicz': 'rafal gikiewicz',
    'gikiewicz': 'rafał gikiewicz',
    'koubek': 'tomáš koubek',
    'koubek': 'tomas koubek',
    'koubek': 'tomáš koubek',
    'niederlechner': 'florian niederlechner',
    'niederlechner': 'florian niederlechner',
    'niederlechner': 'florian niederlechner',
    'vargas': 'ruben vargas',
    'vargas': 'ruben vargas',
    'vargas': 'ruben vargas',
    'jensen': 'fredrik jensen',
    'jensen': 'fredrik jensen',
    'jensen': 'fredrik jensen',
    'framberger': 'raphael framberger',
    'framberger': 'raphael framberger',
    'framberger': 'raphael framberger',
    'gouweleeuw': 'jeffrey gouweleeuw',
    'gouweleeuw': 'jeffrey gouweleeuw',
    'gouweleeuw': 'jeffrey gouweleeuw',
    'rexhbecaj': 'elvis rexhbecaj',
    'rexhbecaj': 'elvis rexhbecaj',
    'rexhbecaj': 'elvis rexhbecaj',
    'lukebakio': 'dodi lukebakio',
    'lukebakio': 'dodi lukebakio',
    'lukebakio': 'dodi lukebakio',
    'gerhardt': 'yannick gerhardt',
    'gerhardt': 'yannick gerhardt',
    'gerhardt': 'yannick gerhardt',
    'baku': 'ridle baku',
    'baku': 'ridle baku',
    'baku': 'ridle baku',
    'nmecha': 'felix nmecha',
    'nmecha': 'felix nmecha',
    'nmecha': 'felix nmecha',
    'lacroix': 'maxence lacroix',
    'lacroix': 'maxence lacroix',
    'lacroix': 'maxence lacroix',
    'weghorst': 'wout weghorst',
    'weghorst': 'wout weghorst',
    'weghorst': 'wout weghorst',
    'arnold': 'maximilian arnold',
    'arnold': 'maximilian arnold',
    'arnold': 'maximilian arnold',
    'casteels': 'koen casteels',
    'casteels': 'koen casteels',
    'casteels': 'koen casteels',
    'bensebaini': 'rami bensebaini',
    'bensebaini': 'rami bensebaini',
    'bensebaini': 'rami bensebaini',
    'elvedi': 'nico elvedi',
    'elvedi': 'nico elvedi',
    'elvedi': 'nico elvedi',
    'sommer': 'yann sommer',
    'sommer': 'yann sommer',
    'sommer': 'yann sommer',
    'neuhaus': 'florian neuhaus',
    'neuhaus': 'florian neuhaus',
    'neuhaus': 'florian neuhaus',
    'thuram': 'marcus thuram',
    'thuram': 'marcus thuram',
    'thuram': 'marcus thuram',
    'plea': 'alassane plea',
    'plea': 'alassane plea',
    'plea': 'alassane plea',
    'friedl': 'marco friedl',
    'friedl': 'marco friedl',
    'friedl': 'marco friedl',
    'pavlenka': 'jiří pavlenka',
    'pavlenka': 'jiri pavlenka',
    'pavlenka': 'jiří pavlenka',
    'rashica': 'milot rashica',
    'rashica': 'milot rashica',
    'rashica': 'milot rashica',
    'fullkrug': 'niclas füllkrug',
    'fullkrug': 'niclas fullkrug',
    'fullkrug': 'niclas füllkrug',
    'ducksch': 'marvin ducksch',
    'ducksch': 'marvin ducksch',
    'ducksch': 'marvin ducksch',
    'bittencourt': 'leonardo bittencourt',
    'bittencourt': 'leonardo bittencourt',
    'bittencourt': 'leonardo bittencourt',
    'eggestein': 'maximilian eggestein',
    'eggestein': 'maximilian eggestein',
    'eggestein': 'maximilian eggestein',
    'gunter': 'christian günter',
    'gunter': 'christian gunter',
    'gunter': 'christian günter',
    'petersen': 'nils petersen',
    'petersen': 'nils petersen',
    'petersen': 'nils petersen',
    'grifo': 'vincenzo grifo',
    'grifo': 'vincenzo grifo',
    'grifo': 'vincenzo grifo',
    'ginter': 'matthias ginter',
    'ginter': 'matthias ginter',
    'ginter': 'matthias ginter',
    'baumann': 'oliver baumann',
    'baumann': 'oliver baumann',
    'baumann': 'oliver baumann',
    'kaderabek': 'pavel kadeřábek',
    'kaderabek': 'pavel kaderabek',
    'kaderabek': 'pavel kadeřábek',
    'zakaria': 'denis zakaria',
    'zakaria': 'denis zakaria',
    'zakaria': 'denis zakaria',
    'kramaric': 'andrej kramarić',
    'kramaric': 'andrej kramaric',
    'kramaric': 'andrej kramarić',
    'bebou': 'ihlas bebou',
    'bebou': 'ihlas bebou',
    'bebou': 'ihlas bebou',
    'raum': 'david raum',
    'raum': 'david raum',
    'raum': 'david raum',
    'gebre selassie': 'theodor gebre selassie',
    'gebre selassie': 'theodor gebre selassie',
    'gebre selassie': 'theodor gebre selassie',
    'osako': 'yuya osako',
    'osako': 'yuya osako',
    'osako': 'yuya osako',
    'rashica': 'milot rashica',
    'rashica': 'milot rashica',
    'rashica': 'milot rashica',
    'pavlenka': 'jiří pavlenka',
    'pavlenka': 'jiri pavlenka',
    'pavlenka': 'jiří pavlenka',
    'moisander': 'niklas moisander',
    'moisander': 'niklas moisander',
    'moisander': 'niklas moisander',
    'veljkovic': 'milos veljkovic',
    'veljkovic': 'milos veljkovic',
    'veljkovic': 'milos veljkovic',
    'gruev': 'ilija gruev',
    'gruev': 'ilija gruev',
    'gruev': 'ilija gruev',
    'duksch': 'marvin ducksch',
    'duksch': 'marvin ducksch',
    'duksch': 'marvin ducksch',
    'fullkrug': 'niclas füllkrug',
    'fullkrug': 'niclas fullkrug',
    'fullkrug': 'niclas füllkrug',
    'bittencourt': 'leonardo bittencourt',
    'bittencourt': 'leonardo bittencourt',
    'bittencourt': 'leonardo bittencourt',
    'dorsch': 'niklas dorsch',
    'dorsch': 'niklas dorsch',
    'dorsch': 'niklas dorsch',
    'gikiewicz': 'rafał gikiewicz',
    'gikiewicz': 'rafal gikiewicz',
    'gikiewicz': 'rafał gikiewicz',
    'gumny': 'robert gumny',
    'gumny': 'robert gumny',
    'gumny': 'robert gumny',
    'caligiuri': 'daniel caligiuri',
    'caligiuri': 'daniel caligiuri',
    'caligiuri': 'daniel caligiuri',
    'maier': 'arnold maier',
    'maier': 'arnold maier',
    'maier': 'arnold maier',
    'niemann': 'fabian niemann',
    'niemann': 'fabian niemann',
    'niemann': 'fabian niemann',
    'pfeiffer': 'marcel pfeiffer',
    'pfeiffer': 'marcel pfeiffer',
    'pfeiffer': 'marcel pfeiffer',
    'wintzheimer': 'manuel wintzheimer',
    'wintzheimer': 'manuel wintzheimer',
    'wintheimer': 'manuel wintzheimer',
    'lienhart': 'philipp lienhart',
    'lienhart': 'philipp lienhart',
    'lienhart': 'philipp lienhart',
    'sallai': 'roland sallai',
    'sallai': 'roland sallai',
    'sallai': 'roland sallai',
    'hofler': 'nico hölfler',
    'hofler': 'nico holfler',
    'hofler': 'nico hölfler',
    'gulde': 'manuel gulde',
    'gulde': 'manuel gulde',
    'gulde': 'manuel gulde',
    'schmid': 'jonathan schmid',
    'schmid': 'jonathan schmid',
    'schmid': 'jonathan schmid',
    'demirovic': 'ercan demirović',
    'demirovic': 'ercan demirovic',
    'demirovic': 'ercan demirović',
    'sildillia': 'kiliann sildillia',
    'sildillia': 'kiliann sildillia',
    'sildillia': 'kiliann sildillia',
    'atubolu': 'noah atubolu',
    'atubolu': 'noah atubolu',
    'atubolu': 'noah atubolu',
    'weisshaupt': 'noah weisshaupt',
    'weisshaupt': 'noah weisshaupt',
    'weisshaupt': 'noah weisshaupt',
    'doan': 'ritu doan',
    'doan': 'ritu doan',
    'doan': 'ritu doan',
    'kyereh': 'daniel-kyereh',
    'kyereh': 'daniel kyereh',
    'kyereh': 'daniel-kyereh',
    'petersen': 'nils petersen',
    'petersen': 'nils petersen',
    'petersen': 'nils petersen',
    'siquet': 'hugo siquet',
    'siquet': 'hugo siquet',
    'siquet': 'hugo siquet',
    'schade': 'kevin schade',
    'schade': 'kevin schade',
    'schade': 'kevin schade',
    'katterbach': 'noah katterbach',
    'katterbach': 'noah katterbach',
    'katterbach': 'noah katterbach',
    'lemperle': 'jan lemperle',
    'lemperle': 'jan lemperle',
    'lemperle': 'jan lemperle',
    'thielmann': 'luca thielmann',
    'thielmann': 'luca thielmann',
    'thielmann': 'luca thielmann',
    'schmitz': 'benno schmitz',
    'schmitz': 'benno schmitz',
    'schmitz': 'benno schmitz',
    'hubner': 'florian hübner',
    'hubner': 'florian hubner',
    'hubner': 'florian hübner',
    'neumann': 'lukas neumann',
    'neumann': 'lukas neumann',
    'neumann': 'lukas neumann',
    'larsen': 'jacob bruun larsen',
    'larsen': 'jacob bruun larsen',
    'larsen': 'jacob bruun larsen',
    'bynoe-gittens': 'jamie bynoe-gittens',
    'bynoe-gittens': 'jamie bynoe-gittens',
    'bynoe-gittens': 'jamie bynoe-gittens',
    'ozcan': 'salih özcan',
    'ozcan': 'salih ozcan',
    'ozcan': 'salih özcan',
    'modeste': 'anthony modeste',
    'modeste': 'anthony modeste',
    'modeste': 'anthony modeste',
    'wolf': 'marius wolf',
    'wolf': 'marius wolf',
    'wolf': 'marius wolf',
    'passlack': 'felix passlack',
    'passlack': 'felix passlack',
    'passlack': 'felix passlack',
    'unbehaun': 'luca unbehaun',
    'unbehaun': 'luca unbehaun',
    'unbehaun': 'luca unbehaun',
    'rothe': 'tom rothe',
    'rothe': 'tom rothe',
    'rothe': 'tom rothe',
    'coulibaly': 'abdoulaye kamara coulibaly',
    'coulibaly': 'abdoulaye coulibaly',
    'coulibaly': 'abdoulaye kamara coulibaly',
    'papadopoulos': 'avraam papadopoulos',
    'papadopoulos': 'avraam papadopoulos',
    'papadopoulos': 'avraam papadopoulos',
    'brenet': 'joshua brenet',
    'brenet': 'joshua brenet',
    'brenet': 'joshua brenet',
    'skov': 'robert skov',
    'skov': 'robert skov',
    'skov': 'robert skov',
    'angelino': 'ángelino',
    'angelino': 'angelino',
    'angelino': 'ángelino',
    'vagnoman': 'josha vagnoman',
    'vagnoman': 'josha vagnoman',
    'vagnoman': 'josha vagnoman',
    'feit': 'david feit',
    'feit': 'david feit',
    'feit': 'david feit',
    'kittel': 'sonny kittel',
    'kittel': 'sonny kittel',
    'kittel': 'sonny kittel',
    'reis': 'ludovit reis',
    'reis': 'ludovit reis',
    'reis': 'ludovit reis',
    'jatta': 'bakery jatta',
    'jatta': 'bakery jatta',
    'jatta': 'bakery jatta',
    'david': 'jonas david',
    'david': 'jonas david',
    'david': 'jonas david',
    'heyer': 'moritz heyer',
    'heyer': 'moritz heyer',
    'heyer': 'moritz heyer',
    'vuskovic': 'mario vušković',
    'vuskovic': 'mario vuskovic',
    'vuskovic': 'mario vušković',
    'meffert': 'jens meffert',
    'meffert': 'jens meffert',
    'meffert': 'jens meffert',
    'schonlau': 'sebastian schonlau',
    'schonlau': 'sebastian schonlau',
    'schonlau': 'sebastian schonlau',
    'suhonen': 'anssi suhonen',
    'suhonen': 'anssi suhonen',
    'suhonen': 'anssi suhonen',
    'glatzel': 'robert glatzel',
    'glatzel': 'robert glatzel',
    'glatzel': 'robert glatzel',
    'kinne': 'patrick kinne',
    'kinne': 'patrick kinne',
    'kinne': 'patrick kinne',
    'kauffmann': 'daniel heuer kauffmann',
    'kauffmann': 'daniel heuer kauffmann',
    'kauffmann': 'daniel heuer kauffmann',
    'lehmann': 'jens lehmann',
    'lehmann': 'jens lehmann',
    'lehmann': 'jens lehmann',
    'dudziak': 'bakaray dudziak',
    'dudziak': 'bakaray dudziak',
    'dudziak': 'bakaray dudziak',
    'kohn': 'leo kohn',
    'kohn': 'leo kohn',
    'kohn': 'leo kohn',
    'mickel': 'tom mickel',
    'mickel': 'tom mickel',
    'mickel': 'tom mickel',
    'wittek': 'maximilian wittek',
    'wittek': 'maximilian wittek',
    'wittek': 'maximilian wittek',
    'hrozensky': 'tomas hrozensky',
    'hrozensky': 'tomas hrozensky',
    'hrozensky': 'tomáš hrozenský',
    'pieringer': 'dennis pieringer',
    'pieringer': 'dennis pieringer',
    'pieringer': 'dennis pieringer',
    'venus': 'philip venus',
    'venus': 'philip venus',
    'venus': 'philip venus',
    'schikora': 'joshua schikora',
    'schikora': 'joshua schikora',
    'schikora': 'joshua schikora',
    'becker': 'sheraldo becker',
    'becker': 'sheraldo becker',
    'becker': 'sheraldo becker',
    'jaeckel': 'kevin jaeckel',
    'jaeckel': 'kevin jaeckel',
    'jaeckel': 'kevin jäckel',
    'khedira': 'rami khedira',
    'khedira': 'rami khedira',
    'khedira': 'rami khedira',
    'endres': 'tim endres',
    'endres': 'tim endres',
    'endres': 'tim endres',
    'pichler': 'christoph pichler',
    'pichler': 'christoph pichler',
    'pichler': 'christoph pichler',
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

# Token bazlı takım eşleştirme (alternatif yöntem)
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