#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_missing_files.py
EXECUTE ESTE ARQUIVO PRIMEIRO para criar database.py e scraper_advanced.py
"""

import os
from pathlib import Path

print("="*70)
print("GERANDO ARQUIVOS FALTANTES - SPORTS ANALYTICS V4.0")
print("="*70)
print()

# ============================================================================
# database.py
# ============================================================================

DATABASE_PY = '''# database.py
"""
Gerenciamento de banco de dados SQLite para armazenamento estruturado.
"""

import sqlite3
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd
from pathlib import Path
import config

logger = logging.getLogger(__name__)


class DatabaseManager:
    """Gerenciador centralizado do banco de dados."""
    
    def __init__(self, db_path: Path = None):
        if db_path is None:
            db_path = config.DATABASE_PATH
        self.db_path = db_path
        self.conn = None
        self._initialize_database()
    
    def _initialize_database(self):
        """Cria as tabelas se não existirem."""
        try:
            self.conn = sqlite3.connect(self.db_path)
            self.conn.row_factory = sqlite3.Row
            self._create_tables()
            logger.info(f"Banco de dados inicializado: {self.db_path}")
        except Exception as e:
            logger.error(f"Erro ao inicializar banco de dados: {e}")
            raise
    
    def _create_tables(self):
        """Cria o schema do banco de dados."""
        cursor = self.conn.cursor()
        
        # Tabela de Ligas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS leagues (
                league_id INTEGER PRIMARY KEY AUTOINCREMENT,
                league_name TEXT NOT NULL UNIQUE,
                country TEXT NOT NULL,
                season TEXT NOT NULL,
                url TEXT NOT NULL,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Tabela de Times
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS teams (
                team_id INTEGER PRIMARY KEY AUTOINCREMENT,
                team_name TEXT NOT NULL,
                league_id INTEGER NOT NULL,
                fbref_id TEXT UNIQUE,
                elo_rating REAL DEFAULT 1500,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (league_id) REFERENCES leagues (league_id)
            )
        """)
        
        # Tabela de Partidas
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS matches (
                match_id INTEGER PRIMARY KEY AUTOINCREMENT,
                league_id INTEGER NOT NULL,
                home_team_id INTEGER NOT NULL,
                away_team_id INTEGER NOT NULL,
                match_date DATE,
                home_goals INTEGER,
                away_goals INTEGER,
                home_xg REAL,
                away_xg REAL,
                home_corners INTEGER,
                away_corners INTEGER,
                home_fouls INTEGER,
                away_fouls INTEGER,
                referee TEXT,
                season TEXT,
                matchweek INTEGER,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (league_id) REFERENCES leagues (league_id),
                FOREIGN KEY (home_team_id) REFERENCES teams (team_id),
                FOREIGN KEY (away_team_id) REFERENCES teams (team_id)
            )
        """)
        
        # Tabela de Cache de Scraping
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS scraping_cache (
                cache_id INTEGER PRIMARY KEY AUTOINCREMENT,
                url TEXT NOT NULL UNIQUE,
                data_type TEXT NOT NULL,
                cached_data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP NOT NULL
            )
        """)
        
        # Índices
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_teams_league ON teams(league_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_matches_teams ON matches(home_team_id, away_team_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_cache_url ON scraping_cache(url)")
        
        self.conn.commit()
        logger.info("Schema do banco de dados criado com sucesso")
    
    def insert_league(self, league_name: str, country: str, season: str, url: str) -> int:
        """Insere ou atualiza uma liga."""
        cursor = self.conn.cursor()
        cursor.execute("""
            INSERT OR REPLACE INTO leagues (league_name, country, season, url, last_updated)
            VALUES (?, ?, ?, ?, ?)
        """, (league_name, country, season, url, datetime.now()))
        self.conn.commit()
        return cursor.lastrowid
    
    def insert_team(self, team_name: str, league_id: int, fbref_id: str = None) -> int:
        """Insere ou atualiza um time."""
        cursor = self.conn.cursor()
        if fbref_id:
            cursor.execute("""
                INSERT OR REPLACE INTO teams (team_name, league_id, fbref_id, last_updated)
                VALUES (?, ?, ?, ?)
            """, (team_name, league_id, fbref_id, datetime.now()))
        else:
            cursor.execute("""
                INSERT INTO teams (team_name, league_id, last_updated)
                VALUES (?, ?, ?)
            """, (team_name, league_id, datetime.now()))
        self.conn.commit()
        return cursor.lastrowid
    
    def bulk_insert_dataframe(self, df: pd.DataFrame, table_name: str, if_exists: str = 'append'):
        """Insere um DataFrame completo em uma tabela."""
        try:
            df.to_sql(table_name, self.conn, if_exists=if_exists, index=False)
            logger.info(f"{len(df)} registros inseridos em {table_name}")
        except Exception as e:
            logger.error(f"Erro ao inserir DataFrame em {table_name}: {e}")
            raise
    
    def query_to_dataframe(self, query: str, params: tuple = None) -> pd.DataFrame:
        """Executa uma query e retorna um DataFrame."""
        try:
            if params:
                return pd.read_sql_query(query, self.conn, params=params)
            return pd.read_sql_query(query, self.conn)
        except Exception as e:
            logger.error(f"Erro ao executar query: {e}")
            raise
    
    def get_team_stats(self, team_id: int, last_n_games: int = 5) -> pd.DataFrame:
        """Retorna estatísticas dos últimos N jogos de um time."""
        query = """
            SELECT *
            FROM matches
            WHERE home_team_id = ? OR away_team_id = ?
            ORDER BY match_date DESC
            LIMIT ?
        """
        return self.query_to_dataframe(query, (team_id, team_id, last_n_games))
    
    def cache_scraping_data(self, url: str, data_type: str, data: str, expiry_hours: int = 24):
        """Armazena dados de scraping em cache."""
        cursor = self.conn.cursor()
        expires_at = datetime.now().timestamp() + (expiry_hours * 3600)
        cursor.execute("""
            INSERT OR REPLACE INTO scraping_cache (url, data_type, cached_data, expires_at)
            VALUES (?, ?, ?, datetime(?, 'unixepoch'))
        """, (url, data_type, data, expires_at))
        self.conn.commit()
    
    def get_cached_data(self, url: str) -> Optional[str]:
        """Recupera dados do cache se ainda válidos."""
        cursor = self.conn.cursor()
        cursor.execute("""
            SELECT cached_data
            FROM scraping_cache
            WHERE url = ? AND expires_at > datetime('now')
        """, (url,))
        result = cursor.fetchone()
        return result['cached_data'] if result else None
    
    def close(self):
        """Fecha a conexão com o banco de dados."""
        if self.conn:
            self.conn.close()
            logger.info("Conexão com banco de dados fechada")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
'''

# ============================================================================
# scraper_advanced.py
# ============================================================================

SCRAPER_ADVANCED_PY = '''# scraper_advanced.py
"""
Sistema avançado de web scraping com cache, retry logic e rate limiting.
"""

import cloudscraper
import pandas as pd
from io import StringIO
from bs4 import BeautifulSoup
import time
import logging
import json
from typing import Optional, List, Dict, Tuple
from datetime import datetime
import hashlib
import config
from database import DatabaseManager

logger = logging.getLogger(__name__)


class AdvancedScraper:
    """Scraper robusto com funcionalidades avançadas."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.scraper = cloudscraper.create_scraper(browser='chrome')
        self.headers = {'User-Agent': config.SCRAPING_CONFIG['user_agent']}
        self.last_request_time = 0
    
    def _rate_limit(self):
        """Implementa rate limiting entre requisições."""
        elapsed = time.time() - self.last_request_time
        delay = config.SCRAPING_CONFIG['delay_between_requests']
        
        if elapsed < delay:
            sleep_time = delay - elapsed
            logger.debug(f"Rate limiting: aguardando {sleep_time:.2f}s")
            time.sleep(sleep_time)
        
        self.last_request_time = time.time()
    
    def _get_cache_key(self, url: str, table_identifier: str) -> str:
        """Gera chave única para cache."""
        return hashlib.md5(f"{url}:{table_identifier}".encode()).hexdigest()
    
    def fetch_with_cache(self, url: str, table_identifier: str, use_cache: bool = True) -> Optional[str]:
        """Busca HTML com sistema de cache."""
        # Tentar cache primeiro
        if use_cache and config.SCRAPING_CONFIG['cache_enabled']:
            cache_key = self._get_cache_key(url, table_identifier)
            cached = self.db.get_cached_data(cache_key)
            if cached:
                logger.info(f"Cache HIT para {table_identifier}")
                return cached
            logger.debug(f"Cache MISS para {table_identifier}")
        
        # Fazer requisição
        self._rate_limit()
        
        max_retries = config.SCRAPING_CONFIG['max_retries']
        retry_wait = config.SCRAPING_CONFIG['retry_wait']
        
        for attempt in range(max_retries):
            try:
                logger.info(f"Buscando {table_identifier} (tentativa {attempt + 1}/{max_retries})")
                
                response = self.scraper.get(url, headers=self.headers, timeout=config.SCRAPING_CONFIG['timeout'])
                response.raise_for_status()
                
                html = response.text
                
                # Armazenar em cache
                if config.SCRAPING_CONFIG['cache_enabled']:
                    cache_key = self._get_cache_key(url, table_identifier)
                    self.db.cache_scraping_data(cache_key, table_identifier, html, config.SCRAPING_CONFIG['cache_expiry_hours'])
                
                logger.info(f"✓ {table_identifier} obtido com sucesso")
                return html
                
            except Exception as e:
                logger.warning(f"Tentativa {attempt + 1} falhou: {e}")
                
                if attempt < max_retries - 1:
                    logger.info(f"Aguardando {retry_wait}s antes da próxima tentativa...")
                    time.sleep(retry_wait)
                else:
                    logger.error(f"Falha após {max_retries} tentativas para {url}")
                    return None
        
        return None
    
    def extract_table(self, html: str, match_str: str, header_row: int = 1) -> Optional[pd.DataFrame]:
        """Extrai tabela do HTML usando pandas."""
        try:
            tables = pd.read_html(StringIO(html), match=match_str, header=header_row)
            if tables:
                df = tables[0]
                logger.debug(f"Tabela '{match_str}' extraída: {df.shape}")
                return df
            else:
                logger.warning(f"Nenhuma tabela encontrada com match='{match_str}'")
                return None
        except Exception as e:
            logger.error(f"Erro ao extrair tabela '{match_str}': {e}")
            return None
    
    def fetch_table(self, url: str, match_str: str, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """Busca e extrai tabela em uma operação."""
        html = self.fetch_with_cache(url, match_str, use_cache)
        if html:
            return self.extract_table(html, match_str)
        return None
    
    def get_team_links(self, league_url: str) -> List[str]:
        """Extrai links de todos os times da liga."""
        logger.info(f"Buscando links de times em: {league_url}")
        
        html = self.fetch_with_cache(league_url, "team_links")
        if not html:
            return []
        
        try:
            soup = BeautifulSoup(html, 'html.parser')
            table = soup.find('table', {'id': lambda x: x and 'results' in x and 'overall' in x})
            
            if not table:
                table = soup.find('table', class_='stats_table')
            
            links = []
            if table:
                team_cells = table.find_all('td', {'data-stat': 'team'})
                
                for cell in team_cells:
                    a_tag = cell.find('a')
                    if a_tag and 'href' in a_tag.attrs:
                        link = a_tag['href']
                        if link.startswith('/en/squads/'):
                            full_link = f"https://fbref.com{link}"
                            links.append(full_link)
            
            unique_links = list(set(links))
            logger.info(f"✓ Encontrados {len(unique_links)} times")
            return unique_links
            
        except Exception as e:
            logger.error(f"Erro ao extrair links de times: {e}")
            return []
    
    def get_team_season_data(self, team_url: str, season: str) -> Dict[str, pd.DataFrame]:
        """Busca todos os dados relevantes de um time em uma temporada."""
        logger.info(f"Buscando dados do time: {team_url}")
        
        data = {}
        
        tables_config = {
            'standard': 'Standard Stats',
            'miscellaneous': 'Miscellaneous Stats'
        }
        
        for key, match_str in tables_config.items():
            df = self.fetch_table(team_url, match_str)
            if df is not None:
                data[key] = df
            else:
                logger.warning(f"Tabela '{key}' não encontrada")
        
        return data
    
    def scrape_league_comprehensive(self, league_url: str, seasons: List[str]) -> Dict:
        """Scraping completo de uma liga incluindo múltiplas temporadas."""
        logger.info(f"Iniciando scraping completo da liga: {league_url}")
        
        comprehensive_data = {
            'league_url': league_url,
            'seasons': seasons,
            'teams_data': {},
            'scrape_timestamp': datetime.now().isoformat()
        }
        
        for season in seasons:
            logger.info(f"\\n{'='*60}")
            logger.info(f"TEMPORADA: {season}")
            logger.info(f"{'='*60}\\n")
            
            team_links = self.get_team_links(league_url)
            
            if not team_links:
                logger.warning(f"Nenhum time encontrado para temporada {season}")
                continue
            
            for i, team_url in enumerate(team_links, 1):
                try:
                    team_name = team_url.split('/')[6].replace('-Stats', '')
                    logger.info(f"\\n[{i}/{len(team_links)}] Time: {team_name}")
                    
                    team_data = self.get_team_season_data(team_url, season)
                    
                    if team_name not in comprehensive_data['teams_data']:
                        comprehensive_data['teams_data'][team_name] = {}
                    
                    comprehensive_data['teams_data'][team_name][season] = team_data
                    
                except Exception as e:
                    logger.error(f"Erro ao processar time {team_url}: {e}")
                    continue
        
        logger.info("\\n✓ Scraping completo finalizado")
        return comprehensive_data
'''

# ============================================================================
# CRIAR ARQUIVOS
# ============================================================================

files_to_create = {
    'database.py': DATABASE_PY,
    'scraper_advanced.py': SCRAPER_ADVANCED_PY
}

for filename, content in files_to_create.items():
    filepath = Path(filename)
    
    if filepath.exists():
        print(f"⚠️  {filename} já existe")
        resp = input(f"   Sobrescrever? (s/n): ")
        if resp.lower() != 's':
            print(f"   ⊘ Pulado: {filename}")
            continue
    
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"✓ Criado: {filename}")

print()
print("="*70)
print("✅ ARQUIVOS CRIADOS COM SUCESSO!")
print("="*70)
print()
print("Agora execute:")
print("python main_orchestrator.py")
print()