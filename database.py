# database.py
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
