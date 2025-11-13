# scraper_advanced.py
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
            logger.info(f"\n{'='*60}")
            logger.info(f"TEMPORADA: {season}")
            logger.info(f"{'='*60}\n")
            
            team_links = self.get_team_links(league_url)
            
            if not team_links:
                logger.warning(f"Nenhum time encontrado para temporada {season}")
                continue
            
            for i, team_url in enumerate(team_links, 1):
                try:
                    team_name = team_url.split('/')[6].replace('-Stats', '')
                    logger.info(f"\n[{i}/{len(team_links)}] Time: {team_name}")
                    
                    team_data = self.get_team_season_data(team_url, season)
                    
                    if team_name not in comprehensive_data['teams_data']:
                        comprehensive_data['teams_data'][team_name] = {}
                    
                    comprehensive_data['teams_data'][team_name][season] = team_data
                    
                except Exception as e:
                    logger.error(f"Erro ao processar time {team_url}: {e}")
                    continue
        
        logger.info("\n✓ Scraping completo finalizado")
        return comprehensive_data
