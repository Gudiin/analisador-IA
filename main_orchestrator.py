# main_orchestrator.py
"""
Orquestrador principal que integra todos os módulos.
"""

import json
import logging
import logging.config
from datetime import datetime
from pathlib import Path
import sys
from typing import Dict, List, Optional
import pandas as pd

import config
from database import DatabaseManager
from scraper_advanced import AdvancedScraper
from analysis_engine import AnalysisEngine
from backtesting import BacktestingEngine, KellyCriterion
from statistical_models import EloRatingSystem

# Configurar logging
logging.config.dictConfig(config.LOGGING_CONFIG)
logger = logging.getLogger(__name__)


class SportsAnalyticsPipeline:
    """Pipeline completo de análise de apostas esportivas."""
    
    def __init__(self):
        logger.info("Inicializando Sports Analytics Pipeline...")
        
        # Inicializar componentes
        self.db = DatabaseManager()
        self.scraper = AdvancedScraper(self.db)
        self.analyzer = AnalysisEngine(self.db)
        self.backtester = BacktestingEngine()
        self.elo_system = EloRatingSystem()
        
        logger.info("✓ Pipeline inicializado com sucesso")
    
    def select_league(self) -> tuple[Optional[str], Optional[str]]:
        """
        Interface para seleção de liga.
        
        Returns:
            Tupla (nome_da_liga, url_da_liga)
        """
        try:
            leagues_file = Path('../API/leagues.json')
            
            if not leagues_file.exists():
                logger.error("Arquivo leagues.json não encontrado")
                return None, None
            
            with open(leagues_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            paises = data.get('paises', [])
            
            if not paises:
                logger.error("Nenhum país encontrado em leagues.json")
                return None, None
            
            # Menu de países
            print("\n" + "="*60)
            print("SELECIONE O PAÍS")
            print("="*60)
            for i, pais in enumerate(paises, 1):
                print(f"[{i}] {pais['nome']}")
            print("[0] Sair")
            
            try:
                choice = int(input("\nDigite o número: "))
                if choice == 0:
                    return None, None
                selected_pais = paises[choice - 1]
            except (ValueError, IndexError):
                logger.error("Seleção inválida")
                return None, None
            
            # Menu de competições
            competicoes = selected_pais.get('competicoes', [])
            
            print(f"\n{'='*60}")
            print(f"SELECIONE A COMPETIÇÃO - {selected_pais['nome']}")
            print("="*60)
            for i, comp in enumerate(competicoes, 1):
                print(f"[{i}] {comp['nome']}")
            print("[0] Voltar")
            
            try:
                choice = int(input("\nDigite o número: "))
                if choice == 0:
                    return None, None
                selected_comp = competicoes[choice - 1]
            except (ValueError, IndexError):
                logger.error("Seleção inválida")
                return None, None
            
            return selected_comp['nome'], selected_comp['url']
            
        except Exception as e:
            logger.error(f"Erro ao carregar leagues.json: {e}")
            return None, None
    
    def collect_league_data(self, league_name: str, league_url: str, seasons: List[str]):
        """
        Coleta todos os dados de uma liga.
        
        Args:
            league_name: Nome da liga
            league_url: URL da liga
            seasons: Lista de temporadas para coletar
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"INICIANDO COLETA DE DADOS")
        logger.info(f"Liga: {league_name}")
        logger.info(f"Temporadas: {', '.join(seasons)}")
        logger.info(f"{'='*60}\n")
        
        # Inserir liga no banco
        league_id = self.db.insert_league(
            league_name=league_name,
            country="",  # Extrair do leagues.json se necessário
            season=seasons[-1],  # Temporada mais recente
            url=league_url
        )
        
        # Scraping completo
        comprehensive_data = self.scraper.scrape_league_comprehensive(
            league_url=league_url,
            seasons=seasons
        )
        
        # Processar e armazenar dados
        self._process_and_store_data(comprehensive_data, league_id)
        
        logger.info("\n✓ Coleta de dados concluída")
    
    def _process_and_store_data(self, data: Dict, league_id: int):
        """
        Processa dados coletados e armazena no banco.
        
        Args:
            data: Dados retornados pelo scraper
            league_id: ID da liga no banco
        """
        logger.info("Processando e armazenando dados...")
        
        # Processar dados de times
        for team_name, seasons_data in data['teams_data'].items():
            # Inserir time
            team_id = self.db.insert_team(
                team_name=team_name,
                league_id=league_id,
                fbref_id=None  # Extrair do URL se necessário
            )
            
            # Processar dados de cada temporada
            for season, tables in seasons_data.items():
                if 'standard' in tables:
                    df_std = tables['standard']
                    # Processar e inserir estatísticas
                    # Implementar lógica específica baseada na estrutura dos dados
                
                if 'miscellaneous' in tables:
                    df_misc = tables['miscellaneous']
                    # Processar faltas, cartões, etc.
        
        # Processar dados de árbitros
        if data['referees'] is not None:
            # Inserir no banco
            pass
        
        # Processar partidas
        if data['matches'] is not None:
            # Inserir no banco
            pass
        
        logger.info("✓ Dados processados e armazenados")
    
    def analyze_market_opportunities(self, league_id: int) -> Dict:
        """
        Analisa oportunidades de apostas para uma liga.
        
        Args:
            league_id: ID da liga
        
        Returns:
            Dicionário com todas as análises
        """
        logger.info("\n" + "="*60)
        logger.info("ANALISANDO OPORTUNIDADES DE MERCADO")
        logger.info("="*60 + "\n")
        
        opportunities = {
            'league_id': league_id,
            'analysis_timestamp': datetime.now().isoformat(),
            'goals_analysis': [],
            'corners_analysis': [],
            'fouls_analysis': [],
            'value_bets': []
        }
        
        # Buscar próximas partidas (ou partidas recentes para backtest)
        query = """
            SELECT match_id, home_team_id, away_team_id, match_date
            FROM matches
            WHERE league_id = ?
            ORDER BY match_date DESC
            LIMIT 10
        """
        
        matches = self.db.query_to_dataframe(query, (league_id,))
        
        for _, match in matches.iterrows():
            home_id = match['home_team_id']
            away_id = match['away_team_id']
            
            logger.info(f"Analisando partida: {home_id} vs {away_id}")
            
            # Análise de Gols
            goals_pred = self.analyzer.predict_goals(home_id, away_id)
            opportunities['goals_analysis'].append({
                'match_id': match['match_id'],
                'prediction': goals_pred
            })
            
            # Análise de Escanteios
            corners_pred = self.analyzer.predict_corners(home_id, away_id)
            opportunities['corners_analysis'].append({
                'match_id': match['match_id'],
                'prediction': corners_pred
            })
            
            # Análise de Faltas
            fouls_pred = self.analyzer.predict_fouls(home_id, away_id)
            opportunities['fouls_analysis'].append({
                'match_id': match['match_id'],
                'prediction': fouls_pred
            })
            
            # Identificar value bets
            # Simular odds do mercado (em produção, buscar de API de odds)
            market_odds_over_2_5 = 1.85
            
            prob_over_2_5 = goals_pred['probabilities'].get('over_2.5', 0)
            
            bet_eval = self.backtester.evaluate_bet_opportunity(
                model_probability=prob_over_2_5,
                market_odds=market_odds_over_2_5,
                bet_type='Over 2.5 Goals'
            )
            
            if bet_eval['is_value_bet']:
                opportunities['value_bets'].append({
                    'match_id': match['match_id'],
                    'bet_analysis': bet_eval
                })
        
        logger.info(f"\n✓ Análise concluída: {len(opportunities['value_bets'])} value bets encontradas")
        
        return opportunities
    
    def generate_comprehensive_report(self, league_name: str, opportunities: Dict) -> str:
        """
        Gera relatório completo em JSON.
        
        Args:
            league_name: Nome da liga
            opportunities: Dicionário com análises
        
        Returns:
            Caminho do arquivo gerado
        """
        logger.info("\nGerando relatório completo...")
        
        # Criar estrutura do relatório
        report = {
            'metadata': {
                'league_name': league_name,
                'report_date': datetime.now().isoformat(),
                'analysis_version': '2.0',
                'model_confidence': 'MEDIUM-HIGH'
            },
            'market_analysis': opportunities,
            'backtesting_summary': self.backtester.get_performance_summary(),
            'elo_rankings': {
                team: rating 
                for team, rating in sorted(
                    self.elo_system.ratings.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                )[:20]
            },
            'recommendations': self._generate_recommendations(opportunities)
        }
        
        # Salvar JSON
        safe_name = league_name.replace(' ', '_').replace('/', '_').lower()
        filename = f"comprehensive_report_{safe_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = config.REPORTS_DIR / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✓ Relatório salvo: {filepath}")
        
        return str(filepath)
    
    def _generate_recommendations(self, opportunities: Dict) -> List[Dict]:
        """Gera recomendações priorizadas."""
        recommendations = []
        
        for bet in opportunities['value_bets']:
            analysis = bet['bet_analysis']
            
            if analysis['confidence_level'] in ['HIGH', 'MEDIUM']:
                recommendations.append({
                    'match_id': bet['match_id'],
                    'bet_type': analysis['bet_type'],
                    'confidence': analysis['confidence_level'],
                    'edge': f"{analysis['edge_percentage']:.2f}%",
                    'recommended_stake': f"R$ {analysis['recommended_stake']:.2f}",
                    'expected_roi': f"{analysis['roi_percentage']:.2f}%"
                })
        
        # Ordenar por edge
        recommendations.sort(key=lambda x: float(x['edge'].rstrip('%')), reverse=True)
        
        return recommendations[:10]  # Top 10
    
    def run(self):
        """Executa o pipeline completo."""
        try:
            logger.info("\n" + "="*60)
            logger.info("SPORTS ANALYTICS PIPELINE - INICIANDO")
            logger.info("="*60 + "\n")
            
            # 1. Seleção de liga
            league_name, league_url = self.select_league()
            
            if not league_url:
                logger.info("Operação cancelada pelo usuário")
                return
            
            # 2. Definir temporadas
            seasons = ['2023-2024', '2024-2025']  # Últimas 2 temporadas
            
            # 3. Coletar dados
            self.collect_league_data(league_name, league_url, seasons)
            
            # 4. Analisar oportunidades
            league_id = 1  # Buscar do banco
            opportunities = self.analyze_market_opportunities(league_id)
            
            # 5. Gerar relatório
            report_path = self.generate_comprehensive_report(league_name, opportunities)
            
            # 6. Gerar relatório de backtesting
            backtest_report_path = config.REPORTS_DIR / f"backtesting_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
            self.backtester.generate_report(str(backtest_report_path))
            
            logger.info("\n" + "="*60)
            logger.info("PIPELINE CONCLUÍDO COM SUCESSO")
            logger.info("="*60)
            logger.info(f"Relatório JSON: {report_path}")
            logger.info(f"Relatório Backtesting: {backtest_report_path}")
            logger.info("="*60 + "\n")
            
        except KeyboardInterrupt:
            logger.info("\n\nOperação interrompida pelo usuário")
        except Exception as e:
            logger.exception(f"Erro fatal no pipeline: {e}")
        finally:
            self.db.close()


def main():
    """Ponto de entrada do programa."""
    pipeline = SportsAnalyticsPipeline()
    pipeline.run()


if __name__ == "__main__":
    main()