# analysis_engine.py
"""
Motor central de análise que orquestra todos os modelos estatísticos.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
import config
from statistical_models import (
    PoissonModel, NegativeBinomialModel, 
    BivariatePoissonModel, EloRatingSystem
)
from database import DatabaseManager

logger = logging.getLogger(__name__)


class AnalysisEngine:
    """Motor principal de análise e predição."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db = db_manager
        self.elo_system = EloRatingSystem()
        self.poisson_model = PoissonModel()
        self.nbinom_model = NegativeBinomialModel()
    
    def apply_temporal_decay(self, df: pd.DataFrame, date_column: str = 'match_date') -> pd.DataFrame:
        """
        Aplica peso exponencial decrescente para jogos mais antigos.
        
        Args:
            df: DataFrame com dados históricos
            date_column: Nome da coluna de data
        
        Returns:
            DataFrame com coluna 'weight' adicionada
        """
        if date_column not in df.columns:
            logger.warning(f"Coluna {date_column} não encontrada, sem decay temporal")
            df['weight'] = 1.0
            return df
        
        df = df.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        df = df.sort_values(date_column)
        
        # Calcular peso exponencial
        decay_rate = config.TEMPORAL_CONFIG['weight_decay']
        n_games = len(df)
        
        weights = np.array([decay_rate ** (n_games - i - 1) for i in range(n_games)])
        weights = weights / weights.sum()  # Normalizar
        
        df['weight'] = weights
        
        logger.debug(f"Decay temporal aplicado: {n_games} jogos, "
                    f"peso recente={weights[-1]:.3f}, peso antigo={weights[0]:.3f}")
        
        return df
    
    def calculate_weighted_average(self, values: List[float], weights: List[float]) -> float:
        """Calcula média ponderada."""
        if len(values) != len(weights):
            raise ValueError("values e weights devem ter o mesmo tamanho")
        
        return np.average(values, weights=weights)
    
    def analyze_team_form(self, team_id: int, n_games: int = 5) -> Dict[str, float]:
        """
        Analisa forma recente de um time.
        
        Args:
            team_id: ID do time no banco
            n_games: Número de jogos a considerar
        
        Returns:
            Dicionário com métricas de forma
        """
        df = self.db.get_team_stats(team_id, n_games)
        
        if df.empty:
            logger.warning(f"Sem dados para time_id={team_id}")
            return {}
        
        df = self.apply_temporal_decay(df)
        
        # Calcular métricas ponderadas
        form = {
            'avg_goals_scored': self.calculate_weighted_average(
                df['home_goals' if df.iloc[0]['is_home'] else 'away_goals'].tolist(),
                df['weight'].tolist()
            ),
            'avg_goals_conceded': self.calculate_weighted_average(
                df['away_goals' if df.iloc[0]['is_home'] else 'home_goals'].tolist(),
                df['weight'].tolist()
            ),
            'avg_corners': self.calculate_weighted_average(
                df['corners'].tolist(),
                df['weight'].tolist()
            ),
            'avg_fouls': self.calculate_weighted_average(
                df['fouls_committed'].tolist(),
                df['weight'].tolist()
            ),
            'avg_possession': self.calculate_weighted_average(
                df['possession'].tolist(),
                df['weight'].tolist()
            )
        }
        
        # Pontos conquistados (vitórias = 3, empates = 1)
        results = []
        for _, row in df.iterrows():
            if row['is_home']:
                if row['home_goals'] > row['away_goals']:
                    results.append(3)
                elif row['home_goals'] == row['away_goals']:
                    results.append(1)
                else:
                    results.append(0)
            else:
                if row['away_goals'] > row['home_goals']:
                    results.append(3)
                elif row['away_goals'] == row['home_goals']:
                    results.append(1)
                else:
                    results.append(0)
        
        form['avg_points'] = self.calculate_weighted_average(results, df['weight'].tolist())
        form['form_rating'] = form['avg_points'] / 3  # 0 a 1
        
        return form
    
    def analyze_h2h(self, team1_id: int, team2_id: int, limit: int = 10) -> Dict[str, any]:
        """
        Analisa histórico de confrontos diretos.
        
        Args:
            team1_id: ID do primeiro time
            team2_id: ID do segundo time
            limit: Número de jogos a analisar
        
        Returns:
            Dicionário com análise H2H
        """
        df = self.db.get_h2h_history(team1_id, team2_id, limit)
        
        if df.empty:
            return {'has_history': False}
        
        df = self.apply_temporal_decay(df)
        
        # Resultados do time1
        team1_wins = 0
        team1_goals = []
        team2_goals = []
        
        for _, row in df.iterrows():
            if row['home_team_id'] == team1_id:
                team1_goals.append(row['home_goals'])
                team2_goals.append(row['away_goals'])
                if row['home_goals'] > row['away_goals']:
                    team1_wins += 1
            else:
                team1_goals.append(row['away_goals'])
                team2_goals.append(row['home_goals'])
                if row['away_goals'] > row['home_goals']:
                    team1_wins += 1
        
        return {
            'has_history': True,
            'matches_played': len(df),
            'team1_win_rate': team1_wins / len(df),
            'avg_team1_goals': self.calculate_weighted_average(team1_goals, df['weight'].tolist()),
            'avg_team2_goals': self.calculate_weighted_average(team2_goals, df['weight'].tolist()),
            'avg_total_goals': self.calculate_weighted_average(
                [g1 + g2 for g1, g2 in zip(team1_goals, team2_goals)],
                df['weight'].tolist()
            )
        }
    
    def calculate_opponent_strength(self, team_id: int, n_games: int = 10) -> float:
        """
        Calcula força média dos adversários enfrentados.
        
        Args:
            team_id: ID do time
            n_games: Número de jogos a considerar
        
        Returns:
            Rating médio dos oponentes
        """
        df = self.db.get_team_stats(team_id, n_games)
        
        if df.empty:
            return config.ELO_CONFIG['initial_rating']
        
        opponent_ratings = []
        
        for _, row in df.iterrows():
            opponent_id = row['away_team_id'] if row['is_home'] else row['home_team_id']
            
            # Buscar rating do oponente
            query = "SELECT elo_rating FROM teams WHERE team_id = ?"
            result = self.db.query_to_dataframe(query, (opponent_id,))
            
            if not result.empty:
                opponent_ratings.append(result.iloc[0]['elo_rating'])
        
        return np.mean(opponent_ratings) if opponent_ratings else config.ELO_CONFIG['initial_rating']
    
    def predict_fouls(self, home_team_id: int, away_team_id: int, 
                     referee_name: Optional[str] = None) -> Dict[str, any]:
        """
        Prediz estatísticas de faltas para uma partida.
        
        Args:
            home_team_id: ID do time mandante
            away_team_id: ID do time visitante
            referee_name: Nome do árbitro (opcional)
        
        Returns:
            Dicionário com predições de faltas
        """
        # Forma recente dos times
        home_form = self.analyze_team_form(home_team_id)
        away_form = self.analyze_team_form(away_team_id)
        
        # Média de faltas
        home_fouls_avg = home_form.get('avg_fouls', 12.0)
        away_fouls_avg = away_form.get('avg_fouls', 12.0)
        
        # Ajuste por árbitro
        if referee_name:
            query = """
                SELECT avg_fouls_per_match 
                FROM referees 
                WHERE referee_name = ?
            """
            ref_data = self.db.query_to_dataframe(query, (referee_name,))
            
            if not ref_data.empty:
                ref_avg = ref_data.iloc[0]['avg_fouls_per_match']
                # Ajustar média considerando tendência do árbitro
                league_avg = 22.0  # Média típica
                adjustment_factor = ref_avg / league_avg
                home_fouls_avg *= adjustment_factor
                away_fouls_avg *= adjustment_factor
        
        # Calcular probabilidades usando Poisson
        home_probs = self.poisson_model.calculate_probabilities(home_fouls_avg)
        away_probs = self.poisson_model.calculate_probabilities(away_fouls_avg)
        
        return {
            'home_team': {
                'expected_fouls': home_fouls_avg,
                'probabilities': home_probs
            },
            'away_team': {
                'expected_fouls': away_fouls_avg,
                'probabilities': away_probs
            },
            'total_expected_fouls': home_fouls_avg + away_fouls_avg,
            'referee_adjusted': referee_name is not None
        }
    
    def predict_corners(self, home_team_id: int, away_team_id: int) -> Dict[str, any]:
        """
        Prediz estatísticas de escanteios usando Negative Binomial.
        
        Args:
            home_team_id: ID do time mandante
            away_team_id: ID do time visitante
        
        Returns:
            Dicionário com predições de escanteios
        """
        # Buscar dados históricos
        home_data = self.db.get_team_stats(home_team_id, n_games=10)
        away_data = self.db.get_team_stats(away_team_id, n_games=10)
        
        # Separar por mandante/visitante
        home_corners_home = home_data[home_data['is_home']]['corners'].tolist()
        away_corners_away = away_data[~away_data['is_home']]['corners'].tolist()
        
        # Fit Negative Binomial (melhor para escanteios)
        if home_corners_home:
            r_home, p_home = self.nbinom_model.fit_parameters(home_corners_home)
            home_probs = self.nbinom_model.calculate_probabilities(r_home, p_home)
        else:
            home_probs = {}
        
        if away_corners_away:
            r_away, p_away = self.nbinom_model.fit_parameters(away_corners_away)
            away_probs = self.nbinom_model.calculate_probabilities(r_away, p_away)
        else:
            away_probs = {}
        
        # Total da partida (simplificado - assumes independence)
        total_corners_expected = np.mean(home_corners_home) + np.mean(away_corners_away)
        
        return {
            'home_team_corners': {
                'expected': np.mean(home_corners_home) if home_corners_home else 5.0,
                'probabilities': home_probs
            },
            'away_team_corners': {
                'expected': np.mean(away_corners_away) if away_corners_away else 5.0,
                'probabilities': away_probs
            },
            'match_total_expected': total_corners_expected
        }
    
    def predict_goals(self, home_team_id: int, away_team_id: int) -> Dict[str, any]:
        """
        Prediz gols usando modelo Poisson Bivariado.
        
        Args:
            home_team_id: ID do time mandante
            away_team_id: ID do time visitante
        
        Returns:
            Dicionário com predições completas
        """
        # Forma recente
        home_form = self.analyze_team_form(home_team_id)
        away_form = self.analyze_team_form(away_team_id)
        
        # H2H
        h2h = self.analyze_h2h(home_team_id, away_team_id)
        
        # Força do calendário
        home_opp_strength = self.calculate_opponent_strength(home_team_id)
        away_opp_strength = self.calculate_opponent_strength(away_team_id)
        
        # Ajustar expectativa por força dos oponentes
        league_avg_rating = 1500
        home_goals_expected = home_form.get('avg_goals_scored', 1.5)
        home_goals_expected *= (league_avg_rating / home_opp_strength)
        
        away_goals_expected = away_form.get('avg_goals_scored', 1.5)
        away_goals_expected *= (league_avg_rating / away_opp_strength)
        
        # Considerar H2H se houver histórico
        if h2h.get('has_history'):
            weight_h2h = 0.3  # 30% de peso para H2H
            home_goals_expected = (
                weight_h2h * h2h['avg_team1_goals'] +
                (1 - weight_h2h) * home_goals_expected
            )
            away_goals_expected = (
                weight_h2h * h2h['avg_team2_goals'] +
                (1 - weight_h2h) * away_goals_expected
            )
        
        # Aplicar vantagem de mandante
        home_advantage = 1.2
        home_goals_expected *= home_advantage
        
        # Usar modelo Poisson Bivariado
        biv_model = BivariatePoissonModel()
        
        # Simular dados para fit (usar forma recente)
        home_hist = self.db.get_team_stats(home_team_id, 10)
        home_goals_list = home_hist['home_goals' if home_hist.iloc[0]['is_home'] else 'away_goals'].tolist()
        away_goals_list = home_hist['away_goals' if home_hist.iloc[0]['is_home'] else 'home_goals'].tolist()
        
        # Ajustar com médias calculadas
        lambda_home = home_goals_expected
        lambda_away = away_goals_expected
        
        biv_model.lambda_home = lambda_home
        biv_model.lambda_away = lambda_away
        biv_model.rho = 0.0  # Simplificado
        
        match_probs = biv_model.calculate_match_probabilities()
        
        return {
            'expected_goals': {
                'home': lambda_home,
                'away': lambda_away,
                'total': lambda_home + lambda_away
            },
            'probabilities': match_probs,
            'context': {
                'home_form_rating': home_form.get('form_rating', 0.5),
                'away_form_rating': away_form.get('form_rating', 0.5),
                'h2h_considered': h2h.get('has_history', False),
                'opponent_strength_adjusted': True
            }
        }