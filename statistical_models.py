# statistical_models.py
"""
Modelos estatísticos avançados para predição de eventos esportivos.
Inclui: Poisson, Poisson Bivariado, Negative Binomial, Dixon-Coles.
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson, nbinom
from scipy.optimize import minimize
import logging
from typing import Dict, Tuple, List, Optional
import config

logger = logging.getLogger(__name__)


class PoissonModel:
    """Modelo Poisson simples para eventos independentes."""
    
    @staticmethod
    def calculate_probabilities(lambda_param: float, max_events: int = 10) -> Dict[str, float]:
        """
        Calcula probabilidades Poisson.
        
        Args:
            lambda_param: Taxa média de eventos (λ)
            max_events: Número máximo de eventos para calcular
        
        Returns:
            Dicionário com probabilidades e odds
        """
        probabilities = {}
        
        for k in range(max_events + 1):
            prob = poisson.pmf(k, lambda_param)
            probabilities[f'P(exactly_{k})'] = prob
            probabilities[f'odds_exactly_{k}'] = 1 / prob if prob > 0 else float('inf')
        
        # Over/Under
        for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
            prob_over = 1 - poisson.cdf(line, lambda_param)
            prob_under = poisson.cdf(line, lambda_param)
            
            probabilities[f'P(over_{line})'] = prob_over
            probabilities[f'P(under_{line})'] = prob_under
            probabilities[f'odds_over_{line}'] = 1 / prob_over if prob_over > 0 else float('inf')
            probabilities[f'odds_under_{line}'] = 1 / prob_under if prob_under > 0 else float('inf')
        
        return probabilities
    
    @staticmethod
    def fit_lambda(observed_data: List[float]) -> float:
        """
        Estima λ a partir de dados observados.
        
        Args:
            observed_data: Lista de contagens observadas
        
        Returns:
            Lambda estimado (média)
        """
        return np.mean(observed_data)


class NegativeBinomialModel:
    """
    Modelo Binomial Negativo para dados sobredispersos.
    Melhor que Poisson quando variância > média (comum em escanteios).
    """
    
    @staticmethod
    def fit_parameters(observed_data: List[float]) -> Tuple[float, float]:
        """
        Estima parâmetros r e p do Binomial Negativo.
        
        Args:
            observed_data: Lista de contagens observadas
        
        Returns:
            Tupla (r, p) - parâmetros do modelo
        """
        mean = np.mean(observed_data)
        var = np.var(observed_data)
        
        # Se variância <= média, usar Poisson
        if var <= mean:
            logger.warning("Variância <= média, considere usar Poisson")
            return mean, 1.0
        
        # Método dos momentos
        r = (mean ** 2) / (var - mean)
        p = mean / var
        
        return r, p
    
    @staticmethod
    def calculate_probabilities(r: float, p: float, max_events: int = 15) -> Dict[str, float]:
        """
        Calcula probabilidades usando Binomial Negativo.
        
        Args:
            r: Parâmetro r (número de sucessos)
            p: Parâmetro p (probabilidade de sucesso)
            max_events: Número máximo de eventos
        
        Returns:
            Dicionário com probabilidades
        """
        probabilities = {}
        
        for k in range(max_events + 1):
            prob = nbinom.pmf(k, r, p)
            probabilities[f'P(exactly_{k})'] = prob
            probabilities[f'odds_exactly_{k}'] = 1 / prob if prob > 0 else float('inf')
        
        # Over/Under para escanteios
        for line in [8.5, 9.5, 10.5, 11.5, 12.5]:
            prob_over = 1 - nbinom.cdf(line, r, p)
            prob_under = nbinom.cdf(line, r, p)
            
            probabilities[f'P(over_{line})'] = prob_over
            probabilities[f'P(under_{line})'] = prob_under
            probabilities[f'odds_over_{line}'] = 1 / prob_over if prob_over > 0 else float('inf')
            probabilities[f'odds_under_{line}'] = 1 / prob_under if prob_under > 0 else float('inf')
        
        return probabilities


class BivariatePoissonModel:
    """
    Modelo Poisson Bivariado para gols (mandante vs visitante).
    Considera correlação entre gols dos dois times.
    """
    
    def __init__(self):
        self.lambda_home = None
        self.lambda_away = None
        self.rho = 0.0  # Parâmetro de correlação
    
    def fit(self, home_goals: List[int], away_goals: List[int], 
            home_advantage: float = 1.2) -> Tuple[float, float, float]:
        """
        Estima parâmetros do modelo.
        
        Args:
            home_goals: Lista de gols marcados pelo mandante
            away_goals: Lista de gols marcados pelo visitante
            home_advantage: Fator de vantagem do mandante
        
        Returns:
            Tupla (lambda_home, lambda_away, rho)
        """
        self.lambda_home = np.mean(home_goals) * home_advantage
        self.lambda_away = np.mean(away_goals)
        
        # Estimar correlação (simplificado)
        # Em produção, usar Maximum Likelihood Estimation
        correlation = np.corrcoef(home_goals, away_goals)[0, 1]
        self.rho = max(-0.1, min(0.1, correlation))  # Limitar entre -0.1 e 0.1
        
        logger.info(f"Modelo ajustado: λ_home={self.lambda_home:.2f}, "
                   f"λ_away={self.lambda_away:.2f}, ρ={self.rho:.4f}")
        
        return self.lambda_home, self.lambda_away, self.rho
    
    def calculate_match_probabilities(self, max_goals: int = 7) -> Dict[str, float]:
        """
        Calcula probabilidades para diferentes resultados de partida.
        
        Args:
            max_goals: Número máximo de gols a considerar
        
        Returns:
            Dicionário com probabilidades de resultados
        """
        if self.lambda_home is None or self.lambda_away is None:
            raise ValueError("Modelo não foi ajustado. Execute fit() primeiro.")
        
        probabilities = {}
        score_matrix = np.zeros((max_goals + 1, max_goals + 1))
        
        # Calcular matriz de probabilidades de placares
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                # Probabilidade básica de Poisson
                prob = poisson.pmf(i, self.lambda_home) * poisson.pmf(j, self.lambda_away)
                
                # Ajuste de correlação (Dixon-Coles simplificado)
                if i == 0 and j == 0:
                    prob *= (1 - self.lambda_home * self.lambda_away * self.rho)
                elif i == 0 and j == 1:
                    prob *= (1 + self.lambda_home * self.rho)
                elif i == 1 and j == 0:
                    prob *= (1 + self.lambda_away * self.rho)
                elif i == 1 and j == 1:
                    prob *= (1 - self.rho)
                
                score_matrix[i, j] = prob
        
        # Normalizar
        score_matrix /= score_matrix.sum()
        
        # Probabilidades de resultados
        prob_home_win = np.sum(np.tril(score_matrix, -1))
        prob_draw = np.sum(np.diag(score_matrix))
        prob_away_win = np.sum(np.triu(score_matrix, 1))
        
        probabilities['home_win'] = prob_home_win
        probabilities['draw'] = prob_draw
        probabilities['away_win'] = prob_away_win
        
        # Odds
        probabilities['odds_home_win'] = 1 / prob_home_win if prob_home_win > 0 else float('inf')
        probabilities['odds_draw'] = 1 / prob_draw if prob_draw > 0 else float('inf')
        probabilities['odds_away_win'] = 1 / prob_away_win if prob_away_win > 0 else float('inf')
        
        # Over/Under gols totais
        for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
            prob_over = 0
            for i in range(max_goals + 1):
                for j in range(max_goals + 1):
                    if i + j > line:
                        prob_over += score_matrix[i, j]
            
            prob_under = 1 - prob_over
            probabilities[f'over_{line}'] = prob_over
            probabilities[f'under_{line}'] = prob_under
            probabilities[f'odds_over_{line}'] = 1 / prob_over if prob_over > 0 else float('inf')
            probabilities[f'odds_under_{line}'] = 1 / prob_under if prob_under > 0 else float('inf')
        
        # Both Teams To Score
        prob_btts_yes = 1 - score_matrix[0, :].sum() - score_matrix[:, 0].sum() + score_matrix[0, 0]
        prob_btts_no = 1 - prob_btts_yes
        probabilities['btts_yes'] = prob_btts_yes
        probabilities['btts_no'] = prob_btts_no
        probabilities['odds_btts_yes'] = 1 / prob_btts_yes if prob_btts_yes > 0 else float('inf')
        probabilities['odds_btts_no'] = 1 / prob_btts_no if prob_btts_no > 0 else float('inf')
        
        # Placar mais provável
        most_likely_idx = np.unravel_index(score_matrix.argmax(), score_matrix.shape)
        probabilities['most_likely_score'] = f"{most_likely_idx[0]}-{most_likely_idx[1]}"
        probabilities['most_likely_score_prob'] = score_matrix[most_likely_idx]
        
        return probabilities


class EloRatingSystem:
    """Sistema Elo para ranking dinâmico de times."""
    
    def __init__(self, k_factor: float = 32, home_advantage: float = 100):
        self.k_factor = k_factor
        self.home_advantage = home_advantage
        self.ratings = {}
    
    def get_rating(self, team: str) -> float:
        """Retorna rating atual do time."""
        return self.ratings.get(team, config.ELO_CONFIG['initial_rating'])
    
    def expected_result(self, rating_a: float, rating_b: float) -> float:
        """
        Calcula resultado esperado (0 a 1).
        
        Args:
            rating_a: Rating do time A
            rating_b: Rating do time B
        
        Returns:
            Probabilidade de A vencer (0.5 = empate esperado)
        """
        return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))
    
    def update_ratings(self, home_team: str, away_team: str, 
                      home_goals: int, away_goals: int):
        """
        Atualiza ratings após uma partida.
        
        Args:
            home_team: Nome do time mandante
            away_team: Nome do time visitante
            home_goals: Gols do mandante
            away_goals: Gols do visitante
        """
        # Ratings atuais
        rating_home = self.get_rating(home_team) + self.home_advantage
        rating_away = self.get_rating(away_team)
        
        # Resultado esperado
        expected_home = self.expected_result(rating_home, rating_away)
        
        # Resultado real (1 = vitória, 0.5 = empate, 0 = derrota)
        if home_goals > away_goals:
            actual_home = 1.0
        elif home_goals == away_goals:
            actual_home = 0.5
        else:
            actual_home = 0.0
        
        # Multiplicador por diferença de gols
        goal_diff = abs(home_goals - away_goals)
        multiplier = np.log(max(goal_diff, 1) + 1) * config.ELO_CONFIG['goal_difference_multiplier']
        
        # Atualizar ratings
        delta = self.k_factor * multiplier * (actual_home - expected_home)
        
        new_rating_home = self.get_rating(home_team) + delta
        new_rating_away = self.get_rating(away_team) - delta
        
        self.ratings[home_team] = new_rating_home
        self.ratings[away_team] = new_rating_away
        
        logger.debug(f"Elo atualizado: {home_team} {new_rating_home:.1f} | "
                    f"{away_team} {new_rating_away:.1f}")
    
    def predict_match(self, home_team: str, away_team: str) -> Dict[str, float]:
        """
        Prediz resultado da partida baseado em Elo.
        
        Args:
            home_team: Nome do time mandante
            away_team: Nome do time visitante
        
        Returns:
            Dicionário com probabilidades
        """
        rating_home = self.get_rating(home_team) + self.home_advantage
        rating_away = self.get_rating(away_team)
        
        prob_home = self.expected_result(rating_home, rating_away)
        prob_away = 1 - prob_home
        
        # Estimar probabilidade de empate (aproximação)
        rating_diff = abs(rating_home - rating_away)
        prob_draw = max(0.1, 0.3 - (rating_diff / 1000))
        
        # Normalizar
        total = prob_home + prob_away + prob_draw
        prob_home /= total
        prob_away /= total
        prob_draw /= total
        
        return {
            'prob_home_win': prob_home,
            'prob_draw': prob_draw,
            'prob_away_win': prob_away,
            'odds_home_win': 1 / prob_home,
            'odds_draw': 1 / prob_draw,
            'odds_away_win': 1 / prob_away,
            'rating_home': rating_home - self.home_advantage,
            'rating_away': rating_away
        }