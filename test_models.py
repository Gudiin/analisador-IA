# test_models.py
"""
Testes unitários para validação dos modelos estatísticos.
"""

import pytest
import numpy as np
from statistical_models import (
    PoissonModel, 
    NegativeBinomialModel, 
    BivariatePoissonModel,
    EloRatingSystem
)
from backtesting import KellyCriterion, BacktestingEngine
from utils import OddsConverter, DataValidator


class TestPoissonModel:
    """Testes para modelo Poisson."""
    
    def test_calculate_probabilities(self):
        """Testa cálculo de probabilidades."""
        probs = PoissonModel.calculate_probabilities(lambda_param=2.0, max_events=5)
        
        # Verificar que probabilidades somam próximo a 1
        total_prob = sum([v for k, v in probs.items() if k.startswith('P(exactly')])
        assert 0.95 <= total_prob <= 1.0
        
        # Verificar que odds são positivas
        assert probs['odds_exactly_2'] > 0
        
        # Verificar over/under são complementares
        prob_over_2_5 = probs['P(over_2.5)']
        prob_under_2_5 = probs['P(under_2.5)']
        assert abs(prob_over_2_5 + prob_under_2_5 - 1.0) < 0.01
    
    def test_fit_lambda(self):
        """Testa estimativa de lambda."""
        data = [1, 2, 2, 3, 2, 1, 3, 2]
        lambda_est = PoissonModel.fit_lambda(data)
        
        assert lambda_est == np.mean(data)
        assert 1.5 <= lambda_est <= 2.5


class TestNegativeBinomialModel:
    """Testes para modelo Binomial Negativo."""
    
    def test_fit_parameters(self):
        """Testa estimativa de parâmetros."""
        # Dados com sobredispersão
        data = [8, 12, 10, 15, 9, 11, 20, 8, 13, 10]
        r, p = NegativeBinomialModel.fit_parameters(data)
        
        assert r > 0
        assert 0 < p <= 1
    
    def test_handles_low_variance(self):
        """Testa comportamento com baixa variância."""
        # Dados com variância baixa (Poisson seria melhor)
        data = [2, 2, 2, 2, 2, 2]
        r, p = NegativeBinomialModel.fit_parameters(data)
        
        # Deve retornar parâmetros válidos mesmo assim
        assert r > 0
        assert p > 0


class TestBivariatePoissonModel:
    """Testes para modelo Poisson Bivariado."""
    
    def test_fit_model(self):
        """Testa ajuste do modelo."""
        home_goals = [2, 1, 3, 2, 1]
        away_goals = [1, 0, 2, 1, 1]
        
        model = BivariatePoissonModel()
        lambda_h, lambda_a, rho = model.fit(home_goals, away_goals)
        
        assert lambda_h > 0
        assert lambda_a > 0
        assert -0.2 <= rho <= 0.2  # Correlação limitada
    
    def test_calculate_probabilities(self):
        """Testa cálculo de probabilidades de partida."""
        model = BivariatePoissonModel()
        model.lambda_home = 1.8
        model.lambda_away = 1.2
        model.rho = 0.0
        
        probs = model.calculate_probabilities()
        
        # Verificar que probabilidades de resultado somam 1
        total = probs['home_win'] + probs['draw'] + probs['away_win']
        assert abs(total - 1.0) < 0.01
        
        # Verificar que tem placar mais provável
        assert 'most_likely_score' in probs
        assert probs['most_likely_score_prob'] > 0


class TestEloRatingSystem:
    """Testes para sistema Elo."""
    
    def test_initial_rating(self):
        """Testa rating inicial."""
        elo = EloRatingSystem()
        assert elo.get_rating('Team A') == 1500
    
    def test_expected_result(self):
        """Testa cálculo de resultado esperado."""
        elo = EloRatingSystem()
        
        # Times iguais
        expected = elo.expected_result(1500, 1500)
        assert abs(expected - 0.5) < 0.01
        
        # Time A muito melhor
        expected = elo.expected_result(1800, 1200)
        assert expected > 0.9
    
    def test_update_ratings(self):
        """Testa atualização de ratings."""
        elo = EloRatingSystem()
        
        # Time A vence
        elo.update_ratings('Team A', 'Team B', 2, 1)
        
        rating_a = elo.get_rating('Team A')
        rating_b = elo.get_rating('Team B')
        
        # A deve subir, B deve cair
        assert rating_a > 1500
        assert rating_b < 1500
        
        # Ratings devem ser conservados (soma constante em sistema fechado)
        assert abs((rating_a + rating_b) - 3000) < 50  # Tolerância por arredondamento


class TestKellyCriterion:
    """Testes para Kelly Criterion."""
    
    def test_kelly_stake(self):
        """Testa cálculo de stake Kelly."""
        kelly = KellyCriterion()
        
        # Value bet clara
        stake = kelly.calculate_kelly_stake(probability=0.6, odds=2.0)
        assert stake > 0
        assert stake <= 0.25  # Máximo do Kelly fracionário
        
        # Sem edge
        stake = kelly.calculate_kelly_stake(probability=0.5, odds=2.0)
        assert stake == 0
        
        # Negative expectation
        stake = kelly.calculate_kelly_stake(probability=0.3, odds=2.0)
        assert stake == 0
    
    def test_expected_value(self):
        """Testa cálculo de EV."""
        kelly = KellyCriterion()
        
        # +EV
        ev = kelly.calculate_expected_value(probability=0.6, odds=2.0, stake=100)
        assert ev > 0
        
        # -EV
        ev = kelly.calculate_expected_value(probability=0.4, odds=2.0, stake=100)
        assert ev < 0
        
        # Breakeven
        ev = kelly.calculate_expected_value(probability=0.5, odds=2.0, stake=100)
        assert abs(ev) < 0.01


class TestBacktestingEngine:
    """Testes para motor de backtesting."""
    
    def test_initialization(self):
        """Testa inicialização."""
        backtester = BacktestingEngine(initial_bankroll=1000)
        assert backtester.current_bankroll == 1000
        assert len(backtester.bet_history) == 0
    
    def test_evaluate_bet_opportunity(self):
        """Testa avaliação de oportunidade."""
        backtester = BacktestingEngine()
        
        # Value bet
        analysis = backtester.evaluate_bet_opportunity(
            model_probability=0.65,
            market_odds=2.0,
            bet_type='Test Bet'
        )
        
        assert analysis['is_value_bet'] == True
        assert analysis['edge'] > 0
        assert analysis['recommended_stake'] > 0
        
        # Não value bet
        analysis = backtester.evaluate_bet_opportunity(
            model_probability=0.45,
            market_odds=2.0,
            bet_type='Test Bet'
        )
        
        assert analysis['is_value_bet'] == False
    
    def test_place_bet_win(self):
        """Testa registro de aposta vencedora."""
        backtester = BacktestingEngine(initial_bankroll=1000)
        
        bet_analysis = {
            'bet_type': 'Test',
            'recommended_stake': 50,
            'market_odds': 2.0,
            'model_probability': 0.6,
            'edge': 0.1,
            'expected_value': 5
        }
        
        initial_bankroll = backtester.current_bankroll
        
        bet = backtester.place_bet(bet_analysis, actual_result=True)
        
        # Banca deve aumentar
        assert backtester.current_bankroll > initial_bankroll
        assert bet['result_type'] == 'WIN'
        assert bet['profit_loss'] > 0
    
    def test_place_bet_loss(self):
        """Testa registro de aposta perdida."""
        backtester = BacktestingEngine(initial_bankroll=1000)
        
        bet_analysis = {
            'bet_type': 'Test',
            'recommended_stake': 50,
            'market_odds': 2.0,
            'model_probability': 0.6,
            'edge': 0.1,
            'expected_value': 5
        }
        
        initial_bankroll = backtester.current_bankroll
        
        bet = backtester.place_bet(bet_analysis, actual_result=False)
        
        # Banca deve diminuir
        assert backtester.current_bankroll < initial_bankroll
        assert bet['result_type'] == 'LOSS'
        assert bet['profit_loss'] < 0


class TestOddsConverter:
    """Testes para conversor de odds."""
    
    def test_decimal_to_probability(self):
        """Testa conversão decimal -> probabilidade."""
        prob = OddsConverter.decimal_to_probability(2.0)
        assert abs(prob - 0.5) < 0.01
        
        prob = OddsConverter.decimal_to_probability(3.0)
        assert abs(prob - 0.333) < 0.01
    
    def test_probability_to_decimal(self):
        """Testa conversão probabilidade -> decimal."""
        odds = OddsConverter.probability_to_decimal(0.5, add_margin=False)
        assert abs(odds - 2.0) < 0.01
        
        odds = OddsConverter.probability_to_decimal(0.25, add_margin=False)
        assert abs(odds - 4.0) < 0.01
    
    def test_decimal_to_american(self):
        """Testa conversão decimal -> americanas."""
        american = OddsConverter.decimal_to_american(2.0)
        assert american == 100
        
        american = OddsConverter.decimal_to_american(1.5)
        assert american == -200


class TestDataValidator:
    """Testes para validador de dados."""
    
    def test_validate_probability(self):
        """Testa validação de probabilidade."""
        assert DataValidator.validate_probability(0.5) == True
        assert DataValidator.validate_probability(0.0) == True
        assert DataValidator.validate_probability(1.0) == True
        assert DataValidator.validate_probability(-0.1) == False
        assert DataValidator.validate_probability(1.1) == False
    
    def test_validate_odds(self):
        """Testa validação de odds."""
        assert DataValidator.validate_odds(2.0) == True
        assert DataValidator.validate_odds(1.5) == True
        assert DataValidator.validate_odds(1.0) == False
        assert DataValidator.validate_odds(0.5) == False


# Fixture para executar todos os testes
@pytest.fixture
def sample_match_data():
    """Dados de exemplo para testes."""
    return {
        'home_goals': [2, 1, 3, 2, 1, 2, 0, 2, 1, 3],
        'away_goals': [1, 0, 2, 1, 1, 0, 0, 1, 2, 1],
        'home_corners': [8, 10, 12, 9, 7, 11, 6, 9, 10, 8],
        'away_corners': [5, 6, 8, 7, 4, 6, 5, 7, 9, 6],
        'home_fouls': [12, 14, 10, 13, 15, 11, 16, 12, 13, 14],
        'away_fouls': [10, 12, 11, 10, 13, 9, 14, 11, 10, 12]
    }


if __name__ == "__main__":
    # Executar testes
    pytest.main([__file__, '-v', '--tb=short'])