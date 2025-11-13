# backtesting.py
"""
Sistema de backtesting para validação de estratégias.
Inclui Kelly Criterion, cálculo de EV e gestão de banca.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import config

logger = logging.getLogger(__name__)


class KellyCriterion:
    """Implementação do Kelly Criterion para gestão de banca."""
    
    @staticmethod
    def calculate_kelly_stake(probability: float, odds: float, fraction: float = 0.25) -> float:
        """
        Calcula tamanho ideal da aposta usando Kelly Criterion.
        
        Args:
            probability: Probabilidade estimada de ganhar (0 a 1)
            odds: Odd oferecida pela casa (formato decimal)
            fraction: Fração do Kelly a usar (default 1/4 para conservador)
        
        Returns:
            Fração da banca a apostar (0 a 1)
        """
        if probability <= 0 or odds <= 1:
            return 0.0
        
        # Fórmula: f = (bp - q) / b
        # onde: b = odds - 1, p = probability, q = 1 - p
        b = odds - 1
        q = 1 - probability
        
        kelly = (b * probability - q) / b
        
        # Aplicar fração conservadora
        kelly_fractional = kelly * fraction
        
        # Nunca apostar mais que o máximo configurado
        max_stake = config.BACKTESTING_CONFIG['max_stake_percentage']
        
        return max(0, min(kelly_fractional, max_stake))
    
    @staticmethod
    def calculate_expected_value(probability: float, odds: float, stake: float = 1.0) -> float:
        """
        Calcula Expected Value (EV) de uma aposta.
        
        Args:
            probability: Probabilidade estimada
            odds: Odd oferecida
            stake: Valor apostado
        
        Returns:
            Expected Value (positivo = +EV, negativo = -EV)
        """
        win_amount = stake * (odds - 1)
        loss_amount = -stake
        
        ev = (probability * win_amount) + ((1 - probability) * loss_amount)
        
        return ev
    
    @staticmethod
    def calculate_roi(ev: float, stake: float) -> float:
        """
        Calcula ROI (Return on Investment) percentual.
        
        Args:
            ev: Expected Value
            stake: Valor apostado
        
        Returns:
            ROI em percentual
        """
        if stake == 0:
            return 0.0
        
        return (ev / stake) * 100


class BacktestingEngine:
    """Motor de backtesting para validação de estratégias."""
    
    def __init__(self, initial_bankroll: float = None):
        self.initial_bankroll = initial_bankroll or config.BACKTESTING_CONFIG['initial_bankroll']
        self.current_bankroll = self.initial_bankroll
        self.bet_history = []
        self.kelly = KellyCriterion()
    
    def reset(self):
        """Reseta o estado do backtesting."""
        self.current_bankroll = self.initial_bankroll
        self.bet_history = []
        logger.info(f"Backtesting resetado. Banca inicial: R$ {self.initial_bankroll:.2f}")
    
    def evaluate_bet_opportunity(self, model_probability: float, market_odds: float,
                                 bet_type: str = 'generic') -> Dict[str, any]:
        """
        Avalia se uma oportunidade de aposta tem valor.
        
        Args:
            model_probability: Probabilidade do nosso modelo
            market_odds: Odd oferecida pelo mercado
            bet_type: Tipo de aposta (para logging)
        
        Returns:
            Dicionário com análise da oportunidade
        """
        # Calcular probabilidade implícita do mercado
        market_probability = 1 / market_odds
        
        # Calcular edge (vantagem)
        edge = model_probability - market_probability
        edge_percentage = edge * 100
        
        # Calcular EV
        ev = self.kelly.calculate_expected_value(model_probability, market_odds, stake=100)
        roi = self.kelly.calculate_roi(ev, stake=100)
        
        # Calcular Kelly stake
        kelly_stake_fraction = self.kelly.calculate_kelly_stake(
            model_probability, 
            market_odds,
            config.BACKTESTING_CONFIG['kelly_fraction']
        )
        
        # Decidir se é uma boa aposta
        min_edge = config.BACKTESTING_CONFIG['min_edge_threshold']
        is_value_bet = edge >= min_edge and ev > 0
        
        # Stake recomendado em valor absoluto
        recommended_stake = self.current_bankroll * kelly_stake_fraction if is_value_bet else 0
        
        analysis = {
            'bet_type': bet_type,
            'model_probability': model_probability,
            'market_odds': market_odds,
            'market_probability': market_probability,
            'edge': edge,
            'edge_percentage': edge_percentage,
            'expected_value': ev,
            'roi_percentage': roi,
            'kelly_stake_fraction': kelly_stake_fraction,
            'recommended_stake': recommended_stake,
            'is_value_bet': is_value_bet,
            'current_bankroll': self.current_bankroll,
            'confidence_level': self._get_confidence_level(model_probability)
        }
        
        return analysis
    
    def _get_confidence_level(self, probability: float) -> str:
        """Retorna nível de confiança baseado em thresholds."""
        thresholds = config.CONFIDENCE_THRESHOLDS
        
        if probability >= thresholds['high']:
            return 'HIGH'
        elif probability >= thresholds['medium']:
            return 'MEDIUM'
        elif probability >= thresholds['low']:
            return 'LOW'
        else:
            return 'VERY_LOW'
    
    def place_bet(self, bet_analysis: Dict, actual_result: bool = None,
                 match_info: Dict = None) -> Dict:
        """
        Registra uma aposta e atualiza a banca (se resultado disponível).
        
        Args:
            bet_analysis: Análise retornada por evaluate_bet_opportunity
            actual_result: True se ganhou, False se perdeu, None se pendente
            match_info: Informações adicionais da partida
        
        Returns:
            Registro da aposta com resultado
        """
        stake = bet_analysis['recommended_stake']
        
        if stake <= 0:
            logger.warning("Stake recomendado é 0, aposta não registrada")
            return {}
        
        bet_record = {
            'timestamp': datetime.now(),
            'bet_type': bet_analysis['bet_type'],
            'stake': stake,
            'odds': bet_analysis['market_odds'],
            'model_probability': bet_analysis['model_probability'],
            'edge': bet_analysis['edge'],
            'expected_value': bet_analysis['expected_value'],
            'bankroll_before': self.current_bankroll,
            'actual_result': actual_result,
            'match_info': match_info or {}
        }
        
        # Se resultado disponível, calcular P&L
        if actual_result is not None:
            if actual_result:  # Ganhou
                profit = stake * (bet_analysis['market_odds'] - 1)
                profit -= profit * config.BACKTESTING_CONFIG['commission']  # Comissão
                bet_record['profit_loss'] = profit
                self.current_bankroll += profit
                bet_record['result_type'] = 'WIN'
            else:  # Perdeu
                bet_record['profit_loss'] = -stake
                self.current_bankroll -= stake
                bet_record['result_type'] = 'LOSS'
            
            bet_record['bankroll_after'] = self.current_bankroll
            bet_record['roi_realized'] = (bet_record['profit_loss'] / stake) * 100
        
        self.bet_history.append(bet_record)
        
        logger.info(f"Aposta registrada: {bet_record['bet_type']} | "
                   f"Stake: R$ {stake:.2f} | "
                   f"Odds: {bet_analysis['market_odds']:.2f} | "
                   f"EV: R$ {bet_analysis['expected_value']:.2f}")
        
        return bet_record
    
    def get_performance_summary(self) -> Dict[str, any]:
        """
        Gera sumário de performance do backtesting.
        
        Returns:
            Dicionário com métricas de performance
        """
        if not self.bet_history:
            return {'status': 'no_bets', 'message': 'Nenhuma aposta registrada'}
        
        df = pd.DataFrame(self.bet_history)
        
        # Filtrar apostas resolvidas
        resolved = df[df['actual_result'].notna()]
        
        if resolved.empty:
            return {'status': 'no_resolved_bets', 'message': 'Nenhuma aposta resolvida'}
        
        # Métricas básicas
        total_bets = len(resolved)
        wins = len(resolved[resolved['result_type'] == 'WIN'])
        losses = len(resolved[resolved['result_type'] == 'LOSS'])
        win_rate = (wins / total_bets) * 100
        
        # Financeiro
        total_staked = resolved['stake'].sum()
        total_profit = resolved['profit_loss'].sum()
        roi = (total_profit / total_staked) * 100
        
        # Banca
        bankroll_change = self.current_bankroll - self.initial_bankroll
        bankroll_change_pct = (bankroll_change / self.initial_bankroll) * 100
        
        # EV vs Reality
        expected_profit = resolved['expected_value'].sum()
        ev_accuracy = (total_profit / expected_profit) * 100 if expected_profit != 0 else 0
        
        # Winning/Losing streaks
        streaks = self._calculate_streaks(resolved)
        
        # Por tipo de aposta
        performance_by_type = {}
        for bet_type in resolved['bet_type'].unique():
            type_data = resolved[resolved['bet_type'] == bet_type]
            performance_by_type[bet_type] = {
                'bets': len(type_data),
                'wins': len(type_data[type_data['result_type'] == 'WIN']),
                'win_rate': (len(type_data[type_data['result_type'] == 'WIN']) / len(type_data)) * 100,
                'total_profit': type_data['profit_loss'].sum(),
                'roi': (type_data['profit_loss'].sum() / type_data['stake'].sum()) * 100
            }
        
        summary = {
            'status': 'success',
            'overview': {
                'total_bets': total_bets,
                'wins': wins,
                'losses': losses,
                'win_rate': win_rate,
                'total_staked': total_staked,
                'total_profit': total_profit,
                'roi': roi
            },
            'bankroll': {
                'initial': self.initial_bankroll,
                'current': self.current_bankroll,
                'change': bankroll_change,
                'change_percentage': bankroll_change_pct
            },
            'model_accuracy': {
                'expected_profit': expected_profit,
                'actual_profit': total_profit,
                'ev_accuracy': ev_accuracy
            },
            'streaks': streaks,
            'performance_by_type': performance_by_type,
            'avg_stake': resolved['stake'].mean(),
            'avg_odds': resolved['odds'].mean(),
            'avg_edge': resolved['edge'].mean() * 100
        }
        
        return summary
    
    def _calculate_streaks(self, df: pd.DataFrame) -> Dict[str, int]:
        """Calcula sequências de vitórias/derrotas."""
        results = df['result_type'].tolist()
        
        current_streak = 1
        max_win_streak = 0
        max_loss_streak = 0
        current_win_streak = 0
        current_loss_streak = 0
        
        for i in range(len(results)):
            if i > 0:
                if results[i] == results[i-1]:
                    current_streak += 1
                else:
                    current_streak = 1
            
            if results[i] == 'WIN':
                current_win_streak = current_streak
                max_win_streak = max(max_win_streak, current_win_streak)
                current_loss_streak = 0
            else:
                current_loss_streak = current_streak
                max_loss_streak = max(max_loss_streak, current_loss_streak)
                current_win_streak = 0
        
        return {
            'max_winning_streak': max_win_streak,
            'max_losing_streak': max_loss_streak,
            'current_streak_type': results[-1] if results else None,
            'current_streak_length': current_streak
        }
    
    def export_to_dataframe(self) -> pd.DataFrame:
        """Exporta histórico de apostas para DataFrame."""
        if not self.bet_history:
            return pd.DataFrame()
        
        return pd.DataFrame(self.bet_history)
    
    def generate_report(self, filepath: str = None):
        """
        Gera relatório detalhado de backtesting.
        
        Args:
            filepath: Caminho para salvar o relatório (opcional)
        """
        summary = self.get_performance_summary()
        
        if summary['status'] != 'success':
            logger.warning(f"Não foi possível gerar relatório: {summary.get('message')}")
            return
        
        # Gerar relatório em texto
        report_lines = [
            "=" * 80,
            "RELATÓRIO DE BACKTESTING",
            "=" * 80,
            "",
            "OVERVIEW:",
            f"  Total de Apostas: {summary['overview']['total_bets']}",
            f"  Vitórias: {summary['overview']['wins']}",
            f"  Derrotas: {summary['overview']['losses']}",
            f"  Win Rate: {summary['overview']['win_rate']:.2f}%",
            f"  Total Investido: R$ {summary['overview']['total_staked']:.2f}",
            f"  Lucro Total: R$ {summary['overview']['total_profit']:.2f}",
            f"  ROI: {summary['overview']['roi']:.2f}%",
            "",
            "BANCA:",
            f"  Inicial: R$ {summary['bankroll']['initial']:.2f}",
            f"  Atual: R$ {summary['bankroll']['current']:.2f}",
            f"  Mudança: R$ {summary['bankroll']['change']:.2f} ({summary['bankroll']['change_percentage']:.2f}%)",
            "",
            "ACURÁCIA DO MODELO:",
            f"  Lucro Esperado (EV): R$ {summary['model_accuracy']['expected_profit']:.2f}",
            f"  Lucro Real: R$ {summary['model_accuracy']['actual_profit']:.2f}",
            f"  Acurácia EV: {summary['model_accuracy']['ev_accuracy']:.2f}%",
            "",
            "SEQUÊNCIAS:",
            f"  Maior Sequência de Vitórias: {summary['streaks']['max_winning_streak']}",
            f"  Maior Sequência de Derrotas: {summary['streaks']['max_losing_streak']}",
            "",
            "PERFORMANCE POR TIPO:",
        ]
        
        for bet_type, perf in summary['performance_by_type'].items():
            report_lines.extend([
                f"  {bet_type}:",
                f"    Apostas: {perf['bets']}",
                f"    Win Rate: {perf['win_rate']:.2f}%",
                f"    Lucro: R$ {perf['total_profit']:.2f}",
                f"    ROI: {perf['roi']:.2f}%",
                ""
            ])
        
        report_lines.append("=" * 80)
        
        report_text = "\n".join(report_lines)
        
        print(report_text)
        
        if filepath:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(report_text)
            logger.info(f"Relatório salvo em: {filepath}")