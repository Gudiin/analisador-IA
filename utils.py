# utils.py
"""
Funções utilitárias e helpers do sistema.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional
import pandas as pd
import numpy as np
from pathlib import Path
import config

logger = logging.getLogger(__name__)


class DataValidator:
    """Validador de dados e qualidade."""
    
    @staticmethod
    def validate_dataframe(df: pd.DataFrame, required_columns: List[str]) -> bool:
        """
        Valida se DataFrame tem colunas necessárias.
        
        Args:
            df: DataFrame para validar
            required_columns: Lista de colunas obrigatórias
        
        Returns:
            True se válido, False caso contrário
        """
        if df is None or df.empty:
            logger.error("DataFrame vazio ou None")
            return False
        
        missing = set(required_columns) - set(df.columns)
        
        if missing:
            logger.error(f"Colunas faltando: {missing}")
            return False
        
        return True
    
    @staticmethod
    def validate_probability(prob: float) -> bool:
        """Valida se probabilidade está entre 0 e 1."""
        return 0 <= prob <= 1
    
    @staticmethod
    def validate_odds(odds: float) -> bool:
        """Valida se odds são válidas (> 1.0)."""
        return odds > 1.0
    
    @staticmethod
    def check_data_quality(df: pd.DataFrame, min_samples: int = 5) -> Dict[str, Any]:
        """
        Verifica qualidade dos dados.
        
        Args:
            df: DataFrame para analisar
            min_samples: Número mínimo de amostras
        
        Returns:
            Dicionário com métricas de qualidade
        """
        quality = {
            'total_rows': len(df),
            'null_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
            'duplicate_rows': df.duplicated().sum(),
            'meets_min_samples': len(df) >= min_samples
        }
        
        # Identificar colunas com muitos nulls
        high_null_cols = [
            col for col, pct in quality['null_percentage'].items() 
            if pct > 20
        ]
        
        quality['high_null_columns'] = high_null_cols
        quality['is_good_quality'] = (
            quality['meets_min_samples'] and 
            len(high_null_cols) == 0 and
            quality['duplicate_rows'] < len(df) * 0.1
        )
        
        return quality


class OddsConverter:
    """Conversor entre diferentes formatos de odds."""
    
    @staticmethod
    def decimal_to_probability(decimal_odds: float, remove_margin: bool = False) -> float:
        """
        Converte odds decimais para probabilidade implícita.
        
        Args:
            decimal_odds: Odds no formato decimal (ex: 2.50)
            remove_margin: Se deve remover margem da casa
        
        Returns:
            Probabilidade (0 a 1)
        """
        if decimal_odds <= 1:
            raise ValueError("Odds decimal deve ser > 1")
        
        prob = 1 / decimal_odds
        
        # Remover margem (simplificado - assume margem uniforme)
        if remove_margin:
            # Típico overround é 5-10%
            prob = prob * 0.95  # Ajuste conservador
        
        return min(prob, 1.0)
    
    @staticmethod
    def probability_to_decimal(probability: float, add_margin: bool = True) -> float:
        """
        Converte probabilidade para odds decimal.
        
        Args:
            probability: Probabilidade (0 a 1)
            add_margin: Se deve adicionar margem da casa
        
        Returns:
            Odds decimal
        """
        if not 0 < probability <= 1:
            raise ValueError("Probabilidade deve estar entre 0 e 1")
        
        odds = 1 / probability
        
        # Adicionar margem da casa
        if add_margin:
            odds = odds * 0.95  # Reduz odds em ~5%
        
        return odds
    
    @staticmethod
    def decimal_to_american(decimal_odds: float) -> int:
        """Converte decimal para odds americanas."""
        if decimal_odds >= 2.0:
            return int((decimal_odds - 1) * 100)
        else:
            return int(-100 / (decimal_odds - 1))
    
    @staticmethod
    def american_to_decimal(american_odds: int) -> float:
        """Converte odds americanas para decimal."""
        if american_odds > 0:
            return (american_odds / 100) + 1
        else:
            return (100 / abs(american_odds)) + 1


class StatisticalTests:
    """Testes estatísticos para validação de modelos."""
    
    @staticmethod
    def calculate_chi_square(observed: List[int], expected: List[float]) -> Dict[str, float]:
        """
        Calcula teste qui-quadrado para goodness-of-fit.
        
        Args:
            observed: Contagens observadas
            expected: Contagens esperadas
        
        Returns:
            Dicionário com estatística e p-value
        """
        from scipy.stats import chisquare
        
        statistic, pvalue = chisquare(observed, expected)
        
        return {
            'chi_square': statistic,
            'p_value': pvalue,
            'is_good_fit': pvalue > 0.05  # Não rejeita H0
        }
    
    @staticmethod
    def calculate_calibration(predicted_probs: List[float], 
                            actual_outcomes: List[int],
                            n_bins: int = 10) -> Dict[str, Any]:
        """
        Calcula calibração do modelo (reliability diagram).
        
        Args:
            predicted_probs: Probabilidades preditas
            actual_outcomes: Resultados reais (0 ou 1)
            n_bins: Número de bins
        
        Returns:
            Dicionário com métricas de calibração
        """
        df = pd.DataFrame({
            'pred': predicted_probs,
            'actual': actual_outcomes
        })
        
        # Criar bins
        df['bin'] = pd.cut(df['pred'], bins=n_bins, labels=False)
        
        # Calcular calibração por bin
        calibration = df.groupby('bin').agg({
            'pred': 'mean',
            'actual': 'mean'
        }).reset_index()
        
        # Brier Score
        brier_score = np.mean((np.array(predicted_probs) - np.array(actual_outcomes)) ** 2)
        
        return {
            'brier_score': brier_score,
            'calibration_data': calibration.to_dict('records'),
            'is_well_calibrated': brier_score < 0.25  # Threshold arbitrário
        }


class ReportGenerator:
    """Gerador de relatórios em múltiplos formatos."""
    
    @staticmethod
    def save_json(data: Dict, filename: str, directory: Path = config.REPORTS_DIR):
        """
        Salva dados em JSON formatado.
        
        Args:
            data: Dados para salvar
            filename: Nome do arquivo
            directory: Diretório de destino
        """
        filepath = directory / filename
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False, default=str)
        
        logger.info(f"JSON salvo: {filepath}")
        return filepath
    
    @staticmethod
    def save_dataframe_excel(df: pd.DataFrame, filename: str, 
                           directory: Path = config.REPORTS_DIR):
        """Salva DataFrame em Excel."""
        filepath = directory / filename
        
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Data')
        
        logger.info(f"Excel salvo: {filepath}")
        return filepath
    
    @staticmethod
    def generate_html_report(data: Dict, title: str = "Relatório de Análise") -> str:
        """
        Gera relatório HTML simples.
        
        Args:
            data: Dicionário com dados do relatório
            title: Título do relatório
        
        Returns:
            String HTML
        """
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <meta charset="utf-8">
            <title>{title}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; border-bottom: 2px solid #ddd; padding-bottom: 5px; }}
                .value-bet {{ background-color: #d4edda; padding: 10px; margin: 10px 0; border-left: 4px solid #28a745; }}
                .metric {{ display: inline-block; margin: 10px 20px; }}
                .metric-label {{ font-weight: bold; color: #666; }}
                .metric-value {{ font-size: 1.5em; color: #333; }}
            </style>
        </head>
        <body>
            <h1>{title}</h1>
            <p>Gerado em: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}</p>
        """
        
        # Adicionar seções dinamicamente
        for section, content in data.items():
            html += f"<h2>{section.replace('_', ' ').title()}</h2>"
            html += f"<pre>{json.dumps(content, indent=2, ensure_ascii=False, default=str)}</pre>"
        
        html += """
        </body>
        </html>
        """
        
        return html


class PerformanceMonitor:
    """Monitor de performance do sistema."""
    
    def __init__(self):
        self.metrics = {}
        self.start_times = {}
    
    def start_timer(self, operation: str):
        """Inicia timer para uma operação."""
        self.start_times[operation] = datetime.now()
    
    def end_timer(self, operation: str):
        """Finaliza timer e registra duração."""
        if operation not in self.start_times:
            logger.warning(f"Timer não iniciado para: {operation}")
            return
        
        duration = (datetime.now() - self.start_times[operation]).total_seconds()
        
        if operation not in self.metrics:
            self.metrics[operation] = []
        
        self.metrics[operation].append(duration)
        
        logger.debug(f"{operation}: {duration:.2f}s")
    
    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Retorna sumário de todas as métricas."""
        summary = {}
        
        for operation, durations in self.metrics.items():
            summary[operation] = {
                'count': len(durations),
                'total': sum(durations),
                'average': np.mean(durations),
                'min': min(durations),
                'max': max(durations),
                'std': np.std(durations)
            }
        
        return summary
    
    def print_summary(self):
        """Imprime sumário formatado."""
        print("\n" + "="*60)
        print("SUMÁRIO DE PERFORMANCE")
        print("="*60 + "\n")
        
        summary = self.get_summary()
        
        for operation, metrics in summary.items():
            print(f"{operation}:")
            print(f"  Execuções: {metrics['count']}")
            print(f"  Tempo Total: {metrics['total']:.2f}s")
            print(f"  Média: {metrics['average']:.2f}s")
            print(f"  Min/Max: {metrics['min']:.2f}s / {metrics['max']:.2f}s")
            print()


def format_currency(value: float, currency: str = "R$") -> str:
    """Formata valor monetário."""
    return f"{currency} {value:,.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Formata porcentagem."""
    return f"{value:.{decimals}f}%"


def calculate_variance_ratio(data: List[float]) -> float:
    """
    Calcula razão variância/média para testar sobredispersão.
    
    Returns:
        > 1: Sobredispersão (usar Negative Binomial)
        = 1: Poisson adequado
        < 1: Subdispersão
    """
    mean = np.mean(data)
    var = np.var(data)
    
    if mean == 0:
        return 0
    
    return var / mean


def load_leagues_config(filepath: str = '../API/leagues.json') -> Dict:
    """Carrega configuração de ligas."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    except FileNotFoundError:
        logger.error(f"Arquivo não encontrado: {filepath}")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Erro ao decodificar JSON: {filepath}")
        return {}


def calculate_confidence_interval(data: List[float], confidence: float = 0.95) -> Tuple[float, float]:
    """
    Calcula intervalo de confiança.
    
    Args:
        data: Lista de valores
        confidence: Nível de confiança (0 a 1)
    
    Returns:
        Tupla (limite_inferior, limite_superior)
    """
    from scipy import stats
    
    mean = np.mean(data)
    se = stats.sem(data)
    margin = se * stats.t.ppf((1 + confidence) / 2, len(data) - 1)
    
    return mean - margin, mean + margin