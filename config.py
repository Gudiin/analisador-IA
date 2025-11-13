# config.py
"""
Configurações centralizadas do sistema de análise de apostas esportivas.
"""

import os
from pathlib import Path

# ============================================================================
# PATHS E DIRETÓRIOS
# ============================================================================
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / 'data'
CACHE_DIR = DATA_DIR / 'cache'
LOGS_DIR = BASE_DIR / 'logs'
REPORTS_DIR = BASE_DIR / 'reports'
DB_DIR = DATA_DIR / 'database'

# Criar diretórios se não existirem
for directory in [DATA_DIR, CACHE_DIR, LOGS_DIR, REPORTS_DIR, DB_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DATABASE
# ============================================================================
DATABASE_PATH = DB_DIR / 'sports_analytics.db'
DATABASE_URL = f'sqlite:///{DATABASE_PATH}'

# ============================================================================
# SCRAPING
# ============================================================================
SCRAPING_CONFIG = {
    'delay_between_requests': 10,  # segundos
    'max_retries': 3,
    'retry_wait': 30,  # segundos
    'timeout': 30,  # segundos
    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'cache_enabled': True,
    'cache_expiry_hours': 24
}

# ============================================================================
# ANÁLISE TEMPORAL
# ============================================================================
TEMPORAL_CONFIG = {
    'seasons_to_analyze': 2,  # Últimas 2 temporadas
    'recent_form_games': 5,   # Últimos 5 jogos para forma recente
    'weight_decay': 0.95,      # Peso exponencial para jogos antigos (95% a cada jogo)
    'min_games_required': 5    # Mínimo de jogos para análise
}

# ============================================================================
# MODELOS ESTATÍSTICOS
# ============================================================================
STATISTICAL_MODELS = {
    'goals': {
        'method': 'bivariate_poisson',  # Poisson bivariado
        'include_home_advantage': True,
        'include_recent_form': True,
        'include_h2h': True
    },
    'corners': {
        'method': 'negative_binomial',  # Melhor para dados sobredispersos
        'home_away_split': True,
        'include_possession': True,
        'include_shots': True
    },
    'fouls': {
        'method': 'poisson',
        'include_referee_tendency': True,
        'include_player_discipline': True
    }
}

# ============================================================================
# ELO RATING
# ============================================================================
ELO_CONFIG = {
    'initial_rating': 1500,
    'k_factor': 32,
    'home_advantage': 100,
    'goal_difference_multiplier': 1.5
}

# ============================================================================
# BACKTESTING
# ============================================================================
BACKTESTING_CONFIG = {
    'initial_bankroll': 1000,
    'min_edge_threshold': 0.05,  # 5% edge mínimo para apostar
    'kelly_fraction': 0.25,       # Kelly conservador (1/4 do Kelly)
    'max_stake_percentage': 0.05, # Máximo 5% da banca por aposta
    'commission': 0.05            # 5% de comissão (média das casas)
}

# ============================================================================
# MERCADOS DE APOSTAS
# ============================================================================
BETTING_MARKETS = {
    'goals': {
        'over_under_lines': [0.5, 1.5, 2.5, 3.5, 4.5],
        'both_teams_to_score': True
    },
    'corners': {
        'total_lines': [8.5, 9.5, 10.5, 11.5, 12.5],
        'first_half_lines': [4.5, 5.5, 6.5],
        'second_half_lines': [4.5, 5.5, 6.5]
    },
    'fouls': {
        'total_lines': [1.5, 2.5, 3.5],
        'team_lines': [20.5, 21.5, 22.5, 23.5, 24.5]
    }
}

# ============================================================================
# LOGGING
# ============================================================================
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'detailed': {
            'format': '%(asctime)s [%(levelname)s] %(name)s [%(filename)s:%(lineno)d]: %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'level': 'INFO',
            'formatter': 'standard',
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'DEBUG',
            'formatter': 'detailed',
            'filename': LOGS_DIR / 'app.log',
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5
        },
        'error_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'ERROR',
            'formatter': 'detailed',
            'filename': LOGS_DIR / 'errors.log',
            'maxBytes': 10485760,
            'backupCount': 5
        }
    },
    'loggers': {
        '': {  # Root logger
            'handlers': ['console', 'file', 'error_file'],
            'level': 'DEBUG',
            'propagate': False
        }
    }
}

# ============================================================================
# FEATURES AVANÇADAS
# ============================================================================
ADVANCED_FEATURES = {
    'expected_goals': True,      # xG analysis
    'elo_rating': True,          # Elo system
    'opponent_strength': True,   # Schedule strength
    'home_advantage': True,      # Home field advantage
    'fatigue_factor': True,      # Days since last match
    'injury_impact': False,      # Requer dados externos
    'weather_impact': False      # Requer API externa
}

# ============================================================================
# DASHBOARD
# ============================================================================
DASHBOARD_CONFIG = {
    'enabled': True,
    'host': '0.0.0.0',
    'port': 8501,
    'theme': 'dark',
    'refresh_interval': 3600  # 1 hora
}

# ============================================================================
# API REST (OPCIONAL)
# ============================================================================
API_CONFIG = {
    'enabled': False,  # Desabilitado por padrão
    'host': '0.0.0.0',
    'port': 8000,
    'debug': False
}

# ============================================================================
# THRESHOLDS DE CONFIANÇA
# ============================================================================
CONFIDENCE_THRESHOLDS = {
    'high': 0.70,    # 70%+ de confiança
    'medium': 0.60,  # 60-70% de confiança
    'low': 0.55      # 55-60% de confiança (não apostar abaixo)
}