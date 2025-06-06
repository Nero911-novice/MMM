# config.py
"""
Настройки и константы для Marketing Mix Model приложения.
"""

# ==========================================
# ЦВЕТОВЫЕ ПАЛИТРЫ
# ==========================================

COLOR_PALETTE = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e', 
    'success': '#2ca02c',
    'danger': '#d62728',
    'warning': '#ff9800',
    'info': '#17a2b8'
}

MEDIA_COLORS = {
    'facebook': '#1877f2',
    'google': '#4285f4',
    'tiktok': '#000000',
    'youtube': '#ff0000',
    'instagram': '#e4405f',
    'offline': '#6c757d',
    'base': '#343a40',
    'search': '#4285f4',
    'social': '#1877f2',
    'display': '#ff7f0e',
    'video': '#ff0000'
}

# ==========================================
# НАСТРОЙКИ МОДЕЛИ
# ==========================================

# Параметры по умолчанию для разных типов каналов
CHANNEL_PRESETS = {
    'paid_search': {
        'decay_range': (0.2, 0.6), 
        'alpha_range': (0.8, 1.5),
        'default_decay': 0.4,
        'default_alpha': 1.0
    },
    'social_media': {
        'decay_range': (0.1, 0.4), 
        'alpha_range': (0.5, 1.2),
        'default_decay': 0.3,
        'default_alpha': 0.8
    },
    'display': {
        'decay_range': (0.3, 0.7), 
        'alpha_range': (0.6, 1.3),
        'default_decay': 0.5,
        'default_alpha': 1.0
    },
    'video': {
        'decay_range': (0.4, 0.8), 
        'alpha_range': (0.7, 1.4),
        'default_decay': 0.6,
        'default_alpha': 1.2
    },
    'offline': {
        'decay_range': (0.5, 0.9), 
        'alpha_range': (0.4, 1.0),
        'default_decay': 0.7,
        'default_alpha': 0.8
    },
    'default': {
        'decay_range': (0.2, 0.7), 
        'alpha_range': (0.5, 1.5),
        'default_decay': 0.5,
        'default_alpha': 1.0
    }
}

# ==========================================
# НАСТРОЙКИ GRID SEARCH
# ==========================================

GRID_SEARCH_MODES = {
    'Быстрый': {
        'decay_steps': 2,
        'alpha_steps': 2,
        'gamma_steps': 2,
        'cv_folds': 2,
        'max_combinations': 64,
        'estimated_time': '2-5 минут'
    },
    'Средний': {
        'decay_steps': 3,
        'alpha_steps': 3,
        'gamma_steps': 3,
        'cv_folds': 3,
        'max_combinations': 512,
        'estimated_time': '5-15 минут'
    },
    'Полный': {
        'decay_steps': 4,
        'alpha_steps': 4,
        'gamma_steps': 3,
        'cv_folds': 3,
        'max_combinations': 2000,
        'estimated_time': '15-60 минут'
    }
}

# ==========================================
# БИЗНЕС-ПРАВИЛА И ПОРОГИ
# ==========================================

# Пороги качества модели
MODEL_QUALITY_THRESHOLDS = {
    'excellent': {'r2': 0.8, 'accuracy': 85, 'score': 95},
    'good': {'r2': 0.7, 'accuracy': 75, 'score': 80},
    'satisfactory': {'r2': 0.5, 'accuracy': 60, 'score': 65},
    'poor': {'r2': 0.0, 'accuracy': 0, 'score': 40}
}

# Пороги ROAS по отраслям
ROAS_BENCHMARKS = {
    'ecommerce': {
        'excellent': 4.0,
        'good': 2.5,
        'acceptable': 1.5,
        'poor': 1.0
    },
    'fmcg': {
        'excellent': 3.0,
        'good': 2.0,
        'acceptable': 1.2,
        'poor': 1.0
    },
    'b2b': {
        'excellent': 5.0,
        'good': 3.0,
        'acceptable': 2.0,
        'poor': 1.0
    }
}

# ==========================================
# НАСТРОЙКИ ИНТЕРФЕЙСА
# ==========================================

# Страницы приложения
APP_PAGES = [
    "🏠 Главная", 
    "📊 Данные", 
    "⚙️ Модель", 
    "📈 Результаты", 
    "💰 Оптимизация", 
    "🔮 Сценарии"
]

# CSS стили для Streamlit
CUSTOM_CSS = """
<style>
.metric-card {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 0.5rem;
    margin: 0.5rem 0;
}
.stAlert > div {
    padding: 1rem;
}
.parameter-explanation {
    background-color: #f8f9fa;
    padding: 0.8rem;
    border-left: 4px solid #17a2b8;
    margin: 0.5rem 0;
}
</style>
"""

# ==========================================
# КЛЮЧЕВЫЕ СЛОВА ДЛЯ ОПРЕДЕЛЕНИЯ ТИПОВ СТОЛБЦОВ
# ==========================================

TARGET_KEYWORDS = ['orders', 'sales', 'revenue', 'заказ', 'продажи', 'выручка']
MEDIA_KEYWORDS = ['spend', 'cost', 'budget', 'расход', 'затрат', 'бюджет']
EXTERNAL_KEYWORDS = ['holiday', 'promo', 'season', 'competitor', 'праздник', 'промо', 'сезон', 'конкурент']

# ==========================================
# НАСТРОЙКИ ДАННЫХ
# ==========================================

# Параметры для генерации демо-данных
DEMO_DATA_CONFIG = {
    'n_periods': 104,  # 2 года еженедельных данных
    'frequency': 'W',
    'start_date': '2023-01-01',
    'seed': 42,
    'base_orders': 8000,
    'noise_level': 0.1,
    'seasonal_amplitude': 0.3,
    'trend_rate': 0.002
}

# Минимальные требования к данным
DATA_REQUIREMENTS = {
    'min_periods': 20,
    'min_media_channels': 1,
    'max_missing_ratio': 0.1,
    'min_variance_threshold': 0.01
}

# ==========================================
# СООБЩЕНИЯ ДЛЯ ПОЛЬЗОВАТЕЛЯ
# ==========================================

HELP_MESSAGES = {
    'adstock_decay': "Доля эффекта, переносимого на следующий период (0.1 = быстрое затухание, 0.8 = медленное)",
    'saturation_alpha': "Форма кривой насыщения (<1 = медленный рост, >1 = S-кривая)",
    'saturation_gamma': "Точка полунасыщения относительно средних расходов",
    'regularization': "Ridge: стабилизация, Lasso: отбор признаков, ElasticNet: баланс",
    'cv_folds': "Больше фолдов = надежнее оценка качества модели",
    'grid_search_mode': "Выберите баланс между скоростью и качеством поиска"
}

BUSINESS_EXPLANATIONS = {
    'waterfall_chart': """
    **Декомпозиция продаж** показывает, откуда приходят ваши заказы:
    - **Base** = заказы, которые идут "сами по себе" (органика, брендинг)
    - **Медиа-каналы** = заказы от конкретной рекламы
    """,
    'roas_interpretation': """
    **ROAS показывает отдачу с каждого рубля рекламы:**
    - ROAS = 3.0 означает 3 рубля дохода с 1 рубля рекламы
    - ROAS < 1.0 = убыточная реклама
    - ROAS > 3.0 = очень эффективная реклама
    """,
    'optimization_logic': """
    **Оптимизация ищет лучшее распределение бюджета:**
    - Анализирует эффективность каждого канала
    - Учитывает эффекты насыщения
    - Находит баланс для максимального результата
    """
}

# ==========================================
# ФОРМАТИРОВАНИЕ ЧИСЕЛ
# ==========================================

def format_number(value, format_type='default'):
    """Форматирование чисел для отображения."""
    if format_type == 'currency':
        return f"{value:,.0f} ₽"
    elif format_type == 'percentage':
        return f"{value:.1f}%"
    elif format_type == 'ratio':
        return f"{value:.2f}"
    elif format_type == 'large_number':
        if value >= 1_000_000:
            return f"{value/1_000_000:.1f}M"
        elif value >= 1_000:
            return f"{value/1_000:.1f}K"
        else:
            return f"{value:.0f}"
    else:
        return f"{value:,.0f}"

def get_roas_color(roas_value, industry='ecommerce'):
    """Получение цвета для ROAS на основе бенчмарков."""
    benchmarks = ROAS_BENCHMARKS.get(industry, ROAS_BENCHMARKS['ecommerce'])
    
    if roas_value >= benchmarks['excellent']:
        return COLOR_PALETTE['success']
    elif roas_value >= benchmarks['good']:
        return COLOR_PALETTE['info']
    elif roas_value >= benchmarks['acceptable']:
        return COLOR_PALETTE['warning']
    else:
        return COLOR_PALETTE['danger']

def interpret_model_quality(r2_score, accuracy_score):
    """Интерпретация качества модели."""
    for level, thresholds in MODEL_QUALITY_THRESHOLDS.items():
        if r2_score >= thresholds['r2'] and accuracy_score >= thresholds['accuracy']:
            return level, thresholds['score']
    
    return 'poor', MODEL_QUALITY_THRESHOLDS['poor']['score']