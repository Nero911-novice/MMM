# data_processor.py
"""
Класс для обработки и подготовки данных для Marketing Mix Model.
"""

import numpy as np
import pandas as pd
from config import DEMO_DATA_CONFIG, DATA_REQUIREMENTS

class DataProcessor:
    """Класс для обработки и подготовки данных для Marketing Mix Model."""
    
    def __init__(self):
        self.data_quality_checks = {}
    
    def generate_demo_data(self, n_periods=None, start_date=None, frequency=None):
        """Генерация демонстрационных данных для MMM."""
        # Используем параметры из конфига или переданные
        config = DEMO_DATA_CONFIG
        n_periods = n_periods or config['n_periods']
        start_date = start_date or config['start_date']
        frequency = frequency or config['frequency']
        
        # Создание временного индекса
        date_range = pd.date_range(start=start_date, periods=n_periods, freq=frequency)
        
        # Установка seed для воспроизводимости
        np.random.seed(config['seed'])
        
        # Создание базовых паттернов
        seasonal_annual = 1 + config['seasonal_amplitude'] * np.sin(2 * np.pi * np.arange(n_periods) / 52)
        seasonal_monthly = 1 + 0.15 * np.sin(2 * np.pi * np.arange(n_periods) / 4.33)
        trend = 1 + config['trend_rate'] * np.arange(n_periods)
        noise = np.random.normal(0, config['noise_level'], n_periods)
        holiday_effect = np.random.choice([0, 0, 0, 0.3], n_periods, p=[0.85, 0.05, 0.05, 0.05])
        
        # Генерация медиа-каналов
        facebook_base = 45000 + 15000 * seasonal_annual + 8000 * np.random.normal(0, 1, n_periods)
        facebook_spend = np.maximum(facebook_base, 5000)
        
        google_base = 67000 + 20000 * seasonal_monthly + 12000 * np.random.normal(0, 1, n_periods)
        google_spend = np.maximum(google_base, 8000)
        
        tiktok_base = 15000 + 25000 * (np.arange(n_periods) / n_periods) + 10000 * np.random.normal(0, 1.5, n_periods)
        tiktok_spend = np.maximum(tiktok_base, 0)
        
        youtube_base = 32000 + 12000 * seasonal_annual + 8000 * np.random.normal(0, 1, n_periods)
        youtube_spend = np.maximum(youtube_base, 2000)
        
        offline_base = 25000 + 8000 * seasonal_annual + 5000 * np.random.normal(0, 0.8, n_periods)
        offline_spend = np.maximum(offline_base, 1000)
        
        # Генерация медиа-показателей
        facebook_impressions = facebook_spend * 50 + np.random.normal(0, facebook_spend * 5, n_periods)
        google_clicks = google_spend * 0.035 + np.random.normal(0, google_spend * 0.007, n_periods)
        
        # Внешние факторы
        promo_activity = np.random.choice([0, 1], n_periods, p=[0.75, 0.25])
        competitor_activity = 0.8 + 0.4 * np.random.beta(2, 2, n_periods)
        
        # Генерация целевой переменной (заказы)
        def apply_adstock_saturation(media, decay=0.5, alpha=1.0, gamma_factor=0.3):
            # Adstock
            adstocked = np.zeros_like(media)
            for i in range(len(media)):
                if i == 0:
                    adstocked[i] = media[i]
                else:
                    adstocked[i] = media[i] + decay * adstocked[i-1]
            
            # Saturation
            gamma = np.mean(adstocked) * gamma_factor
            saturated = np.power(adstocked, alpha) / (np.power(adstocked, alpha) + np.power(gamma, alpha))
            return saturated
        
        # Базовая линия
        base_orders = config['base_orders'] * trend * seasonal_annual
        
        # Эффекты медиа
        facebook_effect = apply_adstock_saturation(facebook_spend, 0.6, 0.8, 0.4) * 0.15
        google_effect = apply_adstock_saturation(google_spend, 0.4, 1.2, 0.3) * 0.12
        tiktok_effect = apply_adstock_saturation(tiktok_spend, 0.3, 1.5, 0.5) * 0.08
        youtube_effect = apply_adstock_saturation(youtube_spend, 0.7, 0.9, 0.35) * 0.10
        offline_effect = apply_adstock_saturation(offline_spend, 0.8, 0.6, 0.6) * 0.06
        
        # Эффекты внешних факторов
        promo_effect = promo_activity * 1500
        competitor_effect = (1 - competitor_activity) * 1000
        holiday_orders = holiday_effect * 2000
        
        # Итоговые заказы
        total_orders = (base_orders + 
                       facebook_effect + google_effect + tiktok_effect + 
                       youtube_effect + offline_effect +
                       promo_effect + competitor_effect + 
                       holiday_orders + noise * 500)
        
        total_orders = np.maximum(total_orders, 1000)
        
        # Создание DataFrame
        demo_data = pd.DataFrame({
            'date': date_range,
            'orders': total_orders.astype(int),
            
            # Медиа-расходы
            'facebook_spend': facebook_spend.astype(int),
            'google_spend': google_spend.astype(int),
            'tiktok_spend': tiktok_spend.astype(int),
            'youtube_spend': youtube_spend.astype(int),
            'offline_spend': offline_spend.astype(int),
            
            # Медиа-показатели
            'facebook_impressions': facebook_impressions.astype(int),
            'google_clicks': google_clicks.astype(int),
            
            # Внешние факторы
            'promo_activity': promo_activity,
            'competitor_activity': competitor_activity.round(2),
            'holiday_effect': holiday_effect,
            
            # Дополнительные переменные
            'seasonal_index': seasonal_annual.round(2),
            'trend_index': trend.round(2)
        })
        
        return demo_data
    
    def validate_data(self, data):
        """Валидация данных для MMM."""
        validation_results = {}
        
        # 1. Проверка обязательных столбцов
        required_columns = ['date']
        missing_required = [col for col in required_columns if col not in data.columns]
        
        validation_results['required_columns'] = {
            'status': len(missing_required) == 0,
            'message': f"Отсутствуют столбцы: {missing_required}" if missing_required else "Все обязательные столбцы присутствуют"
        }
        
        # 2. Проверка формата даты
        try:
            pd.to_datetime(data['date'])
            date_format_ok = True
            date_message = "Формат даты корректный"
        except:
            date_format_ok = False
            date_message = "Некорректный формат даты"
        
        validation_results['date_format'] = {
            'status': date_format_ok,
            'message': date_message
        }
        
        # 3. Проверка пропущенных значений
        missing_counts = data.isnull().sum()
        missing_ratio = missing_counts.sum() / (len(data) * len(data.columns))
        
        validation_results['missing_values'] = {
            'status': missing_ratio <= DATA_REQUIREMENTS['max_missing_ratio'],
            'message': f"Пропусков: {missing_counts.sum()} ({missing_ratio:.2%})"
        }
        
        # 4. Проверка дубликатов дат
        duplicate_dates = data['date'].duplicated().sum()
        validation_results['duplicate_dates'] = {
            'status': duplicate_dates == 0,
            'message': f"Дубликатов дат: {duplicate_dates}" if duplicate_dates > 0 else "Дубликаты отсутствуют"
        }
        
        # 5. Проверка минимального количества данных
        min_periods_ok = len(data) >= DATA_REQUIREMENTS['min_periods']
        validation_results['min_periods'] = {
            'status': min_periods_ok,
            'message': f"Периодов: {len(data)} (мин. {DATA_REQUIREMENTS['min_periods']})"
        }
        
        # 6. Проверка медиа-каналов
        media_columns = [col for col in data.columns if any(keyword in col.lower() 
                        for keyword in ['spend', 'cost', 'budget'])]
        min_media_ok = len(media_columns) >= DATA_REQUIREMENTS['min_media_channels']
        
        validation_results['media_channels'] = {
            'status': min_media_ok,
            'message': f"Медиа-каналов: {len(media_columns)} (мин. {DATA_REQUIREMENTS['min_media_channels']})"
        }
        
        # 7. Проверка вариативности данных
        numeric_columns = data.select_dtypes(include=[np.number]).columns
        low_variance_columns = []
        
        for col in numeric_columns:
            if col != 'date':
                variance = data[col].var()
                if variance < DATA_REQUIREMENTS['min_variance_threshold']:
                    low_variance_columns.append(col)
        
        validation_results['data_variance'] = {
            'status': len(low_variance_columns) == 0,
            'message': f"Столбцы с низкой вариативностью: {low_variance_columns}" if low_variance_columns else "Вариативность данных достаточная"
        }
        
        # 8. Общая оценка
        passed_checks = sum(1 for check in validation_results.values() if check['status'])
        total_checks = len(validation_results)
        quality_score = passed_checks / total_checks * 100
        
        validation_results['overall_quality'] = {
            'status': quality_score >= 80,
            'message': f"Качество данных: {quality_score:.1f}%",
            'score': quality_score
        }
        
        return validation_results
    
    def prepare_model_data(self, data, target_column, media_columns, external_columns=None, control_columns=None):
        """Подготовка данных для обучения MMM модели."""
        df = data.copy()
        df = df.sort_values('date').reset_index(drop=True)
        
        # Формирование списка всех признаков
        all_features = media_columns.copy()
        
        if external_columns:
            all_features.extend(external_columns)
        
        if control_columns:
            all_features.extend(control_columns)
        
        # Проверка наличия столбцов
        missing_columns = [col for col in all_features + [target_column] if col not in df.columns]
        if missing_columns:
            raise ValueError(f"Отсутствуют столбцы: {missing_columns}")
        
        # Обработка пропущенных значений
        for col in all_features:
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col] = df[col].fillna(df[col].median())
                else:
                    df[col] = df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'unknown')
        
        if df[target_column].isnull().any():
            df[target_column] = df[target_column].fillna(df[target_column].median())
        
        # Проверка на отрицательные значения в медиа-каналах
        for col in media_columns:
            if (df[col] < 0).any():
                print(f"Внимание: отрицательные значения в {col} заменены на 0")
                df[col] = df[col].clip(lower=0)
        
        # Проверка на нулевую дисперсию
        for col in all_features:
            if df[col].var() == 0:
                print(f"Внимание: столбец {col} имеет нулевую дисперсию")
        
        # Формирование матрицы признаков
        X = df[all_features].copy()
        y = df[target_column].copy()
        
        return X, y
    
    def split_data(self, data, train_ratio=0.8, date_column='date'):
        """Разделение данных на обучающую и тестовую выборки по времени."""
        df = data.copy()
        df = df.sort_values(date_column)
        
        split_index = int(len(df) * train_ratio)
        
        train_data = df.iloc[:split_index].copy()
        test_data = df.iloc[split_index:].copy()
        
        return train_data, test_data
    
    def detect_outliers(self, data, columns=None, method='iqr', threshold=3):
        """Обнаружение выбросов в данных."""
        if columns is None:
            columns = data.select_dtypes(include=[np.number]).columns
        
        outliers_info = {}
        
        for col in columns:
            if col in data.columns:
                if method == 'iqr':
                    Q1 = data[col].quantile(0.25)
                    Q3 = data[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)]
                
                elif method == 'zscore':
                    z_scores = np.abs((data[col] - data[col].mean()) / data[col].std())
                    outliers = data[z_scores > threshold]
                
                outliers_info[col] = {
                    'count': len(outliers),
                    'percentage': len(outliers) / len(data) * 100,
                    'indices': outliers.index.tolist()
                }
        
        return outliers_info
    
    def apply_transformations(self, data, transformations):
        """Применение трансформаций к данным."""
        df = data.copy()
        
        for col, transform_type in transformations.items():
            if col in df.columns:
                if transform_type == 'log':
                    # Логарифмическое преобразование
                    df[col] = np.log1p(df[col])  # log1p для обработки нулей
                
                elif transform_type == 'sqrt':
                    # Квадратный корень
                    df[col] = np.sqrt(df[col])
                
                elif transform_type == 'normalize':
                    # Нормализация 0-1
                    df[col] = (df[col] - df[col].min()) / (df[col].max() - df[col].min())
                
                elif transform_type == 'standardize':
                    # Стандартизация z-score
                    df[col] = (df[col] - df[col].mean()) / df[col].std()
        
        return df
    
    def create_time_features(self, data, date_column='date'):
        """Создание временных признаков."""
        df = data.copy()
        df[date_column] = pd.to_datetime(df[date_column])
        
        # Основные временные признаки
        df['year'] = df[date_column].dt.year
        df['month'] = df[date_column].dt.month
        df['week'] = df[date_column].dt.isocalendar().week
        df['day_of_year'] = df[date_column].dt.dayofyear
        df['quarter'] = df[date_column].dt.quarter
        
        # Циклические признаки
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['week_sin'] = np.sin(2 * np.pi * df['week'] / 52)
        df['week_cos'] = np.cos(2 * np.pi * df['week'] / 52)
        
        # Праздничные периоды (простая аппроксимация)
        df['is_holiday_season'] = ((df['month'] == 12) | (df['month'] == 1)).astype(int)
        df['is_summer'] = ((df['month'] >= 6) & (df['month'] <= 8)).astype(int)
        
        return df
    
    def get_data_summary(self, data):
        """Получение сводной информации о данных."""
        summary = {
            'shape': data.shape,
            'date_range': {
                'start': data['date'].min() if 'date' in data.columns else None,
                'end': data['date'].max() if 'date' in data.columns else None,
                'periods': len(data)
            },
            'columns': {
                'total': len(data.columns),
                'numeric': len(data.select_dtypes(include=[np.number]).columns),
                'categorical': len(data.select_dtypes(include=['object', 'category']).columns)
            },
            'missing_values': {
                'total': data.isnull().sum().sum(),
                'percentage': data.isnull().sum().sum() / (len(data) * len(data.columns)) * 100
            },
            'media_channels': [col for col in data.columns if any(keyword in col.lower() 
                              for keyword in ['spend', 'cost', 'budget'])],
            'target_candidates': [col for col in data.columns if any(keyword in col.lower() 
                                 for keyword in ['orders', 'sales', 'revenue'])]
        }
        
        return summary