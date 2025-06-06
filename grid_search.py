# grid_search.py
"""
Класс для автоматического подбора оптимальных параметров Adstock и Saturation
для всех медиа-каналов в Marketing Mix Model через Grid Search.
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from itertools import product
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

from config import CHANNEL_PRESETS, GRID_SEARCH_MODES

class MMM_GridSearchOptimizer:
    """
    Класс для автоматического подбора оптимальных параметров Adstock и Saturation
    для всех медиа-каналов в Marketing Mix Model через Grid Search.
    """
    
    def __init__(self, cv_folds=3, scoring='r2', n_jobs=1, verbose=True):
        """
        Инициализация оптимизатора параметров.
        
        Parameters:
        -----------
        cv_folds : int
            Количество фолдов для временной кросс-валидации
        scoring : str
            Метрика для оценки ('r2', 'mape', 'mae')
        n_jobs : int
            Количество параллельных процессов (пока не реализовано)
        verbose : bool
            Вывод прогресса поиска
        """
        self.cv_folds = cv_folds
        self.scoring = scoring
        self.n_jobs = n_jobs
        self.verbose = verbose
        
        # Результаты поиска
        self.search_results = []
        self.best_params = {}
        self.best_score = -np.inf
        
        # Параметры по умолчанию для разных типов каналов
        self.channel_presets = CHANNEL_PRESETS
    
    def _detect_channel_type(self, channel_name):
        """Определение типа канала по названию для предустановок параметров."""
        channel_lower = channel_name.lower()
        
        if any(keyword in channel_lower for keyword in ['google', 'search', 'sem']):
            return 'paid_search'
        elif any(keyword in channel_lower for keyword in ['facebook', 'instagram', 'tiktok', 'social']):
            return 'social_media'
        elif any(keyword in channel_lower for keyword in ['display', 'banner', 'programmatic']):
            return 'display'
        elif any(keyword in channel_lower for keyword in ['youtube', 'video', 'tv']):
            return 'video'
        elif any(keyword in channel_lower for keyword in ['offline', 'radio', 'print', 'ooh']):
            return 'offline'
        else:
            return 'default'
    
    def _generate_parameter_grid(self, media_channels, X_media, 
                                decay_steps=5, alpha_steps=5, gamma_steps=3):
        """
        Генерация сетки параметров для поиска.
        
        Parameters:
        -----------
        media_channels : list
            Список медиа-каналов
        X_media : pd.DataFrame
            Данные по медиа-каналам
        decay_steps : int
            Количество шагов для decay параметра
        alpha_steps : int
            Количество шагов для alpha параметра  
        gamma_steps : int
            Количество шагов для gamma параметра
        """
        param_grid = {}
        
        for channel in media_channels:
            channel_type = self._detect_channel_type(channel)
            presets = self.channel_presets[channel_type]
            
            # Генерация decay параметров
            decay_min, decay_max = presets['decay_range']
            decay_values = np.linspace(decay_min, decay_max, decay_steps)
            
            # Генерация alpha параметров
            alpha_min, alpha_max = presets['alpha_range']
            alpha_values = np.linspace(alpha_min, alpha_max, alpha_steps)
            
            # Генерация gamma параметров (относительно медианы канала)
            channel_data = X_media[channel]
            median_spend = channel_data[channel_data > 0].median() if len(channel_data[channel_data > 0]) > 0 else 1.0
            
            gamma_values = [
                median_spend * 0.3,  # Низкая точка насыщения
                median_spend * 0.7,  # Средняя точка насыщения  
                median_spend * 1.2   # Высокая точка насыщения
            ]
            
            param_grid[channel] = {
                'decay': decay_values,
                'alpha': alpha_values,
                'gamma': gamma_values
            }
        
        return param_grid
    
    def _create_param_combinations(self, param_grid):
        """Создание всех возможных комбинаций параметров."""
        channels = list(param_grid.keys())
        
        # Создаем список всех комбинаций для каждого канала
        channel_combinations = []
        for channel in channels:
            channel_params = param_grid[channel]
            combinations = list(product(
                channel_params['decay'],
                channel_params['alpha'], 
                channel_params['gamma']
            ))
            channel_combinations.append(combinations)
        
        # Генерируем все возможные комбинации между каналами
        all_combinations = list(product(*channel_combinations))
        
        # Преобразуем в удобный формат
        param_combinations = []
        for combination in all_combinations:
            params = {}
            for i, channel in enumerate(channels):
                decay, alpha, gamma = combination[i]
                params[channel] = {
                    'decay': decay,
                    'alpha': alpha,
                    'gamma': gamma
                }
            param_combinations.append(params)
        
        return param_combinations
    
    def _evaluate_params(self, params, model_class, X, y, media_channels):
        """
        Оценка качества параметров через кросс-валидацию.
        
        Parameters:
        -----------
        params : dict
            Параметры adstock и saturation для всех каналов
        model_class : class
            Класс модели MMM
        X : pd.DataFrame
            Матрица признаков
        y : pd.Series
            Целевая переменная
        media_channels : list
            Список медиа-каналов
        """
        try:
            # Подготовка параметров для модели
            adstock_params = {ch: {'decay': params[ch]['decay']} for ch in media_channels}
            saturation_params = {ch: {'alpha': params[ch]['alpha'], 'gamma': params[ch]['gamma']} 
                                for ch in media_channels}
            
            # Временная кросс-валидация
            tscv = TimeSeriesSplit(n_splits=self.cv_folds)
            scores = []
            
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
                y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
                
                # Создание и обучение модели
                model = model_class(
                    adstock_params=adstock_params,
                    saturation_params=saturation_params,
                    regularization='Ridge',
                    alpha=0.01
                )
                
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
                
                # Расчет метрики
                if self.scoring == 'r2':
                    score = r2_score(y_val, y_pred)
                elif self.scoring == 'mape':
                    score = -mean_absolute_percentage_error(y_val, y_pred)  # Отрицательный для максимизации
                elif self.scoring == 'mae':
                    score = -np.mean(np.abs(y_val - y_pred))  # Отрицательный для максимизации
                else:
                    score = r2_score(y_val, y_pred)
                
                scores.append(score)
            
            return np.mean(scores), np.std(scores)
            
        except Exception as e:
            if self.verbose:
                print(f"Ошибка при оценке параметров: {str(e)}")
            return -np.inf, np.inf
    
    def grid_search(self, model_class, X, y, media_channels, 
                   decay_steps=5, alpha_steps=5, gamma_steps=3,
                   max_combinations=1000):
        """
        Основной метод Grid Search для подбора оптимальных параметров.
        
        Parameters:
        -----------
        model_class : class
            Класс модели MarketingMixModel
        X : pd.DataFrame
            Матрица признаков (включая медиа-каналы)
        y : pd.Series
            Целевая переменная
        media_channels : list
            Список медиа-каналов для оптимизации
        decay_steps : int
            Количество шагов для decay параметра
        alpha_steps : int
            Количество шагов для alpha параметра
        gamma_steps : int
            Количество шагов для gamma параметра
        max_combinations : int
            Максимальное количество комбинаций для тестирования
        """
        if self.verbose:
            print("🔍 Запуск Grid Search для оптимизации параметров MMM...")
            print(f"Медиа-каналы: {media_channels}")
            print(f"Метрика оценки: {self.scoring}")
            print(f"Кросс-валидация: {self.cv_folds} фолдов")
        
        # Генерация сетки параметров
        X_media = X[media_channels]
        param_grid = self._generate_parameter_grid(
            media_channels, X_media, decay_steps, alpha_steps, gamma_steps
        )
        
        # Создание комбинаций параметров
        param_combinations = self._create_param_combinations(param_grid)
        
        # Ограничение количества комбинаций для практичности
        if len(param_combinations) > max_combinations:
            if self.verbose:
                print(f"⚠️ Слишком много комбинаций ({len(param_combinations)}). Ограничиваем до {max_combinations}")
            
            # Случайная выборка комбинаций
            np.random.seed(42)
            selected_indices = np.random.choice(len(param_combinations), max_combinations, replace=False)
            param_combinations = [param_combinations[i] for i in selected_indices]
        
        if self.verbose:
            print(f"Тестируем {len(param_combinations)} комбинаций параметров...")
        
        # Поиск лучших параметров
        best_score = -np.inf
        best_params = None
        
        for i, params in enumerate(param_combinations):
            if self.verbose and i % max(1, len(param_combinations) // 20) == 0:
                progress = (i + 1) / len(param_combinations) * 100
                print(f"Прогресс: {progress:.1f}% ({i+1}/{len(param_combinations)})")
            
            # Оценка параметров
            mean_score, std_score = self._evaluate_params(
                params, model_class, X, y, media_channels
            )
            
            # Сохранение результата
            result = {
                'params': params.copy(),
                'mean_score': mean_score,
                'std_score': std_score,
                'iteration': i
            }
            self.search_results.append(result)
            
            # Обновление лучшего результата
            if mean_score > best_score:
                best_score = mean_score
                best_params = params.copy()
                
                if self.verbose:
                    print(f"✅ Новый лучший результат: {self.scoring} = {best_score:.4f}")
        
        # Сохранение результатов
        self.best_score = best_score
        self.best_params = best_params
        
        if self.verbose:
            print(f"\n🎯 Поиск завершен!")
            print(f"Лучший {self.scoring}: {best_score:.4f}")
            print("Оптимальные параметры:")
            for channel, params in best_params.items():
                print(f"  {channel}:")
                print(f"    decay: {params['decay']:.3f}")
                print(f"    alpha: {params['alpha']:.3f}")
                print(f"    gamma: {params['gamma']:.1f}")
        
        return self.best_params, self.best_score
    
    def get_search_results_df(self):
        """Получение результатов поиска в виде DataFrame."""
        if not self.search_results:
            return pd.DataFrame()
        
        # Создание плоской структуры для DataFrame
        flattened_results = []
        
        for result in self.search_results:
            row = {
                'mean_score': result['mean_score'],
                'std_score': result['std_score'],
                'iteration': result['iteration']
            }
            
            # Добавляем параметры каждого канала
            for channel, params in result['params'].items():
                row[f"{channel}_decay"] = params['decay']
                row[f"{channel}_alpha"] = params['alpha']
                row[f"{channel}_gamma"] = params['gamma']
            
            flattened_results.append(row)
        
        return pd.DataFrame(flattened_results)
    
    def plot_search_progress(self):
        """Создание графика прогресса поиска (для использования в Streamlit)."""
        if not self.search_results:
            return None
        
        iterations = [r['iteration'] for r in self.search_results]
        scores = [r['mean_score'] for r in self.search_results]
        
        # Кумулятивный максимум для отображения улучшений
        best_scores = []
        current_best = -np.inf
        for score in scores:
            if score > current_best:
                current_best = score
            best_scores.append(current_best)
        
        fig = go.Figure()
        
        # Все результаты
        fig.add_trace(go.Scatter(
            x=iterations,
            y=scores,
            mode='markers',
            name='Все результаты',
            marker=dict(color='lightblue', size=4),
            opacity=0.6,
            hovertemplate="Итерация: %{x}<br>Результат: %{y:.4f}<extra></extra>"
        ))
        
        # Лучшие результаты
        fig.add_trace(go.Scatter(
            x=iterations,
            y=best_scores,
            mode='lines',
            name='Лучший результат',
            line=dict(color='red', width=2),
            hovertemplate="Итерация: %{x}<br>Лучший результат: %{y:.4f}<extra></extra>"
        ))
        
        fig.update_layout(
            title=f"Прогресс Grid Search (метрика: {self.scoring})",
            xaxis_title="Итерация",
            yaxis_title=f"Значение {self.scoring}",
            height=400,
            template="plotly_white"
        )
        
        return fig
    
    def get_parameter_importance(self, top_n=10):
        """
        Анализ важности параметров на основе результатов поиска.
        
        Parameters:
        -----------
        top_n : int
            Количество топ результатов для анализа
        """
        if not self.search_results:
            return {}
        
        # Сортируем результаты по качеству
        sorted_results = sorted(self.search_results, key=lambda x: x['mean_score'], reverse=True)
        top_results = sorted_results[:top_n]
        
        # Анализ распределения параметров в топе
        parameter_stats = {}
        
        for result in top_results:
            for channel, params in result['params'].items():
                if channel not in parameter_stats:
                    parameter_stats[channel] = {'decay': [], 'alpha': [], 'gamma': []}
                
                parameter_stats[channel]['decay'].append(params['decay'])
                parameter_stats[channel]['alpha'].append(params['alpha'])
                parameter_stats[channel]['gamma'].append(params['gamma'])
        
        # Статистики для каждого параметра
        importance_analysis = {}
        for channel, params in parameter_stats.items():
            importance_analysis[channel] = {}
            for param_name, values in params.items():
                importance_analysis[channel][param_name] = {
                    'mean': np.mean(values),
                    'std': np.std(values),
                    'min': np.min(values),
                    'max': np.max(values),
                    'range': np.max(values) - np.min(values)
                }
        
        return importance_analysis
    
    def plot_parameter_distribution(self, channel=None):
        """Создание графика распределения параметров."""
        if not self.search_results:
            return None
        
        # Если канал не указан, берем первый доступный
        if channel is None and self.search_results:
            channel = list(self.search_results[0]['params'].keys())[0]
        
        # Извлекаем параметры для выбранного канала
        decay_values = []
        alpha_values = []
        gamma_values = []
        scores = []
        
        for result in self.search_results:
            if channel in result['params']:
                decay_values.append(result['params'][channel]['decay'])
                alpha_values.append(result['params'][channel]['alpha'])
                gamma_values.append(result['params'][channel]['gamma'])
                scores.append(result['mean_score'])
        
        # Создание 3D scatter plot
        fig = go.Figure(data=go.Scatter3d(
            x=decay_values,
            y=alpha_values,
            z=gamma_values,
            mode='markers',
            marker=dict(
                size=5,
                color=scores,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title=f"{self.scoring} score")
            ),
            text=[f"Score: {s:.4f}" for s in scores],
            hovertemplate="Decay: %{x:.3f}<br>Alpha: %{y:.3f}<br>Gamma: %{z:.0f}<br>%{text}<extra></extra>"
        ))
        
        fig.update_layout(
            title=f"Распределение параметров для {channel}",
            scene=dict(
                xaxis_title='Decay',
                yaxis_title='Alpha',
                zaxis_title='Gamma'
            ),
            height=600
        )
        
        return fig
    
    def export_results(self, filename='grid_search_results.csv'):
        """Экспорт результатов в CSV файл."""
        if not self.search_results:
            print("Нет результатов для экспорта")
            return False
        
        df = self.get_search_results_df()
        df.to_csv(filename, index=False)
        print(f"Результаты экспортированы в {filename}")
        return True
    
    def get_convergence_analysis(self):
        """Анализ сходимости алгоритма."""
        if not self.search_results:
            return {}
        
        scores = [r['mean_score'] for r in self.search_results]
        
        # Кумулятивный максимум
        cumulative_best = []
        current_best = -np.inf
        for score in scores:
            if score > current_best:
                current_best = score
            cumulative_best.append(current_best)
        
        # Анализ улучшений
        improvements = []
        for i in range(1, len(cumulative_best)):
            if cumulative_best[i] > cumulative_best[i-1]:
                improvements.append(i)
        
        convergence_info = {
            'total_iterations': len(scores),
            'final_best_score': cumulative_best[-1] if cumulative_best else 0,
            'improvement_iterations': improvements,
            'last_improvement': improvements[-1] if improvements else 0,
            'convergence_rate': len(improvements) / len(scores) if scores else 0,
            'score_variance': np.var(scores) if scores else 0
        }
        
        return convergence_info

# Функция для добавления метода Grid Search к существующему классу MMM
def add_grid_search_method():
    """Добавление метода Grid Search в класс MarketingMixModel."""
    
    def auto_optimize_parameters(self, X, y, media_channels, 
                                decay_steps=4, alpha_steps=4, gamma_steps=3,
                                cv_folds=3, scoring='r2', max_combinations=500):
        """
        Автоматическая оптимизация параметров adstock и saturation.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Матрица признаков
        y : pd.Series
            Целевая переменная
        media_channels : list
            Список медиа-каналов для оптимизации
        decay_steps : int
            Количество шагов для поиска decay параметра
        alpha_steps : int
            Количество шагов для поиска alpha параметра  
        gamma_steps : int
            Количество шагов для поиска gamma параметра
        cv_folds : int
            Количество фолдов для кросс-валидации
        scoring : str
            Метрика для оценки ('r2', 'mape', 'mae')
        max_combinations : int
            Максимальное количество комбинаций для тестирования
            
        Returns:
        --------
        dict : Оптимальные параметры
        float : Лучший скор
        MMM_GridSearchOptimizer : Объект оптимизатора с результатами
        """
        
        # Создание оптимизатора
        optimizer = MMM_GridSearchOptimizer(
            cv_folds=cv_folds,
            scoring=scoring,
            verbose=True
        )
        
        # Запуск Grid Search
        best_params, best_score = optimizer.grid_search(
            model_class=self.__class__,
            X=X,
            y=y,
            media_channels=media_channels,
            decay_steps=decay_steps,
            alpha_steps=alpha_steps,
            gamma_steps=gamma_steps,
            max_combinations=max_combinations
        )
        
        # Применение найденных параметров к текущей модели
        if best_params:
            self.adstock_params = {ch: {'decay': best_params[ch]['decay']} 
                                 for ch in media_channels}
            self.saturation_params = {ch: {'alpha': best_params[ch]['alpha'], 
                                         'gamma': best_params[ch]['gamma']} 
                                    for ch in media_channels}
            
            print(f"✅ Параметры модели обновлены. Лучший {scoring}: {best_score:.4f}")
            print("💡 Теперь можно обучить модель с оптимизированными параметрами")
        
        return best_params, best_score, optimizer
    
    return auto_optimize_parameters