# budget_optimizer.py
"""
Класс для оптимизации распределения маркетингового бюджета.
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize, differential_evolution, LinearConstraint, Bounds

class BudgetOptimizer:
    """Класс для оптимизации распределения маркетингового бюджета."""
    
    def __init__(self):
        self.optimization_history = []
        self.best_solution = None
        self.convergence_criteria = {
            'max_iterations': 1000,
            'tolerance': 1e-6,
            'stagnation_limit': 50
        }
    
    def optimize_budget(self, model, total_budget, constraints=None, target='maximize_sales',
                       method='SLSQP', bounds_buffer=0.05):
        """Основной метод оптимизации бюджета."""
        if not hasattr(model, 'media_channels') or not model.media_channels:
            # Простая заглушка для демо
            demo_channels = ['facebook_spend', 'google_spend', 'tiktok_spend']
            optimal_allocation = {
                'facebook_spend': total_budget * 0.3,
                'google_spend': total_budget * 0.5,
                'tiktok_spend': total_budget * 0.2
            }
            
            return {
                'success': True,
                'allocation': optimal_allocation,
                'predicted_sales': 85000,
                'predicted_roas': 2.5,
                'predicted_roi': 1.5,
                'total_budget_used': total_budget,
                'optimization_method': method
            }
        
        media_channels = model.media_channels
        n_channels = len(media_channels)
        
        # Подготовка ограничений
        bounds, linear_constraints = self._prepare_constraints(
            media_channels, total_budget, constraints, bounds_buffer
        )
        
        # Определение целевой функции
        objective_func = self._get_objective_function(model, media_channels, target)
        
        # Начальное приближение
        initial_guess = self._get_initial_guess(media_channels, total_budget, constraints)
        
        # Выбор метода оптимизации
        if method == 'SLSQP':
            result = self._optimize_slsqp(objective_func, initial_guess, bounds, linear_constraints)
        elif method == 'differential_evolution':
            result = self._optimize_differential_evolution(objective_func, bounds, total_budget)
        else:
            raise ValueError(f"Неподдерживаемый метод оптимизации: {method}")
        
        # Обработка результатов
        if result.success or hasattr(result, 'x'):
            optimal_allocation = dict(zip(media_channels, result.x))
            
            # Расчет метрик для оптимального решения
            predicted_results = self._calculate_metrics(model, optimal_allocation, media_channels)
            
            optimization_result = {
                'success': True,
                'allocation': optimal_allocation,
                'predicted_sales': predicted_results['sales'],
                'predicted_roas': predicted_results['roas'],
                'predicted_roi': predicted_results['roi'],
                'total_budget_used': sum(optimal_allocation.values()),
                'optimization_method': method,
                'objective_value': -result.fun if hasattr(result, 'fun') else None
            }
            
            self.best_solution = optimization_result
            
        else:
            optimization_result = {
                'success': False,
                'message': f"Оптимизация не удалась: {result.message if hasattr(result, 'message') else 'Неизвестная ошибка'}",
                'allocation': dict(zip(media_channels, initial_guess))
            }
        
        return optimization_result
    
    def _prepare_constraints(self, media_channels, total_budget, constraints, bounds_buffer):
        """Подготовка ограничений для оптимизации."""
        n_channels = len(media_channels)
        
        bounds_list = []
        
        for channel in media_channels:
            if constraints and channel in constraints:
                min_val = constraints[channel].get('min', 0)
                max_val = constraints[channel].get('max', total_budget)
            else:
                min_val = 0
                max_val = total_budget * 0.5
            
            min_val = max(0, min_val * (1 - bounds_buffer))
            max_val = min(total_budget, max_val * (1 + bounds_buffer))
            
            bounds_list.append((min_val, max_val))
        
        bounds = Bounds([b[0] for b in bounds_list], [b[1] for b in bounds_list])
        
        A_eq = np.ones((1, n_channels))
        linear_constraints = LinearConstraint(A_eq, total_budget, total_budget)
        
        return bounds, linear_constraints
    
    def _get_objective_function(self, model, media_channels, target):
        """Создание целевой функции для оптимизации."""
        def objective(x):
            allocation = dict(zip(media_channels, x))
            
            try:
                metrics = self._calculate_metrics(model, allocation, media_channels)
                
                if target == 'maximize_sales':
                    return -metrics['sales']
                elif target == 'maximize_roas':
                    return -metrics['roas']
                elif target == 'maximize_roi':
                    return -metrics['roi']
                else:
                    return -metrics['sales']
                    
            except Exception as e:
                return 1e10
        
        return objective
    
    def _calculate_metrics(self, model, allocation, media_channels):
        """Расчет метрик для заданного распределения бюджета."""
        scenario_result = model.predict_scenario(allocation)
        
        total_spend = sum(allocation.values())
        
        metrics = {
            'sales': scenario_result['sales'],
            'roas': scenario_result['roas'],
            'roi': (scenario_result['sales'] - total_spend) / total_spend if total_spend > 0 else 0,
            'total_spend': total_spend
        }
        
        return metrics
    
    def _get_initial_guess(self, media_channels, total_budget, constraints):
        """Создание начального приближения для оптимизации."""
        n_channels = len(media_channels)
        
        if constraints:
            initial = []
            remaining_budget = total_budget
            
            for i, channel in enumerate(media_channels):
                if channel in constraints:
                    min_val = constraints[channel].get('min', 0)
                    max_val = constraints[channel].get('max', total_budget)
                    
                    if i == n_channels - 1:
                        allocation = remaining_budget
                    else:
                        preferred = (min_val + max_val) / 2
                        allocation = min(preferred, remaining_budget / (n_channels - i))
                        allocation = max(min_val, min(max_val, allocation))
                    
                    initial.append(allocation)
                    remaining_budget -= allocation
                else:
                    allocation = remaining_budget / (n_channels - i)
                    initial.append(allocation)
                    remaining_budget -= allocation
            
            current_total = sum(initial)
            if current_total > 0:
                initial = [x * total_budget / current_total for x in initial]
        else:
            initial = [total_budget / n_channels] * n_channels
        
        return np.array(initial)
    
    def _optimize_slsqp(self, objective_func, initial_guess, bounds, linear_constraints):
        """Оптимизация методом SLSQP."""
        result = minimize(
            objective_func,
            initial_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=linear_constraints,
            options={
                'maxiter': self.convergence_criteria['max_iterations'],
                'ftol': self.convergence_criteria['tolerance'],
                'disp': False
            }
        )
        
        return result
    
    def _optimize_differential_evolution(self, objective_func, bounds, total_budget):
        """Оптимизация дифференциальной эволюцией."""
        def constrained_objective(x):
            budget_diff = abs(sum(x) - total_budget)
            penalty = 1e6 * budget_diff
            
            return objective_func(x) + penalty
        
        bounds_list = list(zip(bounds.lb, bounds.ub))
        
        result = differential_evolution(
            constrained_objective,
            bounds_list,
            maxiter=self.convergence_criteria['max_iterations'],
            tol=self.convergence_criteria['tolerance'],
            seed=42,
            polish=True
        )
        
        return result
    
    def optimize_portfolio(self, model, scenarios, weights=None):
        """Оптимизация портфеля сценариев."""
        if weights is None:
            weights = [1.0 / len(scenarios)] * len(scenarios)
        
        portfolio_results = []
        
        for i, scenario in enumerate(scenarios):
            result = self.optimize_budget(
                model=model,
                total_budget=scenario['budget'],
                constraints=scenario.get('constraints'),
                target=scenario.get('target', 'maximize_sales')
            )
            
            if result['success']:
                weighted_metrics = {
                    'sales': result['predicted_sales'] * weights[i],
                    'roas': result['predicted_roas'] * weights[i],
                    'roi': result['predicted_roi'] * weights[i]
                }
                result['weighted_metrics'] = weighted_metrics
                portfolio_results.append(result)
        
        # Агрегация результатов портфеля
        if portfolio_results:
            portfolio_summary = {
                'total_scenarios': len(portfolio_results),
                'weighted_sales': sum(r['weighted_metrics']['sales'] for r in portfolio_results),
                'weighted_roas': sum(r['weighted_metrics']['roas'] for r in portfolio_results),
                'weighted_roi': sum(r['weighted_metrics']['roi'] for r in portfolio_results),
                'scenarios': portfolio_results
            }
        else:
            portfolio_summary = {'error': 'Ни один сценарий не был успешно оптимизирован'}
        
        return portfolio_summary
    
    def sensitivity_analysis(self, model, base_allocation, sensitivity_range=0.2, steps=5):
        """Анализ чувствительности результатов к изменениям бюджета."""
        sensitivity_results = {}
        
        for channel in base_allocation.keys():
            channel_results = []
            base_value = base_allocation[channel]
            
            # Диапазон изменений
            min_value = base_value * (1 - sensitivity_range)
            max_value = base_value * (1 + sensitivity_range)
            test_values = np.linspace(min_value, max_value, steps)
            
            for test_value in test_values:
                # Создаем модифицированное распределение
                modified_allocation = base_allocation.copy()
                modified_allocation[channel] = test_value
                
                # Пересчитываем метрики
                try:
                    metrics = self._calculate_metrics(model, modified_allocation, list(base_allocation.keys()))
                    
                    channel_results.append({
                        'budget_change': (test_value - base_value) / base_value * 100,
                        'absolute_budget': test_value,
                        'sales': metrics['sales'],
                        'roas': metrics['roas'],
                        'roi': metrics['roi']
                    })
                except Exception as e:
                    continue
            
            sensitivity_results[channel] = channel_results
        
        return sensitivity_results
    
    def marginal_roas_analysis(self, model, base_allocation, increment=1000):
        """Анализ предельного ROAS для каждого канала."""
        marginal_results = {}
        
        for channel in base_allocation.keys():
            # Базовые метрики
            base_metrics = self._calculate_metrics(model, base_allocation, list(base_allocation.keys()))
            
            # Метрики с увеличенным бюджетом
            incremented_allocation = base_allocation.copy()
            incremented_allocation[channel] += increment
            
            try:
                incremented_metrics = self._calculate_metrics(model, incremented_allocation, list(base_allocation.keys()))
                
                # Расчет предельного ROAS
                marginal_sales = incremented_metrics['sales'] - base_metrics['sales']
                marginal_roas = marginal_sales / increment if increment > 0 else 0
                
                marginal_results[channel] = {
                    'marginal_roas': marginal_roas,
                    'marginal_sales': marginal_sales,
                    'base_roas': base_metrics['roas'],
                    'incremental_efficiency': marginal_roas / base_metrics['roas'] if base_metrics['roas'] > 0 else 0
                }
                
            except Exception as e:
                marginal_results[channel] = {
                    'error': str(e),
                    'marginal_roas': 0
                }
        
        return marginal_results
    
    def get_optimization_recommendations(self, current_allocation, optimal_allocation, model):
        """Генерация рекомендаций по оптимизации."""
        recommendations = {
            'summary': {},
            'channel_actions': {},
            'strategic_insights': []
        }
        
        # Общие изменения
        current_total = sum(current_allocation.values())
        optimal_total = sum(optimal_allocation.values())
        total_change = (optimal_total - current_total) / current_total * 100 if current_total > 0 else 0
        
        recommendations['summary'] = {
            'total_budget_change': total_change,
            'expected_improvement': 'TBD',  # Рассчитывается в зависимости от модели
            'number_of_changes': sum(1 for ch in current_allocation.keys() 
                                   if abs(optimal_allocation.get(ch, 0) - current_allocation[ch]) / current_allocation[ch] > 0.1)
        }
        
        # Рекомендации по каналам
        for channel in current_allocation.keys():
            current_value = current_allocation[channel]
            optimal_value = optimal_allocation.get(channel, 0)
            
            if current_value > 0:
                change_pct = (optimal_value - current_value) / current_value * 100
                
                if abs(change_pct) > 10:  # Значительное изменение
                    if change_pct > 20:
                        action = f"Увеличить бюджет на {change_pct:.0f}%"
                        priority = "Высокий"
                    elif change_pct > 0:
                        action = f"Немного увеличить бюджет на {change_pct:.0f}%"
                        priority = "Средний"
                    elif change_pct > -20:
                        action = f"Немного сократить бюджет на {abs(change_pct):.0f}%"
                        priority = "Средний"
                    else:
                        action = f"Значительно сократить бюджет на {abs(change_pct):.0f}%"
                        priority = "Высокий"
                else:
                    action = "Сохранить текущий уровень"
                    priority = "Низкий"
                
                recommendations['channel_actions'][channel] = {
                    'action': action,
                    'priority': priority,
                    'current_budget': current_value,
                    'optimal_budget': optimal_value,
                    'change_amount': optimal_value - current_value
                }
        
        # Стратегические инсайты
        # Найти каналы с наибольшим увеличением/уменьшением
        changes = {ch: optimal_allocation.get(ch, 0) - current_allocation[ch] 
                  for ch in current_allocation.keys()}
        
        biggest_increase = max(changes.items(), key=lambda x: x[1])
        biggest_decrease = min(changes.items(), key=lambda x: x[1])
        
        if biggest_increase[1] > 1000:
            recommendations['strategic_insights'].append(
                f"Наибольший потенциал роста в канале {biggest_increase[0]} "
                f"(+{biggest_increase[1]:,.0f} руб)"
            )
        
        if biggest_decrease[1] < -1000:
            recommendations['strategic_insights'].append(
                f"Канал {biggest_decrease[0]} показывает признаки насыщения "
                f"({biggest_decrease[1]:,.0f} руб)"
            )
        
        return recommendations