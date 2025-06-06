# visualizer.py
"""
Класс для создания визуализаций результатов Marketing Mix Model.
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import COLOR_PALETTE, MEDIA_COLORS, get_roas_color

class Visualizer:
    """Класс для создания визуализаций результатов Marketing Mix Model."""
    
    def __init__(self):
        self.color_palette = COLOR_PALETTE
        self.media_colors = MEDIA_COLORS
        
    def create_waterfall_chart(self, contributions, title="Декомпозиция продаж по каналам"):
        """Создание waterfall диаграммы для визуализации вкладов каналов."""
        # Проверка входных данных
        if not contributions or len(contributions) == 0:
            # Создаем простой bar chart если нет данных для waterfall
            fig = go.Figure()
            fig.add_annotation(
                text="Нет данных для отображения",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(title=title, height=400)
            return fig
        
        # Подготовка данных
        channels = list(contributions.keys())
        values = list(contributions.values())
        
        # Проверка на корректность значений
        values = [float(v) if v is not None and not np.isnan(float(v)) else 0 for v in values]
        
        # Сортировка по убыванию (исключая Base)
        if 'Base' in contributions:
            base_value = contributions['Base']
            other_contributions = {k: v for k, v in contributions.items() if k != 'Base'}
            sorted_others = sorted(other_contributions.items(), key=lambda x: abs(x[1]), reverse=True)
            
            channels = ['Base'] + [item[0] for item in sorted_others]
            values = [base_value] + [item[1] for item in sorted_others]
        else:
            sorted_items = sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
            channels = [item[0] for item in sorted_items]
            values = [item[1] for item in sorted_items]
        
        # Создание цветов
        colors = []
        for channel in channels:
            channel_lower = str(channel).lower()
            if any(key in channel_lower for key in self.media_colors.keys()):
                # Найти подходящий цвет
                for key, color in self.media_colors.items():
                    if key in channel_lower:
                        colors.append(color)
                        break
                else:
                    colors.append(self.color_palette['primary'])
            else:
                colors.append(self.color_palette['primary'])
        
        try:
            # Создание waterfall графика
            fig = go.Figure(go.Waterfall(
                name="Вклады",
                orientation="v",
                measure=["absolute"] + ["relative"] * (len(channels) - 1),
                x=channels,
                y=values,
                text=[f"{val:,.0f}" for val in values],
                textposition="outside",
                connector={"line": {"color": "gray"}},
                marker_color=colors
            ))
            
            fig.update_layout(
                title={
                    'text': title,
                    'x': 0.5,
                    'xanchor': 'center',
                    'font': {'size': 16}
                },
                showlegend=False,
                xaxis_title="Каналы",
                yaxis_title="Вклад в продажи",
                height=500,
                template="plotly_white"
            )
            
        except Exception as e:
            # Fallback к обычному bar chart если waterfall не работает
            fig = go.Figure(data=[
                go.Bar(x=channels, y=values, marker_color=colors,
                       text=[f"{val:,.0f}" for val in values], textposition='outside')
            ])
            
            fig.update_layout(
                title={
                    'text': title + " (Bar Chart)",
                    'x': 0.5,
                    'xanchor': 'center'
                },
                xaxis_title="Каналы",
                yaxis_title="Вклад в продажи",
                height=500,
                template="plotly_white"
            )
        
        return fig
    
    def create_roas_comparison(self, roas_data, title="ROAS по каналам"):
        """Создание сравнительной диаграммы ROAS."""
        try:
            # Проверка входных данных
            if roas_data is None or roas_data.empty or 'ROAS' not in roas_data.columns or 'Channel' not in roas_data.columns:
                # Создаем пустой график с сообщением
                fig = go.Figure()
                fig.add_annotation(
                    text="Нет данных для отображения ROAS",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
                fig.update_layout(title=title, height=400)
                return fig
            
            # Сортировка по ROAS
            roas_sorted = roas_data.sort_values('ROAS', ascending=True)
            
            # Создание цветов на основе ROAS
            colors = []
            for channel, roas_val in zip(roas_sorted['Channel'], roas_sorted['ROAS']):
                colors.append(get_roas_color(roas_val))
            
            fig = go.Figure(data=[
                go.Bar(
                    x=roas_sorted['ROAS'],
                    y=roas_sorted['Channel'],
                    orientation='h',
                    marker_color=colors,
                    text=[f"{val:.2f}" for val in roas_sorted['ROAS']],
                    textposition='outside',
                    hovertemplate="<b>%{y}</b><br>ROAS: %{x:.2f}<extra></extra>"
                )
            ])
            
            fig.add_vline(
                x=1, 
                line_dash="dash", 
                line_color="red",
                annotation_text="Точка безубыточности",
                annotation_position="top right"
            )
            
            fig.update_layout(
                title={'text': title, 'x': 0.5, 'xanchor': 'center'},
                xaxis_title="ROAS",
                yaxis_title="Каналы",
                height=400,
                template="plotly_white",
                showlegend=False
            )
            
            return fig
            
        except Exception as e:
            # Fallback к простому графику
            fig = go.Figure()
            fig.add_annotation(
                text=f"Ошибка создания графика: {str(e)[:50]}...",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            fig.update_layout(title=title, height=400)
            return fig
    
    def create_budget_allocation_pie(self, budget_data, title="Распределение бюджета"):
        """Создание круговой диаграммы распределения бюджета."""
        channels = list(budget_data.keys())
        values = list(budget_data.values())
        total_budget = sum(values)
        
        colors = []
        for channel in channels:
            channel_lower = channel.lower()
            colors.append(self.media_colors.get(channel_lower, self.color_palette['primary']))
        
        fig = go.Figure(data=[go.Pie(
            labels=channels,
            values=values,
            marker_colors=colors,
            textinfo='label+percent',
            textposition='outside',
            hovertemplate="<b>%{label}</b><br>Бюджет: %{value:,.0f}<br>Доля: %{percent}<extra></extra>"
        )])
        
        fig.update_layout(
            title={
                'text': f"{title}<br><sub>Общий бюджет: {total_budget:,.0f}</sub>",
                'x': 0.5,
                'xanchor': 'center'
            },
            height=500,
            template="plotly_white"
        )
        
        return fig
    
    def create_optimization_results(self, current_allocation, optimal_allocation, 
                                  title="Результаты оптимизации бюджета"):
        """Создание сравнения текущего и оптимального распределения."""
        channels = list(current_allocation.keys())
        current_values = [current_allocation[ch] for ch in channels]
        optimal_values = [optimal_allocation.get(ch, 0) for ch in channels]
        
        fig = go.Figure()
        
        # Текущее распределение
        fig.add_trace(go.Bar(
            name='Текущее',
            x=channels,
            y=current_values,
            marker_color=self.color_palette['info'],
            opacity=0.7
        ))
        
        # Оптимальное распределение
        fig.add_trace(go.Bar(
            name='Оптимальное',
            x=channels,
            y=optimal_values,
            marker_color=self.color_palette['success']
        ))
        
        # Расчет изменений
        for i, channel in enumerate(channels):
            change = optimal_values[i] - current_values[i]
            change_pct = (change / current_values[i] * 100) if current_values[i] > 0 else 0
            
            fig.add_annotation(
                x=i,
                y=max(optimal_values[i], current_values[i]) + max(optimal_values) * 0.05,
                text=f"{change_pct:+.1f}%",
                showarrow=False,
                font=dict(
                    size=10,
                    color=self.color_palette['success'] if change > 0 else self.color_palette['danger']
                )
            )
        
        fig.update_layout(
            title={'text': title, 'x': 0.5, 'xanchor': 'center'},
            xaxis_title="Каналы",
            yaxis_title="Бюджет",
            barmode='group',
            height=500,
            template="plotly_white"
        )
        
        return fig
    
    def create_time_series_plot(self, data, y_columns, x_column='date', title="Временные ряды"):
        """Создание графика временных рядов."""
        fig = go.Figure()
        
        colors = [self.color_palette['primary'], self.color_palette['secondary'], 
                 self.color_palette['success'], self.color_palette['warning']]
        
        for i, col in enumerate(y_columns):
            if col in data.columns:
                color = colors[i % len(colors)]
                fig.add_trace(go.Scatter(
                    x=data[x_column],
                    y=data[col],
                    mode='lines',
                    name=col.replace('_', ' ').title(),
                    line=dict(color=color, width=2),
                    hovertemplate=f"<b>{col}</b><br>Дата: %{{x}}<br>Значение: %{{y:,.0f}}<extra></extra>"
                ))
        
        fig.update_layout(
            title={'text': title, 'x': 0.5, 'xanchor': 'center'},
            xaxis_title="Дата",
            yaxis_title="Значение",
            height=400,
            template="plotly_white",
            hovermode='x unified'
        )
        
        return fig
    
    def create_correlation_heatmap(self, data, title="Корреляционная матрица"):
        """Создание тепловой карты корреляций."""
        # Выбираем только числовые столбцы
        numeric_data = data.select_dtypes(include=[np.number])
        correlation_matrix = numeric_data.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.round(2).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hovertemplate="<b>%{y} vs %{x}</b><br>Корреляция: %{z:.3f}<extra></extra>"
        ))
        
        fig.update_layout(
            title={'text': title, 'x': 0.5, 'xanchor': 'center'},
            height=600,
            template="plotly_white"
        )
        
        return fig
    
    def create_saturation_curve(self, channel_data, alpha=1.0, gamma=None, 
                               current_spend=None, title="Кривая насыщения"):
        """Создание кривой насыщения для канала."""
        if gamma is None:
            gamma = np.median(channel_data[channel_data > 0]) if len(channel_data[channel_data > 0]) > 0 else 1.0
        
        # Диапазон для построения кривой
        max_spend = channel_data.max() * 2
        spend_range = np.linspace(0, max_spend, 100)
        
        # Расчет кривой насыщения
        saturation_values = np.power(spend_range, alpha) / (np.power(spend_range, alpha) + np.power(gamma, alpha))
        
        fig = go.Figure()
        
        # Кривая насыщения
        fig.add_trace(go.Scatter(
            x=spend_range,
            y=saturation_values,
            mode='lines',
            name='Кривая насыщения',
            line=dict(color=self.color_palette['primary'], width=3),
            hovertemplate="Расходы: %{x:,.0f}<br>Эффективность: %{y:.3f}<extra></extra>"
        ))
        
        # Текущий уровень расходов
        if current_spend is not None:
            current_saturation = np.power(current_spend, alpha) / (np.power(current_spend, alpha) + np.power(gamma, alpha))
            fig.add_trace(go.Scatter(
                x=[current_spend],
                y=[current_saturation],
                mode='markers',
                name='Текущие расходы',
                marker=dict(color=self.color_palette['danger'], size=12, symbol='diamond'),
                hovertemplate=f"Текущие расходы: {current_spend:,.0f}<br>Эффективность: {current_saturation:.3f}<extra></extra>"
            ))
        
        # Зона эффективности
        efficient_spend = gamma * 1.2
        fig.add_vrect(
            x0=0, x1=efficient_spend,
            fillcolor=self.color_palette['success'], opacity=0.1,
            annotation_text="Эффективная зона", annotation_position="top left"
        )
        
        fig.add_vrect(
            x0=efficient_spend, x1=max_spend,
            fillcolor=self.color_palette['warning'], opacity=0.1,
            annotation_text="Зона насыщения", annotation_position="top right"
        )
        
        fig.update_layout(
            title={'text': title, 'x': 0.5, 'xanchor': 'center'},
            xaxis_title="Расходы на рекламу",
            yaxis_title="Эффективность (нормализованная)",
            height=500,
            template="plotly_white"
        )
        
        return fig
    
    def create_model_diagnostics(self, y_true, y_pred, title="Диагностика модели"):
        """Создание диагностических графиков модели."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Прогноз vs Реальность', 'Остатки', 'Q-Q Plot', 'Распределение остатков']
        )
        
        residuals = y_true - y_pred
        
        # 1. Прогноз vs Реальность
        fig.add_trace(
            go.Scatter(x=y_true, y=y_pred, mode='markers', name='Данные',
                      marker=dict(color=self.color_palette['primary'])),
            row=1, col=1
        )
        # Линия идеального прогноза
        min_val, max_val = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
        fig.add_trace(
            go.Scatter(x=[min_val, max_val], y=[min_val, max_val], mode='lines',
                      name='Идеальный прогноз', line=dict(color='red', dash='dash')),
            row=1, col=1
        )
        
        # 2. Остатки vs Прогноз
        fig.add_trace(
            go.Scatter(x=y_pred, y=residuals, mode='markers', name='Остатки',
                      marker=dict(color=self.color_palette['secondary'])),
            row=1, col=2
        )
        fig.add_hline(y=0, line_dash="dash", line_color="red", row=1, col=2)
        
        # 3. Q-Q Plot (упрощенный)
        from scipy import stats
        theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(residuals)))
        sample_quantiles = np.sort(residuals)
        
        fig.add_trace(
            go.Scatter(x=theoretical_quantiles, y=sample_quantiles, mode='markers',
                      name='Q-Q', marker=dict(color=self.color_palette['success'])),
            row=2, col=1
        )
        
        # 4. Гистограмма остатков
        fig.add_trace(
            go.Histogram(x=residuals, name='Распределение', 
                        marker=dict(color=self.color_palette['warning'])),
            row=2, col=2
        )
        
        fig.update_layout(
            title={'text': title, 'x': 0.5, 'xanchor': 'center'},
            height=600,
            template="plotly_white",
            showlegend=False
        )
        
        return fig
    
    def create_media_mix_evolution(self, data, media_columns, title="Эволюция медиа-микса"):
        """Создание графика эволюции медиа-микса во времени."""
        # Вычисляем доли каналов для каждого периода
        media_data = data[media_columns + ['date']].copy()
        media_data['total'] = media_data[media_columns].sum(axis=1)
        
        for col in media_columns:
            media_data[f'{col}_share'] = media_data[col] / media_data['total'] * 100
        
        fig = go.Figure()
        
        colors = [self.media_colors.get(col.lower().replace('_spend', ''), 
                                       self.color_palette['primary']) for col in media_columns]
        
        for i, col in enumerate(media_columns):
            fig.add_trace(go.Scatter(
                x=media_data['date'],
                y=media_data[f'{col}_share'],
                mode='lines',
                name=col.replace('_spend', '').title(),
                line=dict(color=colors[i], width=2),
                stackgroup='one',
                hovertemplate=f"<b>{col}</b><br>Дата: %{{x}}<br>Доля: %{{y:.1f}}%<extra></extra>"
            ))
        
        fig.update_layout(
            title={'text': title, 'x': 0.5, 'xanchor': 'center'},
            xaxis_title="Дата",
            yaxis_title="Доля бюджета (%)",
            height=400,
            template="plotly_white",
            hovermode='x unified'
        )
        
        return fig