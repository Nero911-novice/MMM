# export_manager.py
"""
Система экспорта результатов Marketing Mix Model в Excel и PDF.
Версия 2.1 с расширенными возможностями отчетности и исправленными ошибками.
"""

import pandas as pd
import numpy as np
from datetime import datetime
from io import BytesIO
import base64

# Excel export
try:
    import openpyxl
    from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
    from openpyxl.chart import BarChart, Reference
    EXCEL_AVAILABLE = True
except ImportError:
    EXCEL_AVAILABLE = False

# PDF export
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors
    from reportlab.lib.units import inch
    PDF_AVAILABLE = True
except ImportError:
    PDF_AVAILABLE = False

from config import COLOR_PALETTE, format_number

class ExportManager:
    """Класс для экспорта результатов MMM в различные форматы."""
    
    def __init__(self):
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def check_dependencies(self):
        """Проверка доступности библиотек для экспорта."""
        return {
            'excel': EXCEL_AVAILABLE,
            'pdf': PDF_AVAILABLE
        }
    
    def _safe_get(self, data, key, default=None):
        """Безопасное получение значения из словаря."""
        if data is None:
            return default
        if isinstance(data, dict):
            return data.get(key, default)
        return default
    
    def _safe_access_df(self, data, operation='len'):
        """Безопасный доступ к DataFrame."""
        if data is None:
            return 0 if operation == 'len' else False
        if isinstance(data, pd.DataFrame):
            if operation == 'len':
                return len(data)
            elif operation == 'empty':
                return data.empty
            elif operation == 'exists':
                return True
        return 0 if operation == 'len' else False
    
    def export_to_excel(self, model_results, filename=None):
        """
        Экспорт результатов MMM в Excel файл.
        
        Parameters:
        -----------
        model_results : dict
            Словарь с результатами модели (contributions, roas_data, metrics, etc.)
        filename : str
            Имя файла (опционально)
        """
        if not EXCEL_AVAILABLE:
            raise ImportError("Для экспорта в Excel установите: pip install openpyxl xlsxwriter")
        
        if filename is None:
            filename = f"MMM_Report_{self.timestamp}.xlsx"
        
        # Проверка входных данных
        if model_results is None:
            model_results = {}
        
        # Создание буфера в памяти
        buffer = BytesIO()
        
        # Создание Excel файла
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            
            # Лист 1: Сводка
            self._create_summary_sheet(writer, model_results)
            
            # Лист 2: Декомпозиция продаж
            contributions = self._safe_get(model_results, 'contributions', {})
            if contributions:
                self._create_decomposition_sheet(writer, contributions)
            
            # Лист 3: ROAS анализ
            roas_data = self._safe_get(model_results, 'roas_data')
            if roas_data is not None and not self._safe_access_df(roas_data, 'empty'):
                self._create_roas_sheet(writer, roas_data)
            
            # Лист 4: Метрики качества модели
            model_metrics = self._safe_get(model_results, 'model_metrics', {})
            if model_metrics:
                self._create_metrics_sheet(writer, model_metrics)
            
            # Лист 5: Оптимизация бюджета
            optimization_results = self._safe_get(model_results, 'optimization_results')
            if optimization_results:
                self._create_optimization_sheet(writer, optimization_results)
            
            # Лист 6: Сценарии
            scenarios = self._safe_get(model_results, 'scenarios')
            if scenarios:
                self._create_scenarios_sheet(writer, scenarios)
            
            # Лист 7: Данные для анализа
            raw_data = self._safe_get(model_results, 'raw_data')
            if raw_data is not None and not self._safe_access_df(raw_data, 'empty'):
                self._create_data_sheet(writer, raw_data)
            
            # Лист 8: Инсайты и рекомендации
            self._create_insights_sheet(writer, model_results)
        
        buffer.seek(0)
        return buffer.getvalue(), filename
    
    def _create_summary_sheet(self, writer, model_results):
        """Создание сводного листа."""
        ws_name = "Сводка"
        
        # Создание сводной информации с безопасным доступом
        summary_data = {
            'Параметр': [
                'Дата создания отчета',
                'Качество модели (R²)',
                'Точность прогноза (%)',
                'Количество медиа-каналов',
                'Период анализа',
                'Общий бюджет (тыс. руб)',
                'Средний ROAS',
                'Базовая линия (%)',
                'Статус модели'
            ],
            'Значение': [
                datetime.now().strftime("%d.%m.%Y %H:%M"),
                f"{self._safe_get(model_results, 'r2_score', 0):.3f}",
                f"{self._safe_get(model_results, 'accuracy', 0):.1f}%",
                len(self._safe_get(model_results, 'media_channels', [])),
                self._safe_get(model_results, 'analysis_period', 'Н/Д'),
                f"{self._safe_get(model_results, 'total_budget', 0)/1000:.0f}",
                f"{self._safe_get(model_results, 'avg_roas', 0):.2f}",
                f"{self._safe_get(model_results, 'base_contribution_pct', 0):.1f}%",
                self._safe_get(model_results, 'model_status', 'Неизвестно')
            ]
        }
        
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_excel(writer, sheet_name=ws_name, index=False, startrow=2)
        
        # Форматирование
        workbook = writer.book
        worksheet = writer.sheets[ws_name]
        
        # Заголовок
        worksheet['A1'] = 'ОТЧЕТ ПО MARKETING MIX MODEL'
        worksheet['A1'].font = Font(size=16, bold=True)
        worksheet.merge_cells('A1:B1')
        
        # Стилизация таблицы
        self._style_excel_table(worksheet, start_row=3, end_row=3+len(summary_df), 
                               start_col=1, end_col=2)
    
    def _create_decomposition_sheet(self, writer, contributions):
        """Создание листа декомпозиции продаж."""
        ws_name = "Декомпозиция"
        
        # Подготовка данных
        contrib_data = []
        total_contribution = sum(contributions.values()) if contributions else 1
        
        for channel, value in contributions.items():
            contrib_data.append({
                'Канал': str(channel).replace('_spend', '').replace('_', ' ').title(),
                'Вклад (шт)': int(float(value)) if value is not None else 0,
                'Доля (%)': round(float(value) / total_contribution * 100, 1) if total_contribution > 0 and value is not None else 0,
                'Тип': 'Базовая линия' if channel == 'Base' else 'Медиа-канал'
            })
        
        contrib_df = pd.DataFrame(contrib_data)
        contrib_df = contrib_df.sort_values('Вклад (шт)', ascending=False)
        
        # Запись в Excel
        contrib_df.to_excel(writer, sheet_name=ws_name, index=False, startrow=2)
        
        # Форматирование
        worksheet = writer.sheets[ws_name]
        worksheet['A1'] = 'ДЕКОМПОЗИЦИЯ ПРОДАЖ ПО КАНАЛАМ'
        worksheet['A1'].font = Font(size=14, bold=True)
        worksheet.merge_cells('A1:D1')
        
        self._style_excel_table(worksheet, start_row=3, end_row=3+len(contrib_df), 
                               start_col=1, end_col=4)
        
        # Добавление диаграммы
        self._add_chart_to_sheet(worksheet, contrib_df, 'F3', chart_type='bar')
    
    def _create_roas_sheet(self, writer, roas_data):
        """Создание листа ROAS анализа."""
        ws_name = "ROAS"
        
        if isinstance(roas_data, pd.DataFrame) and not roas_data.empty:
            # Добавление интерпретации ROAS
            roas_with_interpretation = roas_data.copy()
            roas_with_interpretation['Оценка'] = roas_with_interpretation['ROAS'].apply(
                lambda x: 'Отлично' if x >= 3.0 else 'Хорошо' if x >= 2.0 else 'Приемлемо' if x >= 1.5 else 'Плохо'
            )
            roas_with_interpretation['Рекомендация'] = roas_with_interpretation['ROAS'].apply(
                lambda x: 'Увеличить бюджет' if x >= 3.0 else 'Сохранить уровень' if x >= 2.0 else 'Оптимизировать' if x >= 1.5 else 'Сократить/перенастроить'
            )
            
            roas_with_interpretation.to_excel(writer, sheet_name=ws_name, index=False, startrow=2)
            
            # Форматирование
            worksheet = writer.sheets[ws_name]
            worksheet['A1'] = 'АНАЛИЗ ROAS ПО КАНАЛАМ'
            worksheet['A1'].font = Font(size=14, bold=True)
            worksheet.merge_cells('A1:F1')
            
            self._style_excel_table(worksheet, start_row=3, end_row=3+len(roas_with_interpretation), 
                                   start_col=1, end_col=6)
    
    def _create_metrics_sheet(self, writer, model_metrics):
        """Создание листа метрик качества модели."""
        ws_name = "Качество модели"
        
        metrics_data = []
        for metric, value in model_metrics.items():
            if isinstance(value, (int, float)):
                if 'Точность' in metric or '%' in metric:
                    formatted_value = f"{value:.1f}%"
                elif 'Качество' in metric:
                    formatted_value = f"{value:.3f}"
                else:
                    formatted_value = f"{value:,.0f}"
            else:
                formatted_value = str(value)
                
            metrics_data.append({
                'Метрика': metric,
                'Значение': formatted_value,
                'Интерпретация': self._interpret_metric(metric, value)
            })
        
        metrics_df = pd.DataFrame(metrics_data)
        metrics_df.to_excel(writer, sheet_name=ws_name, index=False, startrow=2)
        
        # Форматирование
        worksheet = writer.sheets[ws_name]
        worksheet['A1'] = 'МЕТРИКИ КАЧЕСТВА МОДЕЛИ'
        worksheet['A1'].font = Font(size=14, bold=True)
        worksheet.merge_cells('A1:C1')
        
        self._style_excel_table(worksheet, start_row=3, end_row=3+len(metrics_df), 
                               start_col=1, end_col=3)
    
    def _create_optimization_sheet(self, writer, optimization_results):
        """Создание листа результатов оптимизации."""
        ws_name = "Оптимизация"
        
        allocation = self._safe_get(optimization_results, 'allocation', {})
        if allocation:
            opt_data = []
            total_budget = sum(allocation.values())
            
            for channel, budget in allocation.items():
                opt_data.append({
                    'Канал': str(channel).replace('_spend', '').replace('_', ' ').title(),
                    'Оптимальный бюджет': int(float(budget)) if budget is not None else 0,
                    'Доля (%)': round(float(budget) / total_budget * 100, 1) if total_budget > 0 and budget is not None else 0
                })
            
            opt_df = pd.DataFrame(opt_data)
            opt_df.to_excel(writer, sheet_name=ws_name, index=False, startrow=2)
            
            # Добавление сводки оптимизации
            worksheet = writer.sheets[ws_name]
            summary_start_row = len(opt_df) + 5
            summary_data = [
                ['Прогнозируемые продажи', f"{self._safe_get(optimization_results, 'predicted_sales', 0):,.0f}"],
                ['Прогнозируемый ROAS', f"{self._safe_get(optimization_results, 'predicted_roas', 0):.2f}"],
                ['Прогнозируемый ROI', f"{self._safe_get(optimization_results, 'predicted_roi', 0):.2f}"],
                ['Общий бюджет', f"{self._safe_get(optimization_results, 'total_budget_used', 0):,.0f}"],
                ['Метод оптимизации', self._safe_get(optimization_results, 'optimization_method', 'Н/Д')]
            ]
            
            for i, (param, value) in enumerate(summary_data):
                worksheet.cell(row=summary_start_row + i, column=1, value=param)
                worksheet.cell(row=summary_start_row + i, column=2, value=value)
            
            # Форматирование
            worksheet['A1'] = 'РЕЗУЛЬТАТЫ ОПТИМИЗАЦИИ БЮДЖЕТА'
            worksheet['A1'].font = Font(size=14, bold=True)
            worksheet.merge_cells('A1:C1')
    
    def _create_scenarios_sheet(self, writer, scenarios):
        """Создание листа сценарного анализа."""
        ws_name = "Сценарии"
        
        scenarios_df = pd.DataFrame(scenarios).T
        scenarios_df.to_excel(writer, sheet_name=ws_name, startrow=2)
        
        # Форматирование
        worksheet = writer.sheets[ws_name]
        worksheet['A1'] = 'СЦЕНАРНЫЙ АНАЛИЗ'
        worksheet['A1'].font = Font(size=14, bold=True)
        worksheet.merge_cells('A1:D1')
    
    def _create_data_sheet(self, writer, raw_data):
        """Создание листа с исходными данными."""
        ws_name = "Данные"
        
        if isinstance(raw_data, pd.DataFrame):
            raw_data.to_excel(writer, sheet_name=ws_name, index=False)
            
            # Форматирование заголовков
            worksheet = writer.sheets[ws_name]
            for col in range(1, len(raw_data.columns) + 1):
                cell = worksheet.cell(row=1, column=col)
                cell.font = Font(bold=True)
                cell.fill = PatternFill(start_color="CCCCCC", end_color="CCCCCC", fill_type="solid")
    
    def _create_insights_sheet(self, writer, model_results):
        """Создание листа с автоматическими инсайтами."""
        ws_name = "Инсайты"
        
        # Генерация автоматических инсайтов
        insights = self.create_automated_insights(model_results)
        
        insights_data = []
        
        # Инсайты по производительности
        for insight in insights['performance_insights']:
            insights_data.append({'Категория': 'Производительность', 'Инсайт': insight})
        
        # Возможности оптимизации
        for insight in insights['optimization_opportunities']:
            insights_data.append({'Категория': 'Оптимизация', 'Инсайт': insight})
        
        # Предупреждения
        for insight in insights['risk_alerts']:
            insights_data.append({'Категория': 'Риски', 'Инсайт': insight})
        
        # Факторы успеха
        for insight in insights['success_factors']:
            insights_data.append({'Категория': 'Успех', 'Инсайт': insight})
        
        insights_df = pd.DataFrame(insights_data)
        insights_df.to_excel(writer, sheet_name=ws_name, index=False, startrow=2)
        
        # Форматирование
        worksheet = writer.sheets[ws_name]
        worksheet['A1'] = 'АВТОМАТИЧЕСКИЕ ИНСАЙТЫ И РЕКОМЕНДАЦИИ'
        worksheet['A1'].font = Font(size=14, bold=True)
        worksheet.merge_cells('A1:B1')
        
        self._style_excel_table(worksheet, start_row=3, end_row=3+len(insights_df), 
                               start_col=1, end_col=2)
    
    def _style_excel_table(self, worksheet, start_row, end_row, start_col, end_col):
        """Применение стилей к таблице в Excel."""
        # Стиль заголовков
        for col in range(start_col, end_col + 1):
            cell = worksheet.cell(row=start_row, column=col)
            cell.font = Font(bold=True, color="FFFFFF")
            cell.fill = PatternFill(start_color="4472C4", end_color="4472C4", fill_type="solid")
            cell.alignment = Alignment(horizontal="center")
        
        # Границы таблицы
        thin_border = Border(
            left=Side(style='thin'),
            right=Side(style='thin'),
            top=Side(style='thin'),
            bottom=Side(style='thin')
        )
        
        for row in range(start_row, end_row + 1):
            for col in range(start_col, end_col + 1):
                worksheet.cell(row=row, column=col).border = thin_border
        
        # Автоширина колонок
        for col in range(start_col, end_col + 1):
            column_letter = openpyxl.utils.get_column_letter(col)
            worksheet.column_dimensions[column_letter].auto_size = True
    
    def _add_chart_to_sheet(self, worksheet, data_df, position, chart_type='bar'):
        """Добавление диаграммы в Excel лист."""
        if chart_type == 'bar' and len(data_df) > 0:
            chart = BarChart()
            chart.title = "Распределение по каналам"
            chart.y_axis.title = "Значение"
            chart.x_axis.title = "Каналы"
            
            # Данные для диаграммы
            data = Reference(worksheet, min_col=3, min_row=3, max_row=3+len(data_df), max_col=3)
            categories = Reference(worksheet, min_col=1, min_row=4, max_row=3+len(data_df))
            
            chart.add_data(data, titles_from_data=True)
            chart.set_categories(categories)
            
            worksheet.add_chart(chart, position)
    
    def _interpret_metric(self, metric_name, value):
        """Интерпретация метрики для бизнеса."""
        if 'Качество' in metric_name and isinstance(value, (int, float)):
            if value >= 0.8:
                return "Отличное качество"
            elif value >= 0.7:
                return "Хорошее качество"
            elif value >= 0.5:
                return "Удовлетворительное"
            else:
                return "Требует улучшения"
        
        elif 'Точность' in metric_name and isinstance(value, (int, float)):
            if value >= 85:
                return "Высокая точность"
            elif value >= 75:
                return "Хорошая точность"
            elif value >= 60:
                return "Приемлемая точность"
            else:
                return "Низкая точность"
        
        return "Анализировать индивидуально"
    
    def export_to_pdf(self, model_results, filename=None):
        """
        Экспорт результатов MMM в PDF файл.
        
        Parameters:
        -----------
        model_results : dict
            Словарь с результатами модели
        filename : str
            Имя файла (опционально)
        """
        if not PDF_AVAILABLE:
            raise ImportError("Для экспорта в PDF установите: pip install reportlab")
        
        if filename is None:
            filename = f"MMM_Report_{self.timestamp}.pdf"
        
        # Проверка входных данных
        if model_results is None:
            model_results = {}
        
        buffer = BytesIO()
        doc = SimpleDocTemplate(buffer, pagesize=A4)
        story = []
        
        # Стили
        styles = getSampleStyleSheet()
        title_style = ParagraphStyle(
            'CustomTitle',
            parent=styles['Heading1'],
            fontSize=18,
            spaceAfter=30,
            alignment=1  # Center
        )
        
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading2'],
            fontSize=14,
            spaceAfter=12,
            textColor=colors.HexColor('#4472C4')
        )
        
        # Заголовок отчета
        story.append(Paragraph("ОТЧЕТ ПО MARKETING MIX MODEL", title_style))
        story.append(Paragraph(f"Дата создания: {datetime.now().strftime('%d.%m.%Y %H:%M')}", styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Исполнительное резюме
        story.append(Paragraph("ИСПОЛНИТЕЛЬНОЕ РЕЗЮМЕ", heading_style))
        
        summary_text = f"""
        <b>Качество модели:</b> R² = {self._safe_get(model_results, 'r2_score', 'Н/Д')}<br/>
        <b>Точность прогноза:</b> {self._safe_get(model_results, 'accuracy', 0):.1f}%<br/>
        <b>Статус модели:</b> {self._safe_get(model_results, 'model_status', 'Неизвестно')}<br/>
        <b>Количество каналов:</b> {len(self._safe_get(model_results, 'media_channels', []))}<br/>
        <b>Средний ROAS:</b> {self._safe_get(model_results, 'avg_roas', 0):.2f}<br/>
        """
        story.append(Paragraph(summary_text, styles['Normal']))
        story.append(Spacer(1, 20))
        
        # Ключевые инсайты
        insights = self.create_automated_insights(model_results)
        
        story.append(Paragraph("КЛЮЧЕВЫЕ ИНСАЙТЫ", heading_style))
        
        for insight in insights['performance_insights'][:3]:  # Топ 3 инсайта
            story.append(Paragraph(f"• {insight}", styles['Normal']))
        
        story.append(Spacer(1, 15))
        
        # Декомпозиция продаж
        contributions = self._safe_get(model_results, 'contributions', {})
        if contributions:
            story.append(Paragraph("ДЕКОМПОЗИЦИЯ ПРОДАЖ", heading_style))
            
            contrib_data = [['Канал', 'Вклад', 'Доля (%)']]
            total = sum(contributions.values())
            
            for channel, value in contributions.items():
                contrib_data.append([
                    str(channel).replace('_spend', '').replace('_', ' ').title(),
                    f"{int(float(value)) if value is not None else 0:,}",
                    f"{float(value)/total*100:.1f}%" if total > 0 and value is not None else "0%"
                ])
            
            table = Table(contrib_data)
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            story.append(table)
            story.append(Spacer(1, 20))
        
        # ROAS анализ
        roas_data = self._safe_get(model_results, 'roas_data')
        if roas_data is not None and not self._safe_access_df(roas_data, 'empty'):
            story.append(Paragraph("АНАЛИЗ ROAS", heading_style))
            
            if isinstance(roas_data, pd.DataFrame) and not roas_data.empty:
                roas_table_data = [['Канал', 'ROAS', 'Оценка']]
                
                for _, row in roas_data.iterrows():
                    roas_val = float(row.get('ROAS', 0))
                    assessment = 'Отлично' if roas_val >= 3.0 else 'Хорошо' if roas_val >= 2.0 else 'Приемлемо' if roas_val >= 1.5 else 'Плохо'
                    
                    roas_table_data.append([
                        str(row.get('Channel', 'Неизвестно')),
                        f"{roas_val:.2f}",
                        assessment
                    ])
                
                roas_table = Table(roas_table_data)
                roas_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#4472C4')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, 0), 12),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                story.append(roas_table)
                story.append(Spacer(1, 20))
        
        # Рекомендации
        story.append(Paragraph("РЕКОМЕНДАЦИИ", heading_style))
        
        recommendations = self._generate_recommendations(model_results)
        for rec in recommendations:
            story.append(Paragraph(f"• {rec}", styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Риски и возможности
        story.append(Paragraph("РИСКИ И ВОЗМОЖНОСТИ", heading_style))
        
        for risk in insights['risk_alerts']:
            story.append(Paragraph(f"⚠️ {risk}", styles['Normal']))
        
        for opp in insights['optimization_opportunities']:
            story.append(Paragraph(f"📈 {opp}", styles['Normal']))
        
        story.append(Spacer(1, 20))
        
        # Футер
        footer_text = "Создано с помощью Marketing Mix Model Analytics Platform v2.1"
        story.append(Paragraph(footer_text, styles['Normal']))
        
        # Сборка PDF
        doc.build(story)
        buffer.seek(0)
        
        return buffer.getvalue(), filename
    
    def _generate_recommendations(self, model_results):
        """Генерация рекомендаций на основе результатов модели."""
        recommendations = []
        
        # Рекомендации по качеству модели
        r2_score = self._safe_get(model_results, 'r2_score', 0)
        if r2_score >= 0.8:
            recommendations.append("Модель демонстрирует высокое качество. Рекомендации можно использовать для планирования бюджета.")
        elif r2_score >= 0.5:
            recommendations.append("Модель показывает приемлемое качество. Используйте рекомендации с осторожностью.")
        else:
            recommendations.append("Качество модели низкое. Требуется улучшение данных или параметров модели.")
        
        # Рекомендации по ROAS
        avg_roas = self._safe_get(model_results, 'avg_roas', 0)
        if avg_roas >= 3.0:
            recommendations.append("Средний ROAS высокий. Рассмотрите увеличение общего бюджета.")
        elif avg_roas >= 2.0:
            recommendations.append("ROAS находится в приемлемом диапазоне. Фокусируйтесь на оптимизации распределения.")
        else:
            recommendations.append("ROAS ниже ожидаемого. Требуется пересмотр стратегии или каналов.")
        
        # Рекомендации по декомпозиции
        base_pct = self._safe_get(model_results, 'base_contribution_pct', 0)
        if base_pct > 70:
            recommendations.append("Высокая доля базовой линии. Проверьте эффективность медиа-каналов.")
        elif base_pct < 30:
            recommendations.append("Низкая базовая линия. Возможно переоценивается влияние медиа.")
        else:
            recommendations.append("Здоровое соотношение между базовой линией и медиа-эффектами.")
        
        return recommendations
    
    def export_quick_summary(self, model_results, format='excel'):
        """
        Быстрый экспорт краткой сводки результатов.
        
        Parameters:
        -----------
        model_results : dict
            Словарь с результатами модели
        format : str
            Формат экспорта ('excel' или 'pdf')
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Проверка входных данных
        if model_results is None:
            model_results = {}
        
        if format == 'excel':
            filename = f"MMM_Summary_{timestamp}.xlsx"
            buffer = BytesIO()
            
            # Создание простого Excel с основными результатами
            with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                # Основные метрики
                summary_data = {
                    'Метрика': [
                        'Качество модели (R²)',
                        'Точность прогноза (%)',
                        'Средний ROAS',
                        'Общий бюджет',
                        'Статус модели'
                    ],
                    'Значение': [
                        f"{self._safe_get(model_results, 'r2_score', 0):.3f}",
                        f"{self._safe_get(model_results, 'accuracy', 0):.1f}%",
                        f"{self._safe_get(model_results, 'avg_roas', 0):.2f}",
                        f"{self._safe_get(model_results, 'total_budget', 0):,.0f} руб",
                        self._safe_get(model_results, 'model_status', 'Неизвестно')
                    ]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Сводка', index=False)
                
                # ROAS данные если доступны
                roas_data = self._safe_get(model_results, 'roas_data')
                if roas_data is not None and not self._safe_access_df(roas_data, 'empty'):
                    roas_data.to_excel(writer, sheet_name='ROAS', index=False)
            
            buffer.seek(0)
            return buffer.getvalue(), filename
        
        elif format == 'pdf':
            filename = f"MMM_Summary_{timestamp}.pdf"
            buffer = BytesIO()
            
            # Создание простого PDF
            doc = SimpleDocTemplate(buffer, pagesize=A4)
            story = []
            styles = getSampleStyleSheet()
            
            # Заголовок
            story.append(Paragraph("КРАТКАЯ СВОДКА MMM", styles['Title']))
            story.append(Spacer(1, 20))
            
            # Основные результаты
            summary_text = f"""
            <b>Качество модели:</b> R² = {self._safe_get(model_results, 'r2_score', 0):.3f}<br/>
            <b>Точность:</b> {self._safe_get(model_results, 'accuracy', 0):.1f}%<br/>
            <b>ROAS:</b> {self._safe_get(model_results, 'avg_roas', 0):.2f}<br/>
            <b>Статус:</b> {self._safe_get(model_results, 'model_status', 'Неизвестно')}<br/>
            """
            story.append(Paragraph(summary_text, styles['Normal']))
            
            doc.build(story)
            buffer.seek(0)
            return buffer.getvalue(), filename
    
    def create_automated_insights(self, model_results):
        """
        Автоматическая генерация инсайтов на основе результатов модели.
        
        Returns:
        --------
        dict : Автоматически сгенерированные инсайты
        """
        insights = {
            'performance_insights': [],
            'optimization_opportunities': [],
            'risk_alerts': [],
            'success_factors': []
        }
        
        # Безопасное извлечение данных
        r2_score = self._safe_get(model_results, 'r2_score', 0)
        avg_roas = self._safe_get(model_results, 'avg_roas', 0)
        contributions = self._safe_get(model_results, 'contributions', {})
        
        # Инсайты по производительности
        if r2_score >= 0.8:
            insights['performance_insights'].append("Модель демонстрирует высокую точность предсказаний")
        
        if avg_roas >= 3.0:
            insights['performance_insights'].append("Маркетинговые инвестиции показывают отличную окупаемость")
        
        # Возможности оптимизации
        roas_data = self._safe_get(model_results, 'roas_data')
        if roas_data is not None and not self._safe_access_df(roas_data, 'empty'):
            try:
                best_channel = roas_data.loc[roas_data['ROAS'].idxmax()]
                worst_channel = roas_data.loc[roas_data['ROAS'].idxmin()]
                
                insights['optimization_opportunities'].append(
                    f"Лучший канал ({best_channel['Channel']}) показывает ROAS {best_channel['ROAS']:.2f} - рассмотреть увеличение бюджета"
                )
                
                if worst_channel['ROAS'] < 1.5:
                    insights['optimization_opportunities'].append(
                        f"Канал {worst_channel['Channel']} показывает низкий ROAS {worst_channel['ROAS']:.2f} - требует оптимизации"
                    )
            except (KeyError, IndexError, TypeError):
                insights['optimization_opportunities'].append("Рекомендуется детальный анализ эффективности каналов")
        
        # Предупреждения о рисках
        if r2_score < 0.6:
            insights['risk_alerts'].append("Низкое качество модели - рекомендации могут быть неточными")
        
        base_share = 0
        if contributions:
            total_contribution = sum(contributions.values())
            if total_contribution > 0:
                base_share = contributions.get('Base', 0) / total_contribution
                
                if base_share > 0.8:
                    insights['risk_alerts'].append("Высокая доля органических продаж - возможна недооценка медиа-эффектов")
                elif base_share < 0.2:
                    insights['risk_alerts'].append("Низкая доля органических продаж - возможна переоценка медиа-эффектов")
        
        # Факторы успеха
        if avg_roas >= 2.5:
            insights['success_factors'].append("Эффективная медиа-стратегия с высоким ROAS")
        
        if 0.3 <= base_share <= 0.6:
            insights['success_factors'].append("Здоровый баланс между органическими и медиа-продажами")
        
        return insights
    
    def create_export_data(self, model, data, contributions, roas_data, metrics, optimization_results=None, scenarios=None):
        """
        Подготовка данных для экспорта.
        
        Parameters:
        -----------
        model : MarketingMixModel
            Обученная модель
        data : pd.DataFrame
            Исходные данные
        contributions : dict
            Вклады каналов
        roas_data : pd.DataFrame
            Данные ROAS
        metrics : dict
            Метрики качества модели
        optimization_results : dict
            Результаты оптимизации (опционально)
        scenarios : dict
            Результаты сценариев (опционально)
        """
        
        # Базовые расчеты с безопасной обработкой
        total_contribution = sum(contributions.values()) if contributions else 1
        base_contribution = contributions.get('Base', 0) if contributions else 0
        base_pct = (base_contribution / total_contribution * 100) if total_contribution > 0 else 0
        
        avg_roas = 0
        if roas_data is not None and not self._safe_access_df(roas_data, 'empty'):
            if 'ROAS' in roas_data.columns:
                avg_roas = roas_data['ROAS'].mean()
        
        # Определение статуса модели
        r2_score = metrics.get('Качество прогноза', 0) if metrics else 0
        accuracy = metrics.get('Точность модели (%)', 0) if metrics else 0
        
        if r2_score >= 0.8 and accuracy >= 85:
            model_status = "Отлично - готова к использованию"
        elif r2_score >= 0.7 and accuracy >= 75:
            model_status = "Хорошо - подходит для планирования"
        elif r2_score >= 0.5 and accuracy >= 60:
            model_status = "Удовлетворительно - использовать осторожно"
        else:
            model_status = "Плохо - требует улучшения"
        
        # Безопасная обработка медиа-каналов
        media_channels = []
        if hasattr(model, 'media_channels') and model.media_channels:
            media_channels = model.media_channels
        
        # Безопасная обработка периода анализа
        analysis_period = 'Н/Д'
        if data is not None and not self._safe_access_df(data, 'empty') and 'date' in data.columns:
            try:
                analysis_period = f"{data['date'].min().strftime('%Y-%m-%d')} - {data['date'].max().strftime('%Y-%m-%d')}"
            except:
                analysis_period = 'Н/Д'
        
        # Безопасная обработка общего бюджета
        total_budget = 0
        if data is not None and not self._safe_access_df(data, 'empty'):
            for ch in media_channels:
                if ch in data.columns:
                    try:
                        total_budget += data[ch].sum()
                    except:
                        pass
        
        export_data = {
            'r2_score': r2_score,
            'accuracy': accuracy,
            'model_status': model_status,
            'media_channels': media_channels,
            'analysis_period': analysis_period,
            'total_budget': total_budget,
            'avg_roas': avg_roas,
            'base_contribution_pct': base_pct,
            'contributions': contributions or {},
            'roas_data': roas_data,
            'model_metrics': metrics or {},
            'raw_data': data
        }
        
        if optimization_results:
            export_data['optimization_results'] = optimization_results
        
        if scenarios:
            export_data['scenarios'] = scenarios
        
        return export_data
