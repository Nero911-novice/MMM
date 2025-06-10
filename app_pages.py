# app_pages.py
"""
Все страницы Streamlit приложения для Marketing Mix Model v2.1.
Включает новую страницу экспорта результатов.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import base64
from io import BytesIO

from mmm_model import MarketingMixModel
from data_processor import DataProcessor
from visualizer import Visualizer
from budget_optimizer import BudgetOptimizer
from grid_search import MMM_GridSearchOptimizer, add_grid_search_method
from export_manager import ExportManager
from config import (
    GRID_SEARCH_MODES, HELP_MESSAGES, BUSINESS_EXPLANATIONS, 
    TARGET_KEYWORDS, MEDIA_KEYWORDS, EXTERNAL_KEYWORDS
)

class AppPages:
    """Класс, содержащий все страницы приложения."""
    
    def __init__(self, processor, visualizer, optimizer):
        self.processor = processor
        self.visualizer = visualizer
        self.optimizer = optimizer
        
        # Добавляем менеджер экспорта
        self.export_manager = ExportManager()
        
        # Добавляем метод Grid Search к классу модели
        MarketingMixModel.auto_optimize_parameters = add_grid_search_method()
    
    def show_home(self):
        """Главная страница приложения."""
        st.header("Marketing Mix Model - Система оптимизации рекламных бюджетов")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown("""
            ### Что такое Marketing Mix Modeling?
            
            MMM — это статистический подход для измерения влияния различных маркетинговых каналов 
            на бизнес-метрики и оптимизации распределения рекламного бюджета.
            
            #### Ключевые возможности:
            
            **📊 Анализ атрибуции**
            - Определение вклада каждого канала в продажи
            - Учет эффектов переноса (adstock) и насыщения (saturation)
            - Измерение ROAS по каналам
            
            **🎯 Оптимизация бюджета**
            - Поиск оптимального распределения бюджета
            - Прогнозирование эффекта изменений в медиа-планах
            - Сценарное планирование
            
            **🔮 Прогнозирование**
            - "What-if" анализ различных стратегий
            - Моделирование влияния внешних факторов
            - Планирование медиа-активности на будущие периоды
            
            **📄 Профессиональные отчеты** *(Новое в v2.1)*
            - Экспорт результатов в Excel и PDF
            - Автоматические инсайты и рекомендации
            - Готовые для презентации руководству отчеты
            """)
            
        with col2:
            st.markdown("### Математическая модель")
            st.latex(r'''Sales_t = Base + \sum_{i=1}^{n} Adstock_i(Media_i) \times Saturation_i(Media_i) + Externals_t''')
            
            st.markdown("**Где:**")
            st.markdown("- Base — базовая линия продаж")
            st.markdown("- Adstock — эффект переноса")
            st.markdown("- Saturation — эффект насыщения")
            st.markdown("- Externals — внешние факторы")
            
            if st.button("🎲 Загрузить демо-данные", type="primary"):
                demo_data = self.processor.generate_demo_data()
                st.session_state.data = demo_data
                st.success("Демо-данные загружены!")
                st.rerun()
            
            # Новая кнопка быстрого экспорта
            if st.session_state.model_fitted:
                if st.button("📄 Быстрый экспорт", help="Экспорт результатов в PDF"):
                    st.switch_page("📄 Экспорт")
        
        st.markdown("---")
        
        # Демо метрики
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Период анализа", "24 месяца")
        
        with col2:
            st.metric("Медиа-каналы", "5 каналов")
        
        with col3:
            st.metric("Точность модели", "R² > 0.8")
        
        with col4:
            st.metric("Экспорт", "Excel + PDF", help="Новая функция v2.1")

    def show_data(self):
        """Страница управления данными."""
        st.header("📊 Управление данными")
        
        tab1, tab2, tab3 = st.tabs(["Загрузка данных", "Просмотр данных", "Валидация"])
        
        with tab1:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Загрузка файла")
                uploaded_file = st.file_uploader(
                    "Выберите CSV файл с данными",
                    type=['csv'],
                    help="Файл должен содержать временные ряды продаж, медиа-расходов и внешних факторов"
                )
                
                if uploaded_file is not None:
                    try:
                        data = pd.read_csv(uploaded_file)
                        data['date'] = pd.to_datetime(data['date'])
                        st.session_state.data = data
                        st.success(f"Данные загружены: {len(data)} строк")
                    except Exception as e:
                        st.error(f"Ошибка загрузки: {str(e)}")
            
            with col2:
                st.subheader("Демо-данные")
                if st.button("Сгенерировать демо-данные"):
                    demo_data = self.processor.generate_demo_data()
                    st.session_state.data = demo_data
                    st.success("Демо-данные созданы")
                    st.rerun()
        
        with tab2:
            if st.session_state.data is not None:
                data = st.session_state.data
                
                st.subheader("Обзор данных")
                st.dataframe(data.head(10), use_container_width=True)
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Строк", len(data))
                with col2:
                    st.metric("Столбцов", len(data.columns))
                with col3:
                    st.metric("Период", f"{data['date'].min().strftime('%Y-%m')} - {data['date'].max().strftime('%Y-%m')}")
                with col4:
                    st.metric("Пропуски", data.isnull().sum().sum())
                
                # Временные ряды
                st.subheader("Временные ряды основных метрик")
                metrics_cols = [col for col in data.columns if any(keyword in col.lower() 
                               for keyword in TARGET_KEYWORDS)]
                
                if metrics_cols:
                    fig = px.line(data, x='date', y=metrics_cols[0], 
                                title=f"Динамика {metrics_cols[0]}")
                    st.plotly_chart(fig, use_container_width=True)
            else:
                st.info("Загрузите данные для просмотра")
        
        with tab3:
            if st.session_state.data is not None:
                validation_results = self.processor.validate_data(st.session_state.data)
                
                st.subheader("Результаты валидации")
                
                for check, result in validation_results.items():
                    if result['status']:
                        st.success(f"✅ {check}: {result['message']}")
                    else:
                        st.error(f"❌ {check}: {result['message']}")
            else:
                st.info("Загрузите данные для валидации")

    def show_model(self):
        """Страница конфигурации модели."""
        st.header("⚙️ Конфигурация модели")

        if st.session_state.data is None:
            st.warning("Сначала загрузите данные")
            return

        data = st.session_state.data

        # Добавляем общее объяснение MMM
        with st.expander("📚 Математические основы Marketing Mix Model", expanded=False):
            st.markdown(BUSINESS_EXPLANATIONS.get('mmm_theory', """
            ### Теоретическая основа Marketing Mix Modeling
            
            **Marketing Mix Model** представляет собой эконометрическую модель, основанную на регрессионном анализе временных рядов. 
            Математическая формулация базируется на аддитивной декомпозиции продаж.
            """))
    
        tab1, tab2, tab3, tab4 = st.tabs([
            "Переменные модели", 
            "Параметры трансформации", 
            "🤖 Автоматический подбор",
            "Обучение модели"
        ])

        with tab1:
            # Объяснение переменных модели
            with st.expander("📖 Типология переменных в MMM", expanded=False):
                st.markdown("""
                ### Классификация переменных в Marketing Mix Model
                
                **1. Зависимая переменная (Target Variable)**
                - Основная KPI, которую модель должна объяснить и предсказать
                - Требования: временной ряд с достаточной вариативностью
                - Примеры: заказы, продажи, выручка, конверсии
                
                **2. Медиа-каналы (Media Variables)**
                - Контролируемые маркетинговые активности с известными инвестициями
                - Характеристики: положительная корреляция с target, наличие лагов
                - Примеры: затраты на paid search, display, social media, TV, radio
                
                **3. Внешние факторы (External Variables)**
                - Неконтролируемые переменные, влияющие на целевую метрику
                - Типы: макроэкономические, конкурентные, сезонные
                - Функция: контроль смещения оценок медиа-эффектов
                
                **4. Контрольные переменные (Control Variables)**
                - Факторы, не являющиеся медиа, но влияющие на результат
                - Назначение: снижение необъясненной дисперсии модели
                - Примеры: цена продукта, ассортиментные изменения, промо-активность
                """)
        
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Зависимая переменная")
                target_options = [col for col in data.columns if any(keyword in col.lower() 
                                for keyword in TARGET_KEYWORDS)]
                target_var = st.selectbox("Выберите целевую метрику:", target_options)
                
                st.subheader("Медиа-каналы")
                media_options = [col for col in data.columns if any(keyword in col.lower() 
                               for keyword in MEDIA_KEYWORDS)]
                selected_media = st.multiselect("Выберите медиа-каналы:", media_options, default=media_options[:5])
            
            with col2:
                st.subheader("Внешние факторы")
                external_options = [col for col in data.columns if any(keyword in col.lower() 
                                  for keyword in EXTERNAL_KEYWORDS)]
                selected_external = st.multiselect("Выберите внешние факторы:", external_options, default=external_options)
                
                st.subheader("Контрольные переменные")
                control_options = [col for col in data.columns if col not in selected_media + selected_external + [target_var, 'date']]
                selected_controls = st.multiselect("Выберите контрольные переменные:", control_options)
    
        with tab2:
            # Ручная настройка параметров
            st.subheader("Ручная настройка параметров")
            
            st.subheader("Параметры Adstock (эффект переноса)")
            adstock_params = {}
            for media in selected_media:
                with st.expander(f"Настройки для {media}"):
                    col1, col2 = st.columns(2)
                    with col1:
                        decay = st.slider(f"Decay rate для {media}", 0.0, 0.9, 0.5, 0.1, key=f"decay_{media}",
                                        help=HELP_MESSAGES.get('adstock_decay'))
                    with col2:
                        max_lag = st.slider(f"Max lag для {media}", 1, 12, 6, 1, key=f"lag_{media}",
                                          help="Максимальная продолжительность эффекта в периодах")
                    adstock_params[media] = {'decay': decay, 'max_lag': max_lag}
            
            st.subheader("Параметры Saturation (эффект насыщения)")
            saturation_params = {}
            for media in selected_media:
                with st.expander(f"Saturation для {media}"):
                    alpha = st.slider(f"Alpha для {media}", 0.1, 3.0, 1.0, 0.1, key=f"alpha_{media}",
                                    help=HELP_MESSAGES.get('saturation_alpha'))
                    gamma = st.slider(f"Gamma для {media}", 0.1, 2.0, 0.5, 0.1, key=f"gamma_{media}",
                                    help=HELP_MESSAGES.get('saturation_gamma'))
                    saturation_params[media] = {'alpha': alpha, 'gamma': gamma}

        with tab3:  # Новый таб для Grid Search
            self._show_grid_search_tab(selected_media, target_var, selected_external, selected_controls, data)
        
        with tab4:  # Обучение модели
            self._show_training_tab(selected_media, target_var, selected_external, selected_controls, 
                                  adstock_params, saturation_params, data)

    def _show_grid_search_tab(self, selected_media, target_var, selected_external, selected_controls, data):
        """Таб автоматического подбора параметров."""
        st.subheader("🤖 Автоматический подбор параметров")
        
        # Объяснение Grid Search
        with st.expander("❓ Что такое автоматический подбор параметров?", expanded=False):
            st.markdown("""
            ### Grid Search для Marketing Mix Model
            
            **Автоматический подбор** использует алгоритм Grid Search для поиска оптимальных параметров:
            
            🎯 **Что оптимизируется:**
            - **Adstock decay** - скорость затухания рекламного эффекта
            - **Saturation alpha** - форма кривой насыщения  
            - **Saturation gamma** - точка полунасыщения
            
            🔬 **Как работает алгоритм:**
            1. Генерирует сетку возможных значений параметров
            2. Тестирует каждую комбинацию через кросс-валидацию
            3. Выбирает параметры с лучшим качеством модели
            
            ⚡ **Преимущества:**
            - Автоматический поиск без ручной настройки
            - Научно обоснованный подход
            - Защита от переобучения через кросс-валидацию
            """)
        
        # Настройки Grid Search
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("⚙️ Настройки поиска")
            
            search_mode = st.selectbox(
                "Режим поиска",
                list(GRID_SEARCH_MODES.keys()),
                help=HELP_MESSAGES.get('grid_search_mode')
            )
            
            # Получаем настройки из конфига
            mode_config = GRID_SEARCH_MODES[search_mode]
            decay_steps = mode_config['decay_steps']
            alpha_steps = mode_config['alpha_steps']
            gamma_steps = mode_config['gamma_steps']
            cv_folds = mode_config['cv_folds']
            max_combinations = mode_config['max_combinations']
            estimated_time = mode_config['estimated_time']
            
            scoring_metric = st.selectbox(
                "Метрика оптимизации",
                ["r2", "mape"],
                format_func=lambda x: "R² (качество)" if x == "r2" else "MAPE (точность)"
            )
            
            st.info(f"📊 Режим: {search_mode}")
            st.info(f"⏱️ Примерное время: {estimated_time}")
            
        with col2:
            st.subheader("🎯 Каналы для оптимизации")
            
            optimize_channels = st.multiselect(
                "Выберите каналы",
                selected_media,
                default=selected_media,
                help="Параметры будут оптимизированы только для выбранных каналов"
            )
            
            if optimize_channels:
                st.markdown("**Предварительные диапазоны поиска:**")
                for channel in optimize_channels:
                    # Определяем тип канала для показа предполагаемых диапазонов
                    if 'google' in channel.lower() or 'search' in channel.lower():
                        decay_range = "0.2-0.6"
                        alpha_range = "0.8-1.5"
                    elif 'facebook' in channel.lower() or 'social' in channel.lower():
                        decay_range = "0.1-0.4"
                        alpha_range = "0.5-1.2"
                    else:
                        decay_range = "0.2-0.7"
                        alpha_range = "0.5-1.5"
                    
                    st.caption(f"📺 {channel}: decay {decay_range}, alpha {alpha_range}")
            else:
                st.warning("Выберите хотя бы один канал для оптимизации")
        
        # Кнопка запуска Grid Search
        if st.button("🚀 Запустить автоматический подбор", type="primary", disabled=not optimize_channels):
            if not optimize_channels:
                st.error("Выберите каналы для оптимизации")
                return
            
            # Подготовка данных
            try:
                X, y = self.processor.prepare_model_data(
                    data, target_var, selected_media, selected_external, selected_controls
                )
                
                # Прогресс-бар
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                status_text.text("🔍 Запуск Grid Search...")
                progress_bar.progress(10)
                
                # Создание временной модели для поиска
                temp_model = MarketingMixModel()
                
                # Запуск оптимизации
                best_params, best_score, optimizer = temp_model.auto_optimize_parameters(
                    X=X,
                    y=y,
                    media_channels=optimize_channels,
                    decay_steps=decay_steps,
                    alpha_steps=alpha_steps,
                    gamma_steps=gamma_steps,
                    cv_folds=cv_folds,
                    scoring=scoring_metric,
                    max_combinations=max_combinations
                )
                
                progress_bar.progress(100)
                status_text.text("✅ Поиск завершен!")
                
                # Сохранение результатов в session_state
                st.session_state.grid_search_results = {
                    'best_params': best_params,
                    'best_score': best_score,
                    'optimizer': optimizer,
                    'search_completed': True
                }
                
                st.success(f"🎯 Оптимизация завершена! Лучший {scoring_metric}: {best_score:.4f}")
                st.rerun()
                
            except Exception as e:
                st.error(f"Ошибка при Grid Search: {str(e)}")
                st.info("💡 Попробуйте уменьшить количество комбинаций или проверьте данные")
        
        # Отображение результатов Grid Search
        self._show_grid_search_results(selected_media)

    def _show_grid_search_results(self, selected_media):
        """Отображение результатов Grid Search."""
        if hasattr(st.session_state, 'grid_search_results') and st.session_state.grid_search_results.get('search_completed'):
            results = st.session_state.grid_search_results
            
            st.markdown("---")
            st.subheader("📊 Результаты автоматического подбора")
            
            # Основные результаты
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Лучший результат", 
                    f"{results['best_score']:.4f}",
                    help="Качество модели с найденными параметрами"
                )
            
            with col2:
                total_tested = len(results['optimizer'].search_results)
                st.metric(
                    "Протестировано комбинаций", 
                    f"{total_tested}",
                    help="Количество протестированных комбинаций параметров"
                )
            
            with col3:
                # Кнопка применения найденных параметров
                if st.button("✅ Применить найденные параметры", type="primary"):
                    # Применяем параметры к текущим переменным состояния
                    st.session_state.optimized_adstock_params = {
                        ch: {'decay': results['best_params'][ch]['decay']} 
                        for ch in selected_media if ch in results['best_params']
                    }
                    st.session_state.optimized_saturation_params = {
                        ch: {
                            'alpha': results['best_params'][ch]['alpha'],
                            'gamma': results['best_params'][ch]['gamma']
                        } 
                        for ch in selected_media if ch in results['best_params']
                    }
                    st.success("Параметры применены! Переходите к обучению модели.")
                    st.rerun()
            
            # Таблица оптимальных параметров
            st.subheader("🎯 Найденные оптимальные параметры")
            
            params_data = []
            for channel, params in results['best_params'].items():
                params_data.append({
                    'Канал': channel.replace('_spend', '').title(),
                    'Adstock Decay': f"{params['decay']:.3f}",
                    'Saturation Alpha': f"{params['alpha']:.3f}",
                    'Saturation Gamma': f"{params['gamma']:.0f}",
                    'Интерпретация': self._interpret_parameters(params['decay'], params['alpha'])
                })
            
            params_df = pd.DataFrame(params_data)
            st.dataframe(params_df, use_container_width=True, hide_index=True)
            
            # График прогресса поиска
            st.subheader("📈 Прогресс поиска")
            progress_fig = results['optimizer'].plot_search_progress()
            if progress_fig:
                st.plotly_chart(progress_fig, use_container_width=True)
            
            # Анализ важности параметров
            with st.expander("🔬 Анализ важности параметров", expanded=False):
                importance = results['optimizer'].get_parameter_importance(top_n=20)
                
                st.markdown("**Статистика параметров в топ-20 результатах:**")
                for channel, params in importance.items():
                    st.markdown(f"**{channel}:**")
                    for param_name, stats in params.items():
                        st.caption(
                            f"  {param_name}: среднее={stats['mean']:.3f}, "
                            f"диапазон={stats['min']:.3f}-{stats['max']:.3f}, "
                            f"разброс={stats['std']:.3f}"
                        )

    def _show_training_tab(self, selected_media, target_var, selected_external, selected_controls, 
                          adstock_params, saturation_params, data):
        """Таб обучения модели."""
        st.subheader("Обучение модели")
        
        col1, col2 = st.columns(2)
        with col1:
            train_ratio = st.slider("Доля обучающей выборки", 0.6, 0.9, 0.8, 0.05,
                                  help="Временное разделение: обучение всегда предшествует тесту")
            regularization = st.selectbox("Тип регуляризации", ["Ridge", "Lasso", "ElasticNet"],
                                        help=HELP_MESSAGES.get('regularization'))
        
        with col2:
            alpha_reg = st.slider("Коэффициент регуляризации", 0.001, 1.0, 0.01, 0.001,
                                help="Контролирует силу регуляризации: больше = консервативнее")
            cross_val_folds = st.slider("Число фолдов для кросс-валидации", 3, 10, 5, 1,
                                      help=HELP_MESSAGES.get('cv_folds'))
        
        # Проверяем, есть ли оптимизированные параметры
        use_optimized = False
        if (hasattr(st.session_state, 'optimized_adstock_params') and 
            hasattr(st.session_state, 'optimized_saturation_params')):
            
            use_optimized = st.checkbox(
                "✅ Использовать параметры из автоматического подбора",
                value=True,
                help="Применить найденные через Grid Search оптимальные параметры"
            )
            
            if use_optimized:
                st.success("🤖 Будут использованы автоматически найденные оптимальные параметры")
                # Показываем какие параметры будут использованы
                with st.expander("Параметры для обучения", expanded=False):
                    for channel in st.session_state.optimized_adstock_params:
                        decay = st.session_state.optimized_adstock_params[channel]['decay']
                        alpha = st.session_state.optimized_saturation_params[channel]['alpha']
                        gamma = st.session_state.optimized_saturation_params[channel]['gamma']
                        st.caption(f"{channel}: decay={decay:.3f}, alpha={alpha:.3f}, gamma={gamma:.0f}")
                        
        if st.button("🚀 Обучить модель", type="primary"):
            with st.spinner("Обучение модели..."):
                try:
                    # Проверка входных данных
                    if not selected_media:
                        st.error("Выберите хотя бы один медиа-канал")
                        return
                    
                    # Определяем какие параметры использовать
                    if use_optimized:
                        final_adstock_params = st.session_state.optimized_adstock_params
                        final_saturation_params = st.session_state.optimized_saturation_params
                        st.info("🤖 Используются оптимизированные параметры")
                    else:
                        final_adstock_params = adstock_params
                        final_saturation_params = saturation_params
                        st.info("⚙️ Используются параметры ручной настройки")
                    
                    # Создание и обучение модели
                    model = MarketingMixModel(
                        adstock_params=final_adstock_params,
                        saturation_params=final_saturation_params,
                        regularization=regularization,
                        alpha=alpha_reg
                    )
                    
                    # Подготовка данных
                    X, y = self.processor.prepare_model_data(
                        data, target_var, selected_media, selected_external, selected_controls
                    )
                    
                    # Проверка на минимальное количество данных
                    if len(X) < 20:
                        st.error("Недостаточно данных для обучения модели (минимум 20 наблюдений)")
                        return
                    
                    # Обучение
                    train_size = max(10, int(len(X) * train_ratio))
                    X_train, X_test = X[:train_size], X[train_size:]
                    y_train, y_test = y[:train_size], y[train_size:]
                    
                    # Проверка на пустоту тестовой выборки
                    if len(X_test) == 0:
                        X_test = X_train.tail(5).copy()
                        y_test = y_train.tail(5).copy()
                    
                    model.fit(X_train, y_train)
                    
                    # Валидация
                    train_score = model.score(X_train, y_train)
                    test_score = model.score(X_test, y_test)
                    
                    # Сохранение в состояние
                    st.session_state.model = model
                    st.session_state.model_fitted = True
                    st.session_state.X_train = X_train
                    st.session_state.X_test = X_test
                    st.session_state.y_train = y_train
                    st.session_state.y_test = y_test
                    st.session_state.target_var = target_var
                    st.session_state.selected_media = selected_media
                    st.session_state.selected_external = selected_external
                    st.session_state.selected_controls = selected_controls
                    
                    # Результаты
                    st.success("Модель обучена успешно!")
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("R² (train)", f"{train_score:.3f}")
                    with col2:
                        st.metric("R² (test)", f"{test_score:.3f}")
                    with col3:
                        overfitting = train_score - test_score
                        st.metric("Переобучение", f"{overfitting:.3f}", 
                                 delta=None if abs(overfitting) < 0.1 else "Высокое" if overfitting > 0.1 else "Низкое")
                    
                    # Предупреждения о качестве модели
                    if train_score < 0.5:
                        st.warning("⚠️ Низкое качество модели. Попробуйте добавить больше данных или изменить параметры.")
                    elif overfitting > 0.2:
                        st.warning("⚠️ Высокое переобучение. Увеличьте коэффициент регуляризации.")
                    
                except Exception as e:
                    st.error(f"Ошибка обучения модели: {str(e)}")
                    st.info("Попробуйте изменить параметры модели или проверить качество данных.")

    def _interpret_parameters(self, decay, alpha):
        """Интерпретация найденных параметров для пользователя."""
        
        # Интерпретация decay
        if decay < 0.3:
            decay_interp = "Быстрое затухание"
        elif decay < 0.6:
            decay_interp = "Среднее затухание"
        else:
            decay_interp = "Медленное затухание"
        
        # Интерпретация alpha
        if alpha < 0.8:
            alpha_interp = "Быстрое насыщение"
        elif alpha < 1.2:
            alpha_interp = "Умеренное насыщение"
        else:
            alpha_interp = "Медленное насыщение"
        
        return f"{decay_interp}, {alpha_interp}"

    def show_results(self):
        """Страница результатов анализа."""
        st.header("📈 Результаты анализа")
        
        if not st.session_state.model_fitted:
            st.warning("Сначала обучите модель")
            return
        
        # Проверка наличия необходимых данных
        required_session_vars = ['model', 'X_train', 'X_test', 'y_train', 'y_test', 'selected_media']
        missing_vars = [var for var in required_session_vars if var not in st.session_state or st.session_state[var] is None]
        
        if missing_vars:
            st.error(f"Отсутствуют данные: {missing_vars}. Переобучите модель.")
            return
        
        model = st.session_state.model
        
        tab1, tab2, tab3, tab4 = st.tabs(["Качество модели", "Декомпозиция", "ROAS анализ", "Кривые насыщения"])
        
        with tab1:
            self._show_model_quality_tab(model)
        
        with tab2:
            self._show_decomposition_tab(model)
            
        with tab3:
            self._show_roas_tab(model)
            
        with tab4:
            self._show_saturation_tab(model)

    def _show_model_quality_tab(self, model):
        """Таб качества модели."""
        # Объяснение метрик качества
        with st.expander("❓ Что показывают метрики качества модели?", expanded=False):
            st.markdown("""
            **Метрики качества** показывают, насколько хорошо модель научилась предсказывать ваши продажи:
            
            📊 **Качество прогноза** (было R²):
            - Показывает, какую долю изменений в продажах модель может объяснить
            - **90%** = отлично! Модель понимает 90% того, почему продажи растут или падают
            - **70%** = хорошо, модель улавливает основные закономерности
            - **50%** = слабо, модель видит только половину картины
            
            🎯 **Точность модели** (было MAPE):
            - Показывает, насколько точно модель предсказывает количество заказов
            - **90%** = модель очень точная, ошибается только на 10%
            - **80%** = хорошая точность для бизнес-планирования
            - **60%** = приемлемо, но нужна осторожность
            """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Оценка качества модели")
            
            # Получаем качественную оценку
            quality_assessment = model.get_model_quality_assessment(st.session_state.X_test, st.session_state.y_test)
            
            # Показываем статус модели
            st.markdown(f"### {quality_assessment['status']}")
            st.progress(quality_assessment['quality_score'] / 100)
            st.markdown(f"**Общая оценка:** {quality_assessment['quality_score']}/100")
            
            # Бизнес-объяснение
            st.success(quality_assessment['business_explanation']['quality'])
            st.info(quality_assessment['business_explanation']['accuracy'])
            st.markdown(f"**Рекомендация:** {quality_assessment['recommendation']}")
            
        with col2:
            st.subheader("📈 Прогноз vs Реальность")
            y_pred = model.predict(st.session_state.X_test)
            
            fig = self.visualizer.create_time_series_plot(
                pd.DataFrame({
                    'period': range(len(st.session_state.y_test)),
                    'Реальные заказы': st.session_state.y_test,
                    'Прогноз модели': y_pred
                }),
                ['Реальные заказы', 'Прогноз модели'],
                'period',
                "Точность предсказаний модели"
            )
            st.plotly_chart(fig, use_container_width=True)

    def _show_decomposition_tab(self, model):
        """Таб декомпозиции продаж."""
        st.subheader("Декомпозиция продаж")
        
        # Объяснение что показывает декомпозиция
        with st.expander("❓ Что показывает декомпозиция?", expanded=False):
            st.markdown(BUSINESS_EXPLANATIONS.get('waterfall_chart', """
            **Декомпозиция продаж** показывает, откуда приходят ваши заказы:
            - **Base** = заказы, которые идут "сами по себе" (органика, брендинг)
            - **Медиа-каналы** = заказы от конкретной рекламы
            """))
        
        try:
            # Расчет вкладов каналов
            contributions = model.get_media_contributions(st.session_state.X_train, st.session_state.y_train)
            
            # Проверка на корректность данных
            if contributions and len(contributions) > 0:
                # Анализ декомпозиции
                total_contribution = sum(contributions.values())
                base_share = contributions.get('Base', 0) / total_contribution * 100 if total_contribution > 0 else 0
                
                # Предупреждение если Base слишком большой
                if base_share > 80:
                    st.warning(f"⚠️ Базовая линия составляет {base_share:.1f}% продаж. Возможно, модель плохо улавливает влияние рекламы.")
                    st.info("💡 **Попробуйте**: уменьшить коэффициент регуляризации или изменить параметры adstock/saturation")
                elif base_share < 20:
                    st.warning(f"⚠️ Базовая линия всего {base_share:.1f}%. Возможно, модель переоценивает влияние рекламы.")
                else:
                    st.success(f"✅ Здоровая декомпозиция: Базовая линия {base_share:.1f}%, Медиа {100-base_share:.1f}%")
                
                # Waterfall chart
                fig = self.visualizer.create_waterfall_chart(contributions)
                st.plotly_chart(fig, use_container_width=True)
                
                # Таблица вкладов с улучшенным форматированием
                st.subheader("📋 Детализация вкладов")
                contrib_df = pd.DataFrame(list(contributions.items()), columns=['Канал', 'Вклад'])
                contrib_df['Вклад, %'] = (contrib_df['Вклад'] / contrib_df['Вклад'].sum() * 100).round(1)
                contrib_df['Вклад'] = contrib_df['Вклад'].round(0).astype(int)
                
                st.dataframe(contrib_df, use_container_width=True, hide_index=True)
                
            else:
                st.warning("Не удалось рассчитать вклады каналов. Проверьте качество модели.")
                
        except Exception as e:
            st.error(f"Ошибка при расчете декомпозиции: {str(e)}")

    def _show_roas_tab(self, model):
        """Таб ROAS анализа."""
        st.subheader("ROAS по каналам")

        # Объяснение ROAS
        with st.expander("📚 Что такое ROAS и как его интерпретировать", expanded=False):
            st.markdown(BUSINESS_EXPLANATIONS.get('roas_interpretation', """
            **ROAS показывает отдачу с каждого рубля рекламы:**
            - ROAS = 3.0 означает 3 рубля дохода с 1 рубля рекламы
            - ROAS < 1.0 = убыточная реклама
            - ROAS > 3.0 = очень эффективная реклама
            """))

        try:
            if hasattr(st.session_state, 'data') and st.session_state.data is not None:
                roas_data = model.calculate_roas(st.session_state.data, st.session_state.selected_media)

                if not roas_data.empty:
                    fig = self.visualizer.create_roas_comparison(roas_data)
                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("Детализация ROAS")
                    st.dataframe(roas_data, use_container_width=True)
                else:
                    st.warning("Не удалось рассчитать ROAS. Проверьте данные.")
            else:
                st.warning("Данные для расчета ROAS недоступны.")

        except Exception as e:
            st.error(f"Ошибка при расчете ROAS: {str(e)}")

    def _show_saturation_tab(self, model):
        """Таб кривых насыщения."""
        st.subheader("Кривые насыщения")
        
        # Объяснение кривых насыщения
        with st.expander("❓ Что такое кривые насыщения?", expanded=False):
            st.markdown("""
            **Кривые насыщения** показывают, как эффективность рекламного канала меняется при увеличении бюджета.

            🎯 **Простыми словами:**
            - Представьте, что вы поливаете растение водой
            - Сначала каждая капля воды очень помогает росту
            - Но если лить слишком много - эффект уменьшается
            - То же самое с рекламой!
            """)
        
        # Выбор канала для анализа
        selected_channel = st.selectbox("Выберите канал для анализа:", st.session_state.selected_media)
        
        if selected_channel and hasattr(st.session_state, 'data'):
            channel_data = st.session_state.data[selected_channel]
            current_spend = channel_data.mean()
            
            fig = self.visualizer.create_saturation_curve(
                channel_data, 
                alpha=1.0, 
                gamma=None, 
                current_spend=current_spend,
                title=f"Кривая насыщения для {selected_channel.replace('_spend', '').title()}"
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Интерпретация результатов
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("📊 Анализ текущего уровня")
                median_spend = channel_data[channel_data > 0].median() if len(channel_data[channel_data > 0]) > 0 else current_spend
                
                if current_spend < median_spend * 0.7:
                    st.success("🟢 **Недофинансирован**: Можно увеличить бюджет для лучших результатов")
                elif current_spend < median_spend * 1.2:
                    st.info("🟡 **Оптимальный уровень**: Хорошее соотношение затрат и результата")
                else:
                    st.warning("🟠 **Близко к насыщению**: Дополнительные расходы малоэффективны")
            
            with col2:
                st.subheader("🎯 Ключевые точки")
                st.metric("Текущие расходы", f"{current_spend:,.0f} руб")
                st.metric("Медианные расходы", f"{median_spend:,.0f} руб")

    def show_optimization(self):
        """Страница оптимизации бюджета."""
        st.header("💰 Оптимизация бюджета")
        
        # Главное объяснение раздела
        with st.expander("❓ Что такое оптимизация бюджета?", expanded=False):
            st.markdown("""
            **Оптимизация бюджета** - это автоматический поиск наилучшего способа распределить ваши рекламные деньги.
            
            🎯 **Простой пример:**
            У вас есть 1 млн рублей на рекламу. Вопрос: как их разделить между Facebook, Google, TikTok?
            
            **Интуитивный подход:**
            - Facebook: 300,000 руб (30%)
            - Google: 500,000 руб (50%)  
            - TikTok: 200,000 руб (20%)
            - **Результат:** 5,000 заказов
            
            **После оптимизации:**
            - Facebook: 250,000 руб (25%)
            - Google: 600,000 руб (60%)
            - TikTok: 150,000 руб (15%)
            - **Результат:** 5,400 заказов (+400 заказов!)
            
            💡 **Как это работает:**
            1. Модель анализирует эффективность каждого канала
            2. Находит оптимальное соотношение для максимального результата
            3. Учитывает ваши ограничения (минимум/максимум по каналам)
            
            🎯 **Цели оптимизации:**
            - **Максимум заказов** = получить как можно больше заказов
            - **Максимум ROAS** = получить максимальную отдачу с рубля
            - **Максимум ROI** = получить максимальную прибыль
            """)
        
        if not st.session_state.model_fitted:
            st.warning("⚠️ Сначала обучите модель в разделе 'Модель'")
            return
        
        tab1, tab2 = st.tabs(["⚙️ Настройки оптимизации", "📊 Результаты оптимизации"])
        
        with tab1:
            self._show_optimization_settings_tab()
        
        with tab2:
            self._show_optimization_results_tab()

    def _show_optimization_settings_tab(self):
        """Таб настроек оптимизации."""
        st.subheader("⚙️ Настройки оптимизации")
        
        # Объяснение настроек
        with st.expander("❓ Как настроить оптимизацию?", expanded=False):
            st.markdown("""
            **Настройки помогают адаптировать оптимизацию под ваши бизнес-ограничения:**
            
            💰 **Общий бюджет:**
            - Сколько всего денег у вас есть на рекламу в месяц
            - Система распределит эти деньги между каналами оптимально
            
            🎯 **Цель оптимизации:**
            - **Максимум заказов** = приоритет количеству (подходит для роста)
            - **Максимум ROAS** = приоритет эффективности (подходит для прибыльности)
            - **Максимум ROI** = приоритет чистой прибыли
            
            🚧 **Ограничения по каналам:**
            - **Минимум** = меньше этой суммы тратить нельзя (например, минимум по контракту)
            - **Максимум** = больше этой суммы тратить нельзя (например, лимит команды)
            - Без ограничений система может предложить потратить 0 или 100% на один канал
            
            **Пример ограничений:**
            - Facebook: мин 100к (команда справится), макс 500к (больше не потянем)
            - Google: мин 200к (конкуренция), макс без ограничений
            
            🔬 **Математические основы:**
            Система использует методы нелинейной оптимизации (SLSQP, Differential Evolution) 
            для поиска глобального максимума целевой функции с учетом ограничений.
            """)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("💰 Основные параметры")
            total_budget = st.number_input(
                "Общий месячный бюджет (руб)", 
                min_value=10000, 
                value=1000000, 
                step=50000,
                help="Общая сумма, которую вы готовы тратить на рекламу в месяц"
            )
            
            optimization_target = st.selectbox(
                "Цель оптимизации", 
                ["maximize_sales", "maximize_roas", "maximize_roi"],
                format_func=lambda x: {
                    "maximize_sales": "📈 Максимум заказов (рост объемов)",
                    "maximize_roas": "💰 Максимум ROAS (эффективность)",
                    "maximize_roi": "💎 Максимум ROI (прибыльность)"
                }[x],
                help="Что важнее: больше заказов, выше эффективность или больше прибыли?"
            )
            
            # Показываем текущий расход для сравнения
            current_total = sum(
                st.session_state.data[ch].mean()
                for ch in st.session_state.selected_media
            )
            st.info(f"💡 Текущие расходы: {current_total:,.0f} руб/месяц")

        with col2:
            st.subheader("🚧 Ограничения по каналам")
            
            use_constraints = st.checkbox(
                "Использовать ограничения по каналам", 
                value=False,
                help="Если выключено, система может предложить любое распределение"
            )
            
            constraints = {}
            if use_constraints:
                for channel in st.session_state.selected_media:
                    with st.expander(f"⚙️ {channel.replace('_spend', '').title()}", expanded=False):
                        current_avg = st.session_state.data[channel].mean()
                        
                        col_min, col_max = st.columns(2)
                        with col_min:
                            min_spend = st.number_input(
                                f"Минимум", 
                                min_value=0, 
                                max_value=total_budget//2, 
                                value=max(0, int(current_avg * 0.5)),
                                step=10000,
                                key=f"min_{channel}",
                                help="Меньше этой суммы тратить нельзя/неэффективно"
                            )
                        with col_max:
                            max_spend = st.number_input(
                                f"Максимум", 
                                min_value=min_spend, 
                                max_value=total_budget, 
                                value=min(total_budget, int(current_avg * 2)),
                                step=10000,
                                key=f"max_{channel}",
                                help="Больше этой суммы тратить нельзя/неэффективно"
                            )
                        
                        constraints[channel] = {'min': min_spend, 'max': max_spend}
                        st.caption(f"Сейчас тратите: {current_avg:,.0f} руб/месяц")
            
            st.session_state.optimization_settings = {
                'total_budget': total_budget,
                'target': optimization_target,
                'constraints': constraints if use_constraints else None
            }

    def _show_optimization_results_tab(self):
        """Таб результатов оптимизации."""
        if st.button("🎯 Оптимизировать бюджет", type="primary"):
            if 'optimization_settings' not in st.session_state:
                st.error("Сначала настройте параметры оптимизации")
                return
                
            settings = st.session_state.optimization_settings
            
            with st.spinner("Поиск оптимального распределения..."):
                
                # Запуск оптимизации
                optimal_allocation = self.optimizer.optimize_budget(
                    model=st.session_state.model,
                    total_budget=settings['total_budget'],
                    constraints=settings['constraints'],
                    target=settings['target']
                )
                
                # Сохранение результатов в session_state для экспорта
                st.session_state.optimization_results = optimal_allocation
                
                # Результаты оптимизации
                st.success("Оптимизация завершена!")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Оптимальное распределение")
                    optimal_df = pd.DataFrame(list(optimal_allocation['allocation'].items()), 
                                            columns=['Канал', 'Оптимальный бюджет'])
                    optimal_df['Доля, %'] = (optimal_df['Оптимальный бюджет'] / settings['total_budget'] * 100).round(1)
                    st.dataframe(optimal_df, use_container_width=True)
                    
                    # Метрики оптимального решения
                    st.metric("Прогнозируемые продажи", f"{optimal_allocation['predicted_sales']:,.0f}")
                    st.metric("Прогнозируемый ROAS", f"{optimal_allocation['predicted_roas']:.2f}")
                    st.metric("Прогнозируемый ROI", f"{optimal_allocation['predicted_roi']:.2f}")
                
                with col2:
                    st.subheader("Сравнение распределений")
                    
                    # Определяем текущее распределение по каналам
                    current_allocation = {
                        ch: st.session_state.data[ch].mean()
                        for ch in st.session_state.selected_media
                    }

                    # Сравнительная диаграмма
                    fig = self.visualizer.create_optimization_results(
                        current_allocation,
                        optimal_allocation['allocation']
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                # Добавляем кнопку быстрого экспорта
                st.markdown("---")
                if st.button("📊 Экспортировать результаты оптимизации"):
                    # Переключение на страницу экспорта
                    st.info("Переходите в раздел '📄 Экспорт' для создания отчета")

    def show_scenarios(self):
        """Страница сценарного анализа."""
        st.header("🔮 Сценарный анализ")

        # Добавляем общее объяснение сценарного анализа
        with st.expander("📊 Методология сценарного анализа в маркетинге", expanded=False):
            st.markdown("""
            ### Сценарное планирование в Marketing Mix Modeling
            
            **Определение:**
            Сценарный анализ — систематический метод оценки потенциальных последствий различных 
            стратегических решений в области медиа-инвестиций при неопределенных внешних условиях.
            
            **Теоретические основы:**
            
            1. **Детерминистическое моделирование**
               - Модель предполагает фиксированные параметры медиа-трансформаций
               - Учитывает нелинейные эффекты (adstock, saturation)
               - Включает влияние внешних факторов
            
            2. **Сравнительная статика**
               - Анализ изменения равновесных состояний при изменении параметров
               - Ceteris paribus принцип: все остальное остается неизменным
               - Позволяет изолировать эффекты конкретных решений
            
            **Типология сценариев:**
            
            **1. Оптимистичный сценарий**
            - Благоприятная сезонность (seasonality > 1.0)
            - Низкая конкурентная активность (competition < 1.0)
            - Применение: планирование максимальных результатов
            
            **2. Пессимистичный сценарий**
            - Неблагоприятная сезонность (seasonality < 1.0)
            - Высокая конкурентная активность (competition > 1.0)
            - Применение: оценка рисков и планирование contingency
            
            **3. Базовый сценарий**
            - Нейтральные внешние условия (факторы = 1.0)
            - Применение: стандартное планирование
            
            **Интерпретация факторов:**
            
            **Сезонный фактор:**
            - 1.5 = +50% к базовому спросу (высокий сезон)
            - 1.0 = нейтральный период
            - 0.7 = -30% к базовому спросу (низкий сезон)
            
            **Фактор конкуренции:**
            - 1.3 = увеличение конкурентного давления на 30%
            - 1.0 = стабильная конкурентная среда
            - 0.8 = снижение конкурентного давления на 20%
            """)

        if not st.session_state.model_fitted:
            st.warning("Сначала обучите модель")
            return

        tab1, tab2 = st.tabs(["Создание сценариев", "Сравнение сценариев"])

        with tab1:
            self._show_scenario_creation_tab()

        with tab2:
            self._show_scenario_comparison_tab()

    def _show_scenario_creation_tab(self):
        """Таб создания сценариев."""
        st.subheader("Создание нового сценария")
        
        # Добавляем объяснение создания сценариев
        with st.expander("🎯 Рекомендации по созданию сценариев", expanded=False):
            st.markdown("""
            ### Критерии оценки качества сценария
            
            **Метрики для анализа:**
            
            **ROAS (Return on Ad Spend):**
            - **Отличный результат**: ROAS ≥ 3.0
            - **Хороший результат**: ROAS 2.0-3.0
            - **Приемлемый результат**: ROAS 1.5-2.0
            - **Неудовлетворительный**: ROAS < 1.5
            
            **Прогнозируемые продажи:**
            - Сравнивайте с текущим уровнем и историческими данными
            - Учитывайте сезонные колебания
            - Оценивайте реалистичность с точки зрения операционных возможностей
            
            **Общий бюджет:**
            - Должен соответствовать финансовым возможностям
            - Учитывайте ограничения по cash flow
            - Сравнивайте с текущими расходами на маркетинг
            
            **Рекомендации по построению сценариев:**
            
            1. **Консервативный подход**: изменения бюджета ±20% от текущего уровня
            2. **Агрессивный рост**: увеличение бюджета на 50-100%
            3. **Оптимизация**: перераспределение без изменения общего бюджета
            4. **Кризисный**: снижение бюджета на 30-50%
            
            **Практические примеры сценариев:**
            
            **"Черная пятница"**: 
            - Сезонность: 1.8 (высокий спрос)
            - Конкуренция: 1.4 (все активны)
            - Бюджет: +150% к обычному
            
            **"Экономический кризис"**:
            - Сезонность: 0.7 (низкий спрос)
            - Конкуренция: 0.8 (все экономят)
            - Бюджет: -40% к обычному
            
            **"Запуск нового продукта"**:
            - Сезонность: 1.0 (нейтрально)
            - Конкуренция: 1.2 (ответная реакция)
            - Бюджет: +80% на 3 месяца
            """)
        
        scenario_name = st.text_input("Название сценария", "Сценарий 1")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Настройки каналов")
            scenario_budget = {}
            
            # Показываем текущие расходы для сравнения
            st.markdown("**Текущие среднемесячные расходы:**")
            current_totals = {}
            for channel in st.session_state.selected_media:
                current_value = st.session_state.data[channel].mean()
                current_totals[channel] = current_value
                st.caption(f"{channel}: {current_value:,.0f} руб")
            
            total_current = sum(current_totals.values())
            st.caption(f"**Общий текущий бюджет: {total_current:,.0f} руб**")
            
            st.markdown("**Новое распределение:**")
            for channel in st.session_state.selected_media:
                current_value = current_totals[channel]
                scenario_budget[channel] = st.number_input(
                    f"Бюджет {channel}",
                    min_value=0,
                    value=int(current_value),
                    step=1000,
                    key=f"scenario_{channel}",
                    help=f"Текущий уровень: {current_value:,.0f} руб"
                )
        
        with col2:
            st.subheader("Внешние факторы")
            
            seasonality_factor = st.slider(
                "Сезонный фактор", 0.5, 2.0, 1.0, 0.1,
                help="1.5 = высокий сезон (+50%), 0.7 = низкий сезон (-30%)"
            )
            competition_factor = st.slider(
                "Фактор конкуренции", 0.5, 2.0, 1.0, 0.1,
                help="1.3 = усиление конкуренции (-30% эффективности), 0.8 = ослабление (+20%)"
            )
            
            # Прогноз результатов сценария
            if st.button("📊 Рассчитать прогноз"):
                predicted_results = st.session_state.model.predict_scenario(
                    scenario_budget, seasonality_factor, competition_factor
                )
                
                # Сохранение результатов сценария для экспорта
                if 'scenarios_results' not in st.session_state:
                    st.session_state.scenarios_results = {}
                st.session_state.scenarios_results[scenario_name] = predicted_results
                
                st.markdown("### Результаты прогноза")
                
                col_a, col_b, col_c = st.columns(3)
                
                with col_a:
                    st.metric("Прогнозируемые продажи", f"{predicted_results['sales']:,.0f}")
                
                with col_b:
                    roas_result = predicted_results['roas']
                    if roas_result >= 3.0:
                        st.success(f"**ROAS: {roas_result:.2f}** (Отлично)")
                    elif roas_result >= 2.0:
                        st.info(f"**ROAS: {roas_result:.2f}** (Хорошо)")
                    else:
                        st.warning(f"**ROAS: {roas_result:.2f}** (Приемлемо)")
                
                with col_c:
                    st.metric("Общий бюджет", f"{predicted_results['total_spend']:,.0f}")
                
                # Развернутая интерпретация
                st.markdown("### Интерпретация результатов")
                
                # Анализ ROAS
                if roas_result >= 3.0:
                    st.success("""
                    **Отличные результаты**: ROAS выше 3.0 указывает на высокую эффективность 
                    медиа-инвестиций. Рекомендуется реализация данного сценария.
                    """)
                elif roas_result >= 2.0:
                    st.info("""
                    **Хорошие результаты**: ROAS в диапазоне 2.0-3.0 демонстрирует приемлемую 
                    эффективность. Возможны дополнительные оптимизации распределения.
                    """)
                elif roas_result >= 1.5:
                    st.warning("""
                    **Приемлемые результаты**: ROAS 1.5-2.0 находится на нижней границе 
                    эффективности. Требуется анализ альтернативных стратегий.
                    """)
                else:
                    st.error("""
                    **Неудовлетворительные результаты**: ROAS ниже 1.5 указывает на 
                    неэффективность инвестиций. Необходим пересмотр распределения бюджета.
                    """)
                
                # Анализ продаж
                # Для примера сравниваем с "типичным" уровнем
                baseline_sales = st.session_state.data[st.session_state.target_var].mean() if hasattr(st.session_state, 'target_var') else 50000
                sales_change = ((predicted_results['sales'] - baseline_sales) / baseline_sales * 100) if baseline_sales > 0 else 0
                
                if sales_change > 20:
                    st.success(f"Прогнозируемый рост продаж: +{sales_change:.1f}%. Сильная стратегия роста.")
                elif sales_change > 5:
                    st.info(f"Прогнозируемый рост продаж: +{sales_change:.1f}%. Умеренный рост.")
                elif sales_change > -5:
                    st.warning(f"Изменение продаж: {sales_change:+.1f}%. Стабильные результаты.")
                else:
                    st.error(f"Прогнозируемое снижение продаж: {sales_change:.1f}%. Рискованная стратегия.")
                
                # Дополнительные рекомендации на основе факторов
                st.markdown("### Рекомендации по реализации")
                
                if seasonality_factor > 1.2:
                    st.info("📈 **Высокая сезонность**: Подготовьте операционную команду к увеличенной нагрузке")
                if competition_factor > 1.2:
                    st.warning("⚔️ **Высокая конкуренция**: Рассмотрите дифференциацию или премиум-позиционирование")
                total_spend = sum(scenario_budget.values())
                if total_spend > total_current * 1.5:
                    st.warning("💰 **Значительное увеличение бюджета**: Убедитесь в наличии ресурсов для масштабирования")
                
                # Риски и возможности
                st.markdown("### Риски и возможности")
                
                risk_level = "Низкий"
                if roas_result < 2.0:
                    risk_level = "Высокий"
                elif roas_result < 2.5:
                    risk_level = "Средний"
                
                st.markdown(f"**Уровень риска**: {risk_level}")
                
                if risk_level == "Высокий":
                    st.error("🚨 **Высокий риск**: Рекомендуется тестирование на малом бюджете перед полным внедрением")
                elif risk_level == "Средний":
                    st.warning("⚠️ **Средний риск**: Мониторьте результаты и будьте готовы к корректировкам")
                else:
                    st.success("✅ **Низкий риск**: Сценарий можно внедрять с высокой степенью уверенности")

    def _show_scenario_comparison_tab(self):
        """Таб сравнения сценариев."""
        st.subheader("Сравнение предустановленных сценариев")
        
        # Добавляем объяснение сравнения сценариев
        with st.expander("📈 Методология сравнения сценариев", expanded=False):
            st.markdown("""
            ### Принципы сравнительного анализа стратегий

            **Предустановленные стратегии:**

            **1. Текущий сценарий (Current)**
            - Базовая линия для сравнения
            - Основан на исторических средних расходах
            - Показывает результаты при сохранении status quo

            **2. Digital Focus**
            - 80% бюджета на цифровые каналы, 20% на офлайн
            - Стратегия для повышения измеримости и таргетинга
            - Подходит для D2C брендов и e-commerce

            **3. Balanced**
            - Равномерное распределение между всеми каналами
            - Стратегия диверсификации рисков
            - Подходит для тестирования новых каналов

            **4. Performance**
            - Концентрация на каналах с исторически высоким ROAS
            - 70% бюджета на Google + Facebook, 30% на остальные
            - Стратегия максимизации краткосрочной эффективности

            **Критерии выбора оптимальной стратегии:**

            1. **Максимальные продажи**: выбор сценария с наибольшим объемом продаж
            2. **Максимальный ROAS**: приоритет эффективности инвестиций
            3. **Минимальный риск**: выбор наиболее стабильного сценария
            4. **Бюджетные ограничения**: соответствие финансовым возможностям
            
            **Интерпретация результатов:**
            
            **Если Digital Focus лучше:** 
            - Ваши цифровые каналы более эффективны
            - Стоит пересмотреть офлайн инвестиции
            - Возможно, улучшить трекинг офлайн каналов
            
            **Если Balanced лучше:**
            - Все каналы работают примерно одинаково
            - Нет явных лидеров/аутсайдеров
            - Стратегия диверсификации оправдана
            
            **Если Performance лучше:**
            - Есть четкие каналы-лидеры
            - Стоит концентрировать бюджет
            - Остальные каналы менее эффективны
            """)
        
        # Предустановленные сценарии
        current_avg = {channel: st.session_state.data[channel].mean() 
                      for channel in st.session_state.selected_media}
        total_current = sum(current_avg.values())
        
        scenarios = {
            "Текущий": current_avg,
            "Digital Focus": {
                channel: (total_current * 0.8 / len([ch for ch in st.session_state.selected_media if 'offline' not in ch.lower()]) 
                         if 'offline' not in channel.lower() else total_current * 0.2)
                for channel in st.session_state.selected_media
            },
            "Balanced": {channel: total_current / len(st.session_state.selected_media) 
                       for channel in st.session_state.selected_media},
            "Performance": {
                channel: (total_current * 0.7 / len([ch for ch in st.session_state.selected_media if ch in ['google_spend', 'facebook_spend']])
                         if channel in ['google_spend', 'facebook_spend'] else total_current * 0.3 / (len(st.session_state.selected_media) - 2))
                for channel in st.session_state.selected_media
            }
        }
        
        # Расчет прогнозов для всех сценариев
        scenario_results = {}
        for name, budget in scenarios.items():
            results = st.session_state.model.predict_scenario(budget, 1.0, 1.0)
            scenario_results[name] = results
        
        # Сохранение результатов сценариев для экспорта
        st.session_state.scenarios_results = scenario_results
        
        # Таблица сравнения
        comparison_df = pd.DataFrame(scenario_results).T
        comparison_df = comparison_df.round(2)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Визуализация сравнения
        fig = make_subplots(
            rows=1, cols=3,
            subplot_titles=['Продажи', 'ROAS', 'Бюджет']
        )
        
        scenarios_list = list(scenario_results.keys())
        
        # Продажи
        fig.add_trace(
            go.Bar(x=scenarios_list, y=[scenario_results[s]['sales'] for s in scenarios_list], 
                  name='Продажи', showlegend=False),
            row=1, col=1
        )
        
        # ROAS
        fig.add_trace(
            go.Bar(x=scenarios_list, y=[scenario_results[s]['roas'] for s in scenarios_list], 
                  name='ROAS', showlegend=False),
            row=1, col=2
        )
        
        # Бюджет
        fig.add_trace(
            go.Bar(x=scenarios_list, y=[scenario_results[s]['total_spend'] for s in scenarios_list], 
                  name='Бюджет', showlegend=False),
            row=1, col=3
        )
        
        fig.update_layout(title="Сравнение сценариев", height=400)
        st.plotly_chart(fig, use_container_width=True)
        
        # Добавляем анализ после таблицы сравнения
        if 'scenario_results' in locals():
            st.markdown("### Рекомендации по выбору стратегии")
        
            # Находим лучший сценарий по каждому критерию
            best_sales = max(scenario_results.keys(), key=lambda x: scenario_results[x]['sales'])
            best_roas = max(scenario_results.keys(), key=lambda x: scenario_results[x]['roas'])
            most_efficient = min(scenario_results.keys(), key=lambda x: scenario_results[x]['total_spend'])

            col1, col2, col3 = st.columns(3)

            with col1:
                st.success(f"""
            **Максимальные продажи**: {best_sales}

            Продажи: {scenario_results[best_sales]['sales']:,.0f}

            Рекомендуется для: агрессивного роста объемов
            """)

            with col2:
                st.info(f"""
            **Максимальный ROAS**: {best_roas}

            ROAS: {scenario_results[best_roas]['roas']:.2f}

            Рекомендуется для: оптимизации эффективности
            """)

            with col3:
                st.warning(f"""
            **Наименьший бюджет**: {most_efficient}

            Бюджет: {scenario_results[most_efficient]['total_spend']:,.0f}

            Рекомендуется для: ограниченных ресурсов
            """)

            # Общие рекомендации
            st.markdown("### Стратегические рекомендации")
        
            # Сравнение с текущим сценарием
            current_results = scenario_results.get('Текущий', scenario_results.get('Current'))
            if current_results:
                for name, results in scenario_results.items():
                    if name not in ['Текущий', 'Current']:
                        sales_improvement = ((results['sales'] - current_results['sales']) / current_results['sales'] * 100)
                        roas_improvement = ((results['roas'] - current_results['roas']) / current_results['roas'] * 100)
                    
                        if sales_improvement > 10 and roas_improvement > 5:
                            st.success(f"""
                        **{name}**: Превосходит текущую стратегию по всем показателям
                        - Рост продаж: +{sales_improvement:.1f}%
                        - Улучшение ROAS: +{roas_improvement:.1f}%
                        - **Рекомендация**: Приоритетная стратегия для внедрения
                        """)
                        elif sales_improvement > 5:
                            st.info(f"""
                        **{name}**: Увеличивает продажи при сопоставимой эффективности
                        - Рост продаж: +{sales_improvement:.1f}%
                        - Изменение ROAS: {roas_improvement:+.1f}%
                        - **Рекомендация**: Подходит для фазы роста
                        """)
                        elif roas_improvement > 10:
                            st.info(f"""
                        **{name}**: Повышает эффективность инвестиций
                        - Изменение продаж: {sales_improvement:+.1f}%
                        - Улучшение ROAS: +{roas_improvement:.1f}%
                        - **Рекомендация**: Подходит для оптимизации
                        """)
                        else:
                            st.warning(f"""
                        **{name}**: Незначительные улучшения относительно текущей стратегии
                        - Изменение продаж: {sales_improvement:+.1f}%
                        - Изменение ROAS: {roas_improvement:+.1f}%
                        - **Рекомендация**: Второстепенная альтернатива
                        """)

    def show_export(self):
        """Страница экспорта результатов."""
        st.header("📄 Экспорт результатов")
        
        if not st.session_state.model_fitted:
            st.warning("⚠️ Сначала обучите модель для экспорта результатов")
            return
        
        # Проверка доступности библиотек для экспорта
        dependencies = self.export_manager.check_dependencies()
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Экспорт в Excel")
            
            if dependencies['excel']:
                st.success("✅ Поддержка Excel доступна")
                
                # Настройки экспорта
                include_charts = st.checkbox("Включить диаграммы", value=True)
                include_raw_data = st.checkbox("Включить исходные данные", value=False)
                
                if st.button("📊 Экспорт в Excel", type="primary"):
                    try:
                        # Подготовка данных для экспорта
                        export_data = self._prepare_export_data(include_raw_data)
                        
                        # Генерация Excel файла
                        excel_data, filename = self.export_manager.export_to_excel(
                            export_data, 
                            filename=f"MMM_Report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.xlsx"
                        )
                        
                        # Создание ссылки для скачивания
                        b64 = base64.b64encode(excel_data).decode()
                        href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Скачать Excel файл</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        
                        st.success(f"✅ Excel файл готов: {filename}")
                        
                    except Exception as e:
                        st.error(f"Ошибка создания Excel: {str(e)}")
            else:
                st.error("❌ Для экспорта в Excel установите: pip install openpyxl xlsxwriter")
        
        with col2:
            st.subheader("📄 Экспорт в PDF")
            
            if dependencies['pdf']:
                st.success("✅ Поддержка PDF доступна")
                
                # Настройки PDF
                include_recommendations = st.checkbox("Включить рекомендации", value=True)
                include_methodology = st.checkbox("Включить методологию", value=False)
                
                if st.button("📄 Экспорт в PDF", type="primary"):
                    try:
                        # Подготовка данных для экспорта
                        export_data = self._prepare_export_data(include_raw_data=False)
                        
                        # Генерация PDF файла
                        pdf_data, filename = self.export_manager.export_to_pdf(
                            export_data,
                            filename=f"MMM_Report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.pdf"
                        )
                        
                        # Создание ссылки для скачивания
                        b64 = base64.b64encode(pdf_data).decode()
                        href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Скачать PDF файл</a>'
                        st.markdown(href, unsafe_allow_html=True)
                        
                        st.success(f"✅ PDF файл готов: {filename}")
                        
                    except Exception as e:
                        st.error(f"Ошибка создания PDF: {str(e)}")
            else:
                st.error("❌ Для экспорта в PDF установите: pip install reportlab")
        
        # Информация о содержании отчета
        st.markdown("---")
        st.subheader("📋 Содержание отчета")
        
        with st.expander("Что включается в отчет", expanded=False):
            st.markdown("""
            **Excel отчет содержит:**
            - 📊 Сводка модели (качество, метрики, статус)
            - 🎯 Декомпозиция продаж по каналам
            - 💰 ROAS анализ с интерпретацией
            - 📈 Метрики качества модели
            - 🔧 Оптимизация бюджета (если выполнялась)
            - 🔮 Сценарный анализ (если выполнялся)
            - 📊 Исходные данные (опционально)
            - 📈 Диаграммы и графики (опционально)
            - 💡 Автоматические инсайты и рекомендации
            
            **PDF отчет содержит:**
            - 📄 Исполнительное резюме
            - 📊 Основные результаты в табличном виде
            - 💡 Бизнес-рекомендации на основе анализа
            - 🔬 Интерпретация результатов
            - 📈 Ключевые метрики и их значения
            - ⚠️ Риски и возможности
            """)
        
        # Быстрые действия экспорта
        st.markdown("---")
        st.subheader("⚡ Быстрые действия")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📋 Краткая сводка (Excel)", help="Быстрый экспорт основных результатов"):
                try:
                    export_data = self._prepare_export_data(include_raw_data=False)
                    summary_data, filename = self.export_manager.export_quick_summary(export_data, format='excel')
                    
                    b64 = base64.b64encode(summary_data).decode()
                    href = f'<a href="data:application/vnd.openxmlformats-officedocument.spreadsheetml.sheet;base64,{b64}" download="{filename}">Скачать краткую сводку</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    st.success("Краткая сводка готова!")
                except Exception as e:
                    st.error(f"Ошибка: {str(e)}")
        
        with col2:
            if st.button("📋 Краткая сводка (PDF)", help="Быстрый экспорт основных результатов в PDF"):
                try:
                    export_data = self._prepare_export_data(include_raw_data=False)
                    summary_data, filename = self.export_manager.export_quick_summary(export_data, format='pdf')
                    
                    b64 = base64.b64encode(summary_data).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="{filename}">Скачать краткую сводку PDF</a>'
                    st.markdown(href, unsafe_allow_html=True)
                    st.success("Краткая сводка PDF готова!")
                except Exception as e:
                    st.error(f"Ошибка: {str(e)}")
        
        with col3:
            if st.session_state.optimization_results:
                st.info("💰 Результаты оптимизации включены в экспорт")
            else:
                st.caption("💰 Оптимизация не выполнена")
    
    def _prepare_export_data(self, include_raw_data=False):
        """Подготовка данных для экспорта."""
        try:
            model = st.session_state.model
            
            # Получение метрик качества модели
            if hasattr(st.session_state, 'X_test') and st.session_state.X_test is not None:
                metrics = model.get_model_metrics(st.session_state.X_test, st.session_state.y_test)
            else:
                # Демо метрики если нет тестовых данных
                metrics = {
                    'Качество прогноза': 0.75,
                    'Точность модели (%)': 80.0,
                    'Средняя ошибка': 500,
                    'Типичная ошибка': 750
                }
            
            # Получение вкладов каналов
            if hasattr(st.session_state, 'X_train') and st.session_state.X_train is not None:
                contributions = model.get_media_contributions(
                    st.session_state.X_train, 
                    st.session_state.y_train
                )
            else:
                contributions = {'Base': 50000, 'facebook_spend': 30000, 'google_spend': 25000}
            
            # Получение ROAS данных
            if hasattr(st.session_state, 'data') and st.session_state.data is not None:
                roas_data = model.calculate_roas(st.session_state.data, st.session_state.selected_media)
            else:
                roas_data = pd.DataFrame({
                    'Channel': ['Facebook', 'Google'],
                    'ROAS': [2.5, 3.2],
                    'Total_Spend': [450000, 670000],
                    'Total_Contribution': [1125000, 2144000]
                })
            
            # Формирование данных для экспорта
            export_data = self.export_manager.create_export_data(
                model=model,
                data=st.session_state.data if include_raw_data else None,
                contributions=contributions,
                roas_data=roas_data,
                metrics=metrics,
                optimization_results=getattr(st.session_state, 'optimization_results', None),
                scenarios=getattr(st.session_state, 'scenarios_results', None)
            )
            
            return export_data
            
        except Exception as e:
            st.error(f"Ошибка подготовки данных для экспорта: {str(e)}")
            return None
