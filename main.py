# main.py
"""
Основное приложение Marketing Mix Model.
Точка входа для Streamlit приложения.
"""

import streamlit as st
import warnings
warnings.filterwarnings('ignore')

# Импорты модулей проекта
from data_processor import DataProcessor
from visualizer import Visualizer
from budget_optimizer import BudgetOptimizer
from app_pages import AppPages
from config import CUSTOM_CSS, APP_PAGES

# Конфигурация страницы
st.set_page_config(
    page_title="Marketing Mix Model",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Применение CSS стилей
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

class MMM_App:
    """Главный класс приложения Marketing Mix Model."""
    
    def __init__(self):
        """Инициализация приложения."""
        # Инициализация компонентов
        self.processor = DataProcessor()
        self.visualizer = Visualizer()
        self.optimizer = BudgetOptimizer()
        
        # Инициализация страниц
        self.pages = AppPages(self.processor, self.visualizer, self.optimizer)
        
        # Инициализация состояния сессии
        self._initialize_session_state()
    
    def _initialize_session_state(self):
        """Инициализация переменных состояния сессии."""
        # Основные переменные состояния
        if 'model' not in st.session_state:
            st.session_state.model = None
        if 'data' not in st.session_state:
            st.session_state.data = None
        if 'model_fitted' not in st.session_state:
            st.session_state.model_fitted = False
            
        # Переменные для обученной модели
        session_vars = [
            'X_train', 'X_test', 'y_train', 'y_test',
            'target_var', 'selected_media', 'selected_external', 'selected_controls'
        ]
        for var in session_vars:
            if var not in st.session_state:
                st.session_state[var] = None
        
        # Переменные для Grid Search
        if 'grid_search_results' not in st.session_state:
            st.session_state.grid_search_results = {}
        if 'optimized_adstock_params' not in st.session_state:
            st.session_state.optimized_adstock_params = {}
        if 'optimized_saturation_params' not in st.session_state:
            st.session_state.optimized_saturation_params = {}
            
        # Переменные для оптимизации
        if 'optimization_settings' not in st.session_state:
            st.session_state.optimization_settings = {}
    
    def _render_sidebar(self):
        """Отрисовка боковой панели навигации."""
        with st.sidebar:
            st.header("📊 MMM Navigation")
            
            # Выбор страницы
            page = st.selectbox(
                "Выберите раздел:",
                APP_PAGES,
                help="Навигация по разделам приложения"
            )
            
            st.markdown("---")
            
            # Информация о состоянии
            st.markdown("### 📈 Статус системы")
            
            # Статус данных
            if st.session_state.data is not None:
                st.success("✅ Данные загружены")
                data_info = f"Строк: {len(st.session_state.data)}"
                st.caption(data_info)
            else:
                st.warning("⚠️ Данные не загружены")
            
            # Статус модели
            if st.session_state.model_fitted:
                st.success("✅ Модель обучена")
                if hasattr(st.session_state, 'selected_media') and st.session_state.selected_media:
                    st.caption(f"Каналов: {len(st.session_state.selected_media)}")
            else:
                st.warning("⚠️ Модель не обучена")
            
            # Статус Grid Search
            if (hasattr(st.session_state, 'grid_search_results') and 
                st.session_state.grid_search_results.get('search_completed')):
                st.info("🤖 Grid Search выполнен")
            else:
                st.caption("🔍 Grid Search не выполнен")
            
            st.markdown("---")
            
            # Быстрые действия
            st.markdown("### ⚡ Быстрые действия")
            
            if st.button("🎲 Демо-данные", help="Загрузить демонстрационные данные"):
                demo_data = self.processor.generate_demo_data()
                st.session_state.data = demo_data
                st.success("Демо-данные загружены!")
                st.rerun()
            
            if st.session_state.data is not None and st.button("🔄 Сбросить модель", help="Очистить обученную модель"):
                # Сброс состояния модели
                st.session_state.model = None
                st.session_state.model_fitted = False
                st.session_state.grid_search_results = {}
                st.session_state.optimized_adstock_params = {}
                st.session_state.optimized_saturation_params = {}
                st.success("Модель сброшена!")
                st.rerun()
            
            st.markdown("---")
            
            # Информация о приложении
            with st.expander("ℹ️ О приложении", expanded=False):
                st.markdown("""
                **Marketing Mix Model v2.0**
                
                Система для анализа эффективности 
                маркетинговых каналов и оптимизации 
                рекламного бюджета.
                
                **Возможности:**
                - 📊 Анализ атрибуции
                - 🤖 Автоподбор параметров  
                - 💰 Оптимизация бюджета
                - 🔮 Сценарное планирование
                
                **Разработано на основе:**
                - Эконометрических принципов
                - Машинного обучения
                - Научных исследований в области MMM
                """)
        
        return page
    
    def _render_main_content(self, page):
        """Отрисовка основного контента в зависимости от выбранной страницы."""
        try:
            if page == "🏠 Главная":
                self.pages.show_home()
            elif page == "📊 Данные":
                self.pages.show_data()
            elif page == "⚙️ Модель":
                self.pages.show_model()
            elif page == "📈 Результаты":
                self.pages.show_results()
            elif page == "💰 Оптимизация":
                self.pages.show_optimization()
            elif page == "🔮 Сценарии":
                self.pages.show_scenarios()
            else:
                st.error(f"Неизвестная страница: {page}")
                
        except Exception as e:
            st.error(f"Ошибка при отображении страницы '{page}': {str(e)}")
            st.info("Попробуйте перезагрузить страницу или обратитесь к разработчику.")
            
            # Подробная информация об ошибке в expander
            with st.expander("🔧 Техническая информация об ошибке", expanded=False):
                st.code(str(e))
                
                # Предложения по решению
                st.markdown("**Возможные решения:**")
                if "not fitted" in str(e).lower():
                    st.markdown("- Обучите модель в разделе 'Модель'")
                elif "data" in str(e).lower():
                    st.markdown("- Загрузите данные в разделе 'Данные'")
                elif "session_state" in str(e).lower():
                    st.markdown("- Перезагрузите страницу")
                else:
                    st.markdown("- Проверьте качество загруженных данных")
                    st.markdown("- Убедитесь, что все этапы выполнены последовательно")
    
    def _render_footer(self):
        """Отрисовка подвала приложения."""
        st.markdown("---")
        
        # Краткая статистика
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.session_state.data is not None:
                st.metric("📊 Загружено строк", len(st.session_state.data))
            else:
                st.metric("📊 Загружено строк", 0)
        
        with col2:
            if (hasattr(st.session_state, 'selected_media') and 
                st.session_state.selected_media):
                st.metric("📺 Медиа-каналов", len(st.session_state.selected_media))
            else:
                st.metric("📺 Медиа-каналов", 0)
        
        with col3:
            if st.session_state.model_fitted:
                st.metric("🎯 Статус модели", "Обучена")
            else:
                st.metric("🎯 Статус модели", "Не обучена")
        
        with col4:
            grid_search_status = "Выполнен" if (
                hasattr(st.session_state, 'grid_search_results') and 
                st.session_state.grid_search_results.get('search_completed')
            ) else "Не выполнен"
            st.metric("🤖 Grid Search", grid_search_status)
        
        # Copyright и версия
        st.markdown(
            """
            <div style='text-align: center; color: #666; padding: 20px 0;'>
                <small>
                    Marketing Mix Model v2.0 | 
                    Powered by Streamlit & Scientific Python Stack |
                    © 2024 MMM Analytics
                </small>
            </div>
            """, 
            unsafe_allow_html=True
        )
    
    def run(self):
        """Основной метод запуска приложения."""
        # Заголовок приложения
        st.title("🎯 Marketing Mix Model")
        st.markdown("**Система планирования и оптимизации рекламных бюджетов**")
        
        # Отрисовка боковой панели и получение выбранной страницы
        selected_page = self._render_sidebar()
        
        # Отрисовка основного контента
        self._render_main_content(selected_page)
        
        # Отрисовка подвала
        self._render_footer()

# Функции для обработки ошибок
def handle_streamlit_errors():
    """Обработчик ошибок Streamlit."""
    try:
        # Основной код приложения
        app = MMM_App()
        app.run()
        
    except Exception as e:
        st.error("🚨 Критическая ошибка приложения")
        st.markdown(f"**Описание ошибки:** {str(e)}")
        
        st.markdown("""
        **Возможные причины:**
        - Проблемы с зависимостями Python
        - Нехватка памяти
        - Ошибки в данных
        
        **Рекомендации:**
        1. Перезагрузите страницу (F5)
        2. Очистите кэш браузера
        3. Проверьте качество загружаемых данных
        4. Обратитесь к администратору системы
        """)
        
        # Кнопка для перезагрузки
        if st.button("🔄 Перезагрузить приложение"):
            st.rerun()

def check_dependencies():
    """Проверка наличия всех необходимых зависимостей."""
    required_modules = [
        'streamlit', 'pandas', 'numpy', 'plotly', 
        'sklearn', 'scipy', 'warnings'
    ]
    
    missing_modules = []
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    if missing_modules:
        st.error(f"🚨 Отсутствуют модули: {', '.join(missing_modules)}")
        st.markdown("""
        **Для установки зависимостей выполните:**
        ```bash
        pip install -r requirements.txt
        ```
        """)
        st.stop()

def main():
    """Главная функция приложения."""
    # Проверка зависимостей
    check_dependencies()
    
    # Запуск приложения с обработкой ошибок
    handle_streamlit_errors()

# Точка входа приложения
if __name__ == "__main__":
    main()