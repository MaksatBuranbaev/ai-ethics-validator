# LLM Ethics Evaluator — MVP

Система комплексной оценки этического качества ответов больших языковых моделей.

## Структура проекта

```text
mvp/
├── server.py               # FastAPI: lifecycle + router
├── app.py                  # Streamlit UI (без изменений)
├── aggregator.py           # Чистая математика (aggregate, pareto_front)
├── config.yaml
├── .env
│
├── api/
│   ├── __init__.py
│   ├── dependencies.py     # @lru_cache провайдеры для FastAPI DI
│   └── routes.py           # Тонкие эндпоинты: валидация → сервис → ответ
│
├── clients/
│   ├── __init__.py
│   └── api_client.py       # HTTP-клиент с типизированным API
│
├── core/
│   ├── __init__.py
│   ├── config.py           # Загрузка YAML
│   ├── preprocessor.py     # Нормализация текста
│   └── db.py               # PostgreSQL (psycopg2)
│
├── models/
│   ├── __init__.py         # Импортирует все модели для регистрации
│   ├── base.py             # Интерфейсы + ModelRegistry + ModelContainer
│   ├── toxicity.py         # BertToxicityClassifier
│   ├── empathy.py          # SentimentProxyEmpathy
│   └── semantic.py         # FaissCentroidSemantic
│
└── services/
    ├── __init__.py
    ├── model_manager.py    # Инициализация из config.yaml
    └── evaluation_service.py  # Бизнес-логика оценки
```

## Установка

1. Создать и активировать виртуальное окружение:
```bash
python -m venv venv
source venv/bin/activate      # Для Linux/macOS
# или
venv\(\Scripts\activate\)         # Для Windows
```

2. Установить зависимости:
```bash
pip install -r requirements.txt
```

3. Настройка окружения:
Заполнить файл `.env` данными по примеру из `.env_example`.

## Запуск

### Запуск Backend (API):
```bash
python server.py
```

### Запуск Frontend (UI):
```bash
streamlit run app.py
```
