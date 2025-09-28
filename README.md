# Insight AI RAG Assistant

## Описание проекта
AI-ассистент на основе Retrieval-Augmented Generation (RAG), обученный на личных конспектах, учебниках и ноутбуках по курсу Data Science.

## Этапы проекта
1. **Загрузка и обработка данных** разных типов (ODT, PDF, IPYNB, изображения).
2. **Создание векторной базы знаний** на основе текстов.
3. **Поиск релевантного контекста** с помощью векторного поиска.
4. **Интеграция с LLM** для генерации ответов с учетом контекста.
5. **Реализация FastAPI сервера** и **Telegram-бота** для взаимодействия с пользователем.

## Стек технологий
- Python 3.10
- FastAPI
- LangChain / LlamaIndex
- FAISS / ChromaDB
- Sentence-Transformers
- Hugging Face LLMs
- Docker, boto3, AWS S3
- Streamlit (опционально)

## Запуск проекта
1. Создай виртуальное окружение:
   ```bash
   pipenv install
   pipenv shell
    2. Запусти сервер:
       uvicorn src.app.main:app --reload
    3. (опционально) Запусти Telegram-бота:
       python src/app/telegram_bot.py

---