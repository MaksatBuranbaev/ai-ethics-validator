# server.py
from fastapi import FastAPI
from api.routes import router
from services.model_manager import init_models_from_config

app = FastAPI(title="LLM Ethics Evaluator API", version="1.0.0")

# Инициализация моделей при старте
init_models_from_config()

# Подключение роутеров
app.include_router(router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=True)