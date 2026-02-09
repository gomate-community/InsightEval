from pathlib import Path
import sys
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger
from backend.routers import  chat_router, insight_router

app = FastAPI(
    title="IR-Trends API",
    description="Information Retrieval Trends API for tracking papers, topics, and Q&A",
    version="1.0.0",
)

logger.info("FastAPI应用初始化完成")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 允许所有来源访问
    allow_credentials=False,  # 当allow_origins为["*"]时，必须设置为False
    allow_methods=["*"],
    allow_headers=["*"],
)

logger.info("CORS中间件配置完成")

app.include_router(chat_router.router)
logger.info("Chat路由注册完成")

app.include_router(insight_router.router)
logger.info("Insight路由注册完成")


@app.get("/")
async def root():
    return {"message": "IR-Trends API is running"}


@app.get("/api/health")
async def health_check():
    """健康检查接口"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "version": "1.0.0"
    }


if __name__ == "__main__":
    # uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
    # 在 Docker 环境中，需要从项目根目录导入 backend.main
    import os
    if os.getenv("DOCKER_ENV"):
        uvicorn.run(
            "backend.main:app",
            host="0.0.0.0",
            port=8000,
            reload=False,  # Docker 环境中禁用 reload
            log_level="info",
        )
    else:
        uvicorn.run(
            "main:app",
            host="0.0.0.0",
            port=8000,
            reload=True,
            log_level="info",
        )
