from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import rag_router as router

api_app = FastAPI(title="IPL 2022 RAG Service")

api_app.add_middleware(
    CORSMiddleware
)

api_app.include_router(router)