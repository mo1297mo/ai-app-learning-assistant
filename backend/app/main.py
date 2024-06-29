import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from api_v1.routers.rag import qa_router

app = FastAPI(
    title="AI Starter Kit backend API",
    docs_url="/docs"
)

# CORS settings
origins = [
    "http://localhost:8000",
    "http://localhost:3000",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:3000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Root endpoint
@app.get("/")
async def root(request: Request):
    return {"message": "Server is up and running!"}

# Include QA router
app.include_router(qa_router, prefix="/api/v1", tags=["QA"])

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", reload=True, port=8000)
