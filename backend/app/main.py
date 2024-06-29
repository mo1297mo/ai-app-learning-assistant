from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn



# Create FastAPI app
app = FastAPI(
    title="AI Starter Kit backend API",
    docs_url="/docs"
)

# Set CORS origins
origins = [
    "http://localhost:8000",
    "http://localhost:3000",
    "http://127.0.0.1:8000",
    "http://127.0.0.1:3000",
]

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define a simple root endpoint
@app.get("/")
async def root(request: Request):
    return {"message": "Server is up and running!"}

# Include your routers
app.include_router(qa_router, prefix="/api/v1", tags=["QA"])

# Run the application
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
