#!/usr/bin/env python3
"""
Simple CORS test server
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

app = FastAPI()

# CORS middleware - same configuration as main.py
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5174"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {"message": "CORS test server is running"}

@app.get("/api/v1/health")
async def health():
    return {"status": "ok", "message": "CORS test server is healthy"}

@app.post("/api/v1/query/universal")
async def test_query():
    return {"message": "CORS test endpoint working"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)