#!/usr/bin/env python3

"""
Hiroyuki SLM API Service
FastAPI-based API for Hiroyuki-style chat responses
"""

"""
NOT IN USE
""""

import logging
from fastapi import FastAPI, HTTPException
import uvicorn
from pydantic import BaseModel

from slm_model import HiroyukiSLM

logger = logging.getLogger(__name__)
app = FastAPI()
slm = HiroyukiSLM()

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str
    input: str

@app.get("/health")
async def health_check():
    """APIのヘルスチェックエンドポイント"""
    return {"status": "ok"}

@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    """ユーザーのメッセージに対してひろゆき風の返答を生成するエンドポイント"""
    try:
        response = await slm.generate(request.message)
        return ChatResponse(response=response, input=request.message)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def start():
    """APIサーバーの起動関数"""
    logger.info("Starting Hiroyuki-SLM API server...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
