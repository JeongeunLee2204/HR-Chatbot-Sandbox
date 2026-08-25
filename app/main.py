"""
RAG 로직을 API 서버로 감싸기.

실무에서는 프로토타입(rag.py를 그냥 실행)이랑
"서비스"(다른 시스템이 호출할 수 있는 API)는 다른 물건입니다.
FastAPI가 그 다리 역할을 합니다.

실행: uvicorn app.main:app --reload
문서 자동 생성됨: http://localhost:8000/docs
"""

from fastapi import FastAPI
from pydantic import BaseModel

from app.rag import ask

app = FastAPI(title="HR Chatbot API")


class ChatRequest(BaseModel):
    question: str


class ChatResponse(BaseModel):
    answer: str


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    answer = ask(req.question)
    return ChatResponse(answer=answer)


@app.get("/health")
def health():
    return {"status": "ok"}
