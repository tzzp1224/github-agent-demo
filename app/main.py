# 文件路径: app/main.py
import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
from fastapi.responses import StreamingResponse # 确保引入 StreamingResponse
import uvicorn

from app.core.config import settings
from app.services.agent_service import agent_stream

# === 核心修复点：这里要导入 process_chat_stream ===
# 之前的名字是 process_chat，现在改名了
from app.services.chat_service import process_chat_stream
settings.validate()
app = FastAPI(title="GitHub RAG Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True, 
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def health_check():
    return {"status": "ok"}

@app.get("/analyze")
async def analyze(url: str, session_id: str): 
    if not session_id:
        return {"error": "Missing session_id"}
    return EventSourceResponse(agent_stream(url, session_id))

# === 修改：聊天接口改为流式 ===
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_query = data.get("query")
    session_id = data.get("session_id")
    
    if not user_query: return {"answer": "请输入问题"}
    if not session_id: return {"answer": "Session 丢失"}

    # 使用 StreamingResponse
    from app.services.chat_service import process_chat_stream
    
    return StreamingResponse(
        process_chat_stream(user_query, session_id), 
        media_type="text/plain"
    )

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)