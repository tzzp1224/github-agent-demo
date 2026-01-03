# 文件路径: app/main.py
import sys
import io
# 删除 uuid, Cookie 引用，不再需要在后端生成
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from sse_starlette.sse import EventSourceResponse
import uvicorn

from app.core.config import settings
from app.services.agent_service import agent_stream
from app.services.vector_service import store_manager
from app.utils.llm_client import client

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

# === 修改点 1: 分析接口直接接收 session_id 参数 ===
@app.get("/analyze")
async def analyze(url: str, session_id: str): 
    # 直接使用前端传来的 session_id
    if not session_id:
        return {"error": "Missing session_id"}
    return EventSourceResponse(agent_stream(url, session_id))

# === 修改点 2: 聊天接口从 Body 接收 session_id ===
@app.post("/chat")
async def chat(request: Request):
    data = await request.json()
    user_query = data.get("query")
    # 获取前端传来的 session_id
    session_id = data.get("session_id")
    
    if not user_query: return {"answer": "请输入问题"}
    if not session_id: return {"answer": "Session 丢失，请刷新页面重试"}

    # 从管理器获取对应的 VectorStore
    vector_db = store_manager.get_store(session_id)

    # 1. 混合检索
    relevant_docs = vector_db.search_hybrid(user_query, top_k=3)
    
    context_str = ""
    sources = []
    for doc in relevant_docs:
        file_info = doc['file']
        if 'class' in doc.get('metadata', {}):
            class_name = doc['metadata']['class']
            if class_name: # 只有非空时才加
                file_info += f" (Class: {class_name})"
            
        context_str += f"\n--- 引用自 {file_info} ---\n{doc['content'][:1000]}...\n"
        sources.append(doc['file'])

    # 2. 生成回答
    if not client: return {"answer": "LLM Error"}

    # 如果检索结果为空，直接告诉 LLM
    if not context_str:
        context_str = "No relevant code found in the repository."

    prompt = f"""
    Context:
    {context_str}

    Question:
    {user_query}

    Answer in Chinese.
    """
    
    try:
        res = client.models.generate_content(model=settings.MODEL_NAME, contents=prompt)
        return {"answer": res.text, "sources": list(set(sources))}
    except Exception as e:
        return {"answer": f"Error: {str(e)}"}

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=settings.HOST, port=settings.PORT, reload=True)