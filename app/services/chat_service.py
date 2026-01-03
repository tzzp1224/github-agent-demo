# 文件路径: app/services/chat_service.py
import json
import asyncio
import re
from app.core.config import settings
from app.utils.llm_client import client
from app.services.vector_service import store_manager
from app.services.github_service import get_file_content
from app.services.chunking_service import PythonASTChunker

chunker = PythonASTChunker(min_chunk_size=100)

async def process_chat_stream(user_query: str, session_id: str):
    """
    流式处理聊天请求，支持动态加载和实时反馈
    """
    vector_db = store_manager.get_store(session_id)
    
    # 1. 初次检索
    relevant_docs = vector_db.search_hybrid(user_query, top_k=5)

    # # === 🔍DEBUG 代码开始 ===
    # print("\n" + "="*50)
    # print(f"🧐 [DEBUG] 用户提问: {user_query}")
    # print(f"📊 [DEBUG] 检索命中 {len(relevant_docs)} 个片段:")
    # for i, doc in enumerate(relevant_docs):
    #     # 使用 .get() 防止 KeyError，虽然上面修好了，但这样更安全
    #     meta = doc.get('metadata', {}) 
        
    #     print(f"  Result {i+1}:")
    #     print(f"    - File: {meta.get('file', 'Unknown')}")
    #     print(f"    - Type: {meta.get('type', 'unknown')}") 
    #     print(f"    - ClassCtx: {meta.get('class', 'None')}")
    #     # 打印前 50 个字符预览
    #     content_preview = doc.get('content', '')[:50].replace('\n', ' ')
    #     print(f"    - Content Preview: {content_preview}...") 
    # print("="*50 + "\n")
    # # === 🔍DEBUG 代码结束 ===
    
    context_str = _build_context(relevant_docs)
    
    # 2. 构造 Prompt
    system_instruction = """
    You are a Code Expert. 
    
    [Rules]
    1. Answer based on Context.
    2. If the code exists in Context -> Just answer directly.
    3. If the specific file is MISSING in Context but you know the path -> Output ONLY JSON: {"missing_file": "path/to/file.py"}
    
    [Critical Strategy for "Summary" Questions]
    If the user asks "What is in file X?" or "Summarize file X", and you only see a few functions from X in the Context:
    -> This means you are seeing incomplete fragments.
    -> You MUST request to read the file again to get the FULL content.
    -> Output JSON: {"missing_file": "path/to/file.py"}
    """
    
    prompt = f"""
    {system_instruction}
    
    Context:
    {context_str}
    
    User Query: {user_query}
    """
    
    if not client: 
        yield "❌ LLM Error: Client not initialized"
        return

    try:
        # === 核心修改：第一次调用改为流式 (generate_content_stream) ===
        stream = client.models.generate_content_stream(
            model=settings.MODEL_NAME,
            contents=prompt
        )
        
        # === 智能缓冲逻辑 ===
        buffer = ""
        is_checking_json = True # 标记是否还在检测 JSON 阶段
        is_tool_call = False    # 标记最终是否确认为工具调用
        
        for chunk in stream:
            text_chunk = chunk.text
            
            if is_checking_json:
                buffer += text_chunk
                # 清洗 buffer 以前缀检查
                clean_start = buffer.strip().replace("```json", "").replace("```", "").strip()
                
                # 如果缓冲区还很短，继续积攒 (防止误判)
                if len(clean_start) < 5:
                    continue
                    
                # 检查特征
                if clean_start.startswith("{"):
                    # 看起来像 JSON，继续缓冲，不输出给用户
                    continue 
                else:
                    # 确定不是 JSON，是普通回答！
                    # 1. 把积攒的 buffer 吐出去
                    yield buffer
                    buffer = "" # 清空
                    is_checking_json = False # 停止检测，后续直接透传
            else:
                # 已经确定是普通文本，直接流式输出
                yield text_chunk

        # 流结束了
        # 如果 is_checking_json 依然为 True，说明 LLM 回复很短或者全是 JSON
        missing_file = None
        if is_checking_json and buffer:
            # 尝试解析 JSON
            clean_text = buffer.strip().replace("```json", "").replace("```", "").strip()
            if "missing_file" in clean_text:
                match = re.search(r"\{.*?\}", clean_text, re.DOTALL)
                if match:
                    try:
                        data = json.loads(match.group(0))
                        missing_file = data.get("missing_file")
                        is_tool_call = True
                    except:
                        pass
            
            # 如果不是 JSON，说明是一句很短的话，把它补发给用户
            if not is_tool_call:
                yield buffer

        # === 分支 A: 触发动态加载 (ReAct) ===
        if is_tool_call and missing_file:
            # 实时反馈给前端
            yield f"> 🤔 发现缺少文件: `{missing_file}`\n\n"
            
            if not vector_db.repo_url:
                yield f"> ⚠️ 会话信息丢失 (Repo URL)，无法下载。\n\n"
                return

            new_docs_content = []
            
            # 检查已索引
            if missing_file in vector_db.indexed_files:
                yield f"> 📚 该文件已在知识库中，正在提取细节...\n\n"
                stored_docs = vector_db.get_documents_by_file(missing_file)
                if stored_docs:
                    new_docs_content = stored_docs
                else:
                    yield f"> ⚠️ 索引中未找到内容，尝试重新下载...\n\n"
            
            # 下载
            if not new_docs_content:
                yield f"> 📥 正在下载并分析: `{missing_file}`...\n\n"
                success = await _download_and_index(vector_db, missing_file)
                if success:
                    new_docs_content = vector_db.get_documents_by_file(missing_file)
                else:
                    yield f"> ❌ 下载失败 (文件不存在或网络错误)。\n\n"
                    # 这里可以选择把原始 buffer (JSON) 打印出来，或者忽略
                    return

            # === 二次生成 (Streaming) ===
            supplementary_context = _build_context(new_docs_content)
            
            retry_prompt = f"""
            System: You requested '{missing_file}'. Here is its content.
            Now answer the user's question based on the updated context.
            
            New File Content:
            {supplementary_context}
            
            Original Context:
            {context_str}
            
            User Query: {user_query}
            """
            
            # 第二次流式调用
            stream_retry = client.models.generate_content_stream(
                model=settings.MODEL_NAME,
                contents=retry_prompt
            )
            for chunk in stream_retry:
                yield chunk.text
                await asyncio.sleep(0.01)

    except Exception as e:
        import traceback
        traceback.print_exc()
        yield f"❌ Error: {str(e)}"

# 辅助函数 (保持不变)
def _build_context(docs):
    if not docs: return "No code found."
    context = ""
    for doc in docs:
        file_info = doc['file']
        if 'class' in doc.get('metadata', {}):
            cls = doc['metadata']['class']
            if cls: file_info += f" (Class: {cls})"
        context += f"\n--- File: {file_info} ---\n{doc['content'][:2000]}\n"
    return context

async def _download_and_index(vector_db, file_path):
    try:
        content = get_file_content(vector_db.repo_url, file_path)
        if not content: return False
        
        chunks = await asyncio.to_thread(chunker.chunk_file, content, file_path)
        if not chunks: return False
        
        documents = [c["content"] for c in chunks]
        metadatas = []
        for c in chunks:
            meta = c["metadata"]
            metadatas.append({
                "file": meta["file"],
                "type": meta["type"],
                "name": meta.get("name", ""),
                "class": meta.get("class") or ""
            })
            
        await asyncio.to_thread(vector_db.add_documents, documents, metadatas)
        return True
    except Exception as e:
        print(f"Error: {e}")
        return False