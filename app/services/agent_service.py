# 文件路径: app/services/agent_service.py
import json
import asyncio
import traceback
from app.core.config import settings
from app.utils.llm_client import client
from app.services.github_service import get_repo_structure, get_file_content
from app.services.vector_service import store_manager
from app.services.chunking_service import PythonASTChunker

# === 辅助函数：智能文件树生成 ===
def generate_smart_file_list(file_list, max_token_limit=1000):
    """
    策略：
    1. 优先保留 README 和根目录文件。
    2. 如果文件总数较少 (< 300)，直接返回全量列表。
    3. 如果文件很多，过滤掉非核心后缀，且仅保留前 N 个。
    """
    core_extensions = ('.py', '.js', '.ts', '.go', '.java', '.cpp', '.h', '.rs', '.md', '.json', '.yml', '.yaml', 'Dockerfile')
    priority_files = [f for f in file_list if f.lower().endswith("readme.md")]
    code_files = [f for f in file_list if f.endswith(core_extensions) and f not in priority_files]
    total_files_count = len(file_list)
    
    if total_files_count < 300:
        final_list = priority_files + code_files
        return "\n".join(final_list[:500])
    else:
        truncated_list = priority_files + code_files[:400]
        remaining = len(code_files) - 400
        result = "\n".join(truncated_list)
        if remaining > 0:
            result += f"\n... (and {remaining} more files hidden)"
        return result

async def agent_stream(repo_url: str, session_id: str):
    """
    Agent ReAct 工作流：感知 -> (思考 -> 行动 -> 观察) * N -> 报告
    """
    short_id = session_id[-6:] if session_id else "unknown"
    yield json.dumps({"step": "init", "message": f"🚀 [Session: {short_id}] 正在连接 GitHub..."})
    await asyncio.sleep(0.5)
    
    try:
        # 1. 初始化资源
        vector_db = store_manager.get_store(session_id)
        
        # === 核心修复点：先 Reset，再赋值 URL ===
        # 之前的顺序反了，导致 reset 把 url 清空了
        vector_db.reset_collection() 
        vector_db.repo_url = repo_url  # <--- 必须放在 reset 之后！
        
        chunker = PythonASTChunker(min_chunk_size=50)

        # 2. 获取文件树
        file_list = get_repo_structure(repo_url)
        if not file_list:
            yield json.dumps({"step": "error", "message": "❌ 无法获取文件列表。"})
            return

        yield json.dumps({"step": "fetched", "message": f"📦 发现 {len(file_list)} 个文件，正在构建文件视图..."})
        
        file_tree_str = generate_smart_file_list(file_list)
        
        # 3. ReAct 循环配置
        MAX_ROUNDS = 3
        visited_files = set()
        context_summary = ""
        
        readme_file = next((f for f in file_list if f.lower().endswith("readme.md")), None)

        for round_idx in range(MAX_ROUNDS):
            # --- Phase A: 思考 (Reasoning) ---
            yield json.dumps({"step": "thinking", "message": f"🕵️ [Round {round_idx+1}/{MAX_ROUNDS}] 正在分析架构，规划阅读路径..."})
            
            prompt = f"""
            You are a Source Code Auditor. 
            Goal: Analyze the **INTERNAL IMPLEMENTATION** of the project.
            
            Strict Rules:
            1. **PRIORITIZE SOURCE**: Look for folders like 'app/', 'src/', 'fastapi/', 'core/'.
            2. **Follow Imports**: If you see 'from .routing import APIRouter', you MUST read 'routing.py'.
            3. Read 'README.md' in the first round if available.
            
            Project File List (Core files):
            {file_tree_str}
            
            Files already read: {list(visited_files)}
            
            Knowledge gained (Imports/Definitions):
            {context_summary}
            
            Task:
            Select 1-3 critical files to read next.
            Return ONLY a raw JSON list.
            """
            
            if not client:
                 yield json.dumps({"step": "error", "message": "❌ LLM Client 未初始化。"})
                 return

            # 调用 LLM 决策
            response = await asyncio.to_thread(
                client.models.generate_content,
                model=settings.MODEL_NAME, 
                contents=prompt
            )
            
            target_files = []
            try:
                text = response.text.replace("```json", "").replace("```", "").strip()
                target_files = json.loads(text)
            except:
                pass

            valid_files = [f for f in target_files if f in file_list and f not in visited_files]

            # 第一轮强制读取 README
            if round_idx == 0 and readme_file and readme_file not in valid_files:
                valid_files.insert(0, readme_file)
                yield json.dumps({"step": "plan", "message": f"📘 [策略] 强制追加阅读: {readme_file}"})

            if not valid_files:
                yield json.dumps({"step": "plan", "message": f"🛑 [Round {round_idx+1}] 思考完毕，停止探索。"})
                break
            
            yield json.dumps({"step": "plan", "message": f"👉 [Round {round_idx+1}] 决定阅读: {valid_files}"})
            
            # --- Phase B: 行动 (Acting) ---
            new_knowledge = ""
            
            for i, file_path in enumerate(valid_files):
                yield json.dumps({"step": "download", "message": f"📥 解析源码: {file_path}..."})
                
                content = get_file_content(repo_url, file_path)
                if not content: continue
                
                visited_files.add(file_path)
                
                # 提取 Preview
                lines = content.split('\n')[:100]
                if file_path.endswith('.md'):
                    preview = "\n".join([l for l in lines if l.strip().startswith('#')])
                else:
                    preview = "\n".join([l for l in lines if l.strip().startswith(('import', 'from', 'class', 'def'))])
                
                new_knowledge += f"\n--- File: {file_path} ---\n{preview}\n"

                # AST 切片
                chunks = await asyncio.to_thread(chunker.chunk_file, content, file_path)
                if not chunks: continue
                
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

                if documents:
                    await asyncio.to_thread(vector_db.add_documents, documents, metadatas)
            
            # --- Phase C: 观察 (Observing) ---
            context_summary += new_knowledge
            
            yield json.dumps({"step": "indexing", "message": f"🧠 [Round {round_idx+1}] 知识已吸收，准备下一轮思考..."})

        # Step 4: 最终报告
        yield json.dumps({"step": "generating", "message": "📝 正在撰写技术架构报告..."})
        
        analysis_prompt = f"""
        You are a Tech Lead.
        Files analyzed: {list(visited_files)}
        
        Code Summary (Imports & Signatures):
        {context_summary[:10000]}
        
        Write a technical report (Markdown, Chinese).
        Focus on:
        1. Project Purpose
        2. Core Architecture
        3. Key Classes & Data Flow
        """
        
        try:
            stream = client.models.generate_content_stream(
                model=settings.MODEL_NAME, contents=analysis_prompt
            )
            for chunk in stream:
                yield json.dumps({"step": "report_chunk", "chunk": chunk.text})
                await asyncio.sleep(0.02)
        except Exception:
            resp = client.models.generate_content(model=settings.MODEL_NAME, contents=analysis_prompt)
            yield json.dumps({"step": "report_chunk", "chunk": resp.text})

        yield json.dumps({"step": "finish", "message": "✅ 分析完成！"})

    except Exception as e:
        traceback.print_exc()
        yield json.dumps({"step": "error", "message": f"💥 系统错误: {str(e)}"})