# 文件路径: evaluate_baseline.py
import asyncio
import json
import sys
import os

# 确保能导入 app 模块
sys.path.append(os.getcwd())

from app.services.vector_service import vector_db
from app.services.github_service import get_file_content
from app.core.config import settings
from app.services.chunking_service import PythonASTChunker

# 目标仓库
REPO_URL = "https://github.com/fastapi/fastapi"

async def run_evaluation():
    print("🧪 --- 开始 RAG 基线评估 ---")
    
    # 1. 加载数据集
    try:
        with open("evaluation/golden_dataset.json", "r", encoding="utf-8") as f:
            dataset = json.load(f)
    except FileNotFoundError:
        print("❌ 找不到数据集，请先创建 evaluation/golden_dataset.json")
        return

    # 2. 准备环境 (重置向量库)
    vector_db.reset_collection()
    
    # 3. 构建索引 (Indexing)
    # 为了测试 Retrieve 能力，我们需要确保答案文件在库里。
    # 这里我们收集数据集中提到的所有文件，模拟它们已经被 Agent 选中并索引了。
    target_files = list(set([item["answer_file"] for item in dataset]))
    
    print(f"📦 [AST Mode] 正在准备测试数据...")
    
    documents = []
    metadatas = []
    chunker = PythonASTChunker(min_chunk_size=50) # 初始化切分器
    
    for file_path in target_files:
        print(f"   ⬇️ 下载并AST切分: {file_path}")
        content = get_file_content(REPO_URL, file_path)
        if content:
            # === 核心修改点 ===
            # 旧逻辑: snippet = content[:1000]
            # 新逻辑: 使用 AST 切分出多个完整的块
            file_chunks = chunker.chunk_file(content, file_path)
            
            for chunk in file_chunks:
                documents.append(chunk["content"])
                # 合并元数据，保留文件名
                meta = chunk["metadata"]
                # ChromaDB 的 metadata 值必须是 str, int, float, bool
                # 为了简单，我们确保 file 字段存在
                # 新代码: 增加 class 字段，并确保转为字符串 (ChromaDB 要求 metadata 值为简单类型)
                metadatas.append({
                    "file": meta["file"], 
                    "type": meta["type"], 
                    "name": meta.get("name", ""),
                    "class": meta.get("class") or "" # 处理 None
                })
                
        else:
            print(f"   ⚠️ 警告: 无法下载 {file_path}")

    # 写入向量库
    vector_db.add_documents(documents, metadatas)
    print("✅ 索引构建完成，开始测试检索...")
    print("-" * 30)

    # 4. 执行评估 (Evaluation)
    hits = 0
    total = len(dataset)
    top_k = 3

    for item in dataset:
        query = item["query"]
        expected_file = item["answer_file"]
        
        # 调用现有的搜索接口
        results = vector_db.search(query, top_k=top_k)
        
        # 检查命中情况
        retrieved_files = [res['file'] for res in results]
        is_hit = expected_file in retrieved_files
        
        if is_hit:
            hits += 1
            status = "✅ 命中"
        else:
            status = "❌ 未命中"
            
        print(f"Q: {query[:40]}...")
        print(f"   期望: {expected_file}")
        print(f"   检索: {retrieved_files}")
        print(f"   结果: {status}\n")

    # 5. 输出报告
    hit_rate = (hits / total) * 100
    print("=" * 30)
    print(f"📊 基线评估结果 (Baseline)")
    print(f"🎯 Hit Rate @ {top_k}: {hit_rate:.2f}%")
    print("=" * 30)
    
    # 建议：将结果写入文件以便后续对比
    with open("evaluation/baseline_result.txt", "w") as f:
        f.write(f"Baseline Hit Rate: {hit_rate:.2f}%")

if __name__ == "__main__":
    # 检查 Key
    if not settings.GEMINI_API_KEY:
        print("❌ 请先配置 .env 中的 GEMINI_API_KEY")
    else:
        asyncio.run(run_evaluation())