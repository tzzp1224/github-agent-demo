# 文件路径: app/services/vector_service.py
import chromadb
from chromadb.config import Settings as ChromaSettings
from app.utils.llm_client import client
from app.core.config import settings
from rank_bm25 import BM25Okapi
import re
import time

class VectorStore:
    def __init__(self, session_id: str):
        self.session_id = session_id
        # 初始化 ChromaDB (内存模式)
        self.chroma_client = chromadb.Client(ChromaSettings(anonymized_telemetry=False))
        # === 关键点：使用 session_id 区分 Collection ===
        self.collection_name = f"repo_{session_id}"
        
        # Hybrid Search 组件 (内存级，随实例存在)
        self.bm25 = None
        self.doc_store = [] 
        
        self.reset_collection()

    def reset_collection(self):
        try:
            self.chroma_client.delete_collection(name=self.collection_name)
        except Exception:
            pass
        self.collection = self.chroma_client.create_collection(name=self.collection_name)
        self.bm25 = None
        self.doc_store = []
        print(f"🧹 [Session: {self.session_id}] 数据库已重置")

    def embed_text(self, text):
        if not client: return []
        try:
            result = client.models.embed_content(
                model=settings.EMBEDDING_MODEL,
                contents=text
            )
            return result.embeddings[0].values
        except Exception as e:
            print(f"❌ Embedding Error: {e}")
            return []

    def _tokenize(self, text):
        return [t.lower() for t in re.split(r'[^a-zA-Z0-9]', text) if t.strip()]

    def add_documents(self, documents, metadatas):
        if not documents: return
        
        embeddings = []
        ids = []
        
        # 1. 准备数据 (BM25 + Vector)
        for i, doc in enumerate(documents):
            doc_id = f"{metadatas[i]['file']}_{len(self.doc_store) + i}"
            self.doc_store.append({
                "id": doc_id,
                "content": doc,
                "metadata": metadatas[i]
            })
            
            emb = self.embed_text(doc)
            if emb:
                embeddings.append(emb)
                ids.append(doc_id)

        # 2. 存入 Chroma
        if embeddings:
            self.collection.add(documents=documents, embeddings=embeddings, metadatas=metadatas, ids=ids)
        
        # 3. 重建 BM25
        tokenized_corpus = [self._tokenize(doc['content']) for doc in self.doc_store]
        self.bm25 = BM25Okapi(tokenized_corpus)
        
        print(f"✅ [Session: {self.session_id}] 已索引 {len(documents)} 个片段")

    def search_hybrid(self, query, top_k=3):
        # 1. 向量检索
        vector_results = []
        query_embedding = self.embed_text(query)
        if query_embedding:
            chroma_res = self.collection.query(
                query_embeddings=[query_embedding], n_results=top_k * 2
            )
            if chroma_res['ids']:
                ids = chroma_res['ids'][0]
                docs = chroma_res['documents'][0]
                metas = chroma_res['metadatas'][0]
                for i in range(len(ids)):
                    vector_results.append({
                        "id": ids[i], "content": docs[i], "file": metas[i]['file'], "score": 0
                    })

        # 2. BM25 检索
        bm25_results = []
        if self.bm25:
            tokenized_query = self._tokenize(query)
            doc_scores = self.bm25.get_scores(tokenized_query)
            top_n = min(len(doc_scores), top_k * 2)
            top_indices = sorted(range(len(doc_scores)), key=lambda i: doc_scores[i], reverse=True)[:top_n]
            for idx in top_indices:
                if doc_scores[idx] > 0:
                    item = self.doc_store[idx]
                    bm25_results.append({
                        "id": item["id"], "content": item["content"], "file": item["metadata"]["file"], "score": 0
                    })

        # 3. 加权 RRF
        k = 60
        weight_vector = 1.0
        weight_bm25 = 0.3
        fused_scores = {}

        for rank, item in enumerate(vector_results):
            doc_id = item['id']
            if doc_id not in fused_scores: fused_scores[doc_id] = {"item": item, "score": 0}
            fused_scores[doc_id]["score"] += weight_vector * (1 / (k + rank + 1))
            
        for rank, item in enumerate(bm25_results):
            doc_id = item['id']
            if doc_id not in fused_scores: fused_scores[doc_id] = {"item": item, "score": 0}
            fused_scores[doc_id]["score"] += weight_bm25 * (1 / (k + rank + 1))

        sorted_results = sorted(fused_scores.values(), key=lambda x: x['score'], reverse=True)
        return [res['item'] for res in sorted_results[:top_k]]

# === 新增：会话管理器 ===
class VectorStoreManager:
    def __init__(self):
        self.stores = {} # session_id -> VectorStore
        self.last_access = {} # 用于简单的清理策略 (可选)

    def get_store(self, session_id: str) -> VectorStore:
        if session_id not in self.stores:
            print(f"🆕 创建新会话: {session_id}")
            self.stores[session_id] = VectorStore(session_id)
        self.last_access[session_id] = time.time()
        return self.stores[session_id]

# 全局管理器单例
store_manager = VectorStoreManager()