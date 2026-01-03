# 文件路径: app/services/chunking_service.py
import ast

class PythonASTChunker:
    def __init__(self, min_chunk_size=50, max_chunk_size=2000):
        """
        :param min_chunk_size: 最小字符数，太小的代码段忽略
        :param max_chunk_size: 最大字符数 (约 500-800 Token)，超过此长度的 Class 会被强制拆解
        """
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size

    def chunk_file(self, content: str, file_path: str):
        if not content:
            return []
        
        chunks = []
        try:
            tree = ast.parse(content)
        except SyntaxError:
            # 如果解析失败（比如非 Python 文件），退化为简单的文本分块
            return self._fallback_chunking(content, file_path)

        # 1. 遍历顶层节点
        for node in tree.body:
            # === 策略 A: 处理类 (ClassDef) ===
            if isinstance(node, ast.ClassDef):
                class_code = ast.get_source_segment(content, node)
                if not class_code: continue

                # 判断：如果类比较小，直接作为一个整体
                if len(class_code) <= self.max_chunk_size:
                    chunks.append({
                        "content": class_code,
                        "metadata": {
                            "file": file_path,
                            "type": "class",
                            "name": node.name,
                            "start_line": node.lineno,
                            "class": node.name 
                        }
                    })
                else:
                    # 如果类太大，拆解为方法，但保留类名作为上下文
                    chunks.extend(self._chunk_large_class(node, content, file_path))

            # === 策略 B: 处理独立函数 (FunctionDef) ===
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                func_code = ast.get_source_segment(content, node)
                if func_code and len(func_code) >= self.min_chunk_size:
                    chunks.append({
                        "content": func_code,
                        "metadata": {
                            "file": file_path,
                            "type": "function",
                            "name": node.name,
                            "start_line": node.lineno,
                            "class": "" # 顶层函数无类
                        }
                    })

            # === 策略 C: 其他顶层代码 (如全局变量定义) ===
            # 简单略过，或者你可以选择收集起来做一个 "Global Context" chunk
            # 为了保持向量库干净，这里暂时略过，除非它带有大段注释
        
        # 如果文件里全是散代码（如 __init__.py 或 脚本），没有函数/类
        if not chunks and len(content) > 0:
             return self._fallback_chunking(content, file_path)

        return chunks

    def _chunk_large_class(self, class_node, content, file_path):
        """处理超大类：拆解 Method，但注入 Class 上下文"""
        chunks = []
        class_name = class_node.name
        # 提取类的 Docstring
        docstring = ast.get_docstring(class_node) or "No docstring"
        
        # 构造一个 Context Header，拼接到每个方法前面
        # 这样即使按方法切，LLM 也能知道它属于哪个类，以及类的作用
        context_header = f"class {class_name}:\n    \"\"\"{docstring}\"\"\"\n    # ... (Parent Context)\n"

        for node in class_node.body:
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                method_code = ast.get_source_segment(content, node)
                if not method_code: continue
                
                # 拼接上下文
                full_chunk_content = context_header + "\n" + method_code
                
                chunks.append({
                    "content": full_chunk_content, # <--- 关键：带上了类的上下文
                    "metadata": {
                        "file": file_path,
                        "type": "method",
                        "name": node.name,
                        "start_line": node.lineno,
                        "class": class_name
                    }
                })
        return chunks

    def _fallback_chunking(self, content, file_path):
        """兜底策略：按固定长度切分"""
        chunks = []
        lines = content.split('\n')
        # 简单粗暴：每 100 行切一下
        chunk_size = 100
        for i in range(0, len(lines), chunk_size):
            chunk_content = "\n".join(lines[i:i+chunk_size])
            chunks.append({
                "content": chunk_content,
                "metadata": {
                    "file": file_path,
                    "type": "text_chunk",
                    "name": f"chunk_{i}",
                    "start_line": i+1,
                    "class": ""
                }
            })
        return chunks