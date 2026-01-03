# 文件路径: app/services/chunking_service.py
import ast

class PythonASTChunker:
    def __init__(self, min_chunk_size=50):
        self.min_chunk_size = min_chunk_size

    def chunk_file(self, file_content: str, filename: str):
        """
        基于 AST 解析，支持类方法的细粒度提取。
        """
        chunks = []
        try:
            tree = ast.parse(file_content)
        except SyntaxError:
            return [{"content": file_content[:8000], "metadata": {"file": filename, "type": "raw"}}]

        lines = file_content.splitlines()

        # 辅助函数：提取节点源码
        def get_node_source(node):
            start = node.lineno - 1
            end = node.end_lineno
            return "\n".join(lines[start:end]), start + 1, end

        # 遍历顶层节点
        for node in tree.body:
            # 1. 如果是函数，直接提取
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                source, start, end = get_node_source(node)
                if len(source) < self.min_chunk_size: continue
                
                chunks.append({
                    "content": source,
                    "metadata": {
                        "file": filename,
                        "name": node.name,
                        "type": "function",
                        "class": None, # 顶层函数无类名
                        "start_line": start,
                        "end_line": end
                    }
                })

            # 2. 如果是类，深入提取方法
            elif isinstance(node, ast.ClassDef):
                class_name = node.name
                
                # 可选：提取类定义的“头信息”（Docstring + 类变量），作为单独的 Chunk
                # 这样即使不搜方法，搜类本身也能搜到
                class_header_end = node.lineno # 简单处理，只取定义行附近
                # 尝试找 docstring
                if ast.get_docstring(node):
                    pass # 实际项目中可以专门处理 docstring

                # 遍历类体内的节点
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        method_source, start, end = get_node_source(item)
                        if len(method_source) < self.min_chunk_size: continue

                        # === 关键优化：注入上下文 ===
                        # 在内容前加上类名注释，帮助 Embedding 区分同名方法（如不同类的 run 方法）
                        enriched_content = f"# Class: {class_name}\n{method_source}"

                        chunks.append({
                            "content": enriched_content,
                            "metadata": {
                                "file": filename,
                                "name": item.name,
                                "type": "method",
                                "class": class_name, # 记录所属类
                                "start_line": start,
                                "end_line": end
                            }
                        })

        # 兜底：如果没提取到任何有意义的块（全是赋值语句等），且文件不为空
        if not chunks and file_content.strip():
             chunks.append({
                 "content": file_content[:2000], 
                 "metadata": {"file": filename, "type": "global_script", "name": "script"}
             })
             
        return chunks