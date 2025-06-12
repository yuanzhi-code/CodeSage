"""
代码处理器模块，负责代码的加载、分割和向量化。
"""
import os
from typing import List, Optional, Dict, Any
from pathlib import Path

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import CodeSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank
import lancedb

class CodeProcessor:
    """代码处理器类，用于处理代码仓库中的代码。"""
    
    # 支持的编程语言映射
    LANGUAGE_MAP = {
        '.py': 'python',
        '.js': 'javascript',
        '.ts': 'typescript',
        '.java': 'java',
        '.cpp': 'cpp',
        '.c': 'c',
        '.go': 'go',
        '.rs': 'rust',
        '.rb': 'ruby',
        '.php': 'php',
        '.swift': 'swift',
        '.kt': 'kotlin',
        '.scala': 'scala',
        '.hs': 'haskell',
        '.lua': 'lua',
        '.sh': 'bash',
        '.r': 'r',
        '.jl': 'julia',
        '.pl': 'perl',
        '.sql': 'sql',
    }
    
    # 支持的代码文件扩展名
    SUPPORTED_EXTENSIONS = list(LANGUAGE_MAP.keys())
    
    def __init__(
        self,
        code_repo_dir: str,
        lancedb_uri: str = "./lancedb_data",
        lancedb_table_name: str = "code_snippets",
        embedding_model_name: str = "Qwen/Qwen3-Embedding-0.6B",
        rerank_model_name: str = "Qwen/Qwen3-Reranker-0.6B",
        top_k_rerank: int = 3,
        chunk_lines: int = 40,
        chunk_lines_overlap: int = 15,
        max_chars: int = 1500,
    ):
        """
        初始化代码处理器。

        Args:
            code_repo_dir: 代码仓库目录
            lancedb_uri: LanceDB 数据库路径
            lancedb_table_name: LanceDB 表名
            embedding_model_name: 嵌入模型名称
            rerank_model_name: 重排序模型名称
            top_k_rerank: 重排序后保留的结果数量
            chunk_lines: 代码块最大行数
            chunk_lines_overlap: 代码块重叠行数
            max_chars: 代码块最大字符数
        """
        self.code_repo_dir = Path(code_repo_dir)
        self.lancedb_uri = lancedb_uri
        self.lancedb_table_name = lancedb_table_name
        
        # 初始化模型和处理器
        self._init_models(
            embedding_model_name=embedding_model_name,
            rerank_model_name=rerank_model_name,
            top_k_rerank=top_k_rerank
        )
        
        # 初始化代码分割器
        self._init_code_splitter(
            chunk_lines=chunk_lines,
            chunk_lines_overlap=chunk_lines_overlap,
            max_chars=max_chars
        )
        
        # 初始化向量存储
        self._init_vector_store()
        
        self.index = None

    def _init_models(
        self,
        embedding_model_name: str,
        rerank_model_name: str,
        top_k_rerank: int
    ) -> None:
        """
        初始化嵌入模型和重排序模型。

        Args:
            embedding_model_name: 嵌入模型名称
            rerank_model_name: 重排序模型名称
            top_k_rerank: 重排序后保留的结果数量
        """
        # 初始化嵌入模型
        self.embed_model = HuggingFaceEmbedding(model_name=embedding_model_name)
        Settings.embed_model = self.embed_model
        
        # 初始化重排序模型
        self.reranker = SentenceTransformerRerank(
            model=rerank_model_name,
            top_n=top_k_rerank
        )

    def _init_code_splitter(
        self,
        chunk_lines: int,
        chunk_lines_overlap: int,
        max_chars: int
    ) -> None:
        """
        初始化代码分割器。

        Args:
            chunk_lines: 代码块最大行数
            chunk_lines_overlap: 代码块重叠行数
            max_chars: 代码块最大字符数
        """
        self.code_splitter = CodeSplitter(
            language="python",  # 默认使用 Python，后续会根据文件类型自动判断
            chunk_lines=chunk_lines,
            chunk_lines_overlap=chunk_lines_overlap,
            max_chars=max_chars,
        )

    def _init_vector_store(self) -> None:
        """初始化 LanceDB 向量存储。"""
        self.db = lancedb.connect(self.lancedb_uri)
        self.vector_store = LanceDBVectorStore(
            uri=self.lancedb_uri,
            table_name=self.lancedb_table_name,
            db=self.db
        )

    def _detect_language(self, file_path: Path) -> str:
        """
        根据文件扩展名检测编程语言。

        Args:
            file_path: 文件路径

        Returns:
            str: 编程语言名称
        """
        extension = file_path.suffix.lower()
        return self.LANGUAGE_MAP.get(extension, 'python')  # 默认使用 Python

    def process_code_repo(self) -> int:
        """
        处理整个代码仓库，构建向量索引。

        Returns:
            int: 处理的代码片段数量
        """
        if not self.code_repo_dir.exists():
            raise ValueError(f"代码仓库目录不存在: {self.code_repo_dir}")

        # 加载所有代码文件
        documents = SimpleDirectoryReader(
            input_dir=str(self.code_repo_dir),
            recursive=True,
            required_exts=self.SUPPORTED_EXTENSIONS
        ).load_data()

        # 处理每个文档
        all_nodes = []
        for doc in documents:
            file_path = Path(doc.metadata.get('file_path', ''))
            language = self._detect_language(file_path)
            
            # 更新代码分割器的语言设置
            self.code_splitter.language = language
            
            # 分割代码
            nodes = self.code_splitter.get_nodes_from_documents([doc])
            all_nodes.extend(nodes)

        # 构建向量索引
        self.index = VectorStoreIndex(all_nodes, vector_store=self.vector_store)
        return len(all_nodes)

    def query_code(self, query: str, similarity_top_k: int = 10) -> Dict[str, Any]:
        """
        查询代码。

        Args:
            query: 查询文本
            similarity_top_k: 初步召回的结果数量

        Returns:
            Dict[str, Any]: 查询结果，包含生成的摘要和源代码片段
        """
        if not self.index:
            raise ValueError("请先调用 process_code_repo() 处理代码仓库")

        query_engine = self.index.as_query_engine(
            similarity_top_k=similarity_top_k,
            node_postprocessors=[self.reranker]
        )

        response = query_engine.query(query)
        
        return {
            "summary": response.response,
            "source_nodes": [
                {
                    "file_path": node.metadata.get('file_path', '未知文件'),
                    "score": node.score,
                    "content": node.text
                }
                for node in response.source_nodes
            ]
        }

    def cleanup(self) -> None:
        """清理 LanceDB 数据库。"""
        if os.path.exists(self.lancedb_uri):
            import shutil
            shutil.rmtree(self.lancedb_uri) 