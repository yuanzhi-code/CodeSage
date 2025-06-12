import os
import shutil
from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, Settings
from llama_index.core.node_parser import CodeSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core.postprocessor import SentenceTransformerRerank # 用于集成 reranker
import lancedb

# --- 配置参数 ---
# 存放示例代码的目录
CODE_REPO_DIR = "./my_code_repo_for_lancedb_demo"
# LanceDB 数据库的存储路径
LANCEDB_URI = "./lancedb_data"
# LanceDB 中用于存储代码片段的表名
LANCEDB_TABLE_NAME = "code_snippets"
# 用于生成文本嵌入的 Hugging Face 模型名称。
# 这里选择了一个在中文文本嵌入方面表现优秀的 BGE 模型作为 Qwen3 Embedding 的替代或兼容方案。
# 如果 Qwen 官方提供了专门用于 Embedding 的模型，你可以替换为该模型的名称，
# 例如：'qwen/qwen-text-embedding'（假设其存在且可通过 HuggingFaceEmbedding 加载）。
EMBEDDING_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
# 用于 Rerank 的 Qwen3 模型名称或兼容模型。
# 例如：'BAAI/bge-reranker-base' 或 Qwen 官方提供的 Rerank 模型。
# 确保你选择的模型适合重排序任务。
RERANK_MODEL_NAME = "BAAI/bge-reranker-base"
# Rerank 后希望保留的最佳结果数量
TOP_K_RERANK = 3

def setup_demo_code():
    """
    创建示例代码文件，用于演示代码召回功能。
    如果目录已存在，会先清空再创建。
    """
    if os.path.exists(CODE_REPO_DIR):
        shutil.rmtree(CODE_REPO_DIR) # 清理旧目录
    os.makedirs(CODE_REPO_DIR) # 创建新目录

    # 示例 Python 代码文件 1
    code_content_1 = """
# example_utils.py

def calculate_sum(numbers: list) -> int:
    \"\"\"
    计算列表中所有数字的总和。

    Args:
        numbers (list): 包含整数或浮点数的列表。

    Returns:
        int: 列表中所有数字的和。
    \"\"\"
    total = 0
    for num in numbers:
        total += num
    return total

def get_current_working_directory() -> str:
    \"\"\"
    使用 os 模块获取当前工作目录。

    Returns:
        str: 当前工作目录的路径。
    \"\"\"
    import os
    return os.getcwd()

class FileHandler:
    \"\"\"
    文件处理类，用于读写文件。
    \"\"\"
    def __init__(self, filepath: str):
        self.filepath = filepath

    def read_file(self) -> str:
        \"\"\"读取文件内容。\"\"\"
        with open(self.filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return content

    def write_to_file(self, content: str):
        \"\"\"将内容写入文件。\"\"\"
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.write(content)
"""
    with open(os.path.join(CODE_REPO_DIR, "example_utils.py"), "w", encoding="utf-8") as f:
        f.write(code_content_1)

    # 示例 Python 代码文件 2
    code_content_2 = """
# data_processing.py

import pandas as pd

def load_csv(filepath: str) -> pd.DataFrame:
    \"\"\"
    加载 CSV 文件到 Pandas DataFrame。

    Args:
        filepath (str): CSV 文件的路径。

    Returns:
        pd.DataFrame: 加载的 DataFrame。
    \"\"\"
    df = pd.read_csv(filepath)
    return df

def filter_dataframe(df: pd.DataFrame, column: str, value: any) -> pd.DataFrame:
    \"\"\"
    根据列值过滤 DataFrame。

    Args:
        df (pd.DataFrame): 输入 DataFrame。
        column (str): 要过滤的列名。
        value (any): 要匹配的值。

    Returns:
        pd.DataFrame: 过滤后的 DataFrame。
    \"\"\"
    return df[df[column] == value]

def save_dataframe_to_parquet(df: pd.DataFrame, filepath: str):
    \"\"\"
    将 DataFrame 保存为 Parquet 文件。

    Args:
        df (pd.DataFrame): 要保存的 DataFrame。
        filepath (str): Parquet 文件的保存路径。
    \"\"\"
    df.to_parquet(filepath)
"""
    with open(os.path.join(CODE_REPO_DIR, "data_processing.py"), "w", encoding="utf-8") as f:
        f.write(code_content_2)

def cleanup_demo_data():
    """
    清理演示过程中创建的所有数据，包括示例代码目录和 LanceDB 数据库目录。
    """
    if os.path.exists(CODE_REPO_DIR):
        print(f"清理目录: {CODE_REPO_DIR}")
        shutil.rmtree(CODE_REPO_DIR)
    if os.path.exists(LANCEDB_URI):
        print(f"清理 LanceDB 目录: {LANCEDB_URI}")
        shutil.rmtree(LANCEDB_URI)

# --- 主流程函数 ---
def main():
    # 步骤 0: 清理旧数据并设置新的演示代码文件
    cleanup_demo_data()
    setup_demo_code()
    print("演示环境已准备就绪。")

    print("\n--- 步骤 1: 加载代码文档 ---")
    # 使用 SimpleDirectoryReader 从指定目录加载所有代码文件。
    # LlamaIndex 会将每个文件视为一个 Document 对象。
    documents = SimpleDirectoryReader(input_dir=CODE_REPO_DIR).load_data()
    print(f"已从 '{CODE_REPO_DIR}' 目录加载了 {len(documents)} 个代码文件。")

    print("\n--- 步骤 2: 使用 CodeSplitter 进行 AST 切割 ---")
    # 初始化 CodeSplitter。
    # `language` 参数指定了代码的编程语言，CodeSplitter 会根据该语言的 AST 规则进行解析。
    # `chunk_lines` 控制每个代码块的最大行数。
    # `chunk_lines_overlap` 控制代码块之间的重叠行数，有助于保留上下文。
    # `max_chars` 限制每个代码块的最大字符数。
    code_splitter = CodeSplitter(
        language="python",        # 指定为 Python 语言
        chunk_lines=40,           # 每个块最多 40 行
        chunk_lines_overlap=15,   # 块之间重叠 15 行
        max_chars=1500,           # 每个块最多 1500 字符
    )

    # 将加载的文档（代码文件）分割成更小的、语义完整的节点（代码片段）。
    # 每个节点都是一个独立的检索单元。
    nodes = code_splitter.get_nodes_from_documents(documents)
    print(f"代码文件被分割成 {len(nodes)} 个语义代码片段 (LlamaIndex 节点)。")

    # 打印前三个示例代码片段及其来源文件，以便查看切割效果。
    for i, node in enumerate(nodes[:3]):
        print(f"\n--- 示例代码片段 {i+1} ---")
        print(f"原始文件: {node.metadata.get('file_path', '未知文件')}")
        print("--- 代码内容 ---")
        print(node.text)
        print("-----------------")

    print("\n--- 3A: 配置 Qwen3 Embedding (通过 HuggingFaceEmbedding) ---")
    try:
        # 初始化 HuggingFaceEmbedding 模型。
        # LlamaIndex 会使用这个模型将所有文本（代码片段和查询）转换为向量。
        # Settings.embed_model 用于设置 LlamaIndex 全局默认的嵌入模型。
        embed_model = HuggingFaceEmbedding(model_name=EMBEDDING_MODEL_NAME)
        Settings.embed_model = embed_model
        print(f"成功加载嵌入模型: '{EMBEDDING_MODEL_NAME}'。")
        print("请确保你已安装 `transformers` 库，并且模型能够从 Hugging Face Hub 下载。")
    except Exception as e:
        print(f"错误：加载嵌入模型失败。错误信息：{e}")
        print("请检查模型名称和网络连接。如果问题持续，尝试手动下载模型到本地路径或使用其他可用的 Embedding 模型。")
        cleanup_demo_data() # 如果加载模型失败，清理数据并退出
        return

    print("\n--- 3B: 配置 Qwen3 Reranker (通过 SentenceTransformerRerank) ---")
    try:
        # 初始化重排序模型。
        # SentenceTransformerRerank 是 LlamaIndex 中用于集成 SentenceTransformer 模型的后处理器。
        # Qwen3 Reranker 或兼容模型（如 BGE Reranker）通常是基于 SentenceTransformer 架构的。
        reranker = SentenceTransformerRerank(
            model=RERANK_MODEL_NAME, top_n=TOP_K_RERANK
        )
        print(f"成功加载重排序模型: '{RERANK_MODEL_NAME}'。将保留前 {TOP_K_RERANK} 个重排序结果。")
    except Exception as e:
        print(f"错误：加载重排序模型失败。错误信息：{e}")
        print("请检查模型名称和网络连接。请确保已安装 `sentence-transformers` 库。")
        cleanup_demo_data()
        return

    print("\n--- 4: 配置 LanceDB 并构建向量索引 ---")
    # 连接到 LanceDB 数据库。如果指定的 URI 路径不存在，LanceDB 会自动创建。
    db = lancedb.connect(LANCEDB_URI)

    # 初始化 LanceDBVectorStore。
    # LlamaIndex 将使用它来与 LanceDB 进行交互，存储和检索向量。
    # 如果 LanceDB_TABLE_NAME 表不存在，它会自动创建。
    vector_store = LanceDBVectorStore(uri=LANCEDB_URI, table_name=LANCEDB_TABLE_NAME, db=db)

    # 从处理过的节点和配置的 LanceDB 向量存储构建 LlamaIndex 向量索引。
    # 这一步会将所有代码片段的嵌入向量计算出来并存储到 LanceDB 中。
    print(f"开始构建 LanceDB 向量索引，并将 {len(nodes)} 个代码片段的嵌入向量写入数据库...")
    index = VectorStoreIndex(nodes, vector_store=vector_store)
    print(f"向量索引已成功构建并存储在 LanceDB 表 '{LANCEDB_TABLE_NAME}' 中，位于 '{LANCEDB_URI}'。")

    print("\n--- 5: 执行自然语言查询 (包含 Rerank 步骤) ---")
    # 从构建的索引中创建一个查询引擎。
    # query_engine.as_retriever() 获取检索器，负责从向量数据库中初步召回。
    # query_engine.from_args() 可以方便地将检索器和后处理器（如 reranker）组合起来。
    query_engine = index.as_query_engine(
        similarity_top_k=10, # 从向量数据库初步召回更多结果，以便 reranker 筛选
        node_postprocessors=[reranker] # 将 reranker 作为后处理器添加到查询管道中
    )


    # 准备几个自然语言查询示例。
    queries = [
        "如何使用Python计算一个列表的总和？",
        "我想用pandas读取CSV文件，怎么实现？",
        "如何将DataFrame保存为parquet格式？",
        "如何在Python中获取当前工作目录？",
        "文件处理类的功能是什么？"
    ]

    # 遍历每个查询并打印结果。
    for user_query in queries:
        print(f"\n======== 用户查询: '{user_query}' ========")
        # 执行查询，LlamaIndex 会自动将查询向量化，在 LanceDB 中搜索，然后通过 Reranker 进行重排序。
        response = query_engine.query(user_query)

        print("\n检索到的最相关代码片段摘要 (LLM生成):")
        # response.response 通常是 LLM 根据检索到的上下文生成的答案摘要
        print(response.response)

        print("\n--- 详细的源代码片段及其元数据 (经过 Rerank) ---")
        if not response.source_nodes:
            print("未找到相关代码片段。请尝试不同的查询或检查代码库内容。")
        for i, source_node in enumerate(response.source_nodes):
            print(f"\n--- 检索到的片段 {i+1} ---")
            print(f"原始文件: {source_node.metadata.get('file_path', '未知文件')}")
            # LlamaIndex 在检索结果的 source_node 中提供了相似度分数。
            # 对于经过 Rerank 的结果，通常会显示 Rerank 后的得分。
            print(f"Rerank 分数: {source_node.score:.4f}")
            print("--- 代码内容 ---")
            print(source_node.text)
            print("------------------")

    # --- 6: 清理 ---
    # 演示完成后，清理所有创建的示例数据和 LanceDB 数据库目录。
    cleanup_demo_data()
    print("\n演示数据和 LanceDB 目录已清理完成。")

# 确保脚本作为主程序运行时执行 main 函数。
if __name__ == "__main__":
    main()
