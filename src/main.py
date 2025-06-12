"""
CodeSage - 一个基于 LlamaIndex 和 LanceDB 的代码检索系统。

此模块提供了代码检索系统的主程序入口，包括代码仓库处理、向量化存储和自然语言查询功能。
"""
from codesage.core.code_processor import CodeProcessor
from codesage.utils.file_utils import ensure_dir, cleanup_dir
from codesage.examples.demo_code import create_demo_code

# --- 配置参数 ---
CODE_REPO_DIR = "./my_code_repo_for_lancedb_demo"
LANCEDB_URI = "./lancedb_data"
LANCEDB_TABLE_NAME = "code_snippets"

def main():
    """主程序入口"""
    # 步骤 0: 清理旧数据并设置新的演示代码文件
    # cleanup_dir(CODE_REPO_DIR)
    # cleanup_dir(LANCEDB_URI)
    # ensure_dir(CODE_REPO_DIR)
    # create_demo_code(CODE_REPO_DIR)
    print("演示环境已准备就绪。")

    # 步骤 1: 初始化代码处理器
    processor = CodeProcessor(
        code_repo_dir=CODE_REPO_DIR,
        lancedb_uri=LANCEDB_URI,
        lancedb_table_name=LANCEDB_TABLE_NAME
    )

    # 步骤 2: 处理代码仓库
    print("\n--- 处理代码仓库 ---")
    num_nodes = processor.process_code_repo()
    print(f"代码仓库处理完成，共生成 {num_nodes} 个代码片段。")

    # 步骤 3: 执行查询示例
    print("\n--- 执行查询示例 ---")
    queries = [
        "如何使用Python计算一个列表的总和？",
        "我想用pandas读取CSV文件，怎么实现？",
        "如何将DataFrame保存为parquet格式？",
        "如何在Python中获取当前工作目录？",
        "文件处理类的功能是什么？",
        "如何在JavaScript中计算数组总和？",
        "JavaScript中如何读写文件？"
    ]

    for query in queries:
        print(f"\n======== 用户查询: '{query}' ========")
        result = processor.query_code(query)
        
        print("\n检索到的最相关代码片段摘要:")
        print(result["summary"])
        
        print("\n--- 详细的源代码片段及其元数据 ---")
        for i, node in enumerate(result["source_nodes"]):
            print(f"\n--- 检索到的片段 {i+1} ---")
            print(f"原始文件: {node['file_path']}")
            print(f"相似度分数: {node['score']:.4f}")
            print("--- 代码内容 ---")
            print(node["content"])
            print("------------------")

    # # 步骤 4: 清理
    # processor.cleanup()
    # cleanup_dir(CODE_REPO_DIR)
    # print("\n演示数据和 LanceDB 目录已清理完成。")

if __name__ == "__main__":
    main()
