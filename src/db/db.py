"""
LanceDB 数据库管理模块
"""
import os
from pathlib import Path
import lancedb
from pandas import DataFrame
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from src.core.code_splitter import process_code_repository
from src.model.embedding import model as embedding_model

# 获取项目根目录
ROOT_DIR = Path(__file__).parent.parent.parent

# 定义数据库路径（在项目根目录下）
DB_PATH = ROOT_DIR / "data" / "lance_db"

# 确保数据目录存在
os.makedirs(DB_PATH.parent, exist_ok=True)

# 连接到 LanceDB 数据库
db = lancedb.connect(str(DB_PATH))

# 定义表的名称
TABLE_NAME = "code_chunks_table"

def process_and_store_code(repo_path: str) -> None:
    """
    处理代码仓库并将代码片段存储到数据库中

    Args:
        repo_path: 代码仓库路径
    """
    # 使用 code_splitter 处理代码仓库
    results = process_code_repository(repo_path)
    
    # 转换为DataFrame
    df = DataFrame(results)
    
    # 存储到数据库
    if TABLE_NAME not in db.table_names():
        table = db.create_table(TABLE_NAME, data=df)
        print(f"表 '{TABLE_NAME}' 已创建，包含 {len(df)} 条记录")
    else:
        table = db.open_table(TABLE_NAME)
        table.add(df)
        print(f"已添加 {len(df)} 条记录到表 '{TABLE_NAME}'")

def rerank_results(query: str, results: list) -> list:
    """
    使用 Qwen3-Reranker 对搜索结果进行重排序

    Args:
        query: 搜索查询文本
        results: 初始搜索结果列表

    Returns:
        list: 重排序后的结果列表
    """
    # 初始化重排序模型
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side='left')
    reranker = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B").eval()
    
    # 设置重排序参数
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    max_length = 8192
    
    # 设置提示模板
    prefix = "<|im_start|>system\nJudge whether the code snippet meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
    suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
    suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
    
    # 计算每个文档的最大长度
    max_doc_length = max_length - len(prefix_tokens) - len(suffix_tokens) - 100  # 预留一些空间给查询和指令
    
    # 准备重排序的输入
    task = "Find the code snippet that best matches the given query"
    pairs = []
    for result in results:
        # 截断过长的代码片段
        text = result['text']
        if len(text) > max_doc_length:
            text = text[:max_doc_length] + "..."
        instruction = f"<Instruct>: {task}\n<Query>: {query}\n<Document>: {text}"
        pairs.append(instruction)
    
    # 处理输入
    inputs = tokenizer(
        pairs, padding=True, truncation=True,
        return_attention_mask=True, max_length=max_length,
        return_tensors="pt"
    )
    
    # 计算重排序分数
    with torch.no_grad():
        batch_scores = reranker(**inputs).logits[:, -1, :]
        true_vector = batch_scores[:, token_true_id]
        false_vector = batch_scores[:, token_false_id]
        batch_scores = torch.stack([false_vector, true_vector], dim=1)
        batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
        rerank_scores = batch_scores[:, 1].exp().tolist()
    
    # 更新结果分数
    for i, result in enumerate(results):
        result['score'] = rerank_scores[i]
    
    # 按新分数排序
    results.sort(key=lambda x: x['score'], reverse=True)
    return results

def search_code(query: str, limit: int = 10) -> list:
    """
    搜索相似的代码片段

    Args:
        query: 搜索查询文本
        limit: 返回结果的最大数量

    Returns:
        list: 搜索结果列表
    """
    # 生成查询文本的嵌入向量
    query_vector = embedding_model.encode(query, prompt="<Instruct>: Find the code snippet that best matches the given query.\n\n<Query>: ").tolist()
    
    # 获取表
    table = db.open_table(TABLE_NAME)
    
    # 执行搜索
    results = table.search(query_vector).limit(limit).to_pandas()
    
    # 格式化结果
    formatted_results = []
    for _, row in results.iterrows():
        formatted_results.append({
            "score": float(row['_distance']),
            "text": row['text'],
            "metadata": row['metadata']
        })
    
    # 使用重排序模型重新排序结果
    reranked_results = rerank_results(query, formatted_results)
    
    return reranked_results

def get_db_stats() -> dict:
    """
    获取数据库统计信息

    Returns:
        dict: 统计信息
    """
    if TABLE_NAME not in db.table_names():
        return {"total_entries": 0}
        
    table = db.open_table(TABLE_NAME)
    return {
        "total_entries": table.count_rows()
    }

def save_search_results_to_md(results: list, output_file: str = "search_results.md") -> None:
    """
    将搜索结果保存为Markdown文件

    Args:
        results: 搜索结果列表
        output_file: 输出文件路径
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("# 代码搜索结果\n\n")
        for i, result in enumerate(results, 1):
            f.write(f"## 结果 {i}\n\n")
            f.write(f"### 相似度\n{result['score']:.4f}\n\n")
            f.write(f"### 代码片段\n```\n{result['text']}\n```\n\n")
            f.write(f"### 元数据\n```json\n{result['metadata']}\n```\n\n")
            f.write("---\n\n")

# 使用示例
if __name__ == "__main__":
    # 处理代码仓库
    repo_path = str(ROOT_DIR / "my_code_repo_for_lancedb_demo")
    process_and_store_code(repo_path)
    
    # # 搜索示例
    # results = search_code("并发搜索的代码。")
    # print("\n搜索结果:")
    # for result in results:
    #     print(f"相似度: {result['score']:.4f}")
    #     print(f"代码: {result['text'][:100]}...")
    #     print(f"元数据: {result['metadata']}")
    #     print("---")
    
    # # 保存结果到Markdown文件
    # save_search_results_to_md(results)