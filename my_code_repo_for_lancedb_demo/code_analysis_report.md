# 代码分析报告

生成时间：2025-06-13 00:57:51

## 统计信息

- 总文件数：7
- 总代码块数：30

## 语言分布

- python：7 个文件

## 文件详情

### /Users/yuanzhi/CodeSage/my_code_repo_for_lancedb_demo/code_processor.py

- 语言：python
- 代码块数：8

#### 代码块

##### 代码块 1

```python
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
```

##### 代码块 2

```python
class CodeProcessor:
```

##### 代码块 3

```python
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
```

##### 代码块 4

```python
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
```

##### 代码块 5

```python
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
```

##### 代码块 6

```python
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
```

##### 代码块 7

```python
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
```

##### 代码块 8

```python
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
```

### /Users/yuanzhi/CodeSage/my_code_repo_for_lancedb_demo/example_utils.py

- 语言：python
- 代码块数：1

#### 代码块

##### 代码块 1

```python
# example_utils.py

def calculate_sum(numbers: list) -> int:
    """
    计算列表中所有数字的总和。

    Args:
        numbers (list): 包含整数或浮点数的列表。

    Returns:
        int: 列表中所有数字的和。
    """
    total = 0
    for num in numbers:
        total += num
    return total

def get_current_working_directory() -> str:
    """
    使用 os 模块获取当前工作目录。

    Returns:
        str: 当前工作目录的路径。
    """
    import os
    return os.getcwd()

class FileHandler:
    """
    文件处理类，用于读写文件。
    """
    def __init__(self, filepath: str):
        self.filepath = filepath

    def read_file(self) -> str:
        """读取文件内容。"""
        with open(self.filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        return content

    def write_to_file(self, content: str):
        """将内容写入文件。"""
        with open(self.filepath, 'w', encoding='utf-8') as f:
            f.write(content)
```

### /Users/yuanzhi/CodeSage/my_code_repo_for_lancedb_demo/rerank.py

- 语言：python
- 代码块数：2

#### 代码块

##### 代码块 1

```python
# Requires transformers>=4.51.0
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM

def format_instruction(instruction, query, doc):
    if instruction is None:
        instruction = 'Given a web search query, retrieve relevant passages that answer the query'
    output = "<Instruct>: {instruction}\n<Query>: {query}\n<Document>: {doc}".format(instruction=instruction,query=query, doc=doc)
    return output

def process_inputs(pairs):
    inputs = tokenizer(
        pairs, padding=False, truncation='longest_first',
        return_attention_mask=False, max_length=max_length - len(prefix_tokens) - len(suffix_tokens)
    )
    for i, ele in enumerate(inputs['input_ids']):
        inputs['input_ids'][i] = prefix_tokens + ele + suffix_tokens
    inputs = tokenizer.pad(inputs, padding=True, return_tensors="pt", max_length=max_length)
    for key in inputs:
        inputs[key] = inputs[key].to(model.device)
    return inputs

@torch.no_grad()
def compute_logits(inputs, **kwargs):
    batch_scores = model(**inputs).logits[:, -1, :]
    true_vector = batch_scores[:, token_true_id]
    false_vector = batch_scores[:, token_false_id]
    batch_scores = torch.stack([false_vector, true_vector], dim=1)
    batch_scores = torch.nn.functional.log_softmax(batch_scores, dim=1)
    scores = batch_scores[:, 1].exp().tolist()
    return scores

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-Reranker-0.6B", padding_side='left')
```

##### 代码块 2

```python
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B").eval()
# We recommend enabling flash_attention_2 for better acceleration and memory saving.
# model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-Reranker-0.6B", torch_dtype=torch.float16, attn_implementation="flash_attention_2").cuda().eval()
token_false_id = tokenizer.convert_tokens_to_ids("no")
token_true_id = tokenizer.convert_tokens_to_ids("yes")
max_length = 8192

prefix = "<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n"
suffix = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
prefix_tokens = tokenizer.encode(prefix, add_special_tokens=False)
suffix_tokens = tokenizer.encode(suffix, add_special_tokens=False)
        
task = 'Given a web search query, retrieve relevant passages that answer the query'

queries = ["What is the capital of China?",
    "Explain gravity",
]

documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

pairs = [format_instruction(task, query, doc) for query, doc in zip(queries, documents)]

# Tokenize the input texts
inputs = process_inputs(pairs)
scores = compute_logits(inputs)

print("scores: ", scores)
```

### /Users/yuanzhi/CodeSage/my_code_repo_for_lancedb_demo/embedding.py

- 语言：python
- 代码块数：1

#### 代码块

##### 代码块 1

```python
# Requires transformers>=4.51.0
# Requires sentence-transformers>=2.7.0

from sentence_transformers import SentenceTransformer

# Load the model
model = SentenceTransformer("Qwen/Qwen3-Embedding-0.6B")

# We recommend enabling flash_attention_2 for better acceleration and memory saving,
# together with setting `padding_side` to "left":
# model = SentenceTransformer(
#     "Qwen/Qwen3-Embedding-0.6B",
#     model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto"},
#     tokenizer_kwargs={"padding_side": "left"},
# )

# The queries and documents to embed
queries = [
    "What is the capital of China?",
    "Explain gravity",
]
documents = [
    "The capital of China is Beijing.",
    "Gravity is a force that attracts two bodies towards each other. It gives weight to physical objects and is responsible for the movement of planets around the sun.",
]

# Encode the queries and documents. Note that queries benefit from using a prompt
# Here we use the prompt called "query" stored under `model.prompts`, but you can
# also pass your own prompt via the `prompt` argument
query_embeddings = model.encode(queries, prompt_name="query")
document_embeddings = model.encode(documents)

# Compute the (cosine) similarity between the query and document embeddings
similarity = model.similarity(query_embeddings, document_embeddings)
print(similarity)
# tensor([[0.7646, 0.1414],
#         [0.1355, 0.6000]])
```

### /Users/yuanzhi/CodeSage/my_code_repo_for_lancedb_demo/test_code.py

- 语言：python
- 代码块数：8

#### 代码块

##### 代码块 1

```python
"""
这是一个用于测试代码分割器的示例文件。
包含了各种函数、类、装饰器等Python特性。
"""

import time
import random
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from functools import wraps
import logging
import json
import os
from pathlib import Path

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def timing_decorator(func):
    """计算函数执行时间的装饰器"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        logger.info(f"函数 {func.__name__} 执行时间: {end_time - start_time:.2f} 秒")
        return result
    return wrapper

@dataclass
class DataPoint:
    """数据点类"""
    x: float
    y: float
    label: str
    metadata: Dict[str, Any]
```

##### 代码块 2

```python
class DataProcessor:
```

##### 代码块 3

```python
"""数据处理类"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.data_points: List[DataPoint] = []
        self.processed_data: Dict[str, List[float]] = {}
    
    @timing_decorator
    def load_data(self, file_path: str) -> None:
        """加载数据"""
        try:
            with open(file_path, 'r') as f:
                raw_data = json.load(f)
            
            for item in raw_data:
                point = DataPoint(
                    x=item['x'],
                    y=item['y'],
                    label=item['label'],
                    metadata=item.get('metadata', {})
                )
                self.data_points.append(point)
                
            logger.info(f"成功加载 {len(self.data_points)} 个数据点")
        except Exception as e:
            logger.error(f"加载数据失败: {str(e)}")
            raise
    
    def process_data(self) -> Dict[str, List[float]]:
        """处理数据"""
        results = {
            'x_values': [],
            'y_values': [],
            'distances': []
        }
        
        for point in self.data_points:
            results['x_values'].append(point.x)
            results['y_values'].append(point.y)
            distance = (point.x ** 2 + point.y ** 2) ** 0.5
            results['distances'].append(distance)
        
        self.processed_data = results
        return results
```

##### 代码块 4

```python
def save_results(self, output_path: str) -> None:
        """保存处理结果"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.processed_data, f, indent=2)
        logger.info(f"结果已保存到: {output_path}")
```

##### 代码块 5

```python
class DataAnalyzer:
```

##### 代码块 6

```python
"""数据分析类"""
    
    def __init__(self, data: Dict[str, List[float]]):
        self.data = data
        self.statistics: Dict[str, Dict[str, float]] = {}
    
    def calculate_statistics(self) -> Dict[str, Dict[str, float]]:
        """计算统计数据"""
        for key, values in self.data.items():
            self.statistics[key] = {
                'mean': sum(values) / len(values),
                'min': min(values),
                'max': max(values),
                'std': self._calculate_std(values)
            }
        return self.statistics
    
    def _calculate_std(self, values: List[float]) -> float:
        """计算标准差"""
        mean = sum(values) / len(values)
        squared_diff_sum = sum((x - mean) ** 2 for x in values)
        return (squared_diff_sum / len(values)) ** 0.5
    
    def generate_report(self, output_path: str) -> None:
        """生成分析报告"""
        report = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'statistics': self.statistics,
            'summary': self._generate_summary()
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        logger.info(f"分析报告已保存到: {output_path}")
    
    def _generate_summary(self) -> Dict[str, Any]:
        """生成数据摘要"""
        return {
            'total_points': len(next(iter(self.data.values()))),
            'analysis_time': time.strftime('%Y-%m-%d %H:%M:%S'),
            'data_quality': self._assess_data_quality()
        }
```

##### 代码块 7

```python
def _assess_data_quality(self) -> str:
        """评估数据质量"""
        # 这里只是一个示例，实际应用中可能需要更复杂的评估逻辑
        return "良好" if len(self.data) > 0 else "未知"
```

##### 代码块 8

```python
def main():
    """主函数"""
    # 创建测试数据
    test_data = {
        'x_values': [random.uniform(-10, 10) for _ in range(100)],
        'y_values': [random.uniform(-10, 10) for _ in range(100)],
        'distances': [random.uniform(0, 20) for _ in range(100)]
    }
    
    # 保存测试数据
    with open('test_data.json', 'w') as f:
        json.dump(test_data, f, indent=2)
    
    # 创建数据处理器实例
    processor = DataProcessor({'debug': True})
    
    try:
        # 加载数据
        processor.load_data('test_data.json')
        
        # 处理数据
        processed_data = processor.process_data()
        
        # 创建分析器实例
        analyzer = DataAnalyzer(processed_data)
        
        # 计算统计数据
        statistics = analyzer.calculate_statistics()
        
        # 生成报告
        analyzer.generate_report('analysis_report.json')
        
        logger.info("数据处理和分析完成")
        
    except Exception as e:
        logger.error(f"处理过程中出错: {str(e)}")
        raise

if __name__ == "__main__":
    main()
```

### /Users/yuanzhi/CodeSage/my_code_repo_for_lancedb_demo/data_processing.py

- 语言：python
- 代码块数：1

#### 代码块

##### 代码块 1

```python
# data_processing.py

import pandas as pd

def load_csv(filepath: str) -> pd.DataFrame:
    """
    加载 CSV 文件到 Pandas DataFrame。

    Args:
        filepath (str): CSV 文件的路径。

    Returns:
        pd.DataFrame: 加载的 DataFrame。
    """
    df = pd.read_csv(filepath)
    return df

def filter_dataframe(df: pd.DataFrame, column: str, value: any) -> pd.DataFrame:
    """
    根据列值过滤 DataFrame。

    Args:
        df (pd.DataFrame): 输入 DataFrame。
        column (str): 要过滤的列名。
        value (any): 要匹配的值。

    Returns:
        pd.DataFrame: 过滤后的 DataFrame。
    """
    return df[df[column] == value]

def save_dataframe_to_parquet(df: pd.DataFrame, filepath: str):
    """
    将 DataFrame 保存为 Parquet 文件。

    Args:
        df (pd.DataFrame): 要保存的 DataFrame。
        filepath (str): Parquet 文件的保存路径。
    """
    df.to_parquet(filepath)
```

### /Users/yuanzhi/CodeSage/my_code_repo_for_lancedb_demo/code_splitter.py

- 语言：python
- 代码块数：9

#### 代码块

##### 代码块 1

```python
"""
代码切分器模块，提供基于 AST 的代码切分功能。
"""
from typing import List, Dict, Any
from pathlib import Path
import json
import time
from datetime import datetime

from llama_index.core import Document
from llama_index.core.node_parser import CodeSplitter
```

##### 代码块 2

```python
class ASTCodeSplitter:
```

##### 代码块 3

```python
"""基于 AST 的代码切分器，支持多种编程语言。"""
    
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
    
    def __init__(
        self,
        chunk_lines: int = 40,
        chunk_lines_overlap: int = 15,
        max_chars: int = 1500,
        language: str = "python"
    ):
        """
        初始化代码切分器。

        Args:
            chunk_lines: 代码块最大行数
            chunk_lines_overlap: 代码块重叠行数
            max_chars: 代码块最大字符数
            language: 默认编程语言
        """
        self.chunk_lines = chunk_lines
        self.chunk_lines_overlap = chunk_lines_overlap
        self.max_chars = max_chars
        self.language = language
        
        # 初始化 LlamaIndex 的代码切分器
        self._splitter = CodeSplitter(
            language=language,
            chunk_lines=chunk_lines,
            chunk_lines_overlap=chunk_lines_overlap,
            max_chars=max_chars
        )
```

##### 代码块 4

```python
def _detect_language(self, file_path: Path) -> str:
        """
        根据文件扩展名检测编程语言。

        Args:
            file_path: 文件路径

        Returns:
            str: 编程语言名称
        """
        extension = file_path.suffix.lower()
        return self.LANGUAGE_MAP.get(extension, '')
```

##### 代码块 5

```python
def split_directory(self, directory_path: str) -> List[Dict[str, Any]]:
```

##### 代码块 6

```python
"""
        处理文件夹中的所有代码文件，返回代码块信息。

        Args:
            directory_path: 文件夹路径

        Returns:
            List[Dict[str, Any]]: 代码块信息列表，每个字典包含：
                - file_path: 文件路径
                - language: 编程语言
                - code_blocks: 代码块列表
                - metadata: 元数据信息
        """
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"目录不存在或不是有效的目录: {directory_path}")

        results = []
        
        # 递归遍历目录
```

##### 代码块 7

```python
for file_path in directory.rglob("*"):
            if not file_path.is_file():
                continue
                
            # 检测文件语言
            language = self._detect_language(file_path)
            if not language:
                continue
                
            try:
                # 读取文件内容
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 分割代码
                nodes = self._splitter.split_text(content)
                
                # 处理分割结果
                result = {
                    "file_path": str(file_path),
                    "language": language,
                    "code_blocks": [str(node) for node in nodes],
                    "metadata": {
                        "total_blocks": len(nodes),
                        "file_size": len(content),
                        "last_modified": datetime.fromtimestamp(file_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                    }
                }
                results.append(result)
                
            except UnicodeDecodeError:
                print(f"警告：无法解码文件 {file_path}，跳过处理")
                continue
            except Exception as e:
                print(f"处理文件 {file_path} 时出错：{str(e)}")
                continue

        return results
```

##### 代码块 8

```python
def process_code_repository(repo_path: str) -> Dict[str, Any]:
    """
    处理代码仓库，返回处理结果。

    Args:
        repo_path: 代码仓库路径

    Returns:
        Dict[str, Any]: 处理结果统计信息
    """
    # 初始化代码切分器
    splitter = ASTCodeSplitter(
        chunk_lines=40,
        chunk_lines_overlap=15,
        max_chars=1500
    )

    # 处理代码仓库
    results = splitter.split_directory(repo_path)

    return results
```

##### 代码块 9

```python
def main():
    """主函数"""
    # 获取当前文件所在目录
    current_dir = Path(__file__).parent.parent.parent.parent
    repo_path = current_dir / "my_code_repo_for_lancedb_demo"

    if not repo_path.exists():
        print(f"错误：代码仓库路径不存在: {repo_path}")
        return

    print(f"开始处理代码仓库: {repo_path}")
    
    try:
        # 处理代码仓库
        stats = process_code_repository(str(repo_path))
        
        # 将结果保存到 JSON 文件
        output_file = repo_path / "code_analysis_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\n处理结果已保存到: {output_file}")
        
        # 打印简要统计信息
        total_files = len(stats)
        total_blocks = sum(len(item["code_blocks"]) for item in stats)
        print(f"\n处理完成:")
        print(f"总文件数: {total_files}")
        print(f"总代码块数: {total_blocks}")

    except Exception as e:
        print(f"处理过程中出错: {str(e)}")

if __name__ == "__main__":
    main()
```

