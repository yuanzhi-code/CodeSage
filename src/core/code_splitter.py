"""
代码切分器模块，提供基于 AST 的代码切分功能。
"""
from typing import List, Dict, Any
from pathlib import Path
import json
from llama_index.core.node_parser import CodeSplitter
import numpy as np
from src.model.embedding import model

class ASTCodeSplitter:
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
        chunk_lines: int = 60,
        chunk_lines_overlap: int = 10,
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

    def split_directory(self, directory_path: str) -> List[Dict[str, Any]]:
        """
        处理文件夹中的所有代码文件，返回代码块信息。

        Args:
            directory_path: 文件夹路径

        Returns:
            List[Dict[str, Any]]: 代码块信息列表，每个字典包含：
                - text: 代码文本
                - metadata: 元数据信息
                - vector: 向量嵌入
        """
        directory = Path(directory_path)
        if not directory.exists() or not directory.is_dir():
            raise ValueError(f"目录不存在或不是有效的目录: {directory_path}")

        results = []
        
        # 递归遍历目录
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
                    code = f.read()
                
                # 根据语言处理文件
                if language == 'python':
                    code_blocks = self._splitter.split_text(code)
                    code_block_vectors = model.encode(code_blocks)
                    for i, block in enumerate(code_blocks):
                        chunk = {
                            "text": str(block),
                            "metadata": {
                                "file_path": str(file_path),
                                "type": "CodeBlock",
                                "start_line": 1
                            },
                            "vector": code_block_vectors[i].tolist()
                        }
                        results.append(chunk)
                
            except UnicodeDecodeError:
                print(f"警告：无法解码文件 {file_path}，跳过处理")
                continue
            except Exception as e:
                print(f"处理文件 {file_path} 时出错：{str(e)}")
                continue

        return results
        

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
        chunk_lines=300,
        chunk_lines_overlap=100,
        max_chars=4000
    )

    # 处理代码仓库
    results = splitter.split_directory(repo_path)

    return results

def main():
    """主函数"""
    # 获取当前文件所在目录
    current_dir = Path(__file__).parent.parent.parent
    repo_path = current_dir / "my_code_repo_for_lancedb_demo"

    if not repo_path.exists():
        print(f"错误：代码仓库路径不存在: {repo_path}")
        return

    print(f"开始处理代码仓库: {repo_path}")
    
    try:
        # 处理代码仓库
        stats = process_code_repository(str(repo_path))
        
        # 将结果保存到 JSON 文件
        json_output = repo_path / "code_analysis_results.json"
        with open(json_output, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        print(f"\nJSON 结果已保存到: {json_output}")

    except Exception as e:
        print(f"处理过程中出错: {str(e)}")

if __name__ == "__main__":
    main() 