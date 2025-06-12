
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
