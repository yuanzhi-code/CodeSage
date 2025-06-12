
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
