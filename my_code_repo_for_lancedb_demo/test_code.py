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

class DataProcessor:
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
    
    def save_results(self, output_path: str) -> None:
        """保存处理结果"""
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'w') as f:
            json.dump(self.processed_data, f, indent=2)
        logger.info(f"结果已保存到: {output_path}")

class DataAnalyzer:
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
    
    def _assess_data_quality(self) -> str:
        """评估数据质量"""
        # 这里只是一个示例，实际应用中可能需要更复杂的评估逻辑
        return "良好" if len(self.data) > 0 else "未知"

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