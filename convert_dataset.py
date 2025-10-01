#!/usr/bin/env python3
"""
完整数据集转换脚本
将整个TrainSet转换为YOLO格式数据集
"""

import sys
import os
import argparse
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataset_converter import DatasetConverter

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="将TrainSet转换为YOLO格式数据集",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--source", "-s",
        type=str,
        default="TrainSet",
        help="源数据集目录路径"
    )
    
    parser.add_argument(
        "--output", "-o", 
        type=str,
        default="yolo_dataset",
        help="输出YOLO数据集目录路径"
    )
    
    parser.add_argument(
        "--train-ratio", "-r",
        type=float,
        default=0.8,
        help="训练集比例 (0.0-1.0)"
    )
    
    parser.add_argument(
        "--max-instances", "-m",
        type=int,
        default=None,
        help="每个类别最大实例数 (用于限制数据量，None表示不限制)"
    )
    
    parser.add_argument(
        "--test-mode", "-t",
        action="store_true",
        help="测试模式：只处理少量数据用于快速测试"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true", 
        help="详细输出"
    )
    
    args = parser.parse_args()
    
    # 验证参数
    if not (0.0 < args.train_ratio < 1.0):
        print("❌ 训练集比例必须在0.0到1.0之间")
        sys.exit(1)
    
    if not Path(args.source).exists():
        print(f"❌ 源目录不存在: {args.source}")
        sys.exit(1)
    
    # 测试模式的特殊设置
    if args.test_mode:
        args.max_instances = 2  # 每个类别最多2个实例
        args.output = "test_yolo_dataset"
        print("🧪 测试模式已启用")
    
    print("🔄 TrainSet 到 YOLO 数据集转换器")
    print("=" * 60)
    print(f"源目录: {args.source}")
    print(f"输出目录: {args.output}")
    print(f"训练集比例: {args.train_ratio}")
    if args.max_instances:
        print(f"每类最大实例数: {args.max_instances}")
    print("=" * 60)
    
    try:
        # 创建转换器
        converter = DatasetConverter(args.source, args.output)
        
        # 执行转换
        converter.convert_dataset(
            train_ratio=args.train_ratio,
            max_instances_per_category=args.max_instances
        )
        
        print("\n🎉 数据集转换完成！")
        print(f"📁 输出目录: {Path(args.output).absolute()}")
        
        # 显示使用方法
        print("\n📚 YOLO训练使用方法:")
        print("```python")
        print("from ultralytics import YOLO")
        print("")
        print("# 加载预训练模型")
        print("model = YOLO('yolov8s.pt')")
        print("")
        print("# 训练模型")
        print(f"results = model.train(")
        print(f"    data='{args.output}/dataset.yaml',")
        print(f"    epochs=100,")
        print(f"    imgsz=640,")
        print(f"    batch=16")
        print(f")")
        print("```")
        
        # 显示配置文件信息
        config_path = Path(args.output) / "dataset.yaml"
        mapping_path = Path(args.output) / "class_mapping.json"
        
        print(f"\n📋 重要文件:")
        print(f"   - 数据集配置: {config_path}")
        print(f"   - 类别映射: {mapping_path}")
        
        print(f"\n✨ 下一步建议:")
        print(f"   1. 检查生成的数据集结构")
        print(f"   2. 可视化一些样本确认转换正确")
        print(f"   3. 开始YOLO模型训练")
        
    except KeyboardInterrupt:
        print("\n⚠️ 转换被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ 转换过程中出现错误: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()