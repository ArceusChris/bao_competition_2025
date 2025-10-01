#!/usr/bin/env python3
"""
数据集转换器测试脚本
快速测试数据集转换功能
"""

import os
import sys
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_dataset_converter():
    """测试数据集转换器基本功能"""
    print("=== 数据集转换器测试 ===\n")
    
    try:
        from dataset_converter import DatasetConverter
        from keyword_extractor import KeywordExtractor
        
        # 测试关键词提取器在数据集描述上的表现
        print("1. 测试关键词提取器在数据集描述上的表现:")
        print("-" * 50)
        
        extractor = KeywordExtractor('nltk')
        
        # 读取一些真实的描述文件进行测试
        test_descriptions = []
        
        # 收集一些示例描述
        source_dir = Path("TrainSet")
        if source_dir.exists():
            sample_files = []
            
            # 从不同类别中收集示例
            categories = ["Animal", "Bag", "Human", "Car", "Chair"]
            for category in categories:
                category_path = source_dir / category
                if category_path.exists():
                    instances = list(category_path.iterdir())[:2]  # 每个类别取2个实例
                    for instance in instances:
                        nlp_file = instance / "nlp.txt"
                        if nlp_file.exists():
                            try:
                                with open(nlp_file, 'r', encoding='utf-8') as f:
                                    desc = f.read().strip()
                                    test_descriptions.append((category, instance.name, desc))
                            except:
                                continue
        
        # 测试关键词提取
        for category, instance, description in test_descriptions[:10]:  # 只显示前10个
            main_subject = extractor.extract_main_subject(description)
            keywords = extractor.extract_keywords(description)
            
            print(f"类别: {category}")
            print(f"实例: {instance}")
            print(f"描述: {description}")
            print(f"提取的中心词: {main_subject}")
            print(f"关键词: {keywords}")
            print()
        
        # 测试小规模转换
        print("2. 测试小规模数据集转换:")
        print("-" * 50)
        
        # 创建转换器（限制数据量用于测试）
        converter = DatasetConverter("TrainSet", "test_yolo_dataset")
        
        # 只转换很少的数据用于测试
        print("开始小规模转换测试（每个类别最多1个实例）...")
        converter.convert_dataset(
            train_ratio=0.8,
            max_instances_per_category=1  # 每个类别只取1个实例
        )
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def inspect_dataset_structure():
    """检查数据集结构"""
    print("=== 数据集结构检查 ===\n")
    
    source_dir = Path("TrainSet")
    if not source_dir.exists():
        print("❌ TrainSet 目录不存在")
        return
    
    categories = list(source_dir.iterdir())
    print(f"发现 {len(categories)} 个类别:")
    
    total_instances = 0
    for category in sorted(categories)[:10]:  # 只显示前10个类别
        if category.is_dir():
            instances = [p for p in category.iterdir() if p.is_dir()]
            total_instances += len(instances)
            print(f"  {category.name}: {len(instances)} 个实例")
    
    print(f"\n总计约 {total_instances} 个实例（仅前10个类别）")
    
    # 查看一个具体实例的结构
    print("\n示例实例结构:")
    sample_instance = None
    for category in categories:
        if category.is_dir():
            instances = [p for p in category.iterdir() if p.is_dir()]
            if instances:
                sample_instance = instances[0]
                break
    
    if sample_instance:
        print(f"实例: {sample_instance}")
        for item in sample_instance.iterdir():
            if item.is_dir():
                sub_items = list(item.iterdir())
                print(f"  📁 {item.name}/ ({len(sub_items)} 个文件)")
            else:
                print(f"  📄 {item.name}")

def check_conversion_result():
    """检查转换结果"""
    print("=== 检查转换结果 ===\n")
    
    output_dir = Path("test_yolo_dataset")
    if not output_dir.exists():
        print("❌ 转换输出目录不存在，请先运行转换测试")
        return
    
    # 检查目录结构
    print("YOLO数据集目录结构:")
    for root, dirs, files in os.walk(output_dir):
        level = root.replace(str(output_dir), '').count(os.sep)
        indent = ' ' * 2 * level
        print(f'{indent}{os.path.basename(root)}/')
        subindent = ' ' * 2 * (level + 1)
        for file in files[:5]:  # 每个目录只显示前5个文件
            print(f'{subindent}{file}')
        if len(files) > 5:
            print(f'{subindent}... 还有 {len(files) - 5} 个文件')
    
    # 检查配置文件
    config_files = [
        output_dir / "dataset.yaml",
        output_dir / "class_mapping.json"
    ]
    
    for config_file in config_files:
        if config_file.exists():
            print(f"\n✅ 配置文件存在: {config_file.name}")
            if config_file.name.endswith('.json'):
                import json
                with open(config_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    print(f"   类别数量: {data.get('total_classes', 'N/A')}")
                    print(f"   图片总数: {data.get('total_images', 'N/A')}")
        else:
            print(f"❌ 配置文件缺失: {config_file.name}")

if __name__ == "__main__":
    print("🧪 数据集转换器全面测试\n")
    
    # 步骤1: 检查数据集结构
    inspect_dataset_structure()
    
    print("\n" + "="*60)
    
    # 步骤2: 测试转换功能
    success = test_dataset_converter()
    
    print("\n" + "="*60)
    
    # 步骤3: 检查转换结果
    if success:
        check_conversion_result()
    
    print("\n🏁 测试完成！")
    if success:
        print("✅ 数据集转换器测试通过")
        print("💡 可以运行 'python dataset_converter.py' 进行完整转换")
    else:
        print("❌ 测试失败，请检查代码")