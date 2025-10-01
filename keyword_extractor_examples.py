#!/usr/bin/env python3
"""
关键词提取器使用示例
展示如何在实际项目中使用关键词提取器
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def example_basic_usage():
    """基本使用示例"""
    print("=== 基本使用示例 ===\n")
    
    from keyword_extractor import KeywordExtractor
    
    # 创建提取器实例
    extractor = KeywordExtractor('nltk')  # 可选: 'spacy', 'yake'
    
    # 示例文本
    texts = [
        "A woman holding an umbrella",
        "A yellow hat carried by a man",
        "A beautiful sunset over the mountains",
        "Children playing soccer in the park"
    ]
    
    for text in texts:
        print(f"输入: {text}")
        
        # 提取关键词
        keywords = extractor.extract_keywords(text)
        print(f"关键词: {keywords}")
        
        # 提取中心词
        main_subject = extractor.extract_main_subject(text)
        print(f"中心词: {main_subject}")
        print()

def example_batch_processing():
    """批量处理示例"""
    print("=== 批量处理示例 ===\n")
    
    from keyword_extractor import KeywordExtractor
    
    extractor = KeywordExtractor('nltk')
    
    # 批量文本
    descriptions = [
        "A black cat sitting on a windowsill",
        "Red roses blooming in a garden",
        "Old man reading a newspaper in the park",
        "Young girl playing piano in the living room",
        "Golden retriever running on the beach",
        "Blue bicycle parked near the fountain"
    ]
    
    results = []
    for desc in descriptions:
        result = extractor.analyze_text(desc)
        results.append(result)
        print(f"描述: {desc}")
        print(f"中心词: {result['main_subject']}")
        print(f"关键词: {result['keywords']}")
        print()
    
    return results

def example_comparison_methods():
    """比较不同方法的示例"""
    print("=== 方法比较示例 ===\n")
    
    from keyword_extractor import KeywordExtractor
    
    test_text = "A magnificent eagle soaring high above the snow-capped mountains"
    print(f"测试文本: {test_text}\n")
    
    methods = ['nltk']  # 可以添加 'spacy', 'yake' 如果安装了的话
    
    for method in methods:
        try:
            print(f"--- 使用 {method.upper()} 方法 ---")
            extractor = KeywordExtractor(method)
            
            keywords = extractor.extract_keywords(test_text)
            main_subject = extractor.extract_main_subject(test_text)
            
            print(f"关键词: {keywords}")
            print(f"中心词: {main_subject}")
            print()
            
        except Exception as e:
            print(f"{method} 方法不可用: {e}\n")

def example_integration_with_yolo():
    """与YOLO-World集成的示例"""
    print("=== 与YOLO-World集成示例 ===\n")
    
    from keyword_extractor import KeywordExtractor
    
    extractor = KeywordExtractor('nltk')
    
    # 模拟场景描述
    scene_descriptions = [
        "Find all cats in the image",
        "Detect people walking on the street", 
        "Look for cars parked in the parking lot",
        "Identify dogs playing in the yard",
        "Spot bicycles near the building"
    ]
    
    print("为YOLO-World检测准备关键词:")
    print("-" * 40)
    
    for desc in scene_descriptions:
        keywords = extractor.extract_keywords(desc)
        main_subject = extractor.extract_main_subject(desc)
        
        print(f"原始描述: {desc}")
        print(f"检测目标: {main_subject}")
        print(f"相关词汇: {keywords}")
        print(f"YOLO类别: [{main_subject}]")  # 主要检测目标
        print()

def example_text_preprocessing():
    """文本预处理示例"""
    print("=== 文本预处理示例 ===\n")
    
    from keyword_extractor import KeywordExtractor
    
    extractor = KeywordExtractor('keybert')
    
    # 各种类型的输入文本
    complex_texts = [
        "There's a beautiful, big brown dog running quickly through the green grass.",
        "Multiple colorful balloons are floating high up in the bright blue sky.",
        "The old wooden chair is sitting quietly in the corner of the small room.",
        "Several young children are happily playing with their favorite toys."
    ]
    
    print("复杂文本的关键词提取:")
    print("-" * 35)
    
    for text in complex_texts:
        analysis = extractor.analyze_text(text)
        
        print(f"原文: {text}")
        print(f"简化为 -> 中心词: '{analysis['main_subject']}'")
        print(f"相关词: {analysis['keywords']}")
        print()

def main():
    """运行所有示例"""
    print("🚀 关键词提取器使用示例\n")
    
    # 运行各种示例
    example_text_preprocessing()
    
    print("✅ 所有示例运行完成！")
    print("\n📝 使用说明:")
    print("1. 基本用法: KeywordExtractor('nltk').extract_main_subject(text)")
    print("2. 获取关键词: KeywordExtractor('nltk').extract_keywords(text)")
    print("3. 完整分析: KeywordExtractor('nltk').analyze_text(text)")
    print("4. 支持的方法: 'nltk', 'spacy'(需安装), 'yake'(需安装)")

if __name__ == "__main__":
    main()