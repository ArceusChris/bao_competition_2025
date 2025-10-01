#!/usr/bin/env python3
"""
简单的关键词提取器测试脚本
专门用于测试关键词提取功能
"""

import sys
import os

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_keyword_extractor():
    """测试关键词提取器的基本功能"""
    print("=== 关键词提取器功能测试 ===\n")
    
    try:
        from keyword_extractor import KeywordExtractor
        
        # 初始化提取器
        extractor = KeywordExtractor('nltk')
        
        # 测试用例 - 您给出的示例
        test_cases = [
            ("A woman holding an umbrella", "woman"),
            ("A yellow hat carried by a man", "hat"),
            ("A black cat lying on a chair", "cat"),
            ("Children playing with a ball in the park", ["child", "children"]),  # child的复数形式
            ("A red car parked in the garage", "car"),
            ("The dog is running in the garden", "dog"),
            ("An old book on the table", "book"),
            ("Students studying in the library", ["student", "students"]),
        ]
        
        print("测试关键词提取和中心词识别:")
        print("-" * 60)
        
        success_count = 0
        total_count = len(test_cases)
        
        for i, (text, expected) in enumerate(test_cases, 1):
            print(f"{i}. 输入: {text}")
            
            # 提取关键词和中心词
            keywords = extractor.extract_keywords(text)
            main_subject = extractor.extract_main_subject(text)
            
            print(f"   关键词: {keywords}")
            print(f"   中心词: {main_subject}")
            
            # 检查结果是否符合预期
            if isinstance(expected, list):
                is_correct = main_subject in expected
            else:
                is_correct = main_subject == expected
            
            if is_correct:
                print("   ✓ 正确")
                success_count += 1
            else:
                print(f"   ✗ 错误 (期望: {expected})")
            
            print()
        
        print(f"测试结果: {success_count}/{total_count} 通过")
        print(f"成功率: {success_count/total_count*100:.1f}%")
        
        # 额外的功能测试
        print("\n=== 额外功能测试 ===")
        
        # 测试分析函数
        test_text = "A beautiful butterfly flying in the garden"
        result = extractor.analyze_text(test_text)
        print(f"完整分析结果:")
        print(f"原文: {result['original_text']}")
        print(f"方法: {result['method']}")  
        print(f"关键词: {result['keywords']}")
        print(f"中心词: {result['main_subject']}")
        
        return True
        
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_edge_cases():
    """测试边界情况"""
    print("\n=== 边界情况测试 ===")
    
    try:
        from keyword_extractor import KeywordExtractor
        extractor = KeywordExtractor('nltk')
        
        edge_cases = [
            "",  # 空字符串
            "   ",  # 空白字符
            "The",  # 只有停用词
            "A",  # 单个字母
            "Hello world",  # 简单句子
            "a b c d e f",  # 全是停用词
        ]
        
        for case in edge_cases:
            print(f"输入: '{case}'")
            keywords = extractor.extract_keywords(case)
            main_subject = extractor.extract_main_subject(case)
            print(f"关键词: {keywords}")
            print(f"中心词: '{main_subject}'")
            print()
            
    except Exception as e:
        print(f"边界情况测试失败: {e}")

if __name__ == "__main__":
    success = test_keyword_extractor()
    test_edge_cases()
    
    if success:
        print("🎉 关键词提取器测试完成！")
    else:
        print("❌ 测试失败，请检查代码")