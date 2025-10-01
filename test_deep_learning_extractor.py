#!/usr/bin/env python3
"""
深度学习关键词提取器测试脚本
测试基于深度学习的关键词提取方法
"""

import sys
import os
import time
from pathlib import Path

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_deep_learning_methods():
    """测试深度学习关键词提取方法"""
    print("🧠 深度学习关键词提取器测试")
    print("=" * 60)
    
    from keyword_extractor import KeywordExtractor
    
    # 测试用例
    test_cases = [
        "A woman holding an umbrella",
        "A yellow hat carried by a man",
        "A black cat lying on a chair",
        "Children playing with a ball in the park",
        "A red car parked in the garage",
        "The beautiful sunset over the mountains",
        "A group of students studying in the library",
        "An elderly man reading a newspaper on a bench",
        "A small dog running through the green grass",
        "Two birds sitting on a tree branch"
    ]
    
    # 深度学习方法列表
    dl_methods = [
        ('transformer', 'Sentence Transformer'),
        ('keybert', 'KeyBERT'),
        ('bert', 'BERT Masked Language Model'),
        ('llm', 'Large Language Model')
    ]
    
    results = {}
    
    for method_code, method_name in dl_methods:
        print(f"\n🔬 测试方法: {method_name}")
        print("-" * 50)
        
        try:
            start_time = time.time()
            extractor = KeywordExtractor(method_code)
            init_time = time.time() - start_time
            
            print(f"✅ 模型初始化成功 ({init_time:.2f}秒)")
            
            method_results = []
            total_time = 0
            
            for i, text in enumerate(test_cases, 1):
                try:
                    start_time = time.time()
                    
                    keywords = extractor.extract_keywords(text, max_keywords=5)
                    main_subject = extractor.extract_main_subject(text)
                    
                    extract_time = time.time() - start_time
                    total_time += extract_time
                    
                    result = {
                        'text': text,
                        'keywords': keywords,
                        'main_subject': main_subject,
                        'time': extract_time
                    }
                    method_results.append(result)
                    
                    print(f"{i:2d}. 输入: {text}")
                    print(f"    关键词: {keywords}")
                    print(f"    中心词: {main_subject}")
                    print(f"    用时: {extract_time:.3f}秒")
                    print()
                    
                except Exception as e:
                    print(f"❌ 文本 {i} 处理失败: {e}")
            
            # 计算统计信息
            avg_time = total_time / len(test_cases) if test_cases else 0
            
            results[method_code] = {
                'name': method_name,
                'results': method_results,
                'init_time': init_time,
                'total_time': total_time,
                'avg_time': avg_time,
                'success_count': len(method_results)
            }
            
            print(f"📊 {method_name} 统计:")
            print(f"   成功处理: {len(method_results)}/{len(test_cases)}")
            print(f"   平均用时: {avg_time:.3f}秒/文本")
            print(f"   总用时: {total_time:.2f}秒")
            
        except Exception as e:
            print(f"❌ {method_name} 初始化失败: {e}")
            print(f"   可能缺少依赖包，请参考安装说明")
            results[method_code] = None
    
    # 生成对比报告
    print("\n" + "=" * 60)
    print("📊 方法对比报告")
    print("=" * 60)
    
    successful_methods = {k: v for k, v in results.items() if v is not None}
    
    if successful_methods:
        print(f"{'方法':<20} {'成功率':<10} {'平均用时':<12} {'初始化时间':<12}")
        print("-" * 60)
        
        for method_code, data in successful_methods.items():
            success_rate = f"{data['success_count']}/{len(test_cases)}"
            avg_time = f"{data['avg_time']:.3f}s"
            init_time = f"{data['init_time']:.2f}s"
            
            print(f"{data['name']:<20} {success_rate:<10} {avg_time:<12} {init_time:<12}")
    
    # 展示一些具体对比例子
    if len(successful_methods) > 1:
        print(f"\n🔍 具体示例对比:")
        print("-" * 60)
        
        sample_text = test_cases[0]  # 使用第一个测试用例
        print(f"示例文本: '{sample_text}'")
        print()
        
        for method_code, data in successful_methods.items():
            if data['results']:
                result = data['results'][0]  # 第一个结果
                print(f"{data['name']}:")
                print(f"  关键词: {result['keywords']}")
                print(f"  中心词: '{result['main_subject']}'")
                print()
    
    return results

def test_method_accuracy():
    """测试方法准确性（与预期结果对比）"""
    print("\n🎯 准确性测试")
    print("=" * 60)
    
    from keyword_extractor import KeywordExtractor
    
    # 带有预期结果的测试用例
    test_cases_with_expected = [
        {
            'text': "A woman holding an umbrella",
            'expected_subject': "woman",
            'expected_keywords': ["woman", "umbrella"]
        },
        {
            'text': "A yellow hat carried by a man",
            'expected_subject': "hat",
            'expected_keywords': ["hat", "man", "yellow"]
        },
        {
            'text': "A black cat lying on a chair",
            'expected_subject': "cat", 
            'expected_keywords': ["cat", "chair", "black"]
        },
        {
            'text': "Children playing with a ball in the park",
            'expected_subject': "children",
            'expected_keywords': ["children", "ball", "park"]
        }
    ]
    
    methods_to_test = ['nltk', 'transformer', 'keybert']
    
    for method in methods_to_test:
        print(f"\n📏 测试方法: {method.upper()}")
        print("-" * 40)
        
        try:
            extractor = KeywordExtractor(method)
            
            correct_subjects = 0
            total_tests = len(test_cases_with_expected)
            
            for i, test_case in enumerate(test_cases_with_expected, 1):
                text = test_case['text']
                expected_subject = test_case['expected_subject']
                expected_keywords = test_case['expected_keywords']
                
                # 提取结果
                actual_subject = extractor.extract_main_subject(text)
                actual_keywords = extractor.extract_keywords(text)
                
                # 检查中心词准确性
                subject_correct = actual_subject.lower() in [expected_subject.lower(), 
                                                           expected_subject.lower().rstrip('s')]  # 处理复数
                if subject_correct:
                    correct_subjects += 1
                
                # 检查关键词覆盖率
                keyword_matches = sum(1 for kw in expected_keywords 
                                    if any(kw.lower() in actual_kw.lower() or 
                                          actual_kw.lower() in kw.lower() 
                                          for actual_kw in actual_keywords))
                keyword_coverage = keyword_matches / len(expected_keywords) if expected_keywords else 0
                
                print(f"{i}. {text}")
                print(f"   预期中心词: {expected_subject} | 实际: {actual_subject} | {'✅' if subject_correct else '❌'}")
                print(f"   关键词覆盖: {keyword_coverage:.1%} ({keyword_matches}/{len(expected_keywords)})")
                print(f"   实际关键词: {actual_keywords}")
                print()
            
            accuracy = correct_subjects / total_tests
            print(f"📊 {method.upper()} 中心词准确率: {accuracy:.1%} ({correct_subjects}/{total_tests})")
            
        except Exception as e:
            print(f"❌ {method} 测试失败: {e}")

def install_dependencies():
    """检查并提示安装深度学习依赖"""
    print("🔧 深度学习依赖检查")
    print("=" * 60)
    
    dependencies = {
        'transformers': 'pip install transformers torch',
        'sentence_transformers': 'pip install sentence-transformers',
        'keybert': 'pip install keybert',
        'sklearn': 'pip install scikit-learn',
        'openai': 'pip install openai  # 可选，用于LLM方法'
    }
    
    missing_deps = []
    
    for package, install_cmd in dependencies.items():
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package} - 运行: {install_cmd}")
            missing_deps.append(package)
    
    if missing_deps:
        print(f"\n⚠️  发现 {len(missing_deps)} 个缺失的依赖包")
        print("建议运行以下命令安装:")
        print("pip install transformers torch sentence-transformers keybert scikit-learn")
    else:
        print("\n✅ 所有深度学习依赖都已安装!")
    
    return len(missing_deps) == 0

def main():
    """主函数"""
    print("🚀 深度学习关键词提取器完整测试")
    print("=" * 80)
    
    # 1. 依赖检查
    deps_ok = install_dependencies()
    
    if not deps_ok:
        print("\n⚠️  部分依赖缺失，某些测试可能失败")
        print("是否继续测试? (y/n): ", end="")
        response = input().lower()
        if response != 'y':
            print("测试已取消")
            return
    
    # 2. 深度学习方法测试
    print("\n" + "=" * 80)
    dl_results = test_deep_learning_methods()
    
    # 3. 准确性测试
    test_method_accuracy()
    
    print("\n" + "=" * 80)
    print("🎉 测试完成!")
    print("💡 建议:")
    print("   - transformer: 平衡性能和准确性，推荐日常使用")
    print("   - keybert: 高质量关键词提取，适合学术应用")
    print("   - bert: 深度语义理解，计算资源充足时使用")
    print("   - llm: 最高准确性，需要API密钥")

if __name__ == "__main__":
    main()