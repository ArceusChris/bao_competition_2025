"""
使用示例：展示如何使用改进的YOLO-World系统
"""

import os
import sys
import cv2
import numpy as np

# 添加当前目录到Python路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def example_1_keyword_extraction():
    """示例1: 关键词提取"""
    print("=== 示例1: 关键词提取 ===")
    
    try:
        from keyword_extractor import KeywordExtractor
        
        # 初始化提取器
        extractor = KeywordExtractor('nltk')  # 或 'spacy' 如果已安装
        
        # 测试文本
        test_texts = [
            "A black cat lying on a chair",
            "A red car parked in the garage", 
            "Children playing with a ball in the park",
            "A woman wearing a blue dress walking down the street"
        ]
        
        for text in test_texts:
            print(f"\n输入文本: {text}")
            
            # 提取所有关键词
            keywords = extractor.extract_keywords(text)
            print(f"关键词: {keywords}")
            
            # 提取主要中心词
            main_subject = extractor.extract_main_subject(text)
            print(f"中心词: {main_subject}")
            
    except Exception as e:
        print(f"示例1执行失败: {e}")

def example_2_create_test_image():
    """示例2: 创建测试图像"""
    print("\n=== 示例2: 创建测试图像 ===")
    
    try:
        # 创建一个包含"猫"和"椅子"的简单测试图像
        image = np.ones((400, 600, 3), dtype=np.uint8) * 240  # 浅灰色背景
        
        # 绘制椅子（棕色矩形）
        cv2.rectangle(image, (100, 200), (250, 350), (139, 69, 19), -1)  # 椅子座位
        cv2.rectangle(image, (100, 120), (250, 200), (139, 69, 19), -1)  # 椅子靠背
        
        # 绘制猫（黑色椭圆）
        cv2.ellipse(image, (300, 280), (60, 40), 0, 0, 360, (0, 0, 0), -1)  # 猫身体
        cv2.circle(image, (280, 250), 25, (0, 0, 0), -1)  # 猫头
        
        # 添加一些细节
        cv2.circle(image, (275, 245), 3, (255, 255, 255), -1)  # 猫眼睛
        cv2.circle(image, (285, 245), 3, (255, 255, 255), -1)  # 猫眼睛
        
        # 保存图像
        test_image_path = "example_cat_chair.jpg"
        cv2.imwrite(test_image_path, image)
        print(f"测试图像已保存: {test_image_path}")
        
        return test_image_path
        
    except Exception as e:
        print(f"示例2执行失败: {e}")
        return None

def example_3_complete_workflow():
    """示例3: 完整工作流程"""
    print("\n=== 示例3: 完整工作流程 ===")
    
    # 检查模型文件
    model_path = "yolov8s-world.pt"
    if not os.path.exists(model_path):
        print(f"模型文件 {model_path} 不存在")
        print("请先下载模型或运行: python setup.py")
        return
    
    # 创建测试图像
    test_image = example_2_create_test_image()
    if not test_image:
        return
    
    try:
        from main import ImprovedYOLOWorld
        
        # 初始化系统
        print("初始化系统...")
        system = ImprovedYOLOWorld(
            model_path=model_path,
            keyword_method="nltk",  # 使用nltk避免spacy安装问题
            similarity_method="clip",
            device="cpu"  # 使用CPU确保兼容性
        )
        
        # 处理图像
        text_description = "A black cat lying on a chair"
        print(f"处理文本: {text_description}")
        
        result = system.process_single_image(
            test_image,
            text_description,
            conf_threshold=0.1,
            similarity_threshold=0.2
        )
        
        # 显示结果
        print("\n处理结果:")
        print(f"图片: {result['image_path']}")
        print(f"文本: {result['text_description']}")
        print(f"关键词: {result['keywords']}")
        print(f"检测总数: {result['total_detections']}")
        print(f"高相关度检测: {result['filtered_detections']}")
        
        if result['ranked_detections']:
            print("\n排序结果:")
            for item in result['ranked_detections']:
                det = item['detection']
                print(f"  排名 {item['rank']}: {det['class_name']} "
                      f"(置信度: {det['confidence']:.3f}, "
                      f"相似度: {item['similarity_score']:.3f})")
        
        # 可视化结果
        vis_path = "example_result.jpg"
        system.visualize_results(result, vis_path)
        
        print(f"\n可视化结果已保存: {vis_path}")
        
    except Exception as e:
        print(f"示例3执行失败: {e}")
        import traceback
        traceback.print_exc()

def example_4_batch_processing():
    """示例4: 批量处理"""
    print("\n=== 示例4: 批量处理演示 ===")
    
    try:
        from keyword_extractor import KeywordExtractor
        
        # 模拟批量处理场景
        batch_data = [
            {"text": "A black cat lying on a chair", "expected": "cat"},
            {"text": "A red car parked in the garage", "expected": "car"},
            {"text": "Children playing with a ball", "expected": "ball"},
            {"text": "A dog running in the park", "expected": "dog"},
            {"text": "Books on a wooden table", "expected": "books"}
        ]
        
        extractor = KeywordExtractor('nltk')
        
        print("批量关键词提取结果:")
        for i, item in enumerate(batch_data, 1):
            text = item["text"]
            expected = item["expected"]
            
            keywords = extractor.extract_keywords(text)
            main_subject = extractor.extract_main_subject(text)
            
            print(f"\n{i}. 文本: {text}")
            print(f"   关键词: {keywords}")
            print(f"   中心词: {main_subject}")
            print(f"   期望词: {expected}")
            
            # 简单的准确性检查
            if main_subject.lower() in expected.lower() or expected.lower() in main_subject.lower():
                print("   ✓ 中心词提取正确")
            else:
                print("   ⚠️  中心词可能不够准确")
                
    except Exception as e:
        print(f"示例4执行失败: {e}")

def main():
    """运行所有示例"""
    print("改进的YOLO-World系统使用示例")
    print("="*50)
    
    # 运行各个示例
    example_1_keyword_extraction()
    example_2_create_test_image()
    
    # 只有在有模型的情况下才运行完整示例
    if os.path.exists("yolov8s-world.pt"):
        example_3_complete_workflow()
    else:
        print("\n跳过完整工作流程示例（需要模型文件）")
    
    example_4_batch_processing()
    
    print("\n" + "="*50)
    print("示例运行完成!")
    print("\n使用说明:")
    print("1. 安装依赖: pip install -r requirements.txt")
    print("2. 下载模型: python setup.py")
    print("3. 运行系统: python main.py --image image.jpg --text 'description'")
    print("4. 运行测试: python test_system.py")

if __name__ == "__main__":
    main()