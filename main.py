"""
主程序：整合所有模块实现YOLO-World改进功能
"""

import os
import cv2
import numpy as np
from typing import List, Dict, Any, Tuple
import argparse
import json

from keyword_extractor import KeywordExtractor
from yolo_world_inference import YOLOWorldInference
from semantic_similarity import SemanticSimilarityCalculator, MultiModalSimilarityCalculator

class ImprovedYOLOWorld:
    def __init__(self, 
                 model_path: str = "yolov8s-world.pt",
                 keyword_method: str = "spacy",
                 similarity_method: str = "clip",
                 device: str = "auto"):
        """
        初始化改进的YOLO-World系统
        
        Args:
            model_path: YOLO-World模型路径
            keyword_method: 关键词提取方法
            similarity_method: 语义相似度计算方法
            device: 推理设备
        """
        print("初始化改进的YOLO-World系统...")
        
        # 初始化各个模块
        self.keyword_extractor = KeywordExtractor(keyword_method)
        self.yolo_inference = YOLOWorldInference(model_path, device)
        self.similarity_calculator = SemanticSimilarityCalculator(similarity_method, device)
        
        print("系统初始化完成!")
    
    def process_single_image(self, 
                           image_path: str, 
                           text_description: str,
                           conf_threshold: float = 0.1,
                           similarity_threshold: float = 0.3,
                           extract_main_subject: bool = True) -> Dict[str, Any]:
        """
        处理单张图片
        
        Args:
            image_path: 图片路径
            text_description: 文本描述
            conf_threshold: YOLO置信度阈值
            similarity_threshold: 语义相似度阈值
            extract_main_subject: 是否只提取主要中心词
            
        Returns:
            处理结果字典
        """
        print(f"\n处理图片: {image_path}")
        print(f"文本描述: {text_description}")
        
        # 步骤1: 提取关键词
        if extract_main_subject:
            main_keyword = self.keyword_extractor.extract_main_subject(text_description)
            keywords = [main_keyword] if main_keyword else []
            print(f"提取的中心词: {main_keyword}")
        else:
            keywords = self.keyword_extractor.extract_keywords(text_description)
            print(f"提取的关键词: {keywords}")
        
        if not keywords:
            print("未提取到有效关键词!")
            return {"error": "未提取到有效关键词"}
        
        # 步骤2: YOLO-World检测
        print("进行YOLO-World检测...")
        detection_results = self.yolo_inference.predict(
            image_path, keywords, conf_threshold
        )
        
        print(f"检测到 {len(detection_results['detections'])} 个候选区域")
        
        if not detection_results['detections']:
            print("未检测到任何目标!")
            return {
                "image_path": image_path,
                "text_description": text_description,
                "keywords": keywords,
                "detections": [],
                "ranked_detections": []
            }
        
        # 步骤3: 裁剪检测区域
        cropped_images = self.yolo_inference.crop_detection_regions(
            image_path, detection_results
        )
        
        # 步骤4: 计算语义相关度并排序
        print("计算语义相关度...")
        ranked_results = self.similarity_calculator.rank_detections_by_similarity(
            cropped_images, text_description, detection_results['detections']
        )
        
        # 步骤5: 根据相似度阈值过滤
        filtered_results = self.similarity_calculator.filter_by_similarity_threshold(
            ranked_results, similarity_threshold
        )
        
        print(f"过滤后剩余 {len(filtered_results)} 个高相关度检测结果")
        
        # 整理结果
        final_results = {
            "image_path": image_path,
            "text_description": text_description,
            "keywords": keywords,
            "total_detections": len(detection_results['detections']),
            "filtered_detections": len(filtered_results),
            "ranked_detections": []
        }
        
        for detection_info, similarity_score in filtered_results:
            result_item = {
                "detection": detection_info,
                "similarity_score": similarity_score,
                "rank": len(final_results["ranked_detections"]) + 1
            }
            final_results["ranked_detections"].append(result_item)
            
            print(f"  排名 {result_item['rank']}: {detection_info['class_name']} "
                  f"(置信度: {detection_info['confidence']:.3f}, "
                  f"相似度: {similarity_score:.3f})")
        
        return final_results
    
    def process_batch(self, 
                     image_paths: List[str], 
                     text_descriptions: List[str],
                     **kwargs) -> List[Dict[str, Any]]:
        """
        批量处理图片
        
        Args:
            image_paths: 图片路径列表
            text_descriptions: 文本描述列表
            **kwargs: 其他参数
            
        Returns:
            处理结果列表
        """
        results = []
        for image_path, text_desc in zip(image_paths, text_descriptions):
            result = self.process_single_image(image_path, text_desc, **kwargs)
            results.append(result)
        return results
    
    def visualize_results(self, 
                         result: Dict[str, Any], 
                         output_path: str = None,
                         show_similarity: bool = True) -> np.ndarray:
        """
        可视化处理结果
        
        Args:
            result: 处理结果
            output_path: 输出路径
            show_similarity: 是否显示相似度分数
            
        Returns:
            可视化图片
        """
        image = cv2.imread(result["image_path"])
        
        for item in result["ranked_detections"]:
            detection = item["detection"]
            similarity = item["similarity_score"]
            rank = item["rank"]
            
            x1, y1, x2, y2 = detection["bbox"]
            
            # 根据排名选择颜色
            if rank == 1:
                color = (0, 255, 0)  # 绿色 - 最相关
            elif rank <= 3:
                color = (0, 255, 255)  # 黄色 - 较相关
            else:
                color = (0, 165, 255)  # 橙色 - 一般相关
            
            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 绘制标签
            if show_similarity:
                label = f"#{rank} {detection['class_name']}: {similarity:.3f}"
            else:
                label = f"#{rank} {detection['class_name']}"
                
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"可视化结果已保存到: {output_path}")
        
        return image
    
    def save_results(self, results: List[Dict[str, Any]], output_path: str):
        """保存结果到JSON文件"""
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"结果已保存到: {output_path}")

def main():
    parser = argparse.ArgumentParser(description="改进的YOLO-World系统")
    parser.add_argument("--image", type=str, required=True, help="输入图片路径")
    parser.add_argument("--text", type=str, required=True, help="文本描述")
    parser.add_argument("--model", type=str, default="yolov8s-world.pt", help="模型路径")
    parser.add_argument("--output", type=str, help="输出目录")
    parser.add_argument("--conf", type=float, default=0.1, help="YOLO置信度阈值")
    parser.add_argument("--sim", type=float, default=0.3, help="语义相似度阈值")
    parser.add_argument("--keyword-method", type=str, default="spacy", 
                       choices=["spacy", "nltk", "yake"], help="关键词提取方法")
    parser.add_argument("--similarity-method", type=str, default="clip",
                       choices=["clip", "sentence_transformer", "blip"], 
                       help="相似度计算方法")
    parser.add_argument("--device", type=str, default="auto", help="推理设备")
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.image):
        print(f"图片文件不存在: {args.image}")
        return
    
    if not os.path.exists(args.model):
        print(f"模型文件不存在: {args.model}")
        return
    
    # 创建输出目录
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    
    try:
        # 初始化系统
        system = ImprovedYOLOWorld(
            model_path=args.model,
            keyword_method=args.keyword_method,
            similarity_method=args.similarity_method,
            device=args.device
        )
        
        # 处理图片
        result = system.process_single_image(
            args.image, 
            args.text,
            conf_threshold=args.conf,
            similarity_threshold=args.sim
        )
        
        # 可视化和保存结果
        if args.output:
            # 保存JSON结果
            json_path = os.path.join(args.output, "results.json")
            system.save_results([result], json_path)
            
            # 保存可视化图片
            vis_path = os.path.join(args.output, "visualization.jpg")
            system.visualize_results(result, vis_path)
        else:
            # 只显示结果
            system.visualize_results(result)
            
    except Exception as e:
        print(f"处理失败: {e}")
        import traceback
        traceback.print_exc()

def demo():
    """演示功能"""
    print("=== 改进的YOLO-World系统演示 ===")
    
    # 检查模型文件
    model_path = "yolov8s-world.pt"
    if not os.path.exists(model_path):
        print(f"模型文件 {model_path} 不存在，请下载YOLO-World模型")
        return
    
    # 示例图片和文本
    example_cases = [
        {
            "text": "A black cat lying on a chair",
            "expected_keyword": "cat"
        },
        {
            "text": "A red car parked in the garage", 
            "expected_keyword": "car"
        },
        {
            "text": "Children playing with a ball in the park",
            "expected_keyword": "ball"  # 或 "children"
        }
    ]
    
    try:
        # 初始化系统（使用较轻量的配置用于演示）
        system = ImprovedYOLOWorld(
            model_path=model_path,
            keyword_method="spacy",  # 或 "nltk" 如果spacy未安装
            similarity_method="clip",
            device="cpu"  # 演示使用CPU
        )
        
        print("系统初始化成功!")
        print("演示关键词提取功能:")
        
        for i, case in enumerate(example_cases, 1):
            print(f"\n案例 {i}: {case['text']}")
            keywords = system.keyword_extractor.extract_keywords(case['text'])
            main_subject = system.keyword_extractor.extract_main_subject(case['text'])
            print(f"  关键词: {keywords}")
            print(f"  中心词: {main_subject}")
            print(f"  期望关键词: {case['expected_keyword']}")
            
    except Exception as e:
        print(f"演示失败: {e}")
        print("请确保安装了必要的依赖包")

if __name__ == "__main__":
    # 如果有命令行参数，运行主程序；否则运行演示
    import sys
    if len(sys.argv) > 1:
        main()
    else:
        demo()