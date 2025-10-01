"""
YOLO-World推理模块
用于使用YOLO-World模型进行物体检测
"""

import torch
import cv2
import numpy as np
from ultralytics import YOLOWorld
from typing import List, Tuple, Dict, Any
import os

class YOLOWorldInference:
    def __init__(self, model_path: str = "yolov8s-world.pt", device: str = "auto"):
        """
        初始化YOLO-World推理器
        
        Args:
            model_path: 模型文件路径
            device: 设备 ("cpu", "cuda", "auto")
        """
        self.model_path = model_path
        self.device = self._get_device(device)
        self.model = None
        self.load_model()
    
    def _get_device(self, device: str) -> str:
        """获取推理设备"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def load_model(self):
        """加载YOLO-World模型"""
        try:
            self.model = YOLOWorld(self.model_path)
            if self.device == "cuda":
                self.model.to("cuda")
            print(f"成功加载模型: {self.model_path} 到设备: {self.device}")
        except Exception as e:
            print(f"模型加载失败: {e}")
            raise
    
    def set_classes(self, class_names: List[str]):
        """
        设置检测类别
        
        Args:
            class_names: 类别名称列表
        """
        self.model.set_classes(class_names)
        print(f"设置检测类别: {class_names}")
    
    def predict(self, image_path: str, keywords: List[str], 
                conf_threshold: float = 0.1, 
                iou_threshold: float = 0.5) -> Dict[str, Any]:
        """
        进行预测
        
        Args:
            image_path: 图片路径
            keywords: 关键词列表
            conf_threshold: 置信度阈值
            iou_threshold: IoU阈值
            
        Returns:
            检测结果字典
        """
        # 设置检测类别
        self.set_classes(keywords)
        
        # 进行预测
        results = self.model(image_path, conf=conf_threshold, iou=iou_threshold)
        
        # 解析结果
        return self._parse_results(results[0], image_path)
    
    def _parse_results(self, result, image_path: str) -> Dict[str, Any]:
        """
        解析YOLO输出结果
        
        Args:
            result: YOLO输出结果
            image_path: 图片路径
            
        Returns:
            解析后的结果字典
        """
        # 读取原图
        image = cv2.imread(image_path)
        image_height, image_width = image.shape[:2]
        
        detection_results = {
            'image_path': image_path,
            'image_shape': (image_height, image_width),
            'detections': []
        }
        
        if result.boxes is not None:
            boxes = result.boxes.xyxy.cpu().numpy()  # x1, y1, x2, y2
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            
            for i in range(len(boxes)):
                x1, y1, x2, y2 = boxes[i]
                confidence = confidences[i]
                class_id = class_ids[i]
                class_name = result.names[class_id]
                
                detection = {
                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(confidence),
                    'class_id': class_id,
                    'class_name': class_name,
                    'bbox_area': (x2 - x1) * (y2 - y1)
                }
                
                detection_results['detections'].append(detection)
        
        return detection_results
    
    def crop_detection_regions(self, image_path: str, 
                              detection_results: Dict[str, Any]) -> List[np.ndarray]:
        """
        裁剪检测区域
        
        Args:
            image_path: 图片路径
            detection_results: 检测结果
            
        Returns:
            裁剪的图片区域列表
        """
        image = cv2.imread(image_path)
        cropped_regions = []
        
        for detection in detection_results['detections']:
            x1, y1, x2, y2 = detection['bbox']
            cropped_region = image[y1:y2, x1:x2]
            cropped_regions.append(cropped_region)
        
        return cropped_regions
    
    def visualize_results(self, image_path: str, 
                         detection_results: Dict[str, Any], 
                         output_path: str = None) -> np.ndarray:
        """
        可视化检测结果
        
        Args:
            image_path: 图片路径
            detection_results: 检测结果
            output_path: 输出路径（可选）
            
        Returns:
            可视化后的图片
        """
        image = cv2.imread(image_path)
        
        for detection in detection_results['detections']:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 绘制标签
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), (0, 255, 0), -1)
            cv2.putText(image, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        
        if output_path:
            cv2.imwrite(output_path, image)
        
        return image
    
    def batch_predict(self, image_paths: List[str], 
                     keywords: List[str],
                     conf_threshold: float = 0.1) -> List[Dict[str, Any]]:
        """
        批量预测
        
        Args:
            image_paths: 图片路径列表
            keywords: 关键词列表
            conf_threshold: 置信度阈值
            
        Returns:
            批量检测结果
        """
        results = []
        for image_path in image_paths:
            result = self.predict(image_path, keywords, conf_threshold)
            results.append(result)
        return results

def demo():
    """演示功能"""
    # 初始化推理器
    model_path = "yolov8s-world.pt"
    if not os.path.exists(model_path):
        print(f"模型文件 {model_path} 不存在，请确保模型文件在当前目录")
        return
    
    inferencer = YOLOWorldInference(model_path)
    
    # 测试图片路径（需要替换为实际图片路径）
    test_image = "test_image.jpg"
    if not os.path.exists(test_image):
        print(f"测试图片 {test_image} 不存在")
        return
    
    # 测试关键词
    keywords = ["cat", "chair", "person"]
    
    print("=== YOLO-World推理演示 ===")
    print(f"检测关键词: {keywords}")
    
    # 进行预测
    results = inferencer.predict(test_image, keywords)
    
    print(f"检测到 {len(results['detections'])} 个物体:")
    for i, detection in enumerate(results['detections']):
        print(f"  {i+1}. {detection['class_name']}: {detection['confidence']:.2f}")
        print(f"     边界框: {detection['bbox']}")
    
    # 可视化结果
    vis_image = inferencer.visualize_results(test_image, results, "output_vis.jpg")
    print("可视化结果已保存到: output_vis.jpg")

if __name__ == "__main__":
    demo()