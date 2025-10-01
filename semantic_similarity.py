"""
语义相关度计算模块
用于计算检测框内容与原始文本的语义相关度
"""

import torch
import numpy as np
from transformers import CLIPProcessor, CLIPModel, pipeline
from sentence_transformers import SentenceTransformer
import cv2
from typing import List, Tuple, Dict, Any
from PIL import Image
import torch.nn.functional as F

class SemanticSimilarityCalculator:
    def __init__(self, method: str = "clip", device: str = "auto"):
        """
        初始化语义相关度计算器
        
        Args:
            method: 计算方法 ("clip", "sentence_transformer", "blip")
            device: 设备 ("cpu", "cuda", "auto")
        """
        self.method = method
        self.device = self._get_device(device)
        self.setup_model()
    
    def _get_device(self, device: str) -> str:
        """获取设备"""
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device
    
    def setup_model(self):
        """设置模型"""
        if self.method == "clip":
            self.model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.model.to(self.device)
            
        elif self.method == "sentence_transformer":
            self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
            # 对于图像，我们仍然使用CLIP
            self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_model.to(self.device)
            
        elif self.method == "blip":
            # 使用BLIP进行图像描述，然后计算文本相似度
            self.captioner = pipeline("image-to-text", 
                                     model="Salesforce/blip-image-captioning-base",
                                     device=0 if self.device == "cuda" else -1)
            self.text_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        print(f"语义相关度计算器初始化完成，使用方法: {self.method}")
    
    def calculate_clip_similarity(self, image: np.ndarray, text: str) -> float:
        """
        使用CLIP计算图像-文本相似度
        
        Args:
            image: 图像数组 (BGR格式)
            text: 文本
            
        Returns:
            相似度分数 (0-1)
        """
        # 转换BGR到RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # 处理输入
        inputs = self.processor(
            text=[text], 
            images=pil_image, 
            return_tensors="pt", 
            padding=True
        )
        
        # 移动到设备
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # 计算特征
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits_per_image = outputs.logits_per_image
            similarity = torch.sigmoid(logits_per_image).cpu().item()
        
        return similarity
    
    def calculate_sentence_transformer_similarity(self, image: np.ndarray, text: str) -> float:
        """
        使用Sentence Transformer计算相似度
        先用CLIP为图像生成描述，然后计算文本相似度
        
        Args:
            image: 图像数组
            text: 文本
            
        Returns:
            相似度分数 (0-1)
        """
        # 使用CLIP生成图像描述
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # 生成多个可能的描述
        candidate_descriptions = [
            "a photo of an object",
            "an image showing something",
            "a picture of an item",
            "a visual representation"
        ]
        
        inputs = self.clip_processor(
            text=candidate_descriptions,
            images=pil_image,
            return_tensors="pt",
            padding=True
        )
        
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.clip_model(**inputs)
            logits_per_image = outputs.logits_per_image
            probs = F.softmax(logits_per_image, dim=-1)
            best_desc_idx = torch.argmax(probs).item()
            image_description = candidate_descriptions[best_desc_idx]
        
        # 计算文本相似度
        embeddings1 = self.text_model.encode([image_description])
        embeddings2 = self.text_model.encode([text])
        
        similarity = np.dot(embeddings1[0], embeddings2[0]) / (
            np.linalg.norm(embeddings1[0]) * np.linalg.norm(embeddings2[0])
        )
        
        return max(0, similarity)  # 确保非负
    
    def calculate_blip_similarity(self, image: np.ndarray, text: str) -> float:
        """
        使用BLIP生成图像描述，然后计算文本相似度
        
        Args:
            image: 图像数组
            text: 文本
            
        Returns:
            相似度分数 (0-1)
        """
        # 转换格式
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        
        # 生成图像描述
        caption_result = self.captioner(pil_image)
        image_caption = caption_result[0]['generated_text']
        
        # 计算文本相似度
        embeddings1 = self.text_model.encode([image_caption])
        embeddings2 = self.text_model.encode([text])
        
        similarity = np.dot(embeddings1[0], embeddings2[0]) / (
            np.linalg.norm(embeddings1[0]) * np.linalg.norm(embeddings2[0])
        )
        
        return max(0, similarity)
    
    def calculate_similarity(self, image: np.ndarray, text: str) -> float:
        """
        计算图像与文本的语义相似度
        
        Args:
            image: 图像数组
            text: 文本
            
        Returns:
            相似度分数 (0-1)
        """
        if self.method == "clip":
            return self.calculate_clip_similarity(image, text)
        elif self.method == "sentence_transformer":
            return self.calculate_sentence_transformer_similarity(image, text)
        elif self.method == "blip":
            return self.calculate_blip_similarity(image, text)
        else:
            raise ValueError(f"不支持的方法: {self.method}")
    
    def calculate_batch_similarity(self, images: List[np.ndarray], 
                                 texts: List[str]) -> List[float]:
        """
        批量计算相似度
        
        Args:
            images: 图像列表
            texts: 文本列表
            
        Returns:
            相似度分数列表
        """
        similarities = []
        for image, text in zip(images, texts):
            similarity = self.calculate_similarity(image, text)
            similarities.append(similarity)
        return similarities
    
    def rank_detections_by_similarity(self, 
                                    cropped_images: List[np.ndarray],
                                    original_text: str,
                                    detection_info: List[Dict]) -> List[Tuple[Dict, float]]:
        """
        根据语义相似度对检测结果进行排序
        
        Args:
            cropped_images: 裁剪的图像区域列表
            original_text: 原始文本
            detection_info: 检测信息列表
            
        Returns:
            (检测信息, 相似度分数) 的排序列表
        """
        similarities = []
        for image in cropped_images:
            similarity = self.calculate_similarity(image, original_text)
            similarities.append(similarity)
        
        # 组合并排序
        results = list(zip(detection_info, similarities))
        results.sort(key=lambda x: x[1], reverse=True)  # 按相似度降序排序
        
        return results
    
    def filter_by_similarity_threshold(self, 
                                     ranked_results: List[Tuple[Dict, float]],
                                     threshold: float = 0.3) -> List[Tuple[Dict, float]]:
        """
        根据相似度阈值过滤结果
        
        Args:
            ranked_results: 排序后的结果
            threshold: 相似度阈值
            
        Returns:
            过滤后的结果
        """
        return [result for result in ranked_results if result[1] >= threshold]

class MultiModalSimilarityCalculator:
    """多模态相似度计算器，结合多种方法"""
    
    def __init__(self, methods: List[str] = ["clip"], weights: List[float] = None):
        """
        初始化多模态相似度计算器
        
        Args:
            methods: 使用的方法列表
            weights: 各方法的权重
        """
        self.methods = methods
        self.weights = weights or [1.0 / len(methods)] * len(methods)
        self.calculators = {}
        
        for method in methods:
            self.calculators[method] = SemanticSimilarityCalculator(method)
    
    def calculate_weighted_similarity(self, image: np.ndarray, text: str) -> float:
        """
        计算加权相似度
        
        Args:
            image: 图像数组
            text: 文本
            
        Returns:
            加权相似度分数
        """
        total_similarity = 0.0
        
        for method, weight in zip(self.methods, self.weights):
            similarity = self.calculators[method].calculate_similarity(image, text)
            total_similarity += similarity * weight
        
        return total_similarity

def demo():
    """演示功能"""
    print("=== 语义相关度计算演示 ===")
    
    # 创建一个测试图像（纯色图像）
    test_image = np.ones((100, 100, 3), dtype=np.uint8) * 128  # 灰色图像
    test_text = "A gray square"
    
    try:
        calculator = SemanticSimilarityCalculator("clip")
        similarity = calculator.calculate_similarity(test_image, test_text)
        print(f"CLIP相似度: {similarity:.4f}")
        
    except Exception as e:
        print(f"演示失败，可能需要安装相关依赖: {e}")
        print("请安装: pip install transformers torch torchvision sentence-transformers pillow")

if __name__ == "__main__":
    demo()