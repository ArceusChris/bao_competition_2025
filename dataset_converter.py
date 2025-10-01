"""
数据集转换器：将TrainSet转换为YOLO格式的数据集

该模块能够：
1. 读取TrainSet中的图片、标注和描述
2. 使用关键词提取器从描述中提取类别标签
3. 将边界框格式从绝对坐标转换为YOLO格式的相对坐标
4. 生成YOLO格式的数据集结构和配置文件
"""

import os
import json
import shutil
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import cv2
import yaml
from keyword_extractor import KeywordExtractor

class DatasetConverter:
    """数据集转换器类"""
    
    def __init__(self, source_dir: str, output_dir: str = "yolo_dataset"):
        """
        初始化数据集转换器
        
        Args:
            source_dir (str): 源数据集目录路径 (TrainSet)
            output_dir (str): 输出YOLO数据集目录路径
        """
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        
        # 初始化关键词提取器
        self.keyword_extractor = KeywordExtractor('transformer')
        
        # 类别映射和统计
        self.class_mapping = {}  # {class_name: class_id}
        self.class_counts = {}   # 统计每个类别的样本数量
        self.total_images = 0
        self.failed_conversions = []
        
        # 创建输出目录结构
        self._setup_output_dirs()
    
    def _setup_output_dirs(self):
        """创建YOLO数据集目录结构"""
        directories = [
            self.output_dir,
            self.output_dir / "images" / "train",
            self.output_dir / "images" / "val", 
            self.output_dir / "labels" / "train",
            self.output_dir / "labels" / "val"
        ]
        
        for dir_path in directories:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        print(f"创建YOLO数据集目录结构: {self.output_dir}")
    
    def _extract_class_from_description(self, description: str, category_hint: str = None) -> str:
        """
        从描述中提取类别名称
        
        Args:
            description (str): 图片描述文本
            category_hint (str): 类别提示（来自目录名）
            
        Returns:
            str: 提取的类别名称
        """
        # 使用关键词提取器获取主要对象
        main_subject = self.keyword_extractor.extract_main_subject(description)
        
        if not main_subject:
            # 如果提取失败，使用目录名作为后备
            if category_hint:
                return category_hint.lower()
            else:
                return "unknown"
        
        # 对一些特殊情况进行映射
        class_mapping = {
            "child": "person",
            "man": "person", 
            "woman": "person",
            "human": "person",
            "people": "person",
            "girl": "person",
            "boy": "person",
            "student": "person",
            "pigeon": "bird",
            "cell_phone": "phone",
            "hard_disk": "disk",
            "ipad": "tablet"
        }
        
        return class_mapping.get(main_subject, main_subject)
    
    def _convert_bbox_format(self, bbox: List[int], img_width: int, img_height: int) -> List[float]:
        """
        将绝对坐标边界框转换为YOLO格式的相对坐标
        
        Args:
            bbox (List[int]): [x_min, y_min, x_max, y_max] 绝对坐标
            img_width (int): 图片宽度
            img_height (int): 图片高度
            
        Returns:
            List[float]: [x_center, y_center, width, height] 相对坐标 (0-1)
        """
        x_min, y_min, x_max, y_max = bbox
        
        # 计算中心点和宽高
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min
        
        # 转换为相对坐标
        x_center_rel = x_center / img_width
        y_center_rel = y_center / img_height
        width_rel = width / img_width
        height_rel = height / img_height
        
        # 确保坐标在合理范围内
        x_center_rel = max(0, min(1, x_center_rel))
        y_center_rel = max(0, min(1, y_center_rel))
        width_rel = max(0, min(1, width_rel))
        height_rel = max(0, min(1, height_rel))
        
        return [x_center_rel, y_center_rel, width_rel, height_rel]
    
    def _get_class_id(self, class_name: str) -> int:
        """获取或创建类别ID"""
        if class_name not in self.class_mapping:
            self.class_mapping[class_name] = len(self.class_mapping)
            self.class_counts[class_name] = 0
        
        self.class_counts[class_name] += 1
        return self.class_mapping[class_name]
    
    def _process_instance(self, category_path: Path, instance_name: str, split: str = "train") -> bool:
        """
        处理单个实例的所有图片和标注
        
        Args:
            category_path (Path): 类别目录路径
            instance_name (str): 实例名称
            split (str): 数据分割 ("train" 或 "val")
            
        Returns:
            bool: 是否处理成功
        """
        instance_path = category_path / instance_name
        
        # 检查必要文件是否存在
        color_dir = instance_path / "color"
        nlp_file = instance_path / "nlp.txt"
        groundtruth_file = instance_path / "groundtruth_rect.txt"
        
        if not all(path.exists() for path in [color_dir, nlp_file, groundtruth_file]):
            print(f"⚠️  跳过 {instance_path}: 缺少必要文件")
            return False
        
        try:
            # 读取描述文本
            with open(nlp_file, 'r', encoding='utf-8') as f:
                description = f.read().strip()
            
            # 读取边界框标注
            with open(groundtruth_file, 'r') as f:
                bbox_lines = f.read().strip().split('\n')
            
            # 从描述中提取类别
            category_hint = category_path.name.lower()
            class_name = self._extract_class_from_description(description, category_hint)
            class_id = self._get_class_id(class_name)
            
            # 获取图片文件列表
            image_files = sorted(color_dir.glob("*.jpg"))
            
            if len(image_files) != len(bbox_lines):
                print(f"⚠️  {instance_path}: 图片数量({len(image_files)})与标注数量({len(bbox_lines)})不匹配")
                # 取最小值继续处理
                min_count = min(len(image_files), len(bbox_lines))
                image_files = image_files[:min_count]
                bbox_lines = bbox_lines[:min_count]
            
            # 处理每张图片
            success_count = 0
            for img_file, bbox_line in zip(image_files, bbox_lines):
                if self._process_single_image(img_file, bbox_line, class_id, instance_name, split):
                    success_count += 1
            
            print(f"✅ {instance_path.name}: 成功处理 {success_count}/{len(image_files)} 张图片 (类别: {class_name})")
            return success_count > 0
            
        except Exception as e:
            error_msg = f"处理实例 {instance_path} 时出错: {e}"
            print(f"❌ {error_msg}")
            self.failed_conversions.append(error_msg)
            return False
    
    def _process_single_image(self, img_file: Path, bbox_line: str, 
                            class_id: int, instance_name: str, split: str) -> bool:
        """
        处理单张图片和对应的标注
        
        Args:
            img_file (Path): 图片文件路径
            bbox_line (str): 边界框标注行
            class_id (int): 类别ID
            instance_name (str): 实例名称
            split (str): 数据分割
            
        Returns:
            bool: 是否处理成功
        """
        try:
            # 解析边界框坐标
            coords = bbox_line.strip().split(',')
            if len(coords) != 4:
                print(f"⚠️  边界框格式错误: {bbox_line}")
                return False
            
            bbox = [int(coord) for coord in coords]
            
            # 读取图片获取尺寸
            img = cv2.imread(str(img_file))
            if img is None:
                print(f"⚠️  无法读取图片: {img_file}")
                return False
            
            img_height, img_width = img.shape[:2]
            
            # 转换边界框格式
            yolo_bbox = self._convert_bbox_format(bbox, img_width, img_height)
            
            # 生成新的文件名
            new_filename = f"{instance_name}_{img_file.stem}"
            
            # 复制图片到目标目录
            target_img_path = self.output_dir / "images" / split / f"{new_filename}.jpg"
            shutil.copy2(img_file, target_img_path)
            
            # 创建YOLO格式的标注文件
            target_label_path = self.output_dir / "labels" / split / f"{new_filename}.txt"
            with open(target_label_path, 'w') as f:
                # YOLO格式: class_id x_center y_center width height
                f.write(f"{class_id} {' '.join(map(str, yolo_bbox))}\n")
            
            self.total_images += 1
            return True
            
        except Exception as e:
            print(f"⚠️  处理图片 {img_file} 失败: {e}")
            return False
    
    def convert_dataset(self, train_ratio: float = 0.8, max_instances_per_category: int = None):
        """
        转换整个数据集
        
        Args:
            train_ratio (float): 训练集比例
            max_instances_per_category (int): 每个类别最大实例数（用于限制数据量）
        """
        print("🚀 开始转换数据集...")
        print("-" * 50)
        
        # 遍历所有类别
        category_paths = [p for p in self.source_dir.iterdir() if p.is_dir()]
        category_paths.sort()
        
        for category_path in category_paths:
            category_name = category_path.name
            print(f"\n📁 处理类别: {category_name}")
            
            # 获取该类别下的所有实例
            instance_paths = [p for p in category_path.iterdir() if p.is_dir()]
            instance_paths.sort()
            
            # 限制实例数量（如果指定）
            if max_instances_per_category:
                instance_paths = instance_paths[:max_instances_per_category]
            
            # 按比例分割训练集和验证集
            num_train = int(len(instance_paths) * train_ratio)
            train_instances = instance_paths[:num_train]
            val_instances = instance_paths[num_train:]
            
            print(f"   实例总数: {len(instance_paths)} (训练: {len(train_instances)}, 验证: {len(val_instances)})")
            
            # 处理训练集实例  
            for instance_path in train_instances:
                self._process_instance(category_path, instance_path.name, "train")
            
            # 处理验证集实例
            for instance_path in val_instances:
                self._process_instance(category_path, instance_path.name, "val")
        
        # 生成配置文件
        self._generate_config_files()
        
        # 打印统计信息
        self._print_statistics()
    
    def _generate_config_files(self):
        """生成YOLO数据集配置文件"""
        
        # 生成dataset.yaml文件
        dataset_config = {
            'path': str(self.output_dir.absolute()),
            'train': 'images/train',
            'val': 'images/val',
            'names': {v: k for k, v in self.class_mapping.items()}
        }
        
        yaml_path = self.output_dir / "dataset.yaml"
        with open(yaml_path, 'w', encoding='utf-8') as f:
            yaml.dump(dataset_config, f, default_flow_style=False, allow_unicode=True)
        
        # 生成class_mapping.json文件  
        mapping_path = self.output_dir / "class_mapping.json"
        with open(mapping_path, 'w', encoding='utf-8') as f:
            json.dump({
                'class_to_id': self.class_mapping,
                'id_to_class': {v: k for k, v in self.class_mapping.items()},
                'class_counts': self.class_counts,
                'total_classes': len(self.class_mapping),
                'total_images': self.total_images
            }, f, indent=2, ensure_ascii=False)
        
        print(f"\n📝 生成配置文件:")
        print(f"   - {yaml_path}")
        print(f"   - {mapping_path}")
    
    def _print_statistics(self):
        """打印转换统计信息"""
        print("\n" + "="*60)
        print("📊 数据集转换完成统计")
        print("="*60)
        print(f"总图片数量: {self.total_images}")
        print(f"类别数量: {len(self.class_mapping)}")
        print(f"失败转换: {len(self.failed_conversions)}")
        
        print(f"\n🏷️  类别分布:")
        sorted_classes = sorted(self.class_counts.items(), key=lambda x: x[1], reverse=True)
        for class_name, count in sorted_classes:
            class_id = self.class_mapping[class_name]
            print(f"   {class_id:2d}. {class_name:<15} : {count:4d} 样本")
        
        if self.failed_conversions:
            print(f"\n❌ 失败的转换:")
            for error in self.failed_conversions[:10]:  # 只显示前10个错误
                print(f"   - {error}")
            if len(self.failed_conversions) > 10:
                print(f"   ... 还有 {len(self.failed_conversions) - 10} 个错误")
        
        print(f"\n✅ 数据集保存在: {self.output_dir.absolute()}")


def main():
    """主函数：演示数据集转换"""
    
    # 配置参数
    source_dir = "TrainSet"
    output_dir = "yolo_dataset"
    train_ratio = 0.8  # 80% 训练，20% 验证
    max_instances_per_category = 100  # 限制每个类别最多100个实例用于快速测试
    
    print("🔄 TrainSet 到 YOLO 数据集转换器")
    print("=" * 50)
    
    # 检查源目录是否存在
    if not os.path.exists(source_dir):
        print(f"❌ 源目录不存在: {source_dir}")
        return
    
    # 创建转换器实例
    converter = DatasetConverter(source_dir, output_dir)
    
    # 执行转换
    try:
        converter.convert_dataset(
            train_ratio=train_ratio,
            max_instances_per_category=max_instances_per_category
        )
        
        print("\n🎉 数据集转换完成！")
        print(f"📁 输出目录: {output_dir}")
        print("📚 使用方法:")
        print("   from ultralytics import YOLO")
        print(f"   model = YOLO('yolov8s.pt')")
        print(f"   model.train(data='{output_dir}/dataset.yaml', epochs=10)")
        
    except Exception as e:
        print(f"❌ 转换过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()