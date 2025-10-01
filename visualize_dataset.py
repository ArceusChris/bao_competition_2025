#!/usr/bin/env python3
"""
YOLO数据集可视化工具
用于检查和可视化转换后的YOLO数据集
"""

import os
import sys
import json
import random
from pathlib import Path
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import yaml

def load_dataset_config(dataset_path):
    """加载数据集配置"""
    dataset_path = Path(dataset_path)
    
    # 加载dataset.yaml
    yaml_path = dataset_path / "dataset.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"数据集配置文件不存在: {yaml_path}")
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        dataset_config = yaml.safe_load(f)
    
    # 加载class_mapping.json
    mapping_path = dataset_path / "class_mapping.json"
    if mapping_path.exists():
        with open(mapping_path, 'r', encoding='utf-8') as f:
            class_mapping = json.load(f)
    else:
        class_mapping = None
    
    return dataset_config, class_mapping

def yolo_to_bbox(yolo_coords, img_width, img_height):
    """将YOLO格式坐标转换为绝对坐标边界框"""
    x_center, y_center, width, height = yolo_coords
    
    # 转换为绝对坐标
    x_center_abs = x_center * img_width
    y_center_abs = y_center * img_height
    width_abs = width * img_width
    height_abs = height * img_height
    
    # 计算边界框左上角和右下角坐标
    x_min = int(x_center_abs - width_abs / 2)
    y_min = int(y_center_abs - height_abs / 2)
    x_max = int(x_center_abs + width_abs / 2)
    y_max = int(y_center_abs + height_abs / 2)
    
    return x_min, y_min, x_max, y_max

def visualize_sample(dataset_path, split="train", num_samples=6):
    """可视化数据集样本"""
    dataset_path = Path(dataset_path)
    
    # 加载配置
    dataset_config, class_mapping = load_dataset_config(dataset_path)
    
    # 获取类别名称映射
    id_to_name = dataset_config['names']
    
    # 获取图片和标签目录
    img_dir = dataset_path / "images" / split
    label_dir = dataset_path / "labels" / split
    
    if not img_dir.exists() or not label_dir.exists():
        raise FileNotFoundError(f"数据集目录不存在: {img_dir} 或 {label_dir}")
    
    # 获取所有图片文件
    img_files = list(img_dir.glob("*.jpg"))
    if not img_files:
        raise FileNotFoundError(f"在 {img_dir} 中没有找到图片文件")
    
    # 随机选择样本
    sample_files = random.sample(img_files, min(num_samples, len(img_files)))
    
    # 创建子图
    cols = 3
    rows = (len(sample_files) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(15, 5 * rows))
    if rows == 1:
        axes = [axes] if cols == 1 else axes
    else:
        axes = axes.flatten()
    
    # 生成颜色映射
    num_classes = len(id_to_name)
    colors = plt.cm.Set3(np.linspace(0, 1, num_classes))
    
    for i, img_file in enumerate(sample_files):
        # 读取图片
        img = cv2.imread(str(img_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_height, img_width = img.shape[:2]
        
        # 读取对应的标签文件
        label_file = label_dir / f"{img_file.stem}.txt"
        
        ax = axes[i]
        ax.imshow(img)
        ax.set_title(f"{img_file.name}\n({img_width}x{img_height})", fontsize=10)
        ax.axis('off')
        
        if label_file.exists():
            with open(label_file, 'r') as f:
                lines = f.read().strip().split('\n')
            
            for line in lines:
                if line.strip():
                    parts = line.strip().split()
                    class_id = int(parts[0])
                    yolo_coords = [float(x) for x in parts[1:5]]
                    
                    # 转换坐标
                    x_min, y_min, x_max, y_max = yolo_to_bbox(yolo_coords, img_width, img_height)
                    
                    # 绘制边界框
                    rect = Rectangle(
                        (x_min, y_min), x_max - x_min, y_max - y_min,
                        linewidth=2, edgecolor=colors[class_id % len(colors)], 
                        facecolor='none'
                    )
                    ax.add_patch(rect)
                    
                    # 添加类别标签
                    class_name = id_to_name.get(class_id, f"class_{class_id}")
                    ax.text(
                        x_min, y_min - 5, f"{class_name} ({class_id})",
                        fontsize=8, color=colors[class_id % len(colors)],
                        weight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.7)
                    )
    
    # 隐藏多余的子图
    for i in range(len(sample_files), len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.suptitle(f"YOLO数据集样本可视化 ({split.upper()})", fontsize=16, y=1.02)
    
    # 保存图片
    output_path = dataset_path / f"visualization_{split}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✅ 可视化结果保存到: {output_path}")
    
    plt.show()

def analyze_dataset(dataset_path):
    """分析数据集统计信息"""
    dataset_path = Path(dataset_path)
    
    print("📊 数据集分析报告")
    print("=" * 50)
    
    # 加载配置
    dataset_config, class_mapping = load_dataset_config(dataset_path)
    
    # 基本信息
    print(f"数据集路径: {dataset_path.absolute()}")
    print(f"类别数量: {len(dataset_config['names'])}")
    
    if class_mapping:
        print(f"总图片数量: {class_mapping['total_images']}")
    
    # 分析各个分割的数据量
    for split in ['train', 'val']:
        img_dir = dataset_path / "images" / split
        label_dir = dataset_path / "labels" / split
        
        if img_dir.exists() and label_dir.exists():
            img_files = list(img_dir.glob("*.jpg"))
            label_files = list(label_dir.glob("*.txt"))
            
            print(f"\n{split.upper()} 集:")
            print(f"  图片数量: {len(img_files)}")
            print(f"  标签数量: {len(label_files)}")
            
            if len(img_files) != len(label_files):
                print(f"  ⚠️  图片和标签数量不匹配!")
    
    # 类别分布
    print(f"\n🏷️  类别列表:")
    id_to_name = dataset_config['names']
    for class_id in sorted(id_to_name.keys()):
        class_name = id_to_name[class_id]
        if class_mapping and 'class_counts' in class_mapping:
            count = class_mapping['class_counts'].get(class_name, 0)
            print(f"  {class_id:2d}. {class_name:<15} : {count:4d} 样本")
        else:
            print(f"  {class_id:2d}. {class_name}")
    
    # 检查文件完整性
    print(f"\n🔍 文件完整性检查:")
    issues = []
    
    for split in ['train', 'val']:
        img_dir = dataset_path / "images" / split
        label_dir = dataset_path / "labels" / split
        
        if img_dir.exists():
            img_files = list(img_dir.glob("*.jpg"))
            for img_file in img_files[:10]:  # 只检查前10个文件
                label_file = label_dir / f"{img_file.stem}.txt"
                if not label_file.exists():
                    issues.append(f"缺少标签文件: {label_file}")
    
    if issues:
        print(f"  发现 {len(issues)} 个问题:")
        for issue in issues[:5]:  # 只显示前5个问题
            print(f"    - {issue}")
        if len(issues) > 5:
            print(f"    ... 还有 {len(issues) - 5} 个问题")
    else:
        print(f"  ✅ 未发现明显问题")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="YOLO数据集可视化和分析工具",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "dataset_path",
        type=str,
        help="YOLO数据集路径"
    )
    
    parser.add_argument(
        "--action", "-a",
        choices=["analyze", "visualize", "both"],
        default="both",
        help="执行的操作"
    )
    
    parser.add_argument(
        "--split", "-s",
        choices=["train", "val"],
        default="train",
        help="可视化的数据分割"
    )
    
    parser.add_argument(
        "--samples", "-n",
        type=int,
        default=6,
        help="可视化的样本数量"
    )
    
    args = parser.parse_args()
    
    if not Path(args.dataset_path).exists():
        print(f"❌ 数据集路径不存在: {args.dataset_path}")
        sys.exit(1)
    
    print("🔍 YOLO数据集分析工具")
    print("=" * 50)
    
    try:
        if args.action in ["analyze", "both"]:
            analyze_dataset(args.dataset_path)
        
        if args.action in ["visualize", "both"]:
            print(f"\n📸 可视化 {args.split} 集样本...")
            visualize_sample(args.dataset_path, args.split, args.samples)
            
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main()