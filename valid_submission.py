import os
import cv2
import numpy as np

# --- Configuration ---
submission_dir = 'Submission'
dataset_dir = 'dataset/TestSet_fixed'
output_dir = 'valid/TestSet_fixed_annotated'

# --- Main Script ---

print("Starting annotation process...")

# 确保根输出目录存在
os.makedirs(output_dir, exist_ok=True)

# 遍历数据集中的每个序列文件夹 (e.g., '001', '002', ...)
for seq_name in sorted(os.listdir(dataset_dir)):
    seq_path = os.path.join(dataset_dir, seq_name)
    
    # 跳过非目录文件，例如 list.txt
    if not os.path.isdir(seq_path):
        continue

    print(f"Processing sequence: {seq_name}...")

    # 构建对应的标注文件路径
    submission_file_path = os.path.join(submission_dir, seq_name, f"{seq_name}.txt")

    # 检查标注文件是否存在
    if not os.path.exists(submission_file_path):
        print(f"  - Warning: Annotation file not found at '{submission_file_path}'. Skipping sequence.")
        continue

    # 读取所有行的标注信息
    with open(submission_file_path, 'r') as f:
        annotations = [line.strip() for line in f.readlines()]

    # 遍历序列中的 'color' 和 'depth' 文件夹
    for modality in ['color', 'depth']:
        modality_path = os.path.join(seq_path, modality)
        if not os.path.isdir(modality_path):
            continue

        # 获取所有图片文件并排序，以确保与标注行对应
        image_files = sorted([f for f in os.listdir(modality_path) if f.endswith(('.jpg', '.png'))])

        # 检查图片数量和标注数量是否匹配
        if len(image_files) != len(annotations):
            print(f"  - Warning: Mismatch in sequence {seq_name}/{modality}. Found {len(image_files)} images but {len(annotations)} annotations. Skipping.")
            continue
            
        # 创建保存标注后图片的输出目录
        output_seq_modality_dir = os.path.join(output_dir, seq_name, modality)
        os.makedirs(output_seq_modality_dir, exist_ok=True)

        # 遍历每张图片进行标注
        for i, image_name in enumerate(image_files):
            # 读取标注框
            try:
                coords_str = annotations[i].split(',')
                # 将坐标转换为整数
                x, y, w, h = map(int, map(float, coords_str))
            except (ValueError, IndexError) as e:
                print(f"  - Error parsing annotation for {image_name}: {annotations[i]}. Error: {e}. Skipping frame.")
                continue

            # 读取图片
            image_path = os.path.join(modality_path, image_name)
            image = cv2.imread(image_path)
            
            if image is None:
                print(f"  - Error: Could not read image {image_path}. Skipping.")
                continue

            # 在图片上绘制矩形框
            # (x, y) 是左上角点, (x+w, y+h) 是右下角点
            # (0, 255, 0) 是绿色, 2 是线条粗细
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # 保存标注后的图片
            output_image_path = os.path.join(output_seq_modality_dir, image_name)
            cv2.imwrite(output_image_path, image)

print("Annotation process finished.")
print(f"Annotated files are saved in: {os.path.abspath(output_dir)}")