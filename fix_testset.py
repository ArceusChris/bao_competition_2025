
import os
import shutil
from PIL import Image
import glob
from tqdm import tqdm

def process_dataset(source_base_dir, dest_base_dir):
    """
    Processes the dataset by converting images to JPG and padding the groundtruth file.

    Args:
        source_base_dir (str): The path to the source dataset directory (e.g., 'dataset/TestSet').
        dest_base_dir (str): The path to the destination directory for the fixed dataset (e.g., 'dataset/TestSet_fixed').
    """
    # 检查源目录是否存在
    if not os.path.isdir(source_base_dir):
        print(f"错误：源目录 '{source_base_dir}' 不存在。")
        return

    # 创建目标根目录
    os.makedirs(dest_base_dir, exist_ok=True)
    print(f"已创建或确认目标目录: '{dest_base_dir}'")

    # 查找所有符合 '0xx' 模式的序列文件夹
    sequence_dirs = sorted([d for d in glob.glob(os.path.join(source_base_dir, '0*')) if os.path.isdir(d)])
    if not sequence_dirs:
        print(f"警告：在 '{source_base_dir}' 中未找到符合 '0xx' 模式的序列文件夹。")
        return
        
    print(f"找到了 {len(sequence_dirs)} 个序列文件夹，开始处理...")

    # 使用 tqdm 显示处理进度
    for seq_dir in tqdm(sequence_dirs, desc="处理序列"):
        seq_name = os.path.basename(seq_dir)
        
        # 定义源和目标路径
        source_color_dir = os.path.join(seq_dir, 'color')
        source_gt_path = os.path.join(seq_dir, 'groundtruth.txt')
        
        dest_seq_dir = os.path.join(dest_base_dir, seq_name)
        dest_color_dir = os.path.join(dest_seq_dir, 'color')
        dest_gt_path = os.path.join(dest_seq_dir, 'groundtruth.txt')

        # 创建目标序列和color目录
        os.makedirs(dest_color_dir, exist_ok=True)

        # --- 1. 处理和转换图片 ---
        if not os.path.isdir(source_color_dir):
            tqdm.write(f"警告：在序列 '{seq_name}' 中找不到 'color' 目录，跳过该序列。")
            continue
            
        image_files = sorted(os.listdir(source_color_dir))
        num_images = len(image_files)

        for image_name in image_files:
            source_image_path = os.path.join(source_color_dir, image_name)
            # 去掉原扩展名，统一为 .jpg
            base_filename = os.path.splitext(image_name)[0]
            dest_image_path = os.path.join(dest_color_dir, f"{base_filename}.jpg")

            try:
                with Image.open(source_image_path) as img:
                    # 转换为RGB模式以丢弃alpha通道（适用于PNG等），然后保存为JPG
                    img.convert('RGB').save(dest_image_path, 'jpeg')
            except Exception as e:
                tqdm.write(f"错误：无法转换图片 '{source_image_path}'。错误信息: {e}")

        # --- 2. 补全 groundtruth.txt ---
        gt_lines = []
        if os.path.exists(source_gt_path):
            with open(source_gt_path, 'r') as f:
                gt_lines = f.readlines()

        num_gt_lines = len(gt_lines)
        
        # 如果标注行数少于图片数，则进行补全
        if num_gt_lines < num_images:
            lines_to_add = num_images - num_gt_lines
            padding = ['0,0,0,0\n'] * lines_to_add
            gt_lines.extend(padding)
        
        # 写入新的 groundtruth 文件
        with open(dest_gt_path, 'w') as f:
            f.writelines(gt_lines)

        # --- 3. 复制其他所有文件和文件夹 ---
        for item in os.listdir(seq_dir):
            source_item_path = os.path.join(seq_dir, item)
            dest_item_path = os.path.join(dest_seq_dir, item)

            # 跳过已经特殊处理的 color 目录和 groundtruth.txt
            if item == 'color' or item == 'groundtruth.txt':
                continue

            # 如果目标已存在，则跳过（防止重复复制或覆盖）
            if os.path.exists(dest_item_path):
                continue

            if os.path.isdir(source_item_path):
                # 递归复制整个目录
                shutil.copytree(source_item_path, dest_item_path)
            elif os.path.isfile(source_item_path):
                # 复制单个文件
                shutil.copy2(source_item_path, dest_item_path)

    print("\n处理完成！")

if __name__ == '__main__':
    # 定义源目录和目标目录
    source_directory = 'dataset/TestSet'
    destination_directory = 'dataset/TestSet_fixed'
    
    process_dataset(source_directory, destination_directory)
