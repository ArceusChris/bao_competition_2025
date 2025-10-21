'''
Submission
|---001
|    |---001.txt
|---002
|    |---002.txt
|---...
上为测评系统要求的提交格式
'''

result_path = 'test/tracking_results/spt/unimod1k/rgbd-unsupervised'
dataset_path = 'dataset/TestSet_fixed'  

# 创建Submission文件夹
# 将result_path下的结果文件夹复制到Submission文件夹中
# 删除Submission文件夹下多余的文件，例如Submission/001/001_001_time.value等
# 将test/tracking_results/spt/unimod1k/rgbd-unsupervised/001/001_001.txt复制后的txt文件Submission/001/001_001.txt重命名为001.txt
# 将每个txt文件Submission/001/001.txt的第一行用dataset/TestSet_fixed/001/groundtruth.txt中的第一行替换掉
# 将处理后的Submission文件夹压缩为Submission.zip


import os
import shutil
import glob
from tqdm import tqdm
import natsort

def create_submission(result_path, dataset_path):
    """
    根据模型输出结果和数据集，生成符合提交要求的 zip 文件。
    """
    submission_dir = 'Submission'
    zip_path = 'Submission.zip'

    # 1. 清理旧的提交文件和文件夹
    if os.path.exists(submission_dir):
        shutil.rmtree(submission_dir)
        print(f"已删除旧的目录: {submission_dir}")
    if os.path.exists(zip_path):
        os.remove(zip_path)
        print(f"已删除旧的压缩文件: {zip_path}")

    # 2. 创建 Submission 文件夹
    os.makedirs(submission_dir, exist_ok=True)
    print(f"已创建提交目录: {submission_dir}")

    # 获取所有结果序列的文件夹
    sequence_dirs = [d for d in glob.glob(os.path.join(result_path, '*')) if os.path.isdir(d)]
    if not sequence_dirs:
        print(f"警告: 在 '{result_path}' 中未找到任何结果序列文件夹。")
        return

    print(f"找到 {len(sequence_dirs)} 个序列，开始处理...")

    # 3. 遍历每个序列，处理并生成结果文件
    for seq_path in tqdm(sequence_dirs, desc="处理序列"):
        seq_name = os.path.basename(seq_path)

        # 在 Submission 文件夹下创建对应的序列文件夹
        dest_seq_dir = os.path.join(submission_dir, seq_name)
        os.makedirs(dest_seq_dir, exist_ok=True)

        # 查找该序列所有帧的结果文件（如 001_001.txt, 001_002.txt ...）
        frame_files = glob.glob(os.path.join(seq_path, f'{seq_name}_*.txt'))
        
        # 使用 natsort 进行自然排序，确保 001_1.txt 在 001_10.txt 之前
        sorted_frame_files = natsort.natsorted(frame_files)

        if not sorted_frame_files:
            tqdm.write(f"警告: 在序列 '{seq_name}' 中未找到任何 .txt 结果文件，跳过。")
            continue

        # 合并所有帧的结果到一个列表中
        all_lines = []
        for frame_file in sorted_frame_files:
            with open(frame_file, 'r') as f:
                all_lines.extend(f.readlines())

        # 4. 用 groundtruth 的第一行替换结果的第一行
        gt_path = os.path.join(dataset_path, seq_name, 'groundtruth.txt')
        if os.path.exists(gt_path):
            with open(gt_path, 'r') as f:
                first_gt_line = f.readline()
            
            if all_lines:
                # 替换第一行
                all_lines[0] = first_gt_line
            else:
                tqdm.write(f"警告: 序列 '{seq_name}' 的结果为空，但替换了第一行。")
                all_lines.append(first_gt_line)
        else:
            tqdm.write(f"警告: 找不到序列 '{seq_name}' 的 groundtruth 文件: {gt_path}")

        # ... (代码合并所有行到 all_lines)
        
        # 5. NEW: 格式化每一行的数据，保留一位小数
        formatted_lines = []
        for line in all_lines:
            try:
                # 分割、转换、格式化、再合并
                parts = line.strip().split(',')
                formatted_parts = [f"{float(p):.1f}" for p in parts]
                formatted_lines.append(",".join(formatted_parts) + '\n')
            except ValueError:
                # 如果某一行无法转换（例如是空行或非数字），则保持原样
                formatted_lines.append(line)
        
        # 6. 将处理后的结果写入最终的提交文件
        final_submission_path = os.path.join(dest_seq_dir, f'{seq_name}.txt')
        with open(final_submission_path, 'w') as f:
            f.writelines(formatted_lines) # 写入格式化后的 formatted_lines

    # 6. 将 Submission 文件夹压缩为 Submission.zip
    try:
        shutil.make_archive('Submission', 'zip', root_dir='.', base_dir=submission_dir)
        print(f"\n成功创建提交文件: {zip_path}")
    except Exception as e:
        print(f"\n错误: 压缩文件夹时发生错误: {e}")


if __name__ == '__main__':
    # --- 用户配置 ---
    # 模型输出结果的路径
    result_path = 'test/tracking_results/spt/unimod1k/rgbd-unsupervised'
    # 修正后的测试集路径
    dataset_path = 'dataset/TestSet_fixed'
    # --- 配置结束 ---

    # 安装依赖
    try:
        import natsort
    except ImportError:
        print("正在安装缺失的 'natsort' 包...")
        import subprocess
        import sys
        subprocess.check_call([sys.executable, "-m", "pip", "install", "natsort"])

    create_submission(result_path, dataset_path)
