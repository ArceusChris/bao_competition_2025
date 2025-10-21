# 写一个脚本，处理全部的此类文件find /public/home/lxy/bao_3/UniMod1K/SPT/dataset/ValidationSet/ -type f -name "groundtruth_rect.txt"
# 将每个文件的第一行读取，并另存为“groundtruth.txt”

import os

def process_groundtruth_files(root_dir):
    """
    遍历指定目录，找到所有 groundtruth_rect.txt 文件，
    读取其第一行内容，并写入到同级目录下的 groundtruth.txt 文件中。
    """
    print(f"开始处理目录: {root_dir}")
    
    # os.walk 会递归遍历所有子目录
    for dirpath, _, filenames in os.walk(root_dir):
        if "groundtruth_rect.txt" in filenames:
            # 构建原始文件的完整路径
            rect_file_path = os.path.join(dirpath, "groundtruth_rect.txt")
            
            try:
                with open(rect_file_path, 'r') as f_read:
                    # 读取第一行
                    first_line = f_read.readline()
                
                if first_line:
                    # 构建新文件的完整路径
                    new_file_path = os.path.join(dirpath, "groundtruth.txt")
                    
                    with open(new_file_path, 'w') as f_write:
                        # 写入第一行内容
                        f_write.write(first_line)
                        
                    print(f"成功创建: {new_file_path}")
                else:
                    print(f"警告: 文件为空，跳过 {rect_file_path}")

            except Exception as e:
                print(f"处理文件 {rect_file_path} 时出错: {e}")

    print("\n处理完成。")

if __name__ == "__main__":
    # 设置要处理的根目录
    validation_set_dir = "/public/home/lxy/bao_3/UniMod1K/SPT/dataset/ValidationSet/"
    
    # 检查目录是否存在
    if not os.path.isdir(validation_set_dir):
        print(f"错误: 目录不存在 -> {validation_set_dir}")
    else:
        # 执行处理函数
        process_groundtruth_files(validation_set_dir)