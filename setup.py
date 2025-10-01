"""
依赖包安装和环境配置脚本
"""

import subprocess
import sys
import os

def install_package(package):
    """安装Python包"""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ 成功安装 {package}")
        return True
    except subprocess.CalledProcessError:
        print(f"✗ 安装 {package} 失败")
        return False

def download_spacy_model():
    """下载spaCy英文模型"""
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "en_core_web_sm"])
        print("✓ 成功下载 spaCy 英文模型")
        return True
    except subprocess.CalledProcessError:
        print("✗ 下载 spaCy 英文模型失败")
        return False

def setup_environment():
    """设置环境"""
    print("=== 改进的YOLO-World系统环境配置 ===\n")
    
    # 必需的包列表
    required_packages = [
        "ultralytics",  # YOLO-World
        "torch",
        "torchvision", 
        "transformers",
        "sentence-transformers",
        "spacy",
        "nltk",
        "yake",
        "opencv-python",
        "pillow",
        "numpy",
        "scikit-learn"
    ]
    
    # 可选的包（用于更好的功能）
    optional_packages = [
        "accelerate",  # 加速transformers
        "datasets",    # 数据集处理
        "matplotlib",  # 可视化
        "seaborn"      # 更好的可视化
    ]
    
    print("安装必需的依赖包...")
    failed_packages = []
    
    for package in required_packages:
        if not install_package(package):
            failed_packages.append(package)
    
    if not failed_packages:
        print("\n✓ 所有必需包安装成功!")
    else:
        print(f"\n⚠️  以下包安装失败: {failed_packages}")
        print("请手动安装这些包")
    
    # 下载spaCy模型
    print("\n下载spaCy英文模型...")
    download_spacy_model()
    
    # 安装可选包
    print("\n安装可选依赖包...")
    for package in optional_packages:
        install_package(package)
    
    # 下载NLTK数据
    print("\n下载NLTK数据...")
    try:
        import nltk
        nltk.download('punkt')
        nltk.download('stopwords')
        nltk.download('averaged_perceptron_tagger')
        print("✓ NLTK数据下载完成")
    except Exception as e:
        print(f"✗ NLTK数据下载失败: {e}")
    
    print("\n=== 环境配置完成 ===")
    print("\n使用说明:")
    print("1. 确保你有YOLO-World模型文件 (yolov8s-world.pt)")
    print("2. 运行演示: python main.py")
    print("3. 处理图片: python main.py --image path/to/image.jpg --text 'description'")

def check_gpu():
    """检查GPU可用性"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✓ 检测到 {gpu_count} 个GPU: {gpu_name}")
            return True
        else:
            print("⚠️  未检测到GPU，将使用CPU运行（速度较慢）")
            return False
    except ImportError:
        print("⚠️  PyTorch未安装，无法检查GPU")
        return False

def download_model():
    """下载YOLO-World模型"""
    model_path = "yolov8s-world.pt"
    if os.path.exists(model_path):
        print(f"✓ 模型文件 {model_path} 已存在")
        return True
    
    print(f"下载YOLO-World模型到 {model_path}...")
    try:
        from ultralytics import YOLO
        model = YOLO(model_path)  # 这会自动下载模型
        print(f"✓ 模型下载完成: {model_path}")
        return True
    except Exception as e:
        print(f"✗ 模型下载失败: {e}")
        print("请手动下载模型文件")
        return False

if __name__ == "__main__":
    setup_environment()
    print("\n" + "="*50)
    check_gpu()
    print("\n" + "="*50)
    download_model()