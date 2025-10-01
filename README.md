# 改进的YOLO-World系统

本项目对YOLO-World进行了如下改进：
1. **关键词提取**: 从输入文本中提取中心词
2. **智能检测**: 使用提取的关键词进行目标检测
3. **语义相关度计算**: 计算检测框内容与原文本的语义相关度

## 系统架构

```
改进的YOLO-World系统
├── keyword_extractor.py     # 关键词提取模块
├── yolo_world_inference.py  # YOLO-World推理模块
├── semantic_similarity.py   # 语义相关度计算模块
├── main.py                  # 主程序
├── setup.py                 # 环境配置脚本
└── requirements.txt         # 依赖包列表
```

## 功能模块

### 1. 关键词提取模块 (`keyword_extractor.py`)

支持多种提取方法：
- **spaCy**: 基于词性标注和命名实体识别
- **NLTK**: 基于词性标注和停用词过滤
- **YAKE**: 基于关键词提取算法

示例：
```python
from keyword_extractor import KeywordExtractor

extractor = KeywordExtractor('spacy')
text = "A black cat lying on a chair"
keywords = extractor.extract_keywords(text)  # ['cat', 'chair', 'black']
main_subject = extractor.extract_main_subject(text)  # 'cat'
```

### 2. YOLO-World推理模块 (`yolo_world_inference.py`)

功能：
- 使用提取的关键词设置检测类别
- 进行目标检测并返回结构化结果
- 支持批量处理和结果可视化

示例：
```python
from yolo_world_inference import YOLOWorldInference

inferencer = YOLOWorldInference("yolov8s-world.pt")
results = inferencer.predict("image.jpg", ["cat", "chair"])
```

### 3. 语义相关度计算模块 (`semantic_similarity.py`)

支持多种计算方法：
- **CLIP**: 直接计算图像-文本相似度
- **Sentence Transformer**: 基于文本嵌入的相似度
- **BLIP**: 先生成图像描述再计算文本相似度

示例：
```python
from semantic_similarity import SemanticSimilarityCalculator

calculator = SemanticSimilarityCalculator("clip")
similarity = calculator.calculate_similarity(image, text)
```

## 安装和使用

### 1. 环境配置

运行配置脚本（推荐）：
```bash
python setup.py
```

或手动安装依赖：
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

### 2. 基本使用

#### 命令行使用：
```bash
# 处理单张图片
python main.py --image path/to/image.jpg --text "A black cat lying on a chair"

# 指定输出目录
python main.py --image image.jpg --text "描述文本" --output results/

# 调整参数
python main.py --image image.jpg --text "描述文本" \
    --conf 0.2 --sim 0.4 \
    --keyword-method spacy --similarity-method clip
```

#### Python API使用：
```python
from main import ImprovedYOLOWorld

# 初始化系统
system = ImprovedYOLOWorld(
    model_path="yolov8s-world.pt",
    keyword_method="spacy",
    similarity_method="clip"
)

# 处理图片
result = system.process_single_image(
    "image.jpg", 
    "A black cat lying on a chair"
)

# 可视化结果
system.visualize_results(result, "output.jpg")
```

### 3. 运行演示

```bash
python main.py  # 运行演示程序
```

## 参数说明

### 命令行参数：
- `--image`: 输入图片路径
- `--text`: 文本描述
- `--model`: YOLO-World模型路径 (默认: yolov8s-world.pt)
- `--output`: 输出目录
- `--conf`: YOLO置信度阈值 (默认: 0.1)
- `--sim`: 语义相似度阈值 (默认: 0.3)
- `--keyword-method`: 关键词提取方法 (spacy/nltk/yake)
- `--similarity-method`: 相似度计算方法 (clip/sentence_transformer/blip)
- `--device`: 推理设备 (auto/cpu/cuda)

### 配置选项：
```python
# 关键词提取配置
KeywordExtractor(method='spacy')  # spacy, nltk, yake

# 相似度计算配置  
SemanticSimilarityCalculator(method='clip')  # clip, sentence_transformer, blip

# 多方法融合
MultiModalSimilarityCalculator(
    methods=['clip', 'sentence_transformer'],
    weights=[0.7, 0.3]
)
```

## 输出结果

系统输出包含：
- **关键词提取结果**: 提取的关键词和中心词
- **检测结果**: YOLO-World检测到的所有目标
- **相关度排序**: 按语义相关度排序的检测结果
- **可视化图片**: 标注了排序和相关度的检测框

示例输出：
```json
{
  "image_path": "image.jpg",
  "text_description": "A black cat lying on a chair",
  "keywords": ["cat"],
  "total_detections": 5,
  "filtered_detections": 2,
  "ranked_detections": [
    {
      "detection": {
        "bbox": [100, 150, 300, 400],
        "confidence": 0.85,
        "class_name": "cat"
      },
      "similarity_score": 0.92,
      "rank": 1
    }
  ]
}
```

## 工作流程

1. **文本分析**: 输入文本 → 提取关键词/中心词
2. **目标检测**: 图片 + 关键词 → YOLO-World检测
3. **区域裁剪**: 根据检测框裁剪图片区域
4. **相关度计算**: 裁剪区域 + 原文本 → 语义相关度分数
5. **结果排序**: 按相关度排序并过滤
6. **结果输出**: 生成JSON结果和可视化图片

## 性能优化建议

1. **GPU加速**: 使用CUDA设备可显著提升推理速度
2. **批量处理**: 对多张图片使用批量处理接口
3. **模型选择**: 根据精度/速度需求选择合适的模型
4. **阈值调整**: 根据应用场景调整置信度和相似度阈值

## 依赖包

主要依赖：
- ultralytics (YOLO-World)
- torch, torchvision
- transformers (CLIP, BLIP)
- sentence-transformers
- spacy, nltk, yake (关键词提取)
- opencv-python, pillow (图像处理)

详见 `requirements.txt`

## 故障排除

1. **模型加载失败**: 确保模型文件存在并且路径正确
2. **GPU内存不足**: 使用CPU或减小批量大小
3. **依赖包问题**: 运行 `python setup.py` 重新配置环境
4. **spaCy模型缺失**: 运行 `python -m spacy download en_core_web_sm`

## 扩展功能

- 支持中文文本处理
- 添加更多关键词提取算法
- 集成更多多模态模型
- 支持视频处理
- 添加Web界面