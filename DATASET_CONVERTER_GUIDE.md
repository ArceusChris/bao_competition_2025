# 数据集转换器使用指南

## 📖 概述

本数据集转换器可以将TrainSet格式的数据转换为YOLO格式，并使用关键词提取器自动从图片描述中生成类别标签。

## 🚀 快速开始

### 1. 基本转换

```bash
# 转换整个数据集（默认设置）
python convert_dataset.py

# 或指定参数
python convert_dataset.py --source TrainSet --output yolo_dataset --train-ratio 0.8
```

### 2. 测试模式

```bash
# 快速测试（每个类别只处理2个实例）
python convert_dataset.py --test-mode
```

### 3. 自定义转换

```bash
# 限制每个类别的实例数量
python convert_dataset.py --max-instances 10 --train-ratio 0.9

# 详细输出
python convert_dataset.py --verbose
```

## 📁 输出结构

转换后的YOLO数据集结构：

```
yolo_dataset/
├── images/
│   ├── train/          # 训练集图片
│   │   ├── cat1_00000001.jpg
│   │   ├── cat1_00000002.jpg
│   │   └── ...
│   └── val/            # 验证集图片
│       ├── dog1_00000001.jpg
│       └── ...
├── labels/
│   ├── train/          # 训练集标签 (YOLO格式)
│   │   ├── cat1_00000001.txt
│   │   ├── cat1_00000002.txt
│   │   └── ...
│   └── val/            # 验证集标签
│       ├── dog1_00000001.txt
│       └── ...
├── dataset.yaml        # YOLO数据集配置文件
└── class_mapping.json  # 类别映射和统计信息
```

## 🔍 数据集分析和验证

### 分析数据集

```bash
# 分析转换后的数据集
python visualize_dataset.py yolo_dataset --action analyze
```

### 可视化样本

```bash
# 可视化训练集样本
python visualize_dataset.py yolo_dataset --action visualize --split train --samples 6

# 可视化验证集样本  
python visualize_dataset.py yolo_dataset --action visualize --split val --samples 6

# 同时进行分析和可视化
python visualize_dataset.py yolo_dataset --action both
```

## 🎯 关键词提取器工作原理

### 示例转换

| 原始描述 | 提取的中心词 | YOLO类别 |
|---------|-------------|----------|
| "a piebald cat lying on the ground" | cat | cat |
| "a red bag held by a woman" | bag | bag |
| "a black car parked on the side" | car | car |
| "a man in green T-shirt walking" | man | person |

### 特殊映射规则

转换器包含智能映射规则：

```python
# 人类相关词汇统一映射为 "person"
"man" → "person"
"woman" → "person" 
"child" → "person"
"human" → "person"

# 设备相关词汇
"cell_phone" → "phone"
"hard_disk" → "disk"
"ipad" → "tablet"
```

## 📊 命令行选项

### convert_dataset.py

```bash
python convert_dataset.py [选项]

选项:
  -s, --source SOURCE       源数据集目录 (默认: TrainSet)
  -o, --output OUTPUT       输出目录 (默认: yolo_dataset)  
  -r, --train-ratio RATIO   训练集比例 (默认: 0.8)
  -m, --max-instances N     每类最大实例数 (默认: 无限制)
  -t, --test-mode          测试模式 (快速转换少量数据)
  -v, --verbose            详细输出
  -h, --help               显示帮助信息
```

### visualize_dataset.py

```bash
python visualize_dataset.py DATASET_PATH [选项]

选项:
  -a, --action ACTION      操作类型: analyze/visualize/both (默认: both)
  -s, --split SPLIT        数据分割: train/val (默认: train)
  -n, --samples N          可视化样本数 (默认: 6)
  -h, --help               显示帮助信息
```

## 🔧 使用转换后的数据集

### 1. 使用Ultralytics YOLO

```python
from ultralytics import YOLO

# 加载预训练模型
model = YOLO('yolov8s.pt')

# 训练模型
results = model.train(
    data='yolo_dataset/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16,
    name='custom_model'
)

# 验证模型
metrics = model.val()

# 推理
results = model('path/to/image.jpg')
```

### 2. 自定义训练配置

```python
# 高级训练配置
results = model.train(
    data='yolo_dataset/dataset.yaml',
    epochs=200,
    imgsz=640,
    batch=32,
    lr0=0.01,
    optimizer='Adam',
    augment=True,
    name='advanced_model',
    patience=20,
    save_period=10
)
```

## 📈 性能优化建议

### 1. 数据集大小优化

```bash
# 对于大数据集，可以限制每个类别的实例数
python convert_dataset.py --max-instances 50

# 调整训练/验证比例
python convert_dataset.py --train-ratio 0.9
```

### 2. 类别平衡

转换器会自动统计每个类别的样本数量。如果发现类别不平衡，可以：

1. 调整 `max_instances_per_category` 参数
2. 使用数据增强技术
3. 调整训练时的类别权重

### 3. 质量检查

转换后务必进行质量检查：

```bash
# 1. 分析数据集统计
python visualize_dataset.py yolo_dataset --action analyze

# 2. 可视化样本检查标注正确性
python visualize_dataset.py yolo_dataset --action visualize

# 3. 检查文件完整性
ls yolo_dataset/images/train | wc -l
ls yolo_dataset/labels/train | wc -l
```

## ⚠️ 注意事项

### 1. 文件格式要求

- 源数据集必须包含 `color/`、`nlp.txt`、`groundtruth_rect.txt`
- 图片格式为 `.jpg`
- 标注格式为 `x_min,y_min,x_max,y_max`

### 2. 坐标系统

- 输入: 绝对坐标 (像素值)
- 输出: YOLO相对坐标 (0-1归一化)

### 3. 类别映射

关键词提取可能不完美，建议：

1. 检查生成的 `class_mapping.json`
2. 必要时手动调整类别映射
3. 验证关键词提取的准确性

## 🐛 故障排除

### 常见问题

1. **"缺少必要文件"错误**
   - 检查源数据集结构是否完整
   - 确保每个实例都有 `nlp.txt` 和 `groundtruth_rect.txt`

2. **"图片数量与标注数量不匹配"警告**
   - 这是正常的，转换器会自动处理
   - 使用较小的数量继续转换

3. **关键词提取不准确**
   - 检查 `nlp.txt` 中的描述文本
   - 考虑手动调整类别映射

4. **内存不足**
   - 使用 `--max-instances` 限制数据量
   - 分批处理大数据集

### 日志分析

转换过程中的输出含义：

- ✅ 成功处理: 正常转换
- ⚠️ 警告: 可能的问题，但会继续处理
- ❌ 错误: 严重问题，需要处理

## 📞 技术支持

如果遇到问题：

1. 检查本文档的故障排除部分
2. 使用 `--verbose` 获取详细错误信息
3. 检查数据集结构和格式
4. 验证依赖包是否正确安装

---

**祝您转换顺利！🎉**