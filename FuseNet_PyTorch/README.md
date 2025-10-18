# 使用说明

## 功能概述

`data_utils.py` 已扩展支持两种数据格式：
1. **HDF5 格式** - 原始 NYU 深度数据集格式（向后兼容）
2. **JSON 格式** - COCO 格式的对象检测/分割数据集（新增）

## 自动格式检测

`get_data()` 函数会根据文件扩展名自动选择加载方式：
- `.h5` 或 `.hdf5` → 使用 HDF5 加载器
- `.json` → 使用 JSON (COCO格式) 加载器

---

## JSON 格式要求

### JSON 文件结构 (COCO 格式)
```json
{
  "info": {...},
  "licenses": [...],
  "images": [
    {
      "id": 0,
      "file_name": "000001.jpg",
      "height": "480",
      "width": "960",
      "original_path": "Adapter/adapter1/color/00000001.jpg",
      ...
    }
  ],
  "annotations": [
    {
      "id": 0,
      "image_id": 0,
      "category_id": 1,
      "segmentation": [[x1, y1, x2, y2, ...]],
      "bbox": [x, y, w, h],
      ...
    }
  ],
  "categories": [
    {"id": 1, "name": "adapter"},
    {"id": 2, "name": "pedestal"},
    ...
  ]
}
```

### 目录结构要求
```
datasets/
├── instances_train.json          # 训练集标注
├── instances_test.json           # 测试集标注（可选）
└── <category>/<instance>/
    ├── color/                    # RGB 图像
    │   ├── 00000001.jpg
    │   └── ...
    └── depth/                    # 深度图像（可选）
        ├── 00000001.png
        └── ...
```

**注意**：
- RGB 图像路径通过 `original_path` 字段指定
- 深度图像路径自动从 `/color/` 替换为 `/depth/`
- 如果图像文件不存在，会生成虚拟图像（用于测试）

---

## 数据转换流程

### JSON → PyTorch Dataset

```
instances_train.json (COCO format)
        ↓
读取 JSON 文件
        ↓
加载 RGB 图像 (from original_path)
加载深度图像 (from depth/ folder)
生成分割掩码 (from segmentation polygons)
        ↓
调整大小到 240×320
        ↓
转换数据类型和格式
  - RGB: uint8 (H,W,3) → float32 (3,H,W)
  - Depth: uint8 (H,W) → float32 (1,H,W)
  - Label: polygon → uint8 (H,W) → int64 (H,W)
        ↓
PyTorch Tensors
```

---

## 使用方法

### 方法 1: 使用 JSON 数据训练

```bash
# 使用 JSON 格式数据集
python fusenet_train.py \
    --dataroot ./datasets/instances_train.json \
    --batch_size 8 \
    --lr 0.005
```

### 方法 2: 使用 HDF5 数据训练（原始方法）

```bash
# 使用 HDF5 格式数据集
python fusenet_train.py \
    --dataroot ./datasets/nyu_class_10_db.h5 \
    --batch_size 8 \
    --lr 0.005
```

### 方法 3: 在代码中使用

```python
from utils.data_utils import get_data
from options.train_options import TrainOptions

# 解析参数
opt = TrainOptions().parse()

# 自动检测格式并加载数据
train_data, test_data = get_data(opt, use_train=True, use_test=True)

# 创建 DataLoader
import torch.utils.data
train_loader = torch.utils.data.DataLoader(
    train_data,
    batch_size=opt.batch_size,
    shuffle=True,
    num_workers=opt.num_workers
)
```

---

## 输出张量格式

两种格式的输出完全一致，确保兼容性：

### 每个样本返回的数据

**不使用分类标签 (use_class=False)**:
```python
[rgb_img, depth_img, seg_label]
```
- `rgb_img`: Tensor, shape (3, 240, 320), dtype=float32
- `depth_img`: Tensor, shape (1, 240, 320), dtype=float32
- `seg_label`: ndarray, shape (240, 320), dtype=int64

**使用分类标签 (use_class=True)**:
```python
[rgb_img, depth_img, seg_label, class_label]
```
- `rgb_img`: Tensor, shape (3, 240, 320), dtype=float32
- `depth_img`: Tensor, shape (1, 240, 320), dtype=float32
- `seg_label`: ndarray, shape (240, 320), dtype=int64
- `class_label`: int, 类别索引

---

## 关键类说明

### `CreateDataFromJSON` 类

新增的 Dataset 类，用于处理 COCO JSON 格式：

```python
class CreateDataFromJSON(data.Dataset):
    def __init__(self, json_path, data_root, target_size=(240, 320), use_class=False):
        # json_path: JSON 文件路径
        # data_root: 图像根目录
        # target_size: 目标图像大小 (H, W)
        # use_class: 是否包含分类标签
```

**主要方法**:
- `_load_image()`: 加载并调整 RGB 图像大小
- `_load_depth()`: 加载深度图像（如果存在）
- `_create_segmentation_mask()`: 从多边形注释生成分割掩码

---

## 配置参数

### 在 TrainOptions 中设置

```python
# 使用 JSON 数据
parser.add_argument('--dataroot', type=str, default='./datasets/instances_train.json')
parser.add_argument('--use_class', action='store_true', help='使用分类标签')
```

---




