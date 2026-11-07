# 目标检测多框架对比训练与评测

基于建筑裂缝数据集，对比 YOLOv8s、YOLOv11s、RT-DETR-L、Faster R-CNN 和 RetinaNet。

## 代码架构

```text
configs/                              # 数据预处理、训练和评测配置
script/                               # 数据处理、训练及评测入口
src/                                  # 数据集、模型和评测实现
datasets/CrackDataset/                 # 原始数据
datasets/processed/crack_detection/    # 处理后的图片、YOLO 标签和 COCO 标注
models/pretrained/                    # COCO 预训练权重
models/trained/                       # 训练完成的模型权重
outputs/training/                     # 训练日志、配置快照和曲线
outputs/evaluation/                   # 评测指标、预测结果和可视化
```

## 视觉编码与 Slot Attention 实验

核心模型位于 `src/model/slot_attention.py`：CNN 编码器、槽间竞争与 GRU 更新的 Slot Attention、空间广播解码器，以及各槽 RGB 与 mask 的加权合成。`src/data/slot_scenes.py` 生成简单场景，`src/evaluation/slot_metrics.py` 计算实例分割指标。

### 随机多物体场景训练

默认不再反复拟合同一张图片，而是每次更新生成一批新场景：简单深色背景、1～3 个随机颜色/大小/位置的圆形或方形，物体互不重叠。默认四个 slot，为三个物体与背景提供容量；slot 编号没有固定语义。

```bash
uv run --no-sync python script/test.py
```

默认输入 `64×64`、batch size 为 16、学习率 `4e-4`、10000 次优化器更新，自动选择 CUDA 或 CPU。`--epochs` 在此入口表示**更新次数**，不是遍历有限数据集的次数。训练只使用 RGB 重建 MSE，生成的实例标签**不参与训练损失**。

合成场景使用 `constant_color=True` 解码器：每个 slot 从向量预测一个 RGB 颜色，空间广播 CNN 只预测该 slot 的 mask。该归纳偏置与纯色形状数据相匹配，可减少“每槽画出整张场景、背景随意分配”的退化解；它不使用真实标签，但也不能表示单个物体内部的纹理。真实图片单图模式仍使用通用空间 RGB 解码器。解码方式随模型配置保存，加载权重后不会自动切换。

```bash
uv run --no-sync python script/test.py \
  --epochs 10000 --batch-size 16 --val-size 64 \
  --slot-count 4 --image-size 64 --seed 42
```

训练与验证使用独立随机种子。验证固定在同一组未参与训练的场景上，打印重建 MSE、前景 ARI 与匹配 IoU：

- `foreground_ari`：忽略真实背景后的聚类一致性，不要求 slot 编号与物体编号相同；单物体场景的该指标不足以判断分割质量。
- `matched_iou`：真实前景物体与预测 slot 一对一匹配后的平均 IoU；背景误分到物体 slot 会扩大并集、降低分数。

应同时检查指标与多张验证图片。低重建误差不代表分割正确，单看前景 ARI 也可能遗漏背景误分。

### 输出与推理

每次运行在 **`outputs/test/<运行标识>/`** 建立独立目录：

- `model.pt`：训练完成时保存模型权重、模型/场景配置、种子及验证批大小；不包含优化器状态，不用于断点续训。
- `segmentation.png`：第一张固定验证场景的原图、重建图、预测分割叠加图、真实标签、各槽软 mask 及加权重建区域。
- `validation.png`：固定验证集前四张图片的原图、真实标签、预测分割与重建图，不挑选效果最好的样本。
- `metrics.json`：最终验证指标。
- `feature_maps.png`：第一张验证图的编码器特征通道，各通道单独归一化。

加载权重会复用保存的模型配置、场景种子与验证批大小，只评估并生成图片，不重新训练、不复制权重：

```bash
uv run --no-sync python script/test.py \
  --checkpoint outputs/test/<运行标识>/model.pt
```

对指定真实图片推理：

```bash
uv run --no-sync python script/test.py \
  --checkpoint outputs/test/<运行标识>/model.pt \
  --image datasets/images/nailong.png
```

显式传入 `--image` 且不传 `--checkpoint`，仍可运行单图拟合实验，默认分辨率为 `128×128`。真实图片没有实例标签，因此不输出前景 ARI、匹配 IoU 或 `validation.png`。图片转为 RGB 并直接缩放到正方形，可能改变非正方形图片的比例。

`--image-size` 必须至少为 16 且是 16 的倍数；合成场景模式要求至少两个 slot，最多生成 `min(3, slot_count-1)` 个物体。可用 `--device cpu` 强制 CPU 运行。

纯色解码器的颜色使用 sigmoid 限制到 `[0,1]`；通用解码器的 RGB 保持线性，仅显示时截断。两种模式的 mask 都沿 slot 维 softmax，每像素权重和为 1。入口在最终评估前重设种子，同设备同环境下可复现评估结果。简单场景实验用于检验物体分解能力，**不保证直接迁移到 `nailong.png` 或自然图片**；单图重建仍可能发生槽塌缩。


---
## 具体实现：

### 1. 创建环境

使用 Python 3.12，GPU 训练需要兼容 CUDA 12.8 的 NVIDIA 驱动。以下两种方式任选一种；后续命令均在项目根目录、激活所选环境后执行。

1. 基于 uv 的环境管理方案（需已安装 uv）：

```bash
uv sync --locked --managed-python
source .venv/bin/activate
```

2. 基于 conda 的环境管理方案：

```bash
conda create -n cv_task python=3.12 -y
conda activate cv_task
python -m pip install -r requirements.txt
```

需要下载时，可先在已配置代理的终端执行 `proxy_on`。

### 2. 数据预处理

数据集：[CrackDataset](https://github.com/guoguolord/CrackDataset)。原始数据放入 `datasets/CrackDataset/`，路径与处理参数见 `configs/preprocess.yaml`。

预处理将裂缝掩膜转换为检测框，检查重复与冲突样本，按来源分组划分训练、验证和测试集，同时导出 YOLO 与 COCO 格式。

```bash
# 完整预处理
python script/prepare_dataset.py --config configs/preprocess.yaml --stage all

# 校验已有处理结果；已有数据时可只执行此命令
python script/prepare_dataset.py --stage validate --output datasets/processed/crack_detection
```

### 3. 模型训练

训练前，将对应的 COCO 预训练权重放入 `models/pretrained/`，文件名见配置中的 `models` 字段。

**Ultralytics：YOLOv8s、YOLOv11s、RT-DETR-L**

```bash
python script/train_ultralytics.py \
  --config configs/ultralytics.yaml \
  --epochs 150 \
  --model yolov8s
```

将 `--model` 改为 `yolov11s` 或 `rtdetr_l` 即可切换模型。

**TorchVision：Faster R-CNN、RetinaNet（均使用 ResNet-50-FPN）**

```bash
python script/train_torchvision.py \
  --config configs/torchvision.yaml \
  --epochs 150 \
  --model fasterrcnn_resnet50_fpn
```

训练 RetinaNet 时，将 `--model` 改为 `retinanet_resnet50_fpn`。两种训练脚本均支持 `--batch`、`--workers` 覆盖配置；按显存和实际吞吐调整。

**断点续训：**

```bash
python script/train_ultralytics.py --resume outputs/training/yolov8s_formal/weights/last.pt
```

TorchVision 模型使用 `train_torchvision.py --resume <last.pt路径>`。`--resume` 单独使用，不与其他训练参数组合。

训练日志与曲线位于 `outputs/training/<运行名>/`；训练完成的权重归档到 `models/trained/<运行名>/`，其中 `best.pt` 用于最终评测。

**查看 TensorBoard：**

```bash
tensorboard --logdir outputs/training --host 127.0.0.1 --port 6006
```

本机浏览器访问 `http://127.0.0.1:6006`，选择 SCALARS 查看 loss；远程服务器需通过 SSH 端口转发访问。

### 4. 模型评估

加载指定权重，在验证集或测试集上评测，不更新模型参数。以下以 YOLOv8s 为例：

```bash
python script/evaluate.py \
  --config configs/ultralytics.yaml \
  --model yolov8s \
  --weights models/trained/yolov8s_formal/best.pt \
  --split test \
  --output outputs/evaluation/yolov8s \
  --benchmark
```

其他模型替换 `--model`、`--weights` 和输出目录；Faster R-CNN、RetinaNet 使用 `configs/torchvision.yaml`。`--split` 支持 `val`、`test`；`--benchmark` 额外测量参数量、GFLOPs、单张时延和 FPS，使用固定验证样本进行性能测试。

uv 环境下，可执行 `bash script/evaluate_all.sh` 连续评测五个模型，需先准备好全部权重。

| 输出文件 | 用途 |
| --- | --- |
| `metrics.json` | Precision、Recall、mAP@0.5、mAP@0.5:0.95，以及 TP、FP、FN |
| `pr_curve.png` | IoU＝0.5 时的 Precision–Recall 曲线 |
| `previews/` | 真实框为绿色，预测框为橙色，数字为预测置信度 |
| `predictions_coco.json` | 全部评测图片的预测框，格式为 COCO 的 `[x, y, 宽, 高]` |
| `evaluation_summary.json` | 权重、配置、评测协议及结果的完整记录 |
| `benchmark.json` | 性能测量详情，仅启用 `--benchmark` 时生成 |
| `summary.csv` | 检测与性能指标汇总，便于制作对比表，仅启用 `--benchmark` 时生成 |

指标基于所选划分的全部图片计算，预览只展示部分样本。横向对比前统一输入尺寸、评测阈值和测速硬件，不在测试集上调参。