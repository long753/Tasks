# 基于 H-JEPA 的自适应层级划分实验

本项目研究 Hierarchical JEPA 世界模型及其自适应时间层级扩展。H-JEPA、HWM 和 LeWM 共用模型实现；LeWM 是单层配置，不单独复制算法包。

## 目录职责

- `src/h_jepa/`：层级 JEPA 编码、动作表征、训练损失、规划与评估实现。
- `src/stable_worldmodel/`：HDF5 数据集、仿真环境、策略封装及梯度求解器。
- `configs/train/`、`configs/eval/`、`configs/probing/`：算法及实验配置，不保存机器专属存储路径。
- `configs/data/`：数据采集配置。
- `scripts/train/`、`scripts/eval/`：薄启动入口与批量实验脚本。
- `scripts/data/`、`scripts/slurm/`：数据生成处理及集群启动工具。
- `scripts/test/`：自动化测试。
- `datasets/`：数据文件、评估轨迹及 DROID 清单和归一化统计。
- `models/`：模型检查点、断点续训状态、训练配置和 `normalizer.pt`。
- `outputs/`：日志、评估指标、视频、图表和运行缓存。


## Setup

该项目使用 conda 进行环境管理。 `hjepa`，Python 3.11。从项目根目录运行：

```bash
git clone https://github.com/long753/Tasks/tree/master/task_jepa
cd task_jepa
conda create -n hjepa python=3.11.13
conda activate hjepa
python -m pip install -e ".[train,env]"
```


## 训练模型

训练标准 H-JEPA：
```bash
python scripts/train/train.py \
    --config fourroom_hjepa_l4 \
    --training-mode e2e \
    --run-name fourroom_l4_e2e
```

训练基于 stop gradient 的解偶模型：
```bash
python scripts/train/train.py \
    --config fourroom_hjepa_l4 \
    --training-mode sg \
    --run-name fourroom_l4_sg
```

训练基于逐层训练的解偶模型：
```bash
python scripts/train/train.py \
    --config fourroom_hjepa_l4 \
    --training-mode layerwise \
    --run-name fourroom_l4_layerwise
```

action generation and latent feature function join training:
```bash
python scripts/train/train.py \
    --config fourroom_hjepa_l4 \
    --training-mode wam \
    --run-name fourroom_l4_wam
```


## 测试模型
单次测试
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
python scripts/eval/eval_once.py \
  --config fourroom_l4 \
  --checkpoint ./models/h_jepa/fourroom/fourrom_hjepa_l4/seed42/fourroom_hjepa_l4_object.ckpt \
  --task-index 0 \
  --run-name fourroom_sg_epoch1_task0
```

定量测试
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
python scripts/eval/eval.py \
  --config fourroom_l4 \
  --checkpoint ./models/h_jepa/fourroom/fourrom_hjepa_l4/seed42/fourroom_hjepa_l4_object.ckpt \
  --run-name fourroom_sg_epoch1_eval50
```
其中，第一行指定测试使用的显卡；run-name 指定输出文件夹的名字


## H-JEPA 复现与分析
### antMaze

### fourroom

### pusht

### cube


## 层级解偶
### 方案一：基于 Stop Gradient 的解偶模型


### 方案二：基于逐层训练的解偶模型


## action head 联合训练

