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


## 模型训练

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
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/train/train.py \
  --config fourroom_hjepa_l4 \
  --training-mode layerwise \
  --run-name layerwise_l4_seed32 \
  trainer.devices=2 \
  trainer.max_epochs=1 \
  loader.batch_size=32 \
  num_workers=0 \
  loader.persistent_workers=false \
  loader.prefetch_factor=null \
  +tensorboard=true \
  +cuda_memory_monitor=true \
  planning_eval.enabled=false \
  final_probing_decoding_eval.enabled=false
```

action generation and latent feature function join training:
```bash
python scripts/train/train.py \
    --config fourroom_hjepa_l4 \
    --training-mode wam \
    --run-name fourroom_l4_wam
```


## 实验测试
单次测试
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
python scripts/eval/eval_once.py \
  --config fourroom_l4 \
  --checkpoint ./models/h_jepa/fourroom/fourrom_hjepa_l4/seed42/fourroom_hjepa_l4_object.ckpt \
  --task-index 0 \
  --run-name fourroom_hjepa_l4
```

定量测试
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
python scripts/eval/eval.py \
  --config fourroom_l4 \
  --checkpoint ./models/layerwise_l4_seed32/fourroom_hjepa_l4_object.ckpt \
  --run-name fourroom_layerwise_l4
```


解偶消融实验：
```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6 \
python scripts/eval/eval.py \
  --config fourroom_l4 \
  --checkpoint ./models/h_jepa/fourroom/fourroom_hjepa_l4/seed42/fourroom_hjepa_l4_object.ckpt \
  --run-name fourroom_prefix \
  --prefix-levels 1,2,3
```

## H-JEPA 复现与分析
### antMaze

### fourroom

### pusht

### cube


## 层级解偶
为实现推理端的自适应层级，需要先实现训练时各层级的 encoder, predictor, action_encoder 的解偶。在原始 H-JEPA 中，最显著的层级耦合在于每一层的 encoder 都复用了前一层的 encoder，并且没有使用 no_grad 指令，使得当前层的 encoder 会影响前面几层 encoder 的梯度。虽然这样的设计可以使前几层的 encoder 编码更适合后续 encoder 编码的特征向量，但对于自适应层级而言，这样设计的劣势大于优势。为此，我的第一个方案也是最显而易见的方案，就是把各层 encoder 之间的梯度给断开。然而，由于 end2end 训练架构，各个层级之间必然无法完全实现解偶。为此，我的第二个方案则是逐层训练，训练好一层后就冻结该层的 encoder, predictor, action_encoder 等参数，使用该层的参数继续完成后续的训练。

### 方案一：基于 Stop Gradient 的解偶模型


### 方案二：基于逐层训练的解偶模型


## action head 联合训练
为了展现出 JEPA 带来的 **轻量化** 优势，我认为需要使用类似当前 world action model 的 action head 联合训练。我将现有的 WAM 的 world model 和 action head 的联合方式分为4类：

1. world model 和 action head 和为一体，训练一个模型联合生成未来状态和动作，典型代表就是 ZeroWAM
2. world model 和 action head 和为一体，训练阶段使用一个模型联合生成未来状态和动作，推理阶段只输出动作，典型代表就是 Fast-WAM
3. world model 和 action head 分开，先使用 world model 生成预期的未来状态，再使用 action head 生成动作，典型代表是 VPP2
4. world model 和 action head 分开，先使用 action head 生成候选动作，再使用 world model 生成动作产生的后果，并在线评估最好的 action (有点像 MPC 了)，典型代表是