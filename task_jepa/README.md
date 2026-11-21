# JEPA-based Task-Conditioned World Action Model

本项目在 [H-JEPA](https://github.com/kevinghst/H-JEPA) 基础上研究两件事：用 stop-gradient 和逐层训练检验低层能否独立规划，以及联合训练任务条件世界模型和动作生成器。

## 项目树

- `src/h_jepa/`：H-JEPA、HWM 和单层 LeWM 的模型、损失与规划。
- `src/jepa_wam/`：任务条件 latent flow、动作 flow 和闭环策略。
- `src/stable_worldmodel/`：数据集、仿真环境、策略接口和求解器。
- `src/cost_probe/`：规划代价与表征探针。
- `configs/`：训练、评估、探针和数据采集配置；`scripts/`：对应命令入口。
- `datasets/`、`models/`、`outputs/`：数据和固定任务、权重和训练状态、指标和录像。

## Setup

使用 Conda 环境 `hjepa`（Python 3.11）：

```bash
git clone https://github.com/long753/Tasks.git
cd Tasks/task_jepa
conda create -n hjepa python=3.11
conda activate hjepa
python -m pip install -e ".[train,env]"
```

训练 HDF5 放在 `datasets/`，固定评估任务放在 `datasets/eval_trajs/`。H-JEPA 评估显式指定 `--checkpoint`，检查点旁需有 `normalizer.pt`。按需用 `--datasets`、`--models`、`--outputs` 更换数据、训练产物和结果目录。数据布局与旧检查点迁移见 [AGENTS.md](AGENTS.md)。

## H-JEPA 复现与分析

下面以 FourRoom 预训练 L4 为例。`eval.py` 跑固定任务并保存指标；需要单任务执行录像时，用 `eval_once.py` 替换入口并加 `--task-index 0`。

```bash
python scripts/eval/eval.py --config fourroom_l4 --run-name fourroom_pretrained_l4 \
  --checkpoint models/h_jepa/fourroom/fourroom_hjepa_l4/seed42/fourroom_hjepa_l4_object.ckpt
```

### AntMaze

使用 `--config ant_l2` 和 `models/h_jepa/ant/ant_hjepa_l2/seed42/ant_hjepa_l2_object.ckpt`，固定任务来自 AntMaze 专家轨迹。

### FourRoom

使用 `--config fourroom_l4`（上例）；单层 LeWM 改用 `fourroom_flat` 和 `models/h_jepa/fourroom/fourroom_lewm/seed42/fourroom_lewm_object.ckpt`。下文两组解耦实验也用这 50 个固定任务。

### PushT

使用 `--config pusht_l2` 和 `models/h_jepa/pusht/pusht_hjepa_l2/seed42/pusht_hjepa_l2_object.ckpt`。

### Cube

使用 `--config cube_l2` 和 `models/h_jepa/cube/cube_hjepa_l2/seed42/cube_hjepa_l2_object.ckpt`，评估 OGBench 的 Cube Pickup 任务。

## 层级解耦实验一：基于 Stop Gradient 的解耦模型

### 核心原理

高层 encoder 接收已 `detach` 的下层状态特征。各层损失仍然相加，下层继续由自己的损失训练；动作表征的跨层梯度不截断。

### 训练脚本

```bash
python scripts/train/train.py --config fourroom_hjepa_l4 --training-mode sg \
  --run-name fourroom_sg_example trainer.devices=1 trainer.max_epochs=1 \
  planning_eval.enabled=false final_probing_decoding_eval.enabled=false
```

### 测试脚本

用同一个四层检查点分别评估 L1、L1–L2、L1–L3，不重训；去掉 `--prefix-levels` 即评估完整 L4。

```bash
python scripts/eval/eval.py --config fourroom_l4 --run-name fourroom_sg_prefix \
  --checkpoint models/fourroom_sg_example/fourroom_hjepa_l4_object.ckpt \
  --prefix-levels 1,2,3
```

### 测试结果

SG 在 L1–L2、L1–L3 上优于本次端到端训练，但 L1 未改善，完整 L4 的成功率持平。短前缀仍不足以可靠规划；具体数据见下一节。

## 层级解耦实验二：基于逐层训练的解耦模型

### 核心原理

从 L1 到 L4 依次训练。每阶段只更新当前层的 encoder、predictor 和动作表征；已训练的下层保持冻结，最后不做联合微调。

### 训练脚本

同一入口选择 `layerwise`。`trainer.max_epochs=1` 是**每层**一轮，四个阶段自动执行。

```bash
python scripts/train/train.py --config fourroom_hjepa_l4 --training-mode layerwise \
  --run-name fourroom_layerwise_example trainer.devices=1 trainer.max_epochs=1 \
  planning_eval.enabled=false final_probing_decoding_eval.enabled=false
```

### 测试脚本

复用上节评估命令，将检查点换成 `models/fourroom_layerwise_example/fourroom_hjepa_l4_object.ckpt` 并换一个 `--run-name`。前缀结果汇总在 `outputs/<run-name>/prefix_summary.csv`。

### 测试结果

FourRoom 历史实验成功率（50 个固定任务，最多 210 步，使用原 `goal_state` 判定）：

| 模型 | L1 | L1–L2 | L1–L3 | L1–L4 |
| --- | ---: | ---: | ---: | ---: |
| 预训练 H-JEPA | 6% | 6% | 40% | 88% |
| 端到端训练 | 4% | 2% | 20% | 22% |
| Stop Gradient | 2% | 12% | 28% | 22% |
| 逐层训练 | 12% | 8% | 26% | 72% |

逐层训练的完整模型在这次实验中更强，但截断后仍明显退化。训练预算和检查点来源不完全一致，不能把差异单独归因于解耦，也不支持直接删掉高层。

## Task-Conditioned World Action Model

### 模型架构

在线 encoder 编码当前观测和目标图像（可加 `proprio`）。任务条件世界模型通过 Flow Matching 生成 5 步后的子目标 latent，Action Expert 用当前 latent 和子目标生成动作块。训练 Action Expert 时，默认前 70% 的更新用 EMA 编码的真实未来作子目标，之后用世界模型从同一样本的带噪未来 latent 估计的终点。

### 训练脚本

准备 FourRoom 的训练和验证 HDF5，以及同环境 LeWM 的 ViT `state_dict`（不是完整检查点）。

```bash
python scripts/train/train_wam.py --run-name fourroom_wam_train \
  --encoder-weights models/wam_init/fourroom_lewm_vit.pt \
  --epochs 2 --batch-size 32 --workers 2
```

模型权重保存为 `models/<run-name>/last.pt`，训练与验证指标保存在 `outputs/<run-name>/metrics.csv`。

### 测试脚本

用训练好的 `last.pt` 直接评估，不需要单独提供 `normalizer.pt`。

```bash
ckpt=models/wam_seed42/last.pt

# 单任务：输出真实执行和目标画面的并排录像
python scripts/eval/eval_wam_once.py --checkpoint "$ckpt" \
  --task-index 0 --run-name fourroom_wam_demo

# 定量评估：50 个固定任务，不录视频
python scripts/eval/eval_wam.py --checkpoint "$ckpt" \
  --run-name fourroom_wam_eval

# 同一张 CUDA 卡上依次对比 WAM、LeWM 和 H-JEPA L4
base=models/h_jepa/fourroom
python scripts/eval/eval_cost.py --mode both --run-name fourroom_cost_repro \
  --model wam wam "$ckpt" \
  --model lewm fourroom_flat "$base/fourroom_lewm/seed42/fourroom_lewm_object.ckpt" \
  --model hjepa_l4 fourroom_l4 "$base/fourroom_hjepa_l4/seed42/fourroom_hjepa_l4_object.ckpt"
```

结果写入 `outputs/<run-name>/`：WAM 单独评估看 `metrics.yaml`，录像在 `videos/`；成本对比看 `summary.csv`，明细见 `closed_loop.csv` 和 `replay.csv`。

### 测试结果

正式对比在 RTX 4060 Laptop GPU 上完成：50 个固定任务，按目标图像对应的末帧位置 `goal_xy` 判成功，首次成功即停，最多 210 步。重规划延迟来自 10 段固定轨迹重放，不执行仿真。

| 模型 | 成功率 | 总推理时间（s） | 推理时间/步（ms） | 闭环总耗时（s） | 重规划 P50 / P95（ms） |
| --- | ---: | ---: | ---: | ---: | ---: |
| WAM | 34%（17/50） | 353.5 | 38.0 | 474.7 | 25.8 / 30.1 |
| LeWM | 40%（20/50） | 6042.2 | 622.2 | 6158.7 | 6226.2 / 12343.4 |
| H-JEPA L4 | 86%（43/50） | 2391.7 | 254.1 | 2503.3 | 2814.2 / 3317.3 |

推理时间含观测预处理、规划或读取缓存动作，不含仿真；每步耗时按实际执行步数平均，闭环总耗时另含环境恢复和仿真。WAM 每步推理耗时约为 H-JEPA L4 的 **1/6.7**，但成功率低 **52 个百分点**，不是同等成功率下的加速。该表的成功判定与上文使用原 `goal_state` 的解耦实验不同。