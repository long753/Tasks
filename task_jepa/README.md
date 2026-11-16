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

根目录 `models/` 是模型产物目录；网络结构源码位于 `src/h_jepa/models/`。变更记录位于 `CHANGLOG.md`。本项目以普通目录纳入主仓库，克隆主仓库即可获得源码，不需要初始化子模块。`datasets/`、`models/` 和 `outputs/` 不上传，需要在本地准备。

## Setup

该项目使用 conda 进行环境管理。 `hjepa`，Python 3.11。从项目根目录运行：

```bash
git clone https://github.com/long753/Tasks/tree/master/task_jepa
cd task_jepa
conda create -n hjepa python=3.11.13
conda activate hjepa
python -m pip install -e ".[train,env]"
```


## 训练与评估

训练基线：
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

接下来，我们来实现 action 和 未来状态的联合生成训练：

在当前项目中，我们使用 h-jepa 的方法实现了多层 predictor 的 end2end 训练，这个 predictor（也就是所谓的 world model）无法直接生成 action。因此，在 eval 阶段，我们使用了一种基于 mpc 的方法实现了 planning。但是该方式生成 action 需要进行大量无关计算，为此我设计了下面的改进算法。

我计划新增一个 action head，这个 action head 可以接收当前的 z_t 和 predictor 预测的 z_{t+1} 然后直接生成真实的 action。具体流程如下：模型使用 encoder 将当前状态编码为 z_t，不对动作进行编码；随后，将 z_t 输入给 predictor 预测 z'_{t+1}，并用这个 z'_{t+1} 与真实的后续观测编码得到的 z_{t+1} 做 MSE Loss；然后将 z'_{t+1} 和 z_t 输入给 action head,让他直接输出真实的 action，用这个 action 和真实的 action 做 mse loss。此外，我们依旧要使用原始 JEPA 中的 R。通过这样的方式，训练一个 action head，在推理的时候，我们就先使用 encoder 编码状态得到 z_t 然后使用 predictor 预测 z'_{t+1}，将 z_t 和 z'_{t+1} 输入给 action head 得到 action；机器人执行 action 后再次获得 z_{t+1} 循环往复下去即可。

实现方式：
1. 在 jepa-wam 中实现具体的代码逻辑
2. 在 scripts/train 中复用 train.py 实现模型训练
3. 在 scripts/eval 中复用 eval_once.py 和 eval.py 实现模型评估

请你仔细想一下实现细节，是否还有问题？如果没有问题，请你告诉我你计划如何实现这个任务？

