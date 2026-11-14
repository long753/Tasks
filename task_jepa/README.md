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

## 环境与安装

使用 conda 环境 `hjepa`，Python 3.11。从项目根目录运行：

```bash
conda activate hjepa
python -m pip install --no-deps -e .
```

上述命令用于已有依赖的 `hjepa` 环境，不安装依赖。新环境使用 Python 3.11，并安装训练与仿真依赖：

```bash
python -m pip install -e ".[train,env]"
```

## 训练与评估

在 `task_jepa/` 下运行：

```bash
python scripts/train/train.py --config-name ant_hjepa_l2 \
  --datasets ./datasets --models ./models --outputs ./outputs

python scripts/eval/eval.py --config-name ant_l2 \
  --datasets ./datasets --models ./models --outputs ./outputs \
  policy=ant/ant_hjepa_l2/seed42/ant_hjepa_l2_object.ckpt

python scripts/eval/probing.py --config-name ant \
  --datasets ./datasets --models ./models --outputs ./outputs \
  policy=ant/ant_hjepa_l2/seed42/ant_hjepa_l2_object.ckpt
```

`--datasets`、`--models`、`--outputs` 默认指向项目根目录下的同名目录；相对路径以项目根目录解析，不受启动目录影响。Hydra 入口也接受 `datasets=... models=... outputs=...`。不再读取 `HJEPA_HOME` 或回退到 `~/.stable_worldmodel`。

训练与 probing 需要先准备匹配配置的 HDF5 数据；规划评估需要对应模型检查点和评估轨迹。独立 probing 还需要匹配的训练配置。仓库不包含这些数据和模型产物。