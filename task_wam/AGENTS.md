# Task WAM

本项目从 `task_jepa` 提取单层 Task-Conditioned JEPA WAM，作为后续算法实验的基线。新工程不导入 H-JEPA 或旧工程源码。

## 目录

- `src/jepa_wam/`：在线与 EMA 编码器、任务条件世界模型、Action Expert、HDF5 采样、闭环策略与录像。
- `src/stable_worldmodel/`：从原项目保留的 FourRoom 环境、封装和固定任务恢复，不包含其他环境和梯度求解器。
- `scripts/`：`train.py`、`eval.py`、`eval_once.py`。训练和评估流程留在入口，算法实现在 `src/`。
- `config/fourroom.yaml`：固定任务、环境参数和成功判定协议；训练超参沿用 `scripts/train.py` 的命令参数。
- `datasets/`、`models/`、`outputs/`：本地数据、权重和运行结果，不纳入 Git。

## 环境与数据

使用 `uv`、Python 3.11。本地锁文件面向 Linux x86-64，PyTorch 2.7.1 使用 CUDA 12.8 构建。通过 `uv sync --locked` 安装，命令由 `uv run --locked` 执行，不激活 Conda。
本机下载依赖前，在执行 `uv` 的同一个 Bash 会话中先运行 `proxy_on`；代理设置不写入 `pyproject.toml` 或锁文件。

默认从项目根目录下的 `datasets/` 查找数据。训练需要 `fourroom_tp35_d1.h5` 和 `fourroom_tp35_d1_val.h5`，评估需要 `eval_trajs/fourroom/fourroom_tp35_cross2_goal75_n50_d1.pt`。迁移没复制数据；以下命令用 `--datasets ../task_jepa/datasets` 复用数据。如果把文件放进本项目的 `datasets/`，可省略此参数。

`models/wam_seed42/last.pt` 是已有 WAM 基线权重；`models/wam_init/fourroom_lewm_vit.pt` 是从 FourRoom LeWM 导出的 ViT `state_dict`，仅作初始化。训练和评估使用不同输入文件，不需要保留 LeWM 整模型。

## 运行入口

从项目根目录执行；训练前先确认选定的 GPU 和训练预算。以下为示例命令，不在迁移过程中执行：

```bash
uv run --locked python scripts/train.py --datasets ../task_jepa/datasets \
  --encoder-weights models/wam_init/fourroom_lewm_vit.pt --run-name fourroom_wam_train

uv run --locked python scripts/eval.py --datasets ../task_jepa/datasets \
  --checkpoint models/wam_seed42/last.pt --run-name fourroom_wam_eval

uv run --locked python scripts/eval_once.py --datasets ../task_jepa/datasets \
  --checkpoint models/wam_seed42/last.pt --run-name fourroom_wam_debug --task-index 0
```

所有相对存储路径从项目根目录解析。训练写 `models/<run-name>/last.pt` 和 `outputs/<run-name>/metrics.csv`；评估写 `metrics.yaml`、`tasks.csv` 和 `eval_config.yaml`，单任务另写 `videos/task_NNN.mp4`。新实验使用新运行名。

闭环协议沿用原 WAM：固定任务按目标图像末帧 `goal_xy` 判成功，每任务 210 步预算，使用任务保存的环境种子和 `--seed + task_index` 的策略种子。单任务录像在首次成功或预算耗尽时结束。
