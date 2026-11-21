# 项目说明

本项目研究 Hierarchical JEPA 世界模型及其自适应时间层级扩展。H-JEPA、HWM 和 LeWM 共用模型实现；LeWM 是单层配置，不单独复制算法包。

## 目录职责

- `src/h_jepa/`：层级 JEPA 编码、动作表征、训练损失、规划与评估实现。
- `src/jepa_wam/`：独立的单层任务条件 latent flow、动作 flow、数据采样及闭环策略。
- `src/stable_worldmodel/`：HDF5 数据集、仿真环境、策略封装及梯度求解器。
- `configs/train/`、`configs/eval/`、`configs/probing/`：算法及实验配置，不保存机器专属存储路径。
- `configs/data/`：数据采集配置。
- `scripts/train/`、`scripts/eval/`：薄启动入口与批量实验脚本。
- `scripts/train_wam.py`：WAM 单卡或 DDP 训练入口，脚本内处理命令参数、训练、验证和保存。
- `scripts/data/`：数据采集、转换、划分及评估任务生成工具。
- `datasets/`：训练、验证和 probing 的 HDF5 文件直接放在根目录；`eval_trajs/`、`droid/`、`.cache/` 分别保存固定评估任务、DROID 辅助文件和下载缓存。
- `models/`：模型检查点、断点续训状态、训练配置和 `normalizer.pt`。
- `outputs/`：日志、评估指标、视频、图表和运行缓存。

根目录 `models/` 是模型产物目录；网络结构源码位于 `src/h_jepa/models/` 和 `src/jepa_wam/`。`docs/` 不保留，`README.md` 不由本次迁移改动。变更记录位于 `CHANGELOG.md`。

## 数据目录布局

所有环境共用同一个数据根目录，文件名保留环境前缀，不再在训练文件外套 `antmaze/`、`fourroom/` 子目录。现有配置通过 `--datasets ./datasets` 查找 `datasets/<数据集名称>.h5`，训练、验证、probing 和训练结束后的自动评估使用同一数据根目录。

AntMaze 的四份数据为 `visual_antmaze_medium_explore_stitch_train.h5`、`visual_antmaze_medium_stitch_val.h5`、`visual_antmaze_medium_probing_train.h5`、`visual_antmaze_medium_probing_eval.h5`；FourRoom 的四份数据为 `fourroom_tp35_d1.h5`、`fourroom_tp35_d1_val.h5`、`fourroom_tp35_d1_probing.h5`、`fourroom_tp35_d1_probing_val.h5`。固定评估任务仍位于 `datasets/eval_trajs/<环境>/`，DROID 辅助文件仍位于 `datasets/droid/`。

下载时将数据根目录作为本地目标；已有完整文件不要重复下载。`datasets/SHA256SUMS` 保留下载方的完整校验清单，其中可能包含尚未下载的数据。`datasets/antmaze/` 内保留原下载仓库的 `.git` 和校验清单，不作为训练数据目录；不要把它们当作冗余数据直接删除。下载缓存保留在 `.cache/`，不参与模型训练。

## 环境与安装

使用 conda 环境 `hjepa`，Python 3.11。从项目根目录运行：

```bash
conda activate hjepa
python -m pip install --no-deps -e .
```

### AutoDL 远程环境

远程 H-JEPA 训练代码位于 `/root/autodl-tmp/task_jepa`，独立 Python 3.11 环境位于 `/root/autodl-tmp/conda/envs/hjepa`。环境和代码放在数据盘，不替换服务器自带的 base 环境。已有 FourRoom 训练、验证、probing 四份 HDF5 位于 `/root/autodl-tmp`，不复制到项目目录，也不重复上传；运行时显式传入 `--datasets /root/autodl-tmp`。本次仅上传 `src/h_jepa`、`src/stable_worldmodel`、脚本、配置和安装说明，不上传本地数据、模型、输出或无关算法源码。

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda/envs/hjepa
cd /root/autodl-tmp/task_jepa

# 只展开配置，不训练。
python scripts/train/train.py --config fourroom_hjepa_l4 \
  --training-mode layerwise --datasets /root/autodl-tmp --cfg job --resolve
```

远程环境的 pip 使用清华镜像，版本约束保存在远程项目的 `requirements.remote.txt`。PyTorch、torchvision、torchaudio 保留项目指定版本；CUDA 运行库由镜像构建的依赖解析，不混用本地 cu128 的 NVIDIA 版本约束。无卡阶段只安装环境、导入模块、展开配置和读取数据，不启动训练；启用 GPU 后必须再检查驱动、`torch.cuda.is_available()` 和实际显存，之后再选择训练 batch 和预算。

当前远程安装为 Python 3.11.18、PyTorch 2.7.1（cu126 / CUDA 12.6）、torchvision 0.22.1、torchaudio 2.7.1、stable-pretraining 0.1.7 和 Lightning 2.6.6。GPU 尚未分配时 `torch.cuda.is_available()` 为 false 属于预期，不代表已验证 GPU 兼容。若之后选用 RTX 5090 等 Blackwell 显卡，需改用 cu128 构建；[PyTorch 2.7 的官方说明](https://pytorch.org/blog/pytorch-2-7/) 将该架构支持与 CUDA 12.8 轮子配套。

清华镜像不提供 `tfp-nightly`，已仅对 `tfp-nightly==0.26.0.dev20261113` 使用官方 PyPI 补齐原版本，其余依赖走清华镜像。pip 缓存在当前用户拥有的 `/root/autodl-tmp/conda/pip-cache`，不改已有数据缓存的所有权。

上游 decord 0.6.0 轮子的文件名声明 py3-none，但内部 WHEEL 错写 cp36-cp36m，RECORD 的 top_level.txt 哈希也不符。远程保留了校验官方 PyPI SHA-256 后重建的 `/root/autodl-tmp/conda/wheels/repaired/decord-0.6.0-py3-none-manylinux2010_x86_64.whl`：只修正 WHEEL/RECORD，所有 Python 源码和原生库逐项哈希不变；已通过 pip check 及真实 CPU 视频解码。重建环境时应安装该修复轮子，不能通过关闭依赖检查掩盖元数据问题。

### zju3090 数据盘与独立容器

`ssh zju3090` 登录到共享账号 star。宿主为 Ubuntu 18.04 / glibc 2.27，项目指定的 PyTorch 2.7.1 Linux 轮子要求 glibc 2.28 及以上，因此环境必须在 Ubuntu 22.04 容器中运行，不能在宿主直接激活后运行 Python。长期使用的独立容器名为 `chenlong`，复用服务器已有镜像，不修改共享的 Anaconda、宿主系统库或 Docker 配置。

代码当前由用户移动到 `/data/hdd6t/chenlong/home/task_jepa`，在容器里可写作 `~/task_jepa`；不恢复已移除的 workspace 布局。独立 Conda 管理器位于 `/data/hdd6t/chenlong/conda/base`，hjepa 环境位于 `/data/hdd6t/chenlong/conda/envs/hjepa`。容器将数据盘目录同路径绑定，HOME、Conda 包缓存、pip/Hugging Face 缓存及 TMPDIR 全部指向数据盘，不占用宿主 `/home`。容器以 UID/GID 1000（star）运行，根文件系统只读，关闭 Docker 文件日志；默认网桥缺失，因此只给这个容器使用 host 网络，不启动额外网络服务。

直接进入容器即可，默认工作目录固定为 Home（`/data/hdd6t/chenlong/home`），提示符显示 `~`，标准 `.bashrc` 初始化独立 Conda base 环境。原 enter_hjepa.sh 已被移除，不恢复该脚本，也不恢复旧 workspace 路径：

```bash
ssh zju3090
docker exec -it chenlong bash
conda activate hjepa
cd ~/task_jepa
```

容器 HOME 中的标准 `.bashrc` 提供 ll、la、l、..、... 等别名；刷新 `.bashrc` 不会切换已激活的环境或更改当前目录。ll 是 ls 的别名，不是缺少一个需要安装的程序。用 `exit` 退出当前 shell，不会停止或删除容器。

不能通过 apt 往只读 `/usr` 等系统目录安装工具，且当前 UID 1000 不是 root；不要通过修改共享宿主环境或添加 sudo 来绕过存储限制。独立 Conda base 和 hjepa 前缀在数据盘上可写。需要安装 Vim 时，可在容器中执行 `conda install -n base --override-channels -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge vim`，安装后在 base 使用，其他环境可用 `conda run -n base --no-capture-output vim <文件>`。这只是安装方式说明，本次未执行任何工具下载；若实际失败的是 Conda 安装而不是 apt，应按具体命令和报错另行排查。

容器现按用户要求以 `--gpus all` 映射全部 8 张 RTX 3090，已核对容器内可见 UUID 集合与宿主完全一致；这只是可见性，不代表独占或获得所有卡的使用许可。启动任何训练前必须先确认分配卡号，并在 Python 启动前设定 CUDA_VISIBLE_DEVICES；单卡显式指定 trainer.devices=1，多卡则让 trainer.devices 与 torchrun 的进程数及所选卡数一致，避免默认 auto 使用全部可见卡。例如确认获分配宿主 GPU 6 后可设 CUDA_VISIBLE_DEVICES=6，此时该进程中的 cuda:0 对应宿主 GPU 6。当前驱动为 530.41.03，依赖选择 cu118 构建，不混用 AutoDL 的 cu126/cu128 库。

容器重启策略为 unless-stopped。数据盘保留 `setup/create_container.sh`、可选的 `setup/hjepa.bashrc`、HOME 下的 `.bashrc` 和容器专用账户映射；可选配置只激活 hjepa，不再强制跳转项目目录。主动删除容器后可重建，不必重装数据盘上的 Conda；这些文件不修改宿主账户。容器名称冲突时不会自动覆盖其他容器。现已开放全部卡，切换获分配的训练卡通常只需在新 Python 进程启动前改变 CUDA_VISIBLE_DEVICES，不再为每次选卡重建容器；只有修改容器级设备策略时才需要重建。不得通过停止他人任务、重置 GPU 或重启共享 Docker 服务来腾卡。

此次只上传代码，不上传数据、模型或旧输出。用户现已将 FourRoom 四份 HDF5 放回 `~/task_jepa/datasets/` 根目录，不再嵌套 fourroom/；可省略 --datasets，或显式使用 `--datasets ./datasets`，不再使用旧的空 `/data/hdd6t/chenlong/datasets`。`datasets/eval_trajs/fourroom/*.pt` 仍是固定评估任务，不能作为训练 HDF5 输入。

用户已自行通过阿里源补齐 hjepa 依赖。已核对 Python 3.11.18、PyTorch 2.7.1+cu118 / torchvision 0.22.1+cu118 / torchaudio 2.7.1+cu118、stable-pretraining 0.1.7、Lightning 2.6.6；pip check、核心训练模块导入及 CPU 张量运算通过。Conda 个人镜像配置保持原样，不自动恢复下载或把用户选择的源切回清华。

经用户明确确认 GPU 6、7 已分配并允许两步诊断，已在这两张卡完成真实 FourRoom 四层 SG 双卡冒烟：训练前验证、两次 optimizer 更新、验证以及模型/完整续训状态保存均通过，进程正常退出；未启动完整训练或操作其他显卡。此结果只证明受限路径可运行，不代表性能、收敛或全部评估环境已验证。

双卡使用外部 torchrun 启动；当前直接 `python train.py trainer.devices=2` 的 Lightning/Hydra 子进程参数兼容问题尚未修复。以下示例为正式训练命令，不会自动执行：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/train/train.py --config fourroom_hjepa_l4 --training-mode sg \
  --run-name fourroom_l4_sg_2gpu_fixed --datasets ./datasets \
  --models ./models --outputs ./outputs \
  trainer.devices=2 trainer.max_epochs=7 loader.batch_size=16 \
  num_workers=0 loader.persistent_workers=false loader.prefetch_factor=null \
  planning_eval.enabled=false final_probing_decoding_eval.enabled=false
```

当前容器 `/dev/shm` 为 1 GiB，双卡多 worker 配合默认 prefetch_factor=8 曾触发 DataLoader Bus error；暂用上述零 worker 配置，代价是数据加载可能较慢。重建容器扩大共享内存需要另外确认，不自动重建。ViT 工厂默认 use_mask_token=false：当前训练不提供 bool_masked_pos，启用的 mask_token 永远没有梯度，会使默认 DDP 在第二个 batch 报 unused parameters；修复不启用 find_unused_parameters 来隐藏死参数，也不改变 SG 截断。已有完整模型对象未迁移；新训练使用新运行名，不承诺含 mask_token 旧结构的 Trainer 检查点可以直接续训。

### FourRoom 三组单轮对照脚本

`scripts/train/train_fourroom_ep1.sh` 在项目内依次训练 SG-L3、E2E-L3、E2E-L4，均从现有共享训练入口启动，不复制算法或另加 Python 训练入口。须在 chenlong 容器激活 hjepa 后执行；要求显式设置两张已分配显卡，未指定时会拒绝启动，不默认占用所有卡：

```bash
cd ~/task_jepa
conda activate hjepa
CUDA_VISIBLE_DEVICES=6,7 bash scripts/train/train_fourroom_ep1.sh \
  --run-name fourroom_ep1_compare
```

`--run-name` 在这里是前缀，三组运行名分别为 `<前缀>/sg_l3_seed42`、`<前缀>/e2e_l3_seed42`、`<前缀>/e2e_l4_seed42`。脚本沿用每卡 batch 32、全局 batch 64、seed 42、bf16、零 worker、prefetch_factor=null、persistent_workers=false 和关闭训练后规划/probing；各层学习率、损失和模型结构沿用对应配置，固定 trainer.max_epochs=1。默认使用项目根目录下 datasets、models、outputs，可透传同名存储参数。相同前缀重启沿用原训练入口的完整状态恢复；新实验使用新前缀。任一模型失败即停止后续队列，不自动重试、重建容器或清理其他进程。

在命令末尾加 `--cfg job --resolve` 只展开三组配置。已在远程 hjepa 中通过实际 Bash、torchrun 和每组两个进程验证模式、层数、预算、批量及数据路径；未执行这三组 GPU 前向或训练，尤其不保证 E2E 与 SG 显存峰值相同。若只有一轮预算，应让任务正常结束，不以 Ctrl+C 代替轮次设置；此前多卡 Ctrl+C 曾遗留 rank 进程，重新启动前需确认原进程已退出。

用户已选择三组各自完整的单轮学习率计划，不修改训练源码：它不同于旧 SG-L4 的“计划七轮、仅运行第一轮”，因此旧 SG-L4 只能作为参考，不声称严格单变量对照。L3 与 L4 的采样时间窗口、有效样本数和单轮计算量也不同。当前 E2E/SG 入口在模型构建后由 SPT Manager 设置全局种子；脚本传入 seed=42，但本次未修改初始化顺序，不承诺各组具有相同初始权重。

## 命令参数与实验目录

- `--config`：选择对应 `configs/train/`、`configs/eval/` 或 `configs/probing/` 中的实验配置，不写 `.yaml`。
- `--checkpoint`：评估和 probing 加载的具体检查点文件。相对路径以项目根目录解析，不再与模型输出目录拼接，也不猜测最新检查点。
- `--datasets`：数据根目录，默认项目的 `datasets/`。
- `--models`：训练模型和解码器的保存根目录，默认项目的 `models/`。单次规划评估不接受此参数，只使用 `--checkpoint` 选择输入。
- `--outputs`：日志、指标、图片和运行缓存的根目录，默认项目的 `outputs/`。
- `--run-name`：本次实验名；省略时由配置名、时间戳和随机标识自动生成。可使用相对层级名称，但不能使用绝对路径或 `..`。

相对存储路径以项目根目录解析，不依赖当前工作目录。训练产物保存到 `models/<run-name>/`，结果保存到 `outputs/<run-name>/`；独立 probing 的解码器保存到 `models/<run-name>/decoders/`。训练结束时自动评估的结果仍放在该训练运行目录内的评估子目录。

不再使用旧的配置选择参数、`policy=...`、`output.dir=...`、`output_model_name=...` 或 `subdir=...`。算法超参数仍可使用 `seed=42`、`level2.stride=3` 等覆盖；可用 `--cfg job --resolve` 只检查配置，不启动任务。存储位置只通过普通命令参数输入，不新增路径 YAML，也不读取 `HJEPA_HOME`。数据采集配置中的 `policy` 仍表示专家策略，不是模型检查点输入。

```bash
python scripts/train/train.py --config ant_hjepa_l2 --run-name ant_l2_exp1 \
  --datasets ./datasets --models ./models --outputs ./outputs

python scripts/eval/eval.py --config ant_l2 --run-name ant_l2_eval \
  --checkpoint ./models/h_jepa/ant/ant_hjepa_l2/seed42/ant_hjepa_l2_object.ckpt \
  --datasets ./datasets --outputs ./outputs

python scripts/eval/probing.py --config ant --run-name ant_l2_probe \
  --checkpoint ./models/ant_l2_exp1/ant_hjepa_l2_object.ckpt \
  --datasets ./datasets --models ./models --outputs ./outputs
```

模型文件名由配置内的 `model_name` 决定，实验目录只由 `--run-name` 决定。下载模型可能存放在 `models/h_jepa/` 等子目录；评估直接提供其实际文件路径。旧模型旁仍需 `normalizer.pt`；独立 probing 还需要匹配的 `config.yaml` 和 probing 数据。旧评估配置仅作历史记录，运行时使用当前配置和新参数。

下载的旧完整模型可能仍引用 `models.*` Python 类路径。先运行一次离线迁移，再使用原检查点路径评估：

```bash
python scripts/train/migrate_checkpoint.py \
  --checkpoint ./models/h_jepa/pusht/pusht_hjepa_l2/seed42/pusht_hjepa_l2_object.ckpt
```

工具处理 Torch ZIP / pickle 协议 2，只修改 `GLOBAL` 类引用为 `h_jepa.models.*`，不反序列化模型，不修改普通字符串或张量数据。所有非 pickle 归档成员经 SHA-256 校验一致后，原文件保留为相邻的 `.ckpt.legacy`，新归档原子替换检查点；不会覆盖已有备份。已迁移文件再次执行不变。迁移只处理指定文件，不向评估加载器添加旧模块别名。

批量训练的 `--run-name` 是前缀，脚本自动追加环境、模型和 seed；批量评估使用 `--checkpoint-root` 指定输入检查点根目录，而不是把它与单次评估的输出参数混用。已移除的测试及集群脚本不在本次接口调整中恢复。

`scripts/test/test_variation_reset.py` 是本次新增的环境变体恢复回归测试，不恢复此前删除的其他测试。可在 `hjepa` 中运行 `python -m unittest discover -s scripts/test -p test_variation_reset.py -v`。

## 单任务可视化调试

`scripts/eval/eval_once.py` 从固定评估任务文件中选择一个任务，复用正式评估的模型加载、状态恢复和闭环规划逻辑，不训练世界模型。`--task-index` 默认为 0，表示任务文件内的下标，不是原始数据集的 episode 编号；`--video-fps` 默认为 10，仅控制播放速度。

```bash
python scripts/eval/eval_once.py --config ant_l2 --run-name debug_ant_l2 \
  --checkpoint ./models/h_jepa/ant/ant_hjepa_l2/seed42/ant_hjepa_l2_object.ckpt \
  --task-index 0 --video-fps 10 --datasets ./datasets --outputs ./outputs
```

结果保存在 `outputs/debug_ant_l2/`：`videos/task_000.mp4` 并排显示真实环境与目标画面，标注步数及状态；同目录保留 `eval_config.yaml`、`metrics.yaml`、`planning_compute.json`、`chunks/` 和 `hydra/`。视频从恢复后的实际初始状态开始，在首次成功或预算耗尽时结束；评估本身仍执行原有预算。指标还记录任务下标、视频相对路径、帧率与帧数。

单任务入口固定只运行一个环境和一个任务，保留配置中的动作优化迭代数和执行预算；默认逐任务规划随机种子与正式评估相同。即使指定的运行目录已有分块缓存，调试入口也会重新执行并生成视频。正常 `eval.py` 保留缓存恢复行为且不录制视频。调试耗时包含渲染和视频写入，不用于正式性能比较。

该入口用于仿真环境，需要配置中的 `load_eval_trajs_path` 指向已生成的任务文件。DROID 的离线 clip 评估不执行真实机器人动作，继续使用 `eval.py`，不能用离线输入视频冒充执行录像。

视频编码需要 `imageio[ffmpeg]`（已包含在 `env` 可选依赖中）；已有环境可用 `python -m pip install 'imageio[ffmpeg]'` 补齐。

## 四层检查点的前缀评估

在本地 `hjepa` 环境运行。传入一个完整四层检查点，同一批固定任务分别用 L1、L1–L2、L1–L3 评估；不再评估完整 L4，不重新训练或改写检查点。

```bash
python scripts/eval/eval.py --config fourroom_l4 \
  --checkpoint ./models/h_jepa/fourroom/fourroom_hjepa_l4/seed42/fourroom_hjepa_l4_object.ckpt \
  --run-name fourroom_prefix --prefix-levels 1,2,3
```

评估复用同一配置的任务、seed、环境步数预算及保留层的求解器参数。被截断的上层不参与编码、求解或代价计算。不要用 `fourroom_flat` 或 `*_project` 代替这组实验：它们改变规划参数或代价空间。

`outputs/<run-name>/prefix_l1/`、`prefix_l2/`、`prefix_l3/` 各有独立的逐任务结果、规划记录及分块缓存。`prefix_summary.csv` 汇总成功率、成功任务平均步数、累积求解器耗时及总评估耗时；没有成功任务时平均步数留空。使用相同评估配置和未改动的输入文件可恢复分块；更换模型、任务文件或参数时使用新运行名。对比逐层训练与端到端训练时，应在相同截断深度和同一批任务上比较。

## FourRoom 单层 WAM

使用 `scripts/train_wam.py`，在 `hjepa` 环境中直接训练，不调用 H-JEPA 的训练或评估入口。`--encoder-weights` 接收同环境 LeWM 的 ViT `state_dict`，不能传旧完整模型对象。初始化文件是 `models/wam_init/fourroom_lewm_vit.pt`，取自 `jepa-world-models/h-jepa` 的 FourRoom LeWM seed42；仅导出 ViT，略去未使用的 mask token，不修改旧检查点。

在 `chenlong` 容器确认获分配的 GPU 后，从项目根目录运行。以下双卡配置跑两个完整 epoch；`--batch-size` 为每卡大小，全局 batch 为 64：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=6,7 \
python -m torch.distributed.run --standalone --nproc_per_node=2 \
  scripts/train_wam.py \
  --encoder-weights models/wam_init/fourroom_lewm_vit.pt \
  --run-name fourroom_wam_h5_ep2_io \
  --horizon 5 --goal-max 75 --teacher-fraction 0.7 --ema-decay 0.99 \
  --epochs 2 --batch-size 32 --workers 2
```

单卡将启动器换成 `python scripts/train_wam.py`，且只暴露一张卡。双卡同时训练和验证，SIGReg 以全局 batch 为统计口径，仅 rank 0 写结果。按当前数据量，双卡两轮是 46,406 次更新；其中 32,485 次使用 EMA 子目标。`--workers 2` 是每卡两个数据 worker；与逐帧读取优化配合，在 GPU 6、7 短测达约 8 次更新/秒，峰值已分配显存约 1.8 GiB/卡。两轮按此速率约 1 小时 37 分（另计验证），不是整轮实测。WAM 的 `last.pt` 是模型权重，不是可续训的优化器检查点。

默认图像加 `proprio`，`--image-only` 可关闭状态输入。每个样本的短程动作与未来帧来自同条轨迹，最终任务用后续真实到达帧重标注。前 70% 优化器更新给 Action Expert 真实未来帧的 EMA 编码；之后使用同一样本带噪 latent 的世界模型终点估计，动作 loss 经此回传。完整训练不传 `--max-steps` 或 `--val-batches`。训练日志及有效切换步数在 `outputs/<run-name>/` 的 `metrics.csv` 与 `config.json`，每轮结束保存 `models/<run-name>/last.pt`。使用新运行名，不覆盖旧结果。

部署时用 `jepa_wam.policy.WAMPolicy.from_checkpoint()` 加载，供 `stable_worldmodel.World` 使用：原有 `goal` 及 `goal_proprio` 作为固定任务输入，两条 ODE 从独立噪声采样，默认执行两步后重规划。训练期间测的是带监督的 validation loss，不等价于纯噪声采样的固定任务成功率。

### WAM 闭环评估

在本地 `hjepa` 中运行，直接加载 WAM 的 `last.pt`，不需要另传 LeWM 初始化权重或 `normalizer.pt`：

```bash
checkpoint=models/fourroom_wam_h5_ep5_b64_gpu6/last.pt
python scripts/eval/eval_wam_once.py --checkpoint "$checkpoint" \
  --run-name fourroom_wam_debug0 --task-index 0
python scripts/eval/eval_wam.py --checkpoint "$checkpoint" \
  --run-name fourroom_wam_eval50
```

批量入口顺序执行 50 个固定任务，每个一次、预算 210 个环境步；单任务入口按文件下标选一个任务。两者均默认每条流做 8 次 Euler 更新，每执行 2 步重新采样，分别用任务保存的环境种子和 `--seed + task_index` 初始化环境及策略。所有任务从噪声运行两条流，不使用 EMA 教师或真实动作作为监督。

`outputs/<run-name>/` 保存 `eval_config.yaml`、逐任务 `tasks.csv` 和汇总 `metrics.yaml`；只有单任务会写 `videos/task_NNN.mp4`，从实际恢复后的状态录到首次成功或预算耗尽。录像耗时不与批量结果比较。新评估使用新运行名，不覆盖既有结果。

WAM 按固定任务片段最后一帧的 `goal_xy` 判成功，与输入的 `goal` 和 `goal_proprio` 一致。固定文件的原 `goal_state` 可能是采集时更远的目标；旧 H-JEPA 结果如果按那个位置判定，不能直接对比。重评对照时至少统一成功目标（原 `fourroom_l4` 配置可传 `eval.callables.1.args.goal_state.value=goal_xy`）及环境随机种子，再做逐任务比较。

## FourRoom 计算成本对照

在本地 `hjepa` 环境从项目根目录运行。重复 `--model NAME CONFIG CHECKPOINT` 指定有序列表；`wam` 配置加载 WAM 的 `last.pt`，`fourroom_flat` / `fourroom_lN` 等使用相应的现有评估配置和检查点。各模型使用同一张 CUDA 卡，依次加载、预热和测试。

```bash
base=models/h_jepa/fourroom
python scripts/eval/eval_cost.py --mode both --run-name fourroom_cost \
  --model wam wam models/wam_seed42/last.pt \
  --model lewm fourroom_flat "$base/fourroom_lewm/seed42/fourroom_lewm_object.ckpt" \
  --model hjepa_l4 fourroom_l4 "$base/fourroom_hjepa_l4/seed42/fourroom_hjepa_l4_object.ckpt"
```

- `--mode closed-loop`：所有模型用相同环境种子跑 50 个固定任务，按末帧位置 `goal_xy` 判成功，首次成功后立即结束当前任务；失败才跑满 210 步。保留各模型原有重规划节奏。`closed_loop.csv` 记录逐任务成功、首次成功步数、实际环境步数、重规划次数、总推理秒数、每步平均推理毫秒数和闭环总秒数。
- `--mode replay`：默认从 50 个任务中等距选 10 段轨迹，逐帧把同一输入重放给各策略，不执行仿真。CUDA 同步后计动作返回 CPU 的延迟；`replay.csv` 区分首次规划、后续重规划和缓存动作。因为重规划节奏不同，后续重规划的采样帧不一定成对。
- `--mode both` 默认执行两项。用 `--closed-tasks 1 --replay-tasks 1` 做冒烟。`outputs/<run-name>/summary.csv` 每模型一行，含闭环成功数、成功率、成功任务平均步数、总推理时间、每步平均推理时间、闭环总耗时及重放的 P50/P95；没有成功任务时，成功任务平均步数留空。`eval_config.yaml` 记录模型、设备、任务选择、步数上限和早停规则。新运行使用新目录，不录视频或复用旧结果。

推理时间含观测预处理、规划或取缓存动作，以及动作返回 CPU；不含状态恢复、仿真、模型加载和预热。闭环总耗时另含状态恢复和仿真。模型级每步平均推理时间用总推理时间除以实际执行总步数。

## 训练模式与跨层状态 stop-gradient

训练统一使用 `scripts/train/train.py`，通过普通命令参数 `--training-mode e2e|sg|layerwise` 选择模式，省略时默认 `e2e`。`e2e` 保留原端到端行为；`sg` 启用跨层状态 stop-gradient；`layerwise` 是下述贪心逐层训练。三种模式共用模型、数据、优化器和训练装配，不复制算法，也不保留独立 SG 脚本。

```bash
python scripts/train/train.py --config ant_hjepa_l4 --training-mode e2e \
  --run-name ant_l4_e2e --datasets ./datasets --models ./models --outputs ./outputs

python scripts/train/train.py --config ant_hjepa_l4 --training-mode sg \
  --run-name ant_l4_sg --datasets ./datasets --models ./models --outputs ./outputs
```

FourRoom 使用 `--config fourroom_hjepa_l4`。SG 仅 detach 传给高层 encoder 的下层状态特征；fusion 时同时处理像素和 proprio 输入，原始下层输出仍用于本层 loss。四层 loss 仍相加，动作表征链路、预测目标策略、SIGReg、时间窗口和优化器分组不变；低层 encoder 仍由自己的损失训练，不被冻结。

保存的 `models/<run-name>/config.yaml` 在 SG 模式下记录 `training_variant: state_stop_gradient` 和 `detach_lower_state: true`，端到端模式记录 `end_to_end` 和 `false`。`--cfg job --resolve` 也会显示这两项。模式只通过 `--training-mode` 选择，不接受 `training_mode=...`、`training_variant=...` 或 `detach_lower_state=...` 等另一套覆盖写法。不同模式应使用不同 `--run-name`，避免混用续训目录。模型结构和类路径不变，仍使用现有 eval/probing 入口；训练专用截断不影响规划对候选动作的梯度。

梯度回归可在 `hjepa` 中运行 `python -m unittest discover -s scripts/test -p test_state_stop_gradient.py -v`。仅做训练冒烟时必须显式限制步数，例如 `+trainer.max_steps=2 trainer.max_epochs=1 +trainer.limit_train_batches=2 +trainer.limit_val_batches=1 trainer.num_sanity_val_steps=0 +trainer.log_every_n_steps=1`，并使用 `planning_eval.enabled=false final_probing_decoding_eval.enabled=false` 关闭完整后处理；不要把只限制 epoch 当作两步冒烟。

## 贪心逐层训练

```bash
python scripts/train/train.py --config ant_hjepa_l4 --training-mode layerwise \
  --run-name ant_l4_layerwise --datasets ./datasets --models ./models --outputs ./outputs \
  trainer.max_epochs=50
```

依次训练 L1、L2、L3、L4；每阶段只有当前层的完整 JEPA 可训练，包括 encoder、predictor、动作表征和已启用的辅助模块。已训练低层参数被冻结、始终保持 eval，训练前缀在 no_grad 中编码，动作队列等 buffer 不再更新；尚未轮到的高层不执行。每阶段只计算当前层原有损失，不进行最后的联合微调。FourRoom 使用 `--config fourroom_hjepa_l4`，层数由现有配置决定。

`trainer.max_epochs` 和可选 `trainer.max_steps` 都是每层预算，不是整次实验预算。上例是每层最多 50 epoch，四层共最多 200 个阶段 epoch，不自动判断收敛。每阶段重建 optimizer/scheduler，只包含当前层参数，学习率沿用 `levelN.lr`。采样窗口只使用当前层所需的层级前缀，状态/动作时间对齐规则不变；所有阶段共用首次保存的 normalizer。

`src/h_jepa/layerwise_training.py` 负责真实的阶段调度、冻结阶段续训及完成提交，复用 `main_hjepa.py` 的 `run_training`。根配置记录 `training_variant: layerwise`、`detach_lower_state: true`，各阶段配置另记录 `active_level`。阶段产物位于 `models/<run-name>/stages/levelN/`，包括配置、normalizer、SPT 产物及 `lightning_resume/last.ckpt`；日志位于 `outputs/<run-name>/stages/levelN/`。命令级 Hydra 输出仍在根运行目录的 `hydra/`。

阶段完整状态保存成功后才原子提交 `progress.json`；不能把中断时已有的检查点当作完成标志。使用相同参数和 `--run-name` 重启：跳过已完成阶段，恢复未完成阶段的参数、optimizer、scheduler 和循环计数；采样顺序沿用现有恢复规则，不承诺逐 batch 位级复现。恢复要求训练配置一致，更换预算或其他配置需使用新运行名。阶段内的 global_step 独立计数。

全部层完成后才在 `models/<run-name>/` 导出完整 `<model_name>_object.ckpt`，与根 `config.yaml`、`normalizer.pt` 配套，继续使用原 eval/probing 入口。仅在最后阶段结束时执行原配置启用的训练后规划评估/probing，结果位于根输出运行目录，不在低层阶段运行完整模型评估。

在 `hjepa` 中运行 `python -m unittest discover -s scripts/test -p test_state_stop_gradient.py -v` 检查共享真实模型的 SG/逐层冻结、buffer 和动作规划梯度，再运行 `python -m unittest discover -s scripts/test -p test_layerwise_resume.py -v` 检查真实 Trainer 中断恢复。恢复同一阶段时，SPT 会生成新的日志子目录，查看指标应包含该阶段所有 CSV。逐层冒烟沿用上节限制，并指定 `--training-mode layerwise`：每层最多 2 步，四层合计最多 8 步，必须关闭训练后评估及 probing。

## 核心阅读顺序

1. `configs/train/ant_hjepa_l2.yaml`：实验层数、步长和窗口。
2. `src/h_jepa/main_hjepa.py`、`hjepa_utils.py`：训练装配和逐层损失。
3. `src/h_jepa/models/jepa.py`、`hjepa.py`：状态/动作时间窗口及层级编码。
4. `src/h_jepa/hierarchical_solver.py`：自顶向下的规划及跨层代价。

自适应层级需要保持状态与动作边界、持续时间、训练裁剪及规划时间语义一致。

## 代码检查

在 `hjepa` 环境运行：

```bash
# 使用真实临时 HDF5、检查点及新命令入口做针对性的冒烟验证。
ruff check .
ty check
```

不能通过关闭检查规则或排除源码目录掩盖迁移问题。完整训练与性能复现不属于目录重构的冒烟验证。

## 训练显存监控

训练默认不采集 CUDA 分配历史。需要排查 OOM 时，给 `scripts/train/train.py` 或批量脚本追加 `+cuda_memory_monitor=true`；不改变 batch、精度、优化器或训练模式。仍须先确认获分配的卡。以下在 `chenlong` 的 `hjepa` 环境执行，使用新的运行名前缀：

```bash
CUDA_VISIBLE_DEVICES=6,7 bash scripts/train/train_fourroom_ep1.sh \
  --run-name fourroom_ep1_monitored +cuda_memory_monitor=true
```

每个 rank 在 `outputs/<run-name>/cuda_memory/rank-*.jsonl` 写独立、实时刷新的日志；重启不会覆盖旧日志。训练开始后，从另一个终端查看首组：

```bash
tail -F outputs/fourroom_ep1_monitored/sg_l3_seed42/cuda_memory/rank-*.jsonl
```

`l1_images` 是实际送入 L1 的图像数；`peak_mib` 是当步 PyTorch 已分配峰值，`allocated_mib` 是步末仍在使用的分配，`reserved_mib` 包括 PyTorch 缓存，`device_free_mib` 是整张卡的余量。另有各阶段采样、碎片计数和分配重试。首次观测到已分配峰值达显卡容量 75% 时以及 CUDA OOM 时分别保存 `.pickle` 快照，包含有界的分配历史和调用栈。监控会引入额外开销，仅排障时开启；进程若被直接杀死，不保证能生成快照，但已刷新的日志保留。

再次失败时提供运行名，保留对应 `cuda_memory/`、原始异常及 `hydra/`。先分辨步末累积、单步峰值和分配碎片，再决定改哪里。

## TensorBoard 训练曲线

`+tensorboard=true` 为训练与验证 loss（总量和逐层）及学习率写 TensorBoard 事件，不替换 CSV 或 WandB。若同时启用 `+cuda_memory_monitor=true`，各 rank 还会写逐步的 `cuda/` 标量，包含实际图像批量、已分配显存、峰值、预留显存、碎片和设备空闲。JSONL 与 OOM 快照照常保留，事件文件不能替代快照排查。监控默认关闭；没有 TensorBoard 的 `hjepa` 环境可用 `python -m pip install 'tensorboard>=2.16,<3'` 补装。

在 `chenlong` 中激活 `hjepa`，确认卡号分配后，在 `~/task_jepa` 启动训练：

```bash
CUDA_VISIBLE_DEVICES=6,7 bash scripts/train/train_fourroom_ep1.sh \
  --run-name fourroom_ep1_tb +tensorboard=true +cuda_memory_monitor=true
```

在容器的另一个 tmux 窗格启动只监听回环的 TensorBoard：

```bash
tensorboard --logdir outputs/fourroom_ep1_tb --host 127.0.0.1 --port 6006
```

在本地另开终端执行 `ssh -N -L 6006:127.0.0.1:6006 zju3090`，然后访问 `http://127.0.0.1:6006`。容器使用 host 网络，无需公开端口。训练曲线在 `outputs/<run-name>/tensorboard/version_*`，逐卡曲线在同目录的 `cuda/rank-*` 子运行；不同训练尝试使用独立事件文件目录。验证曲线在执行验证后才有数据。

## 待续：FourRoom SG-L3 显存异常

SG-L3 曾两次耗尽显存，获准的双卡诊断已完成但未复现。未修改训练源码、运行完整训练或覆盖原实验。后续 GPU 操作前仍须确认分配和许可，不停止用户或他人的任务。

### 已确认的证据

- 服务器 `zju3090` 的容器 `chenlong`，项目位于 `/data/hdd6t/chenlong/home/task_jepa`，使用独立 Conda `hjepa`。训练数据已平铺到项目 `datasets/`。
- `scripts/train/train_fourroom_ep1.sh` 顺序运行 SG-L3、E2E-L3、E2E-L4，每组一个 epoch，每卡 batch 32、两卡、bf16、workers 0；关闭训练后规划评估和 probing。第一组失败会停止脚本，后两组尚未开始。
- 首次失败运行名为 `fourroom_ep1_compare/sg_l3_seed42`，约第 962 个训练 batch 在 L1 的 ViT MLP 激活处 OOM。报错为申请 110 MiB 时 GPU 仅剩 58 MiB；PyTorch allocated 21.95 GiB、reserved 中未分配部分 1.20 GiB。它不是之前的 DDP unused-parameter 或共享内存错误。
- 用户重跑在 `global_step=757` 再次 OOM：rank 1（宿主 GPU 7）的 L1 ViT MLP `fc2` 前向分配 26 MiB 失败；PyTorch 已分配 21.81 GiB，另预留未分配 1.34 GiB。两次最后写出的指标分别为 step 949 和 749，loss 有限；均未达到 2000 步的恢复检查点间隔。`ChildFailedError` 是退出汇总，不是首因。
- 对照为此前已跑完一个 epoch 的 `fourroom_l4_sg_b32`。已核对两组共有的 L1–L3 网络、图像尺寸、batch、精度与 loader 参数，没有发现 L3 被配置成更大模型。
- CPU 使用真实模型构造与裁剪函数验证：L3 参数量 21,663,942，序列输入 13 帧；L4 参数量 26,851,986，序列输入 25 帧。各抽样 2000 次，实际送入 L1 编码的帧数分别观察到 8–9 和 10–12；每卡 batch 32 对应 ViT 图像批量 256–288 和 320–384。因此不能把本次异常解释成“L3 理所当然比 L4 更耗显存”。
- 已检查层级编码、loss、动作队列和 SPT 手动优化流程：动作队列 detach 且容量受限，loss 日志 detach，未发现明确的无界计算图累积。此结论不等于已排除框架内部保留或 CUDA 后端问题。
- 两次失败、旧 SG-L4 和本次长诊断的包快照及已记录的 CUDA 相关环境变量相同。两组 20 步双卡短测在各自观测到的所有图像批量上实测为 Flash SDPA；SG-L3/SG-L4 已分配显存峰值分别为 7.69/10.35 GiB。不能据此倒推出原失败步使用的 CUDA 内核。
- SG-L3 诊断以每卡 batch 32、bf16 跑完 1000 次更新，覆盖原失败区间。两卡预热后步末 allocated 约 0.615 GiB，峰值 7.69 GiB，reserved 最高约 10.15 GiB，分配重试及 OOM 均为 0。诊断用隔离目录和监测钩子，并将常规验证频率设为每两轮以跳过提前停止时的验证；不是原失败进程的逐步重放。

### 后续排查

1. 不把千步通过视为修复。现有数据不支持本次诊断中存在持续泄漏或不断恶化的碎片，但无法解释原失败步的 21–22 GiB 活跃分配。
2. 下次获准训练时，启用上节监控，在独立运行中保留每个 rank 的日志以及峰值／OOM 快照；结合当步图像批量和分配调用栈定位大块显存的来源。
3. 定位后再改张量生命周期、缓冲或后端设置。不要自动减 batch（会改变 SIGReg 批统计）、启用 unused-parameter 检测或把 `expandable_segments` 当作已验证的修复。

原始日志：`outputs/fourroom_ep1_compare/sg_l3_seed42/spt/runs/20261117/{170942/a2aa68772bd2,174300/467ba905b9a2}/`；旧 SG-L4：`outputs/fourroom_l4_sg_b32/spt/runs/20261117/122111/c2b7770da5fc/`。诊断数据位于服务器 `outputs/fourroom-memdiag.bCGcOe/`，每组的 `rank-*.jsonl` 记录显存时间序列，`l3-extended/console.log` 保存 1000 步诊断日志。L3 短测已完成 20 次更新，退出时的模型序列化错误来自诊断钩子携带文件句柄，仅影响短测导出；移除钩子后，L4 短测和 L3 长诊断均正常退出。
