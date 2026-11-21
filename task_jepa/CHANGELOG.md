# 修改记录

## 2026-11-21：成本评估成功早停与推理计时

- `eval_cost.py` 的闭环任务在首次成功时停止，210 步为上限；旧 H-JEPA 和 WAM 评估入口继续使用原默认行为。逐任务和模型汇总分别报告同步后的总策略推理时间、按实际步数计算的平均推理时间，以及包含仿真的闭环总耗时。
- 本地 `hjepa` 仅做冒烟：WAM 任务 0 在 150 步成功即停，任务 1 失败并执行满 210 步；76 帧重放和汇总口径核对通过。修改文件通过 Ruff 和 ty，未运行正式评估。

## 2026-11-21：FourRoom 计算成本对照

- 新增 `scripts/eval/eval_cost.py`。重复传入 `--model NAME CONFIG CHECKPOINT` 按顺序测试模型，支持固定 210 步闭环与离线观测重放。闭环统一任务种子和末帧目标，重放分别统计首次规划、后续重规划及缓存动作的延迟。
- WAM 评估复用同一套固定任务打包与恢复配置，原命令和默认行为不变。用真实 WAM、LeWM、H-JEPA L4 检查点完成单任务闭环与 76 帧重放冒烟；更新文件通过 Ruff 和 ty。

## 2026-11-21：FourRoom WAM 闭环评估

- 新增 `scripts/eval/eval_wam.py` 和 `eval_wam_once.py`。主脚本直接处理参数、加载检查点、遍历固定任务并汇总；`src/jepa_wam/evaluation.py` 负责环境执行与录制。单任务输出真实运行视频，批量入口只输出 50 个任务的逐项结果与统计。
- 两个入口使用同一任务预算与种子规则。恢复任务文件中的环境种子，按末帧 `goal_xy` 判成功，避免把采集时更远的 `goal_state` 当成模型看到的终点。旧 H-JEPA 评估行为不变。
- 在本地 `hjepa` 用最终 `last.pt` 实测：50 个任务成功 17 个（34%），成功任务平均 140.24 步。单独重跑任务 0 和 49 与批测结果一致；MP4 分别在成功步和预算终点结束。新增 Python 文件通过 Ruff 和 ty。

## 2026-11-20：WAM FourRoom 数据吞吐

- `GoalClips` 保留每个 worker 的像素数据集句柄和 32 MiB HDF5 chunk cache，用连续切片直接写入目标数组，替代稀疏索引；不改数据格式、采样和动作对齐。
- 本地真实样本逐字节一致，多 worker 读取和 Ruff/ty 通过。zju3090 双卡每卡 batch 32 的短测由原版零 worker 的 2.79 秒/步、原版 4 worker 的 0.67 秒/步，降到新读法配合 2 worker 的 0.126 秒/步。GPU 7 随后被其他进程占用，未测试更大 batch 或启动正式训练。

## 2026-11-20：WAM 双卡过夜训练

- `scripts/train_wam.py` 支持 torchrun 数据并行，各 rank 分片采样，按全局优化器步数进行 70/30 切换；SIGReg 使用跨卡批统计，聚合训练和验证指标，只由 rank 0 写盘。
- 本地 CPU 两进程通过真实 FourRoom 四步联训、验证、检查点重载和拼接 batch 的 SIGReg 梯度对照；单卡 CUDA batch 32 四步冒烟与 Ruff/ty 通过。

## 2026-11-20：任务条件单层 WAM

- 新增 `src/jepa_wam/` 和 `scripts/train_wam.py`。使用 FourRoom LeWM ViT 初始化，联合训练任务条件 latent flow、动作 flow 和在线编码器；EMA 目标只编码真实短程未来，按优化器步数切换教师子目标和可反传的预测子目标。
- 本地 `hjepa` 用真实 FourRoom 数据执行四次 CUDA 更新（3 次 EMA、1 次预测）及一批验证，检查动作对齐、切换后的梯度和 EMA，并重载检查点完成固定任务的五步闭环。新增代码 Ruff/ty 通过；未执行完整训练。

## 2026-11-19：四层检查点前缀评估

- `scripts/eval/eval.py` 支持 `--prefix-levels 1,2,3`，一次加载完整检查点，依次评估 L1、L1–L2、L1–L3；各深度独立缓存，写入 `prefix_summary.csv`。不传参数时保留原评估行为。
- 本地 `hjepa` 用真实 FourRoom 固定任务和四层检查点验证三个深度的实际求解、同配置恢复和配置变更拒绝；变更文件 Ruff/ty 通过。未运行完整评估，未同步服务器。

## 2026-11-18：TensorBoard 训练监控

- 训练支持 `+tensorboard=true`，写入训练与验证 loss、逐层指标和学习率；同时启用 CUDA 监控时，分别写入各 rank 的显存标量。保留现有 CSV、WandB、JSONL 和 OOM 快照。
- 本地和 `zju3090` 的 `hjepa` 已补装 TensorBoard 2.21.0，不升级现有包。在真实 FourRoom 两步训练后核对了事件数值和浏览器标量页面；变更文件 Ruff/ty 检查通过。

## 2026-11-18：训练 CUDA 显存监控

- 统一训练入口支持 `+cuda_memory_monitor=true`。逐 rank 写入显存峰值、步末分配、分配器碎片及 L1 实际图像数；首次达到显卡容量 75% 的已分配峰值和 CUDA OOM 分别保存分配快照。
- 监控默认关闭，不改训练参数。本地 `hjepa` 的真实 FourRoom SG-L3 两步与模型保存、独立 CUDA 高峰/OOM 快照冒烟及变更文件 Ruff/ty 检查通过。

## 2026-11-18：新增 FourRoom 三组单轮训练脚本

- 新增 scripts/train/train_fourroom_ep1.sh，顺序启动 SG-L3、E2E-L3、E2E-L4；每组独立运行目录、每卡 batch 32、两卡、seed 42、bf16、零 worker，固定训练一轮并关闭训练后评估。
- 使用 torchrun 避免现有多卡参数重启兼容问题；要求显式选择两张已分配显卡，沿用普通存储参数与批量运行名前缀，任一失败停止队列。
- 按用户选择使用完整单轮学习率计划，不修改共享训练源码；记录与旧七轮预算 SG-L4 第一轮、不同层数窗口及初始化种子时机的比较限制。
- Bash 语法、帮助与显卡选择保护已验证；在远程 hjepa 中执行实际脚本的 --cfg job --resolve，三组共六个进程的模式、层数和配置均通过。没有启动这三组训练或 GPU 模型计算，也未实测其显存峰值。
- 脚本与说明同步远程；更新用户已扁平化的 datasets 布局说明，README、现有数据、模型和输出不变。

## 2026-11-18：迁移本地 FourRoom 四层 baseline 检查点

- 按用户指定，仅迁移 models/h_jepa/fourroom/fourroom_hjepa_l4/seed42/fourroom_hjepa_l4_object.ckpt 中的 14 处旧 models.* 类引用，修复加载时的 ModuleNotFoundError。
- 原文件保留为相邻 .ckpt.legacy；逐成员校验 577 个非 pickle 归档成员不变，包括所有张量存储。
- 在本地 Conda hjepa 中通过 CPU 加载，确认模型为 h_jepa.models.hjepa.HJEPA、层数为 4，邻接 normalizer.pt 也可加载；未启动 GPU 规划评估。源码、README 和其他检查点未修改。

## 2026-11-17：修复真实双卡 SG 的未使用 ViT 参数

- 在用户确认获分配的 GPU 6、7 上做受限诊断，定位 DDP 在 global_step=1 后失败的唯一未使用参数：L1 ViT embeddings.mask_token。当前训练未提供像素掩码输入，与 SG 的跨层截断无关。
- 将 create_hf_vit 的 use_mask_token 默认值改为 false，避免创建永远不参与当前 loss 的参数；保留显式掩码选项，不通过 find_unused_parameters 掩盖问题，不改损失或优化器。
- 新增真实 ViT 普通图像反向的梯度覆盖回归；hjepa 中 8 项现有/新增梯度回归、两个变更 Python 文件的 ruff 和 ty 定向检查通过。
- 修复已同步远程，使用真实 FourRoom 训练/验证 HDF5、bf16、双卡各 batch 16 完成两步训练和验证，正常保存模型与续训状态并退出；未执行完整训练或训练后评估。
- 更新 torchrun、用户当前数据路径及 1 GiB 共享内存下的零 worker 启动说明；保留逐 rank 诊断日志，清理本次临时模型和暂存文件，不删除用户原有失败运行。README 不变。

## 2026-11-17：清华源配置与手动安装交接

- 恢复 zju3090 的 hjepa 依赖安装，复用既有 cu118 PyTorch 三件套、tfp-nightly 和数据盘缓存；常规依赖使用清华 pip 镜像。
- 新建个人数据盘 CONDARC，将 defaults 和 conda-forge 映射到清华，保留 SSL 校验；未修改共享 Anaconda、宿主配置或其他人的任务。
- 用户要求自行下载后，仅终止本次启动的精确匹配 pip 进程，确认以 143 退出；保留完整缓存，停止后续自动安装、解码修复与环境验证。
- 在 AGENTS.md 记录手动安装及既有 decord 修复命令；其余依赖和 CPU/GPU 验证仍未完成，不声称已可训练。README 不变。

## 2026-11-17：chenlong 开放全部显卡

- 按用户明确要求，将自有容器创建参数从指定 GPU 0 改为 --gpus all，并仅停止、删除及重建带有本项目所有权标签的 chenlong 容器。
- 保留 Home 默认目录、只读根文件系统、UID/GID、账户映射、重启策略及全部数据盘挂载；既有 Python 环境与项目文件保留。适配用户当前的 ~/task_jepa 位置，不恢复旧目录。
- 实测 Docker DeviceRequests 为全部 GPU，容器内可见的 8 张 RTX 3090 UUID 集合与宿主完全一致。未启动训练、恢复下载、重置 GPU、重启 Docker 服务或操作其他容器。
- 更新选卡说明：显卡可见不代表独占；使用前确认分配，启动 Python 前设置 CUDA_VISIBLE_DEVICES，并显式设置 trainer.devices=1，防止默认 auto 使用所有可见卡。依赖与 GPU 前向验证仍处于暂停状态。

## 2026-11-17：默认 Home 与工具安装方式说明

- 按用户确认重建同名 chenlong 容器，将固定默认工作目录设为数据盘上的 Home，保留代码、Conda、缓存及用户当前布局，不下载镜像或依赖。
- 确认用户已将项目移动到 ~/workspace/task_jepa，原 enter_hjepa.sh 已移除；不恢复旧脚本或旧路径。更新创建配置，移除可选 hjepa profile 的强制项目跳转及创建脚本中的失效入口提示。
- 实测普通 docker exec 进入后提示符为 (base) chenlong:~$，可选 hjepa profile 同样从 Home 开始；进入项目后提示符为 ~/workspace/task_jepa，刷新 bashrc 不改变当前目录。
- 核对容器 UID 1000、只读根文件系统，以及数据盘 Conda 前缀可写；说明 apt 的权限/存储限制和通过 Conda 安装 Vim 的方式，未实际安装工具、恢复下载或启动训练。更新本地和远程使用说明，README 不变。

## 2026-11-17：长期容器交互与 workspace 入口调整

- 按用户要求将容器名称改为 chenlong，更新入口和创建配置，适配用户已移动的 /data/hdd6t/chenlong/workspace/task_jepa；不复制或移动代码回旧路径，不保留旧路径别名。
- 先在线重命名保留会话；发现旧默认工作目录无法在线修改后，经用户明确同意重建同名容器，修正默认 workspace、账户映射及 unless-stopped 策略。复用原镜像和数据盘挂载，保留既有 Conda、代码和缓存。
- 在容器标准 HOME/.bashrc 中配置 ll/la/l 等交互别名和 Conda 初始化，直接 docker exec 进入默认 base，项目入口进入 hjepa；刷新配置保持已有环境。未下载安装其他命令行工具。
- 已实际验证直接进入和项目入口两条交互路径、别名执行、workspace 工作目录、hjepa 切换和已有 Python 环境保留；退出 shell 后容器仍运行。更新本地与远程项目的说明，继续保持依赖下载和训练暂停。

## 2026-11-17：zju3090 独立容器部署（按用户要求暂停）

- 代码已上传至 /data/hdd6t/chenlong/task_jepa，独立 Conda 和 Python 3.11.18 hjepa 环境位于数据盘；未上传数据、模型或旧输出。
- 复用既有 Ubuntu 22.04 镜像创建 chenlong-hjepa，只读根文件系统、单卡映射、数据盘 HOME/缓存/临时目录；默认 Docker 网桥缺失，仅对自有容器使用 host 网络，未修改共享 Docker 或 Anaconda。
- 数据盘 enter_hjepa.sh 的交互及普通命令入口已验证，能够激活 hjepa 并进入项目目录。
- 用户要求先停止下载并说明用法，已终止安装任务及容器内残留 pip 安装进程，保留已有文件和缓存。训练依赖、GPU 前向验证与最终收尾未完成，不标记为已完成部署；等待用户明确恢复后继续。

## 2026-11-16：AutoDL 无卡环境部署

- 将 H-JEPA 与 stable_worldmodel 核心源码、脚本、配置和安装说明上传到 /root/autodl-tmp/task_jepa；未上传本地 datasets、models、outputs 或无关算法包。服务器已有四份 FourRoom HDF5 继续位于 /root/autodl-tmp，未移动或重复上传。
- 在数据盘创建独立 conda hjepa 环境 /root/autodl-tmp/conda/envs/hjepa，Python 3.11.18，不替换自带 base 环境。安装 PyTorch 2.7.1 cu126 / torchvision 0.22.1 / torchaudio 2.7.1、stable-pretraining 0.1.7、Lightning 2.6.6 及训练/仿真依赖。
- 按用户建议切换清华 pip 镜像；版本约束保留在远程 requirements.remote.txt。只对清华未收录的 tfp-nightly 从官方 PyPI 补齐原固定版本，CUDA 运行库由镜像的 PyTorch 构建解析，避免混用本地 cu128 约束。缓存放到当前用户拥有的数据盘目录。
- 修复上游 decord 0.6.0 的 WHEEL 标签与 RECORD 元数据：先核对官方 PyPI 归档 SHA-256，再逐文件确认源码和原生库不变，仅重建包元信息；修复轮子保留在数据盘，pip check 通过，真实 CPU 视频解码的最大颜色误差为 1。
- 无卡验证通过核心依赖导入、四份真实 HDF5 的压缩图像读取、项目外目录的 layerwise 配置展开，以及真实验证样本的四层 CPU 编码/预测，输出均有限。没有调用 Trainer.fit 或 optimizer.step，也没有生成模型检查点或训练输出。
- 当前 torch.cuda.is_available() 为 false，符合未启用 GPU 的状态；未声称 GPU 已验证。后续启用 GPU 后再核对驱动、显存和 CUDA 兼容，RTX 5090 等 Blackwell 型号需切换 cu128 构建后再训练。更新 AGENTS.md 的远程激活与数据参数说明，README 不改动。
- 远程另保存 installed-versions.remote.txt 作为实际安装版本清单；安装临时目录已清理，保留版本约束和修复轮子供复现。

## 2026-11-16：贪心逐层训练

- train.py 新增 --training-mode layerwise，保留 e2e 默认值和 sg；不增加独立启动脚本或转发 main 模块。
- 每阶段只优化当前层完整 JEPA，冻结其他层参数并清除旧梯度；低层始终 eval，显式训练前缀 no_grad，冻结动作队列等 buffer。编码、裁剪和数据采样限制到当前层，只计算该层原有损失，不追加联合微调。
- 复用共享训练装配；每层独立 optimizer、scheduler 和训练预算，使用 levelN.lr，并共用 normalizer。新增实质性的阶段调度模块及阶段内完整状态恢复。
- 阶段状态和日志分别放到 models、outputs 运行目录的 stages/levelN 下；先保存完整检查点，再原子提交完成标志。恢复跳过已完成阶段、续训未完成阶段，拒绝不同训练配置混用同一运行目录。
- 全部阶段完成后导出根目录完整模型，复用原训练后评估和 probing；低层阶段不导出未训练完整的模型对象，也不执行完整模型后处理。
- 新增真实模型冻结/梯度及真实 Trainer 中断恢复回归，更新 AGENTS.md；README、已有数据、预训练模型及 cost_probe 不改动。
- 在 conda hjepa 中，8 个相关变更 Python 文件通过针对性的 ty、ruff 检查；7 项共享真实模型 SG/逐层梯度回归及 3 项真实 Trainer 续训回归通过，未关闭规则或修改 cost_probe。
- 从项目外目录展开 layerwise 配置，并使用真实 Ant 训练/验证各 200 帧临时切片完成四层受限冒烟，各层 global_step=2；确认 optimizer、损失和采样窗口按层隔离，非当前层权重及 buffer 逐项不变，四层共用相同 normalizer。
- 在 L2 第 1 步检查点落盘后注入中断，再使用原训练命令恢复：L1 检查点未重写，L2 从第 1 步续至第 2 步，后续阶段完成；中断前未导出不完整的最终模型。
- 完成后的同命令重启不重复训练或重写检查点；最终完整模型和 normalizer 可读取。恢复阶段的多个 SPT 日志 CSV 已合并核验损失有限，临时数据、检查点及输出已清理；未运行完整训练或训练后评估。

## 2026-11-16：统一训练模式入口

- 统一使用 scripts/train/train.py，通过 --training-mode e2e|sg 选择训练模式，默认 e2e，保留原有不带模式参数的训练命令。
- 删除 scripts/train/train_sg.py，不保留转发脚本或别名。两个模式继续使用共享 run_training，模型、损失、梯度截断和优化器实现不变。
- 训练专用参数由公共命令解析层处理，并在 Hydra 配置展开阶段注入已有的 training_variant 和 detach_lower_state，不增加重复的模式字段；拒绝用普通配置覆盖另设模式开关。
- 更新 AGENTS.md 和训练入口示例；README、既有数据及模型不改动。
- conda hjepa 中三个变更 Python 文件通过针对性的 ty、ruff 检查，现有 4 项梯度回归全部通过；从项目外目录验证默认模式与显式 e2e 的配置完全一致，sg 只改变已有的两项模式记录，非法模式及内部字段覆盖均被拒绝。
- 从真实 Ant 训练和验证 HDF5 各截取 200 帧到临时目录，分别通过统一入口完成 e2e、sg 四层两步冒烟；两组均在 global_step=2 停止，四层损失有限，保存模式正确，模型和 normalizer 可加载。
- 两组均显式设置 max_steps=2、max_epochs=1、limit_train_batches=2、limit_val_batches=1，并关闭训练后规划评估及 probing；临时数据、检查点和输出已清理，未启动完整训练。

## 2026-11-16：精简 SG 启动层级

- 将 Hydra 配置装配和 SG 模式选择直接放入 scripts/train/train_sg.py，删除仅转发调用的 src/h_jepa/main_hjepa_sg.py。
- SG 脚本直接调用共享 run_training(cfg, detach_lower_state=True)，保留公共训练装配与模型内的梯度截断；不复制训练流程，不修改原 train.py、梯度策略或评估接口。
- 更新 AGENTS.md 的入口说明；不保留已删除模块的兼容导入或别名。
- 在 conda hjepa 中通过精简入口的 ty、ruff 检查；从项目外目录使用真实 Ant 数据片段完成四层两步冒烟，确认 global_step=2、四层损失有限、保存模式为 state_stop_gradient，模型与 normalizer 可加载。
- 冒烟显式设置 max_steps=2 并关闭训练后规划评估及 probing；临时数据与训练产物已清理，未运行完整训练。

## 2026-11-16：独立的跨层状态 SG 训练入口

- 新增独立入口 `scripts/train/train_sg.py`，固定启用跨层状态 stop-gradient；原 train.py 文件和端到端默认梯度策略保持不变。
- 原训练主体提取为共享装配，两个入口复用数据、模型、优化器和产物保存；保存配置分别记录 state_stop_gradient / end_to_end 模式及实际 detach_lower_state 值。
- 只在训练层级编码时为上层构造浅拷贝输入并 detach 状态特征，fusion 同时截断像素和 proprio 分支；不覆盖供本层 loss 使用的原始 embedding，不冻结 encoder，不修改动作梯度链路或 loss 配方。
- 不修改通用 JEPA.encode、规划器及模型结构，保持原检查点格式和评估接口。新增四层普通/fusion 梯度边界、本层更新、前向等价、动作梯度等价及跨层规划动作梯度回归。
- 验证限定于 conda hjepa 的回归检查和受限训练冒烟，不运行完整训练、训练后规划评估或 probing；README、已有数据和预训练模型、cost_probe 不作修改。

### SG 验证（仅冒烟）

- 在 conda hjepa 中通过 4 项回归测试，覆盖普通和 fusion 四层模型的逐层梯度边界、本层 encoder 更新、前向数值及动作梯度等价，以及 SG 训练后跨层规划对候选动作的有效梯度。
- 六个变更 Python 文件通过针对性的 ty、ruff 检查；未修改或重新检查已约定保留的 cost_probe。
- 从下载的真实 AntMaze、FourRoom 训练和验证文件各截取 200 帧到临时 HDF5，不生成随机替代观测或动作，不修改原文件；分别通过 Ant 原入口、Ant SG、FourRoom SG 的四层训练冒烟。
- 每组均强制 max_steps=2、max_epochs=1、limit_train_batches=2，使用 batch_size=2、单 GPU、32 位精度和逐步日志；训练后规划评估及 probing 关闭，未启动完整训练。
- 各组 checkpoint 的 global_step 均为 2，四层训练损失均有限，保存的配置正确区分 end_to_end / state_stop_gradient，模型对象与 normalizer 可正常读取。当前 stable-pretraining 的 RegistryLogger 在输出运行目录的 spt 子树保存 CSV 指标。
- 冒烟用数据切片、检查点、输出和临时验证脚本均已清理；已有 datasets、预训练模型、README 及 cost_probe 保持不变。


## 2026-11-16：统一 AntMaze 与 FourRoom 数据目录

- 将两个环境的训练、验证和 probing 共 8 个 HDF5 文件移动到 datasets 根目录，保留原文件名，与现有训练、probing 配置的 datasets/<name>.h5 查找约定一致；不修改算法配置或训练源码。
- 移动均为同一文件系统内重命名，逐文件核对 inode 和字节数保持不变；全部 8 个文件通过下载方 SHA256SUMS 校验。
- 两份下载清单内容相同，将 FourRoom 清单移到 datasets/SHA256SUMS；该清单保留尚未下载的 PushT 条目，不代表已经下载该数据。删除移动后为空的 FourRoom 下载目录。
- 保留 datasets/eval_trajs、datasets/droid、datasets/.cache，以及 AntMaze 下载仓库的 .git 和原校验清单，不删除用户仓库元数据。
- 在 hjepa 中从项目外目录展开 AntMaze、FourRoom 的四层训练 CLI 配置，确认数据根目录解析正确；按训练、验证及 probing 配置核对全部 8 份文件的必要列，并通过真实 HDF5Dataset 读取样本。未启动正式训练。
- 更新 AGENTS.md 的数据目录说明，并将变更记录链接指向当前已有的 CHANGELOG.md；不恢复已不存在的 CHANGLOG.md，不修改 README、模型或 cost_probe。

## 2026-11-14：修复主仓库上传与说明

- 将主仓库中错误的 `task_jepa` 子模块指针替换为普通目录，直接跟踪源码、配置、脚本和项目说明，无需子模块初始化。
- 保留数据、模型、实验输出和本地配置的忽略规则，不上传运行产物或本地凭据。
- 更新根目录 README 的 JEPA 项目介绍；补充本项目 README 的依赖安装、训练评估入口和本地数据准备说明。

## 2026-11-14：源码布局与存储迁移

- 将算法实现与公共环境基础设施迁入 `src/h_jepa/`、`src/stable_worldmodel/`，使用可编辑安装解析导入，不依赖切换到算法目录。
- 将实验配置移入根目录 `configs/`，训练、评估、数据工具和集群入口按用途放入 `scripts/`；自动化测试放入 `scripts/test/`。
- 通过命令参数 `--datasets`、`--models`、`--outputs` 分别控制数据、模型及输出目录；Hydra 入口兼容同名 `key=value` 参数。没有新增存储路径 YAML。
- 默认存储位置为项目根目录下的 `datasets/`、`models/`、`outputs/`，相对路径不随启动目录改变。移除 `HJEPA_HOME` 和用户主目录缓存回退。
- 从原外部存储目录迁回 Ant LeWM、两层和三层 H-JEPA 的 seed42 检查点、归一化统计及历史评估结果，分别归入 `models/ant/` 和 `outputs/ant/`。
- 将评估轨迹、DROID 清单和统计移入 `datasets/`，将原算法目录内的 Hydra 日志迁入根目录 `outputs/`。下载缓存随所属产物迁移；外部目录的 `.git` 和 `.vscode` 未改动。
- 旧检查点 pickle 中的 `models.*` 类引用一次性迁为 `h_jepa.models.*`；转换时逐项确认所有非 pickle 归档内容保持一致，包括全部权重数据。不保留旧包名别名或运行时兼容 shim。
- 模型旁继续保存 `normalizer.pt` 和续训状态；日志、评估指标及缓存不再写入模型目录。
- 修正 `.gitignore` 的根目录产物规则，避免忽略网络结构源码。
- `docs/` 当前不存在，重构不再创建；`README.md` 保持不变。项目说明放在 `AGENTS.md`。
- 在 `hjepa` 环境安装更新后的可编辑包；补充真实 HDF5 读取、模型路径隔离和输出位置的回归测试。
- 补齐 OmegaConf 映射、PyTorch 模型、Gym 包装器和第三方仿真接口的类型契约，修复检查发现的规则问题，不新增检查豁免。
- 将专家策略使用的 `distrax` 补入环境依赖，并安装到 `hjepa` 环境。

原外部目录没有训练 HDF5 数据，下载模型也未附原始训练配置；目录迁移不生成伪造的数据或训练元信息。

### 验证

- `hjepa` 环境中 `ty check` 与 `ruff check .` 均通过；未新增规则屏蔽或源码排除。
- 四项真实文件/HDF5 存储回归测试通过，训练、评估、probing 命令均完成配置解析验证。
- 三个迁回的 Ant 检查点均完成加载及有限值预测。
- 从 `/tmp` 启动，使用带空格的自定义目录，完成合成 HDF5 数据两批 GPU 训练和一轮解码评估；确认指标写入 outputs，检查点及解码器写入 models。
- 修复 `stable-pretraining` 缓存目录同时承载日志与模型的耦合：保留 RegistryLogger 的输出目录，单独重定向模型检查点及 Hugging Face 导出目录。
- 单批冒烟配置曾触发第三方调度器除零；最终使用两批验证，未修改调度算法。完整训练、性能复现和实际 SLURM 提交未运行。
- 使用迁回的两层 H-JEPA 和已有 Ant 评估轨迹完成真实 MuJoCo 分层规划冒烟，结果写入指定 outputs；该小预算验证不代表任务成功率复现。
- 旧检查点加载存在 scikit-learn 1.7.2 → 1.9.1 的版本警告；本次未屏蔽警告，也未声称完整复现。
