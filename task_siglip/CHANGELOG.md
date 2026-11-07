# 数据准备调试日志

## 十二份权重统一对比与六次训练 loss 曲线

- 扩展现有 `scripts/test_model.py`，自动遍历当前六个模型目录的 best/last 共12份权重，不评测中间快照，不修改训练代码、模型或 checkpoint。预读元数据确认 `pretrained_model` 为 frozen_dino_bert，其他模型为原 custom/official；报告使用真实 checkpoint epoch，不按目录名称猜测。
- 报告标题按实际权重数生成，不再写死“八模型”。继续沿用完整COCO val2017、Flickr8k test、CIFAR100测试集及同一组COCO验证图文抽样；主指标仍为双向R@1、零样本Top1、图像/文本有效秩及比例和总方差，不新增加权总分。
- 新增 `generate_loss_curves`，只读取各训练目录的 `metrics.jsonl`，每次训练生成一张包含 train/val 的原始损失图，不将 best/last 误当成两次训练。统一图存放于 `outputs/compare_models/loss_curves/<模型目录名>.png`，报告新增六组曲线与原数据来源链接。
- 按用户要求，`siglip-small-coco-50` 固定仅使用第1–49轮；不查找TensorBoard、train.log或云端的第50轮记录，不插值、不补造数据，也不将任何第50轮loss值写入本次图表或汇总。`siglip-tiny-coco-50` 仍保留它实际已有的完整1–50轮。
- Small COCO-50 的 last.pt 元数据实际为epoch49、best.pt为epoch11；Tiny COCO-50的best/last分别为epoch23/50。其余记录范围：pretrained_model、Small COCO-20、Tiny Flickr8k-20、官方Tiny-20均为1–20轮。
- 将缺失的 `outputs/siglip-small-coco-50/loss_curve.png`、`outputs/siglip-tiny-coco-50/loss_curve.png` 补齐；原有四张loss_curve.png不覆盖。六张汇总图样式统一、逐轮数据带标记，Small图的横轴和标题明确截止49轮。不同batch/数据集的原始loss不进行跨模型高低排名。
- `summary.json` 新增 `loss_curves`，保存六组来源、实际epoch、原始train/val数值、epoch限制、汇总图与原图路径；原有指标CSV保持结构。协议记录Small的49轮限制，`comparison.md`仍分任务性能与特征诊断两张表，另附六次训练曲线索引。
- 新增 `--history_root`（默认 outputs），训练日志子目录名对应模型目录名；其余现有参数保持不变。默认仍输出 `outputs/compare_models/`。

### 使用

```bash
uv run python -m scripts.test_model
```

### 验证记录

- Ruff format/check、ty、CLI帮助检查通过。临时边界验证覆盖：Small源数据即使含第50轮也只绘至49轮，Tiny的50轮不会被误删；原始metrics不变，已有原图不被覆盖；报告标题与曲线数据由实际输入生成。临时验证文件已清理。
- 已生成并解码六张真实汇总PNG，实际查看Small和Tiny两张50轮实验曲线。两张补图与对应汇总图逐字节相同；实施前记录的四张原图SHA256保持不变。
- 使用本机CUDA实际完成全部12份权重统一评测，耗时897.47秒；每份均使用COCO的5000图/25014条文本、Flickr8k的1000图/5000条文本及CIFAR100的10000张测试图片，无旧结果缓存复用。
- 完成结果验收：12组(model_dir, weights)唯一且与目标清单完全一致，实际epoch匹配checkpoint元数据，CSV与JSON指标一致，12份分类明细的图像数与Top1一致；特征使用与此前一致的1024组固定样本，24组图像/文本统计均无非有限向量或零向量，有效秩比例公式成立。
- 六组汇总loss数值逐项匹配来源metrics.jsonl，原始训练记录轮次未变；Small曲线数据确为1–49轮、Tiny为1–50轮，报告的六个曲线及来源链接有效。最终文件为 `outputs/compare_models/comparison.md`、`summary.csv`、`summary.json` 及 `loss_curves/` 六张PNG；未修改训练代码、模型结构或任何模型权重。

## 共享评测入口支持冻结 DINO/BERT 权重

- 修复 `scripts/evaluation.py` 无法加载 `models/siglip-dino-bert-mini-frozen-coco/` 的问题。真实 checkpoint 的 backend 为 `frozen_dino_bert`，保存的是 `encoder_configs`、`pretrained_sources`，没有旧 SigLIP checkpoint 的 `config` 字段；共享加载器现在先识别 backend，再按对应格式重建。
- `src/evaluation/common.py` 复用独立脚本的模型工厂，从 checkpoint 的编码器配置构造结构并加载完整 state_dict，包括已训练的两个投影和 logit 参数。使用 checkpoint 指定的 BERT tokenizer、DINO 图像处理器，不误用原 SigLIP 预处理，也不重新加载远端预训练编码器权重。
- 冻结模型采用惰性导入，避免独立脚本复用共享评测统计时产生循环导入。原 custom/official 加载及编码路径保留；没有修改 `scripts/train.py`、loss、checkpoint、用户改写的 evaluation.py 示例或旧模型权重。
- `FrozenSigLIP.image_features/text_features` 新增可选 `normalize` 参数，默认仍为 True，训练和独立评测的计算方式不变。共享编码函数取得投影后、L2归一化前的256维特征，再按任务选择是否归一化，避免特征诊断把原始范数误记为1。
- 工厂支持依据已有 checkpoint 配置重建随机结构但不加载其权重，供 `--random_init` 诊断使用；该模式会随机初始化编码器和投影，不等于独立冻结实验中“预训练编码器+随机投影”的 before 状态。随机参照记录 epoch0、初始化种子和配置来源。

### 使用

```bash
uv run python -m scripts.evaluation \
  --checkpoint models/siglip-dino-bert-mini-frozen-coco/last.pt \
  --manifest data/manifests/coco_val2017.jsonl

uv run python -m scripts.evaluation --task zero-shot \
  --checkpoint models/siglip-dino-bert-mini-frozen-coco/last.pt \
  --dataset cifar100 --data_root data

uv run python -m scripts.evaluation --task feature-diagnostics \
  --checkpoint models/siglip-dino-bert-mini-frozen-coco/last.pt \
  --manifest data/manifests/coco_val2017.jsonl --num_samples 1024
```

### 实际验证

- Ruff format/check、ty 检查通过，独立训练入口 CLI 导入正常。
- 使用 last.pt 完成全量 COCO 检索（5000图/25014描述，含20组可视化HTML）、完整CIFAR100分类（10000图）及1024对特征诊断；结果按现有约定写入该模型的 outputs 目录，生成的PNG可解码，并实际查看特征相似度图。
- COCO 图→文/文→图R@1分别为11.16%/8.99%，CIFAR100 Top1为12.74%；与训练结束时独立脚本保存的对应指标完全一致，抽样图文也完全相同。修复只是接入共享评测，没有重新训练或改变权重。
- 使用真实图文验证归一化特征与独立模型接口逐位一致、原始特征确为未归一化的投影输出、重建logits与模型forward逐位一致、已训练投影权重正确加载。
- 原自写Tiny和官方Tiny均通过16图/80描述的检索冒烟；新模型best.pt通过CPU同规模检索。随机结构CPU诊断通过，同种子编码器权重可复现且不同于训练权重。临时清单与报告已清理，没有覆盖旧模型评测。

## 本地冻结 DINO + BERT-Mini 的线性图文对齐实验

- 实现代码集中在新增 `scripts/train_pretrained_siglip.py`，不导入或修改 `scripts/train.py`，也不修改已有模型、loss、共享评测模块或依赖。仅做一组预训练编码器冻结实验，不增加随机初始化对照组、不自动解冻、不加 LoRA 或 MLP。
- 视觉编码器为 `facebook/dino-vits16`，使用 CLS 特征；文本编码器为 `google/bert_uncased_L-4_H-256_A-4`，使用 attention mask 排除 padding 后平均池化。使用各自的图像处理器和 tokenizer，不沿用 SigLIP 的预处理。两种原始 pooler 和任务头均不保留。
- 两个编码器 `requires_grad=False`，模型切换 train/eval 时仍强制保持编码器 eval，关闭 BERT dropout。只在编码器前向中使用 no_grad，投影层正常计算梯度，不使用会影响训练反向保存的 inference tensor。
- 投影为独立的 `Linear(384,256)`、`Linear(256,256)`，带 bias；在构建投影前独立设置种子42，权重 Xavier uniform、bias为零，投影后 L2 归一化。logit_scale/logit_bias 初始值为2.3/-10，复用原有 SigLIP loss。
- 实测总参数32934786，可训练164354；只训练两层投影的权重/bias及两个logit参数。优化器AdamW betas=(0.9,0.999)，矩阵weight decay0.01，bias和logit参数不衰减；默认20epochs、真实batch16、LR1e-4、5% warmup+cosine最低1e-6、BF16、4 workers、图像224、文本最长64。
- 模型写入 `models/siglip-dino-bert-mini-frozen-coco/`，训练记录写入 `outputs/siglip-dino-bert-mini-frozen-coco/`。保存best/last和每10轮快照、metrics.jsonl、TensorBoard和最终loss_curve.png。checkpoint保存state_dict、编码器结构配置、来源、优化器/调度器状态和训练参数，不pickle脚本中的整个模型对象。
- 自带 `--mode train|smoke|evaluate`。评测重建编码器结构并加载checkpoint，不重新加载预训练模型权重；仍使用本地缓存的配套tokenizer/processor。backend为 `frozen_dino_bert`，现已接入共享评测加载器，通用 `scripts/evaluation.py` 入口可用，原独立评测入口也保持可用。
- 正式训练自动先评测随机投影的before状态，结束后自动评测best、last。使用完整COCO val2017（5000图/25014描述）、Flickr8k test（1000图/5000描述）、CIFAR100 test（10000图）；检索复用多描述归属Recall，分类沿用原prompt。完整任务评测使用FP32，训练/验证loss使用BF16；before/best/last的评测精度一致。
- 每个状态在相同种子下抽取COCO验证集1024张图、每图一条描述，复用已提取的投影后特征计算有效秩、比例、总方差等；冻结编码器原始特征不变，不把投影几何变化误认为骨干参数改变。结果写入 `evaluation/before.json`、`best.json`、`last.json` 和前后对比 `evaluation/comparison.md`，没有伪造尚未训练的结果行。
- 训练都在本地新tmux窗口中执行：预检窗口 `siglip-local:check-frozen-4fb20a27`，正式窗口 `siglip-local:train-frozen-4fb20a27`，不复用或终止其他任务。与云端Small batch256/50轮训练完全独立。

### 使用

```bash
uv run python -m scripts.train_pretrained_siglip

uv run python -m scripts.train_pretrained_siglip --mode evaluate \
  --checkpoint models/siglip-dino-bert-mini-frozen-coco/best.pt
```

### 已完成的预检

- 两个预训练checkpoint、配置、图像处理器和词表已下载缓存；没有通过随机模型替代缺失权重。Ruff format/check、ty、CLI帮助检查通过。
- 本地COCO训练集118287图/591753描述、COCO验证集、Flickr8k测试集全部图片路径存在，CIFAR100完整测试集可离线读取；`scripts/train.py` SHA256在实施前后保持不变。
- 在本地RTX4060 Laptop GPU的tmux新窗口完成64次真实batch16更新和全部5000张验证图。两编码器state_dict指纹完全未变、梯度均为None、BERT特征重复计算及训练前后完全一致；六个可训练参数张量确实改变且梯度有限。
- 实际写出checkpoint，按结构配置重建另一份模型并重载，真实样本logits逐位一致；checkpoint没有依赖pickle的脚本类路径。临时预检模型、日志和启动器已清理，正式训练不使用预检更新过的权重。
- 预检CUDA分配峰值约0.319 GiB（不含桌面/其他进程和CUDA上下文），训练步稳态均值约0.02362秒，全量验证约7.69秒。按每轮7392步、20轮及验证/保存估算训练约1.01小时；训练前及结束后的三次完整任务评测不计入此数字，正式启动后结合基线实测说明耗时范围。
- 实测证据保存于正式输出目录 `preflight_benchmark.json`。没有修改README或原有训练入口，没有留下临时测试权重。
- 正式训练已在本地 `siglip-local:1`（窗口名 `train-frozen-4fb20a27`）启动，已完成完整before评测，并确认第1/20轮至少64/7392批参数更新、loss有限。启动及before评测合计约87.4秒；正式稳态训练步约0.02661秒，含验证/保存预计20轮约1.14小时，建议预留1–1.5小时，结束后的best/last完整评测另约3分钟。
- 启动阶段CUDA分配峰值约1.10 GiB，包含完整检索候选池评分，不是整张GPU的总占用；本地预检窗口已清理，正式窗口已选中。启动命令、stdout/stderr、真实运行配置、首64批观测和实测估算均保存在独立输出目录；请保持笔记本接电且不休眠，tmux不能避免系统休眠或关机中断。

## 云端自写 Small：batch256、50 epochs

- 本次部署到 `ssh -p 13710 root@connect.westc.seetacloud.com`，项目为 `/root/task_siglip`，数据盘为 `/root/autodl-tmp/data`。这是自写 SigLIP-Small 从头训练，不是此前讨论的本地 DINO/BERT 冻结编码器实验，也不加载官方 SigLIP 预训练权重。
- 新服务器实测 GPU 为 RTX 4080 SUPER，显存 32760 MiB，部署前空闲 32230 MiB；CUDA/BF16 可用，环境为 Python 3.12 venv、PyTorch 2.11.0+cu128。沿用已验证的 `uv run --no-sync` 环境，没有联网更新依赖。
- COCO 清单训练 118287 条、验证 5000 条，全部图片路径、ID 唯一，缺失图片、空文件和无效 caption 均为零；另抽样解码 64 张训练图、16 张验证图。图片路径已为服务器绝对路径，不需要重传图片或清单；tokenizer/processor 离线缓存齐全。
- 模型、encoder、transformer、loss、数据代码及 Small 配置的 SHA256 与本地一致；服务器训练脚本差异仅为类型注解和等价的参数调用写法，没有覆盖或修改服务器源文件。
- 正式配置：真实 batch256、50 epochs、BF16、4 workers、seed42、LR `1e-4`、AdamW betas `(0.9, 0.999)`，矩阵参数 weight decay `0.01`、向量参数不衰减。保留 `--no_pin_memory`，不使用梯度累积冒充大负样本池。
- 不续接旧 20 轮 checkpoint；从随机初始化开始，按完整 COCO 的每轮 462 步、总 23100 步重新建立 5% warmup + cosine、最低 LR `1e-6`。旧 batch128/20轮实验及其产物保持原样。
- 模型目录为 `/root/autodl-tmp/data/models/siglip-small-coco-bs256-ep50/`；日志目录为 `/root/autodl-tmp/data/outputs/siglip-small-coco-bs256-ep50/`。正式命令保存在后者 `launch_train.sh`，stdout/stderr 写入 `train.log`。沿用每轮 best/last、每10轮快照、TensorBoard 和训练结束后绘制 loss 曲线的行为。
- 本次预检使用 tmux 新窗口 `siglip:check-bs256-01e88210`，正式训练启动器使用另一个新窗口 `siglip:small-bs256-ep50-01e88210`；没有复用旧窗口执行训练。
- 真实预检执行 16384 张 COCO 训练图、64 次 batch256 AdamW 更新以及全部 5000 张验证图（20批，包括末批136张）；验证 loss/梯度/参数有限、参数确实更新、Adam/scheduler 都到64步，checkpoint 后端为 custom、模型参数量46546946，并实际写出 best/last。
- 预检显存分配峰值 14.44 GiB，保留峰值15.12 GiB。排除前四批预热后，每步均值约0.596秒；全量验证约13.88秒，两次 checkpoint 保存约1.73秒。计入启动、验证和保存后，估算每轮约294秒、50轮约4.09小时；建议预留4–5小时，不保证固定完成时刻。
- 预检记录与配置、源文件哈希、数据核对结果保存在正式输出目录 `preflight_benchmark.json`。临时预检脚本、模型权重和日志在正式启动前清理，正式训练不加载预检权重。
- 正式训练已在 `siglip:1`（窗口名 `small-bs256-ep50-01e88210`）启动并持续运行；最终交付前观测到第1/50轮已完成至少230/462批，GPU Python进程PID3801占用15800 MiB。完整启动观测已写入 `preflight_benchmark.json` 的 `formal_startup`，预检窗口已清理，正式窗口已选中。

## 八份 best/last 权重的统一对比

- `scripts/test_model.py` 自动遍历 `models` 下的模型子目录，仅比较每个目录的 `best.pt`、`last.pt`；不将 `epoch_010.pt` 等快照混入。当前四个子目录共八份权重。先检查配对权重是否齐全，不静默跳过缺失文件。
- 主报告只保留四类指标：COCO/Flickr8k 双向检索 R@1、CIFAR100 零样本 Top1、图像/文本有效秩及其比例、图像/文本总方差。不生成任意加权总分，也不把有效秩当成性能排名。
- 检索分别使用完整 `coco_val2017.jsonl`、`flickr8k_test.jsonl` 的全部图片和描述，沿用现有 `collect_captions`、编码和 Recall 实现，保留同图多描述的正样本归属。增加 Flickr8k 测试避免只用 COCO 评价 Flickr8k 模型，但两个数据集的分数不能直接混合成同一难度的总分。
- 几何统计固定 COCO 验证集的 1024 张不同图片和每图一条描述，默认 seed 42，所有权重共用同一组样本；复用完整检索已编码的单位长度特征，不额外编码图片。样本数参数仅影响特征统计，不缩小检索候选池。
- 复用 `feature_statistics` 的中心化协方差有效秩与总方差定义。比例为 `effective_rank / min(N-1, D)`，避免只比较 Tiny/Small 的绝对有效秩。保留无定义方向的状态、异常数量和 null/N/A，不将其替换为正常分数。
- CIFAR100 复用现有 zero-shot 实现和统一 prompt，使用完整测试集且不训练分类头。旧实现的原始分类 JSON/混淆矩阵按模型目录和 best/last 分别保存在 `details/`；主对比表仅引用 Top1，不将其余明细扩展为主指标。
- 默认输出目录 `outputs/compare_models/`：`comparison.md` 提供任务性能与特征诊断两张表，`summary.csv` 用于表格处理，`summary.json` 保留协议、固定样本、来源 checkpoint、实际 epoch 和完整精简对比结果。模型按目录名排列，不暗示综合优劣。
- 支持 `--models_dir`、`--data_root`、`--output_dir`、`--batch_size`、`--num_samples`、`--seed`、`--device`；仅显式 `--download` 时允许下载缺失的 CIFAR100。一次只加载一份权重，避免八个模型同时占用显存。
- 各模型训练数据、epoch、batch 不相同；本报告比较现有权重的实际能力，不能单独据此归因架构、初始化或模型规模。没有修改训练、模型、loss、依赖和 README。

### 运行

```bash
uv run python -m scripts.test_model
```

也可直接运行 `uv run python scripts/test_model.py`。默认自动选择 CUDA（可用时），`--device cpu` 可改为 CPU；所有数据默认从现有 `data` 读取。

### 验证

- Ruff format/check、ty 与 CLI 帮助检查通过。
- 用真实 COCO 清单核对 1024 个抽样描述的 caption 索引及图片归属，与完整候选池映射一致。
- 临时报告验证覆盖无定义特征统计：JSON 保留 null，CSV 不填伪造分数，Markdown 显示 N/A；有效秩比例在 CSV 中保留原始小数。临时报告已清理。
- 已在本机 RTX 4060 Laptop GPU 上完整运行四个目录的 best/last 共八份真实权重，未复用旧评测结果：每份均测试 COCO 的 5000 图/25014 条描述、Flickr8k 的 1000 图/5000 条描述和 CIFAR100 的 10000 张测试图，特征统计共用固定 1024 对样本。
- 实际生成并查看 `comparison.md` 两张完整八行对比表；核对八份权重没有遗漏或重复、候选池一致、16 组图像/文本统计可计算、有效秩比例和方差范围有效、CSV 与 JSON 一致且 JSON 不含 NaN/Inf。没有留下临时报告或修改原评测文件。

## 图文特征坍塌诊断

- `scripts/evaluation.py` 新增 `--task feature-diagnostics`，复用现有 checkpoint、manifest、device、batch size、seed 和输出目录参数；新增 `--num_samples`（默认 1024）和仅供诊断使用的 `--random_init`。原默认检索和 zero-shot 入口保持不变。
- 核心实现位于 `src/evaluation/feature_diagnostics.py`。按独立局部随机数生成器选择不同图片，每张固定一条 caption，并保存图片 ID、路径、清单索引、caption 索引和文本；相同清单、种子和样本数在不同权重下得到相同配对。清单不足请求数量时使用全部图片，报告实际数量；至少需要两张图片。
- `src/evaluation/common.py` 的批量图像/文本编码函数增加关键字参数 `normalize`，默认仍为 True。诊断取得归一化前的原始特征，单独统计范数、非有限值和零向量，再计算单位长度特征的几何统计，不使用 logit scale/bias。
- 几何统计包括均值向量范数、中心化总方差、非对角余弦分布、协方差特征值谱、有效秩及其占最大可用秩的比例、最大特征值占比。协方差使用 CPU float64、除以 N；有效秩定义为协方差特征值归一化后的熵的指数，最大可用秩为 `min(N-1, D)`。
- 零方差时有效秩报告为 0，最大特征值占比为 null；出现零向量或 NaN/Inf 时明确记录数量，将无定义的方向统计和对齐统计标记为不可用，不静默删除样本、不用 epsilon 伪造正常特征。极小但非零的向量使用双精度显式归一化，避免被默认 epsilon 改变方向范数。
- 对齐诊断包含真实配对/其他图片描述的余弦分布、正负均值差和双向 Recall。额外使用非零循环移位生成没有原配对的错误归属标签，作为独立负对照；不把它作为真实 ground truth，也不混淆“文本换序但同步保留真实归属”的不变性。
- 随机初始化参照复用来源 checkpoint 的模型配置、后端和 tokenizer，不加载其中的模型权重；报告明确记录 `config_source_checkpoint`、初始化种子和 epoch 0，不将它冒充已训练 checkpoint。
- 每次输出到 `outputs/<模型目录名>/feature_diagnostics/<manifest名>_<checkpoint名>/`，保存 `report.json`、`cosine_distributions.png`、`covariance_spectrum.png`；随机参照使用 `random_init_seed42` 等独立后缀。图表使用无窗口 Agg 后端，特征值谱使用支持零值的 symlog 坐标。
- `status=ok` 只表示方向统计可计算，不是“模型健康”的自动结论。没有设置通用坍塌阈值；低有效秩必须结合总方差、余弦分布、随机参照和训练过程解释。

### 使用

```bash
uv run python -m scripts.evaluation \
  --task feature-diagnostics \
  --checkpoint models/siglip-small-coco-bs128/best.pt \
  --manifest data/manifests/coco_val2017.jsonl \
  --num_samples 1024 --batch_size 32 --seed 42
```

- 随机初始化参照：在上述命令末尾追加 `--random_init`。
- 比较训练阶段：将 checkpoint 换为 `epoch_010.pt` 或 `last.pt`。
- 检查训练集：将 manifest 换为 `data/manifests/coco_train2017.jsonl`，其余抽样参数保持不变。
- 诊断 Recall 只使用抽取的一图一描述候选池，不可直接与完整 5000 图、25014 描述的检索报告比较。

### 已完成的验证

- Ruff format/check、ty 检查通过。临时合成验证覆盖完全重复向量、中心化已知秩、低秩、随机且未对齐的模态、极小非零范数、低方差但高有效秩、零向量及 NaN/Inf；正确配对的单位基向量 R@1 为 100%，错误归属负对照 R@1 为 0%。
- 实际抽取 COCO 验证记录检查可复现的图片/caption 选择；完全相同方向及无定义方向均能输出合法 JSON 和 PNG，临时验证图表已清理。
- 在本机 RTX 4060 Laptop GPU 上完成 Small 的 `epoch_010`、`best`、`last`、随机初始化参照共四种状态的训练集/验证集诊断，每次 1024 对；自动确认同一 split 的四次样本完全相同、JSON 不含 NaN/Inf、两张 PNG 可解码，并人工查看真实 best 权重的图表。
- 官方 Tiny 后端也完成真实图像的原始特征诊断；原默认检索使用 16 张 COCO 图及其全部 80 条描述通过，原 zero-shot 入口使用完整 CIFAR10 测试集 10000 张图通过。兼容性结果位于临时目录，运行后删除，没有覆盖用户原评测报告。
- CPU 模式另用 8 张真实 COCO 图完成官方 Tiny 的随机初始化诊断全流程，确认后端标记、epoch 0 和配置来源记录正确；临时产物已清理。
- 初步观察：Small best 在验证样本上的图像/文本总方差分别约 0.910/0.875，非对角平均余弦约 0.089/0.124，有效秩约 31.8/25.3（特征维度 384）；不符合所有样本几乎同向的整体坍塌。训练后期有效秩继续增加，不能仅凭其低于特征维度就认定维度坍塌或初始化错误。
- 没有修改 README、训练脚本、loss、编码器、初始化公式或依赖；按 README 的约定在本文件记录说明，没有留下临时脚本或测试权重。

## 服务器上的自写 Small COCO 训练

- `scripts/train.py` 增加 `--data_root`（默认 `data`）和 `--log_dir`（默认仍为 `outputs/<模型目录名>`）。原 `--output_dir` 继续指定 checkpoint 目录，原本地默认行为保持不变。
- 服务器项目位于 `/root/task_siglip`，数据位于 `/root/autodl-tmp/data`。补齐服务器专用 COCO train/val manifest，将图片路径转换为服务器绝对路径；不修改本地原清单，不复制或重复下载图片，不创建数据软链接。
- 服务器验证了 118,287 条训练记录和 5,000 条验证记录的全部图片路径：缺失、空文件、重复路径、无效 caption 均为零；另抽样解码 64 张训练图和 16 张验证图。
- 仅同步 tokenizer 与图像处理器缓存，不同步官方预训练模型权重。服务器已验证环境通过 `uv run --no-sync` 使用，避免启动时的联网依赖同步；Ruff、ty 检查通过。
- 正式运行使用自写 `scripts/train.py`、`configs/small.yaml`，20 epochs、真实 batch 128、BF16、LR `1e-4`、weight decay `0.01`、4 workers、seed 42；不恢复旧 checkpoint。
- checkpoint 目录为 `/root/autodl-tmp/data/models/siglip-small-coco-bs128/`，训练输出目录为 `/root/autodl-tmp/data/outputs/siglip-small-coco-bs128/`。启动命令保存为后者的 `launch_train.sh`，终端输出保存为 `train.log`，使用 tmux 会话 `siglip` 的现有窗格。
- 正式启动前，临时自写 Small（46,546,946 参数）完成 32 次 batch128 训练更新、40 批全量验证和 checkpoint 写入；loss 与参数有限，Adam/scheduler 均到 32 步，后端为 `custom`。GPU 分配峰值为 7.54 GiB。
- 排除前四批预热后的训练步均值约 0.346 秒；结合全量验证、checkpoint 保存和每轮 924 步，估算 20 轮约 1.87 小时，实际运行建议预留 2–3 小时。此为短程实测外推，不是完成时间保证。
- 实测报告保存在正式输出目录的 `preflight_benchmark.json`；临时测试模型、日志和部署脚本在正式启动前清理，不将测试权重作为正式初始模型。
- 首次正式运行在第 25 批后于 DataLoader 的锁页内存线程报 `CUDA error: invalid argument`；服务器锁页额度为 unlimited，底层原因尚未定位。新增显式 `--no_pin_memory` 开关，关闭训练和验证 DataLoader 的可选锁页内存优化；默认行为保持不变，不捕获或吞掉异常，不跳过样本。
- 服务器正式启动命令已加入 `--no_pin_memory`。保留失败日志为 `train.failed-pin-memory.log`，从相同初始化重新训练；确认第 1 轮至少完成 128/924 批（45 秒），Python GPU 进程占用约 8.3 GiB。模型、batch、学习率及 20 轮预算不变。上述预检报告是在默认锁页内存启用时测得，重启后的正式进度也支持预留 2–3 小时。

## 官方 Tiny 结构的独立训练入口

- 新增 `scripts/train_offical_siglip.py`，保留用户指定的 `offical` 拼写；默认读取 `configs/tiny.yaml`，不需要显式传 `--config`。支持模块方式和直接运行脚本。
- 新增 `src/model/official_siglip.py`：使用 `SiglipModel(SiglipConfig(...))` 随机初始化官方结构，映射当前 Tiny 的视觉 192 维/12 层、文本 256 维/4 层、192 维投影及全部尺寸参数；参数量为 17,385,986。显式使用 YAML 中的 logit scale/bias，不调用模型 `from_pretrained`，只复用现有 tokenizer 和图像处理器。
- `scripts/train.py` 提取共享 `run_training`，两个入口复用同一数据采样、预处理、独立 sigmoid loss、AdamW 分组、cosine/warmup、AMP、验证和保存过程，不复制训练循环。原自写入口仍要求 `--config`，原默认模型目录不变。
- 官方入口默认模型目录为 `models/siglip-tiny-official-<dataset>/`，保存 `best.pt`、`last.pt`、每 10 轮的 `epoch_010.pt` 等。指标、TensorBoard、`loss_curve.png` 写到 `outputs/<model_dir.name>/`，`--output_dir` 仍表示模型目录；恢复训练继续使用原日志截断与 TensorBoard purge 规则。
- 新 checkpoint 记录 `model_backend`。`src/evaluation/common.py` 按后端加载与编码，原评测入口可直接处理新官方 checkpoint；旧 checkpoint 缺失该字段时仍使用自写模型，保持旧评测协议。
- 没有修改自写模型、编码器、loss、Tiny 配置、依赖或 README，也没有加载官方预训练权重或启动正式全量训练。

### 使用

```bash
uv run --locked python -m scripts.train_offical_siglip \
  --dataset coco \
  --num_epochs 20 \
  --batch_size 16 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --device cuda \
  --precision bf16 \
  --num_workers 4 \
  --seed 42

uv run --locked python -m scripts.evaluation \
  --checkpoint models/siglip-tiny-official-coco/best.pt \
  --manifest data/manifests/coco_val2017.jsonl
```

- 比较实现时，两个训练入口使用相同的 batch、epoch、学习率和数据设置；相同 seed 不保证两种实现的初始权重逐元素一致。恢复同一次训练时使用新入口的 `--resume`，并保持原总训练计划。

### 实际验证

- 默认 Tiny 配置使用真实 COCO 训练/验证图片完成 CUDA BF16 训练，实际参数量匹配，Adam 与 scheduler 均完成 2 次更新，官方 checkpoint 可严格重建。保存后的 logits、独立 loss 及归一化图文特征与原生 HF 输出一致。
- 临时小配置使用 4 张训练图、3 张验证图完成 11 轮和第 10 轮快照恢复，覆盖验证尾批为 1；恢复后权重变化，Adam/scheduler 到 22 步，metrics 与 TensorBoard 的 epoch 均为 1–11 且不重复，scalar 与实际指标一致，loss 曲线已查看。模型目录不含 metrics，训练输出分别位于 models/、outputs/。
- 原自写训练入口同样完成真实图文训练；旧自写 checkpoint 的检索入口正常。新官方 Tiny checkpoint 完成实际检索与案例生成，官方 checkpoint 的 zero-shot 分支完整处理 10,000 张 CIFAR-10 图片；这些仅用于接口烟测，不作为模型质量结果。
- 修改文件的 Ruff、ty 检查通过，直接运行入口的 `--help` 正常。临时训练模型、日志和评测文件在验证后清理，不覆盖用户实验。

## 对齐 Transformer 与两塔编码器初始化

- `src/model/transformer.py`：融合 QKV 按三个独立 `[D,D]` 切片做 Xavier uniform，attention 输出投影同样使用 Xavier，bias 全零；Transformer MLP 两层使用 Xavier uniform，bias 使用 `std=1e-6` 的 normal 初始化。
- `src/model/encoder.py`：Patch Conv、文本投影和 MAP attention 的输出投影使用 LeCun normal；本地 `_lecun_normal_` 先在标准正态的 `[-2,2]` 内截断，再按 `sqrt(1/fan_in)/0.87962566103423978` 缩放，与官方方差修正规则一致，不依赖 transformers 的私有初始化函数。
- MAP query 改为 Xavier uniform，pooling MLP 对齐 Transformer MLP 的初始化。保留已经匹配的 MHA in-projection 与零 bias、embedding 和 LayerNorm 初始化；保留 `logit_scale=2.3`、`logit_bias=-10.0`。
- 没有改变模型结构、参数名和形状、forward、loss、训练配置或已有 checkpoint。MLP 使用局部 Linear 变量完成初始化后再组装 Sequential，保持原有 state_dict 键，不增加辅助类或运行时防御分支。
- 数值验证：四种 Conv/Linear 形状在相同随机种子下与官方 LeCun 初始化逐元素一致；实际 Tiny 两塔的 48 个 Q/K/V 切片、各 MLP、投影、pooling 与 LayerNorm 的初始化检查通过。
- 临时新模型使用 4 张真实 COCO 图片及描述，完成两次 FP32 与一次 BF16 参数更新，216 个参数的梯度均有限，两塔及 logit 参数实际更新；默认无配置模型的前向正常。没有保存临时权重或启动完整训练。
- 修改前后加载同一现有 COCO 第 50 轮 checkpoint，216 个 tensor 的名称/形状及固定输入 logits 完全相同；Ruff 和 ty 检查通过。
- 初始化变化只影响新建参数。验证新初始化的效果需要从头训练，恢复旧 checkpoint 会覆盖初始化；本次不改动旧评测结果。

## 修复 SigLIPModel 的无配置构造

- `src/model/siglip.py` 在 `config is None` 分支补上 `init_logit_scale=2.3`、`init_logit_bias=-10.0`，与当前 Tiny 配置一致，修复默认构造时变量未赋值的 `UnboundLocalError`。
- 显式配置分支、参数名称及形状不变；没有修改其他层的初始化、训练配置或已有 checkpoint。
- 离线实际执行 `SigLIPModel()`，完成两组图文的前向、独立 sigmoid loss 和反向传播；两塔及 logit 参数梯度均有限且非零。另验证显式传入不同 scale/bias 时配置值生效，已有 COCO 第 50 轮 checkpoint 严格加载及前向正常。
- `ruff check src/model/siglip.py`、`ty check src/model/siglip.py` 均通过。

## 官方预训练 SigLIP 的 CIFAR 对照

- 新增独立实验脚本 `scripts/test_pretrained_siglip.py`，默认完整评测 CIFAR-10、CIFAR-100 test split。使用 `main` 保护，不会在 pytest 导入时下载模型或运行整套分类实验。
- 加载 [Google 官方 SigLIP Base/16 224 权重](https://huggingface.co/google/siglip-base-patch16-224)，固定 revision 为 `7fd15f0689c79d79e38b1c2e2e2370a7bf2761ed`，通过 `SiglipModel.from_pretrained(..., use_safetensors=True)` 加载真实预训练参数，不使用自训练 Tiny checkpoint。
- 保持原实验的类别顺序、`a photo of the {}.` 模板、固定长度文本 padding、慢版图像处理器和 FP32 推理。使用官方文本、图像特征接口，归一化后计算所有类别的余弦相似度；每个数据集的类别文本只编码一次，不训练分类头。
- 输出协议与自训练模型一致：CIFAR-10 Top-1，CIFAR-100 Top-1 / Top-5，每类样本数、正确数、准确率和原始混淆矩阵；另存按真实类别逐行归一化的混淆矩阵 PNG。metadata 记录官方模型、权重 revision 和实际 dtype。
- 按本次要求，默认保存到单数目录 `output/siglip-base-patch16-224/zero_shot/`，文件为 `cifar10_pretrained.json`、`cifar100_pretrained.json` 和对应的 `_confusion.png`，不覆盖已有 `outputs/` 下的自训练结果。数据继续复用 `data/cifar10/`、`data/cifar100/`。
- 没有修改自训练模型、训练脚本、统一评测入口、已有评测核心、依赖或 README。

### 使用

```bash
# 默认测试两个数据集；官方权重首次自动下载到 Hugging Face 缓存。
uv run --locked python -m scripts.test_pretrained_siglip --device cuda --batch_size 32

# 只测试一个数据集。
uv run --locked python -m scripts.test_pretrained_siglip --dataset cifar10 --device cuda
```

- `--output_dir` 可更改结果目录；`--data_root` 可更改数据根目录；只有 CIFAR 数据未缓存时才需要加 `--download`。

### 实际验证

- 官方 safetensors 已下载；实际加载参数量为 203,155,970。在真实 CIFAR 图片上，脚本使用的 token IDs、像素输入与官方 `AutoProcessor` 完全一致；缓存特征计算恢复的 logits 与官方原始 `forward()` 一致，预测类别也一致，通过 `-W error`。
- 完整回归 `uv run --locked python -m pytest -q -W error`：12 passed；新脚本 Ruff、ty 检查通过。
- 实际执行独立 CLI，以 CUDA、FP32、batch size 32 完整评测两个测试集，各 10,000 张图像，通过 `-W error`：CIFAR-10 Top-1 为 91.03%；CIFAR-100 Top-1 为 70.15%、Top-5 为 89.75%。两个 JSON 和两个 PNG 均已保存到上述 `output/` 目录。
- 与现有 COCO Tiny 第 50 轮结果逐项比较，两个数据集的类别顺序、prompts、test split 和样本数完全一致；混淆矩阵各行样本数、对角线汇总与每类统计均校验通过，并已查看两个实际生成的混淆矩阵。

## 统一评测入口与 CIFAR 零样本分类

- `scripts/evaluation.py` 只负责参数解析、任务分发和终端摘要。`--task retrieval` 为默认值，原检索命令不需要修改；`--task zero-shot` 不需要 `--manifest`。
- 核心迁到 `src/evaluation/`：`common.py` 加载 checkpoint、tokenizer、图像处理器并编码归一化特征；`retrieval.py` 负责全候选池 Recall 和案例排名；`zero_shot.py` 负责 CIFAR 分类；`reporting.py` 负责 JSON、原离线 HTML 和混淆矩阵绘图。移除脚本中的旧实现，不保留转发别名。
- 零样本分类使用 CIFAR-10 / CIFAR-100 官方 test split，不训练分类头、不更新模型参数。固定模板为 `a photo of the {}.`，类别名中的下划线替换为空格，保持原类别顺序；类别文本特征只编码一次，图像分批与所有类别计算余弦相似度。
- CIFAR-10 输出 Top-1，CIFAR-100 输出 Top-1 / Top-5；均记录每类样本数、正确数、准确率和原始混淆矩阵。按真实类别逐行归一化绘制 PNG，空行保持为零，不用归一化结果覆盖原始计数。计数在设备端累加，最后才转到 CPU；百分比从整数计数计算。
- 结果保存到 `outputs/<run>/zero_shot/<dataset>_<checkpoint>.json` 和对应的 `_confusion.png`，两个数据集和 best / last 不互相覆盖。原检索 JSON、HTML、缩略图路径及内容协议保持不变。
- `--data_root` 默认 `data`，两个数据集分别缓存到 `data/cifar10/`、`data/cifar100/`；只有显式传入 `--download` 才请求下载。复用现有 torchvision、matplotlib 等依赖，没有修改训练实现、模型、独立 loss 或 README。

### 实际验证

- 现有 COCO Tiny 第 50 轮 `last.pt` 配合 3 张真实 Flickr8k dev 图片、15 条描述，迁移前后的完整检索 JSON 和 `samples.json` 完全一致；这是重构一致性检查，不是新的完整检索 benchmark。
- 实际浏览器打开新入口生成的 HTML：两种查询、真实匹配展示正常，3 组案例的 15 个图片元素全部加载，没有水平溢出或脚本，并查看了图查文、文查图页面截图。
- 同一现有 checkpoint 实际通过统一 CLI 完整评测两个测试集，各 10,000 张图像：CIFAR-10 Top-1 为 15.47%；CIFAR-100 Top-1 为 4.13%、Top-5 为 12.20%。JSON 与 PNG 已保存到 `outputs/siglip-tiny-coco-cosine-beta0999/zero_shot/`，逐类样本数和总数校验通过。
- 官方源首次下载过慢并超时，本次验证改从 [原始包镜像](https://huggingface.co/datasets/ThunderStruct/CIFAR) 获取两个原始 tar 包，均通过 torchvision 提供的官方 MD5 校验，再按原格式解压。没有给生产代码增加镜像回退或关闭校验。
- 绘图显式转换为 NumPy，修复 PyTorch 张量隐式转换在 NumPy 2 下的弃用警告；真实结果绘图通过 `-W error`。原恢复训练测试的中断钩子参数名对齐 `torch.save(obj, f)`，兼容现有关键字调用，只修改测试，不修改训练代码。
- 完整回归 `uv run --locked python -m pytest -q -W error`：12 passed；修改代码和测试的 Ruff 检查、`ty check scripts/evaluation.py src/evaluation` 通过。

### 使用

```bash
uv run --locked python -m scripts.evaluation \
  --task zero-shot \
  --checkpoint models/siglip-tiny-coco-cosine-beta0999/last.pt \
  --dataset cifar10 \
  --batch_size 64 \
  --device cuda
```

- CIFAR-100 将 `--dataset cifar10` 改为 `--dataset cifar100`；首次没有本地数据时添加 `--download`。本次已缓存两个数据集，不需要重复下载。

## 模型输出 logits 与独立 loss 的脚本适配

- `scripts/train.py` 的训练与验证改为先执行 `logits = model(texts, pixels)`，再调用 `src.model.loss.siglip_cross_entropy_loss(logits, positive_mask)`；mask 不再传给模型，也不再从返回值中读取 `["loss"]`。移除旧的可空 loss 断言，损失仍在原 autocast 上下文内计算。
- `scripts/evaluation.py` 明确说明 forward 返回 logits；原有全量候选池检索本身不依赖模型返回的 loss，继续使用归一化编码器特征的余弦相似度，不重复计算训练损失，也不改为小 batch 内排名。既有 Recall、可视化和输出格式保持不变。
- 没有修改用户编写的 `src/model/siglip.py`、`src/model/loss.py`，没有增加兼容旧返回字典的分支，也没有修改模型参数、优化器、调度、TensorBoard 或 checkpoint 结构。
- 训练回归的验证集改为 3 张、batch size 2，实际覆盖验证尾批为 1 时的 mask 切片与独立损失调用。
- 实际小模型配合 4 张真实训练图片、3 张验证图片完成 2 轮、4 次参数更新，两塔与 logit 参数均更新。batch size 1 和 2 的独立损失数值与 `binary_cross_entropy_with_logits(..., reduction="sum") / batch_size` 一致，logits 梯度也一致。新 checkpoint 及已有 COCO 第 50 轮 checkpoint 均完成小候选池检索与 HTML 生成；已有用户权重和评测报告未改动。

- GPU BF16 同样完成 2 轮、4 次更新，验证尾批为 1，loss 与 Adam 状态均有限；完整回归 10 passed，无警告，修改脚本/测试的 Ruff 检查及训练、评测脚本的 ty 检查均通过。

## 2026-11-03：双向检索案例、训练曲线与周期快照

- `scripts/evaluation.py` 增加 `--num_examples`（默认 20，0 时仅指标）和 `--seed`（默认 42）。不重复编码，随机选取不同图片及各自一条真实描述；同一 manifest 与 seed 的查询不随 checkpoint 改变，两个方向仍在完整候选池中检索。
- 生成离线 HTML、`samples.json` 和缓存缩略图：图片查询显示 Top-5 描述、余弦相似度、按 owner 标记的全部真实匹配以及该图所有标注描述；文本查询显示 Top-5 图片，并独立展示真实图片及其全池排名，即使没有进入 Top-5 也不会隐藏。相同字符串归属不同图片时不混淆标注。HTML 转义文本，无脚本或外部网络依赖。
- 案例目录为 `outputs/<run>/visualizations/<manifest>_<checkpoint>/`，包含 `index.html`、`samples.json`、`images/`。原完整检索指标 JSON 和 Recall 计算保持不变。
- `scripts/train.py` 默认模型目录改为 `models/<run>/`，仅产生 checkpoint；`--output_dir` 仍指定模型目录。训练指标统一写到 `outputs/<model_dir.name>/metrics.jsonl`，TensorBoard 写到该目录的 `tensorboard/`，每 epoch 记录并 flush `loss/train`、`loss/val`；正常结束绘制两条 loss 曲线到 `loss_curve.png`。writer 在异常退出时也会关闭。
- 每轮更新 `last.pt`，验证 loss 改善时更新 `best.pt`，每 10 个 epoch 额外保存完整恢复状态的 `epoch_010.pt`、`epoch_020.pt` 等。模型、初始化、AdamW beta2=0.999、学习率调度和 best 的选模标准不变。
- 恢复训练沿用 checkpoint 的 best loss，metrics 仅保留至恢复 epoch，TensorBoard 以恢复点 purge，避免未来或重复 epoch 混入曲线；从头训练则重置本次 run 的日志。既有实验文件不自动搬迁或删除，旧模型目录中的 metrics 不自动迁移。
- 通过 uv 增加并锁定 `tensorboard`、`matplotlib`；没有修改 README。

### 实际验证

- 当前训练入口配合临时小模型和真实 Flickr8k 图片，完成 11 轮及第 10 轮快照恢复：模型目录只有 best、last、epoch_010 三个 pt，Adam/scheduler 最终 22 步，metrics 和 TensorBoard 的 epoch 均为 1–11 且不重复，两条 scalar 与实际 loss 对应，PNG 可解码并已查看。额外复现并修复同目录从头重训混入旧历史的问题：epoch 序列由 1,2,1 修正为 1。
- 当前评测 CLI 使用现有 COCO Tiny 第 50 轮权重、完整 Flickr8k dev 的 1000 图/5000 描述生成 20 组案例；实际浏览器检查两种展示、140 个图片元素全部加载、匹配标记可见、真实图片未入 Top-5 时仍单独展示，无水平溢出或脚本。TensorBoard 实际界面也已打开并确认两条 loss 曲线。
- 已将这份实际案例报告保存为 `outputs/siglip-tiny-coco-cosine-beta0999/visualizations/flickr8k_dev_last/index.html`；没有覆盖已有指标 JSON 或用户模型权重。临时长训练未启动。

### 使用

```bash
# 指标与默认 20 组可视化；可改成其他 checkpoint 或 manifest。
uv run --locked python -m scripts.evaluation \
  --checkpoint models/siglip-tiny-coco-cosine-beta0999/last.pt \
  --manifest data/manifests/flickr8k_dev.jsonl \
  --num_examples 20 --seed 42

# 查看各次训练的 TensorBoard 曲线。
uv run --locked tensorboard --logdir outputs
```

- 最终回归：`uv run --locked python -m pytest -q -W error`，10 passed，无警告；修改脚本与测试的 Ruff 检查通过，`ty check scripts/train.py scripts/evaluation.py` 通过。

## 2026-11-03：恢复 cosine，补齐 beta2=0.999 对照

- 按用户选择将当前 AdamW `betas=(0.9, 0.99)` 改为 `(0.9, 0.999)`，补齐与最早固定 LR、beta2=0.999 实验对应的 cosine 对照；其余模型、数据、初始化与输出目录逻辑不变。
- `scripts/train.py` 恢复前 5% step 的线性 warmup，随后 cosine 衰减至 `1e-6`；`--lr` 表示峰值学习率。每次实际参数更新后推进 scheduler，FP16 梯度溢出跳过参数更新时不推进。
- checkpoint 保存并恢复 scheduler 状态；移除固定 LR 版本在恢复时覆盖参数组学习率的逻辑，避免把衰减中的学习率重置到峰值。恢复同一次训练时应保持原计划的 `--num_epochs`、`--lr`、batch size 和训练集大小不变。本次对照从头训练，不恢复旧固定 LR 或 beta2=0.95 的权重。
- 现有训练测试更新为覆盖学习率衰减及中断恢复后的逐更新学习率轨迹，同时保留权重变化和 optimizer 更新进度检查。
- 实际执行当前 Tiny 配置的 GPU BF16 小样本：24 张真实 Flickr8k train 图片、4 张 dev 图片，计划 2 轮、24 次更新，warmup 1 步；学习率由 0 升至 `1e-4`，最终为 `1e-6`。第 1 轮中断后恢复，后 12 次更新的学习率与连续训练完全相同，optimizer 与 scheduler 均到 24 步，恢复后视觉权重实际更新。正式 20 轮实验未启动，已有实验权重不变。

```bash
uv run --locked python -m scripts.train \
  --config configs/tiny.yaml \
  --dataset flickr8k \
  --num_epochs 20 \
  --batch_size 16 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --device cuda \
  --precision bf16 \
  --num_workers 4 \
  --seed 42 \
  --output_dir models/siglip-tiny-flickr8k-cosine-beta0999
```

- 回归验证：`uv run --locked python -m pytest -q tests/test_train.py -W error`，2 passed，无警告。

## 2026-11-03：固定学习率对照训练

- `scripts/train.py` 移除 warmup、cosine 调度器、逐步调度调用，以及 scheduler checkpoint 的保存和恢复；学习率由 `--lr` 指定，全程固定。新训练继续使用 AdamW `betas=(0.9, 0.95)`，不修改模型、初始化、数据或依赖。
- 恢复训练仍加载模型、optimizer 和 scaler；加载 optimizer 后将所有参数组的学习率设为本次 `--lr`，避免旧 checkpoint 的学习率覆盖命令行指定值。对照实验应从头训练，不恢复已有 cosine 权重。
- 保持当前脚本的输出逻辑不变：`last.pt`、`best.pt`、`metrics.jsonl` 保存在同一输出目录；下方显式指定新的 models 目录，不覆盖已完成的实验。README 不变。
- 更新现有训练测试的模型初始化配置和显式输出路径，检查正常训练的固定学习率，以及恢复时命令行学习率优先于 checkpoint，同时保留 optimizer 更新进度。
- 实际执行当前 Tiny 配置的 GPU BF16 小样本训练：4 张真实 Flickr8k train 图片、4 张 dev 图片，前两轮共 4 次更新均为 `1e-4`；恢复到第 3 轮并指定 `--lr 0.0003`，新增 2 次更新均为该值，Adam 总更新次数为 6，视觉卷积权重实际变化，checkpoint 不含 scheduler 状态。正式 20 轮训练未启动。

```bash
uv run --locked python -m scripts.train \
  --config configs/tiny.yaml \
  --dataset flickr8k \
  --num_epochs 20 \
  --batch_size 16 \
  --lr 1e-4 \
  --weight_decay 0.01 \
  --device cuda \
  --precision bf16 \
  --num_workers 4 \
  --seed 42 \
  --output_dir models/siglip-tiny-flickr8k-fixed-lr-beta095
```

- 回归验证：`uv run --locked python -m pytest -q tests/test_train.py -W error`，2 passed，无警告。

## 模型与评测结果分目录保存

- `scripts/train.py` 默认 checkpoint 目录改为 `models/<config.name>-<dataset>/`，保存 last.pt、best.pt；训练指标仍写在 `outputs/<模型目录名>/metrics.jsonl`，不混入模型目录。
- 无显式 `--output_dir` 时，即使 `--resume` 指向旧 outputs 目录，后续 checkpoint 也写入新的 models 默认目录，不再继承旧 checkpoint 的父目录。原 `--output_dir` 自定义模型目录功能保留，显式路径优先。
- `scripts/evaluation.py` 除终端摘要外，新增 JSON 文件保存；默认 `outputs/<checkpoint父目录名>/<manifest文件名去后缀>_<checkpoint文件名去后缀>.json`，也可通过评测的 `--output_dir` 指定结果目录。
- JSON 包含 checkpoint 路径、模型名称、epoch、manifest 路径、完整图片/描述候选数、百分比单位，以及两个方向的 R@1/5/10；dev/test 和 best/last 使用不同文件名，不混淆结果。

```text
models/siglip-tiny-flickr8k/
  last.pt
  best.pt
outputs/siglip-tiny-flickr8k/
  metrics.jsonl
  flickr8k_dev_best.json
```

- 当前推荐入口：`uv run python scripts/train.py --config configs/tiny.yaml --dataset flickr8k`；评测新模型使用 `uv run python scripts/evaluation.py --checkpoint models/siglip-tiny-flickr8k/best.pt --manifest data/manifests/flickr8k_dev.jsonl`。
- 已有 outputs 下的用户 checkpoint 不自动搬动或删除，旧路径仍可作为 resume/checkpoint 输入。实际使用现有 epoch 10 模型生成了 `outputs/siglip-tiny-flickr8k/flickr8k_dev_best.json`，包含完整 1000 图片、5000 描述的 dev 结果；源 checkpoint 大小与修改时间不变。
- `uv run --locked python -m pytest -q -W error`：6 passed，无警告；测试覆盖默认 models 保存、旧 outputs 路径恢复不回写源文件、训练日志分离，以及真实小模型的评测 JSON 持久化。ty 和修改文件的 Ruff 检查通过。
- 真实 GPU 小样本在临时项目根目录运行原训练 CLI：默认模型进入 models，指标进入 outputs；从旧 outputs/legacy/last.pt 恢复后 models 中 epoch 到 2、Adam 更新进度到 4，原 legacy 文件不变。临时训练产物已清理。
- 模型、损失、数据、依赖和 README 不变；本次修改脚本、对应测试及本文件，没有为目录调整增加额外框架。

## 双向图文检索评测：`scripts/evaluation.py`

- 新增简洁评测入口，只有 `encode_images()`、`encode_texts()`、`compute_recall()`、`main()`，共 115 行；不修改模型、训练代码、配置或依赖。
- 从 checkpoint 的 config 和 model_state_dict 恢复架构与参数；沿用训练时的 tokenizer、图像预处理尺寸和文本固定长度。使用 model.eval() 与 inference_mode，分别调用已有两个编码器。
- 每张图片只编码一次，每条描述都编码；两侧输出分别做 L2 归一化并保存至 CPU。完整分数矩阵为图片行、文本列的余弦相似度，不需要 logit scale/bias 或 sigmoid。
- 读取完整 captions 列表并显式记录每条描述对应的图片索引，支持每图描述数量不一致，不假设五条或依赖 image_id/文件名格式。Dataset 的固定首条描述仅用于图片读取接口，文本候选库仍包含所有描述。
- 分别输出 image_to_text 和 text_to_image 的 R@1/5/10，单位为百分比。图片找文本时任意一条正确描述进入前 K 即命中；文本找图片使用该描述所属图片作为正例。
- batch_size 仅控制编码，不划分检索候选库；全部特征提取完成后才在完整矩阵上计算 Top-K。K 大于候选数时使用全部可用候选，不会发生 topk 越界。第一版直接保存完整 CPU 分数矩阵，未加入分块缓存或多卡框架；COCO 5k 规模矩阵约需 500 MB，此外还有特征与 Top-K 工作区。

```bash
# 在 dev 上调试与比较 checkpoint。
uv run python scripts/evaluation.py \
  --checkpoint outputs/siglip-tiny-flickr8k/best.pt \
  --manifest data/manifests/flickr8k_dev.jsonl \
  --batch_size 32

# 冻结模型与设置后，才使用 test 报告最终结果。
uv run python scripts/evaluation.py \
  --checkpoint outputs/siglip-tiny-flickr8k/best.pt \
  --manifest data/manifests/flickr8k_test.jsonl \
  --batch_size 32
```

- 可选 `--device auto/cpu/cuda`；checkpoint 和 manifest 可用项目相对路径或绝对路径。打印加载的 epoch、配置名称、实际图片/描述候选数和六项指标，并返回指标字典便于交互使用；脚本不写入数据、checkpoint 或模型输出目录。
- `tests/test_evaluation.py` 覆盖非均匀且交错的多描述正例、图片/文本重排不变性、完整候选库中 R@1/5/10 的精确排名、小候选集边界，以及真实小编码器的归一化和不同 batch size 特征一致性。无网络或模型 mock。
- `uv run --locked python -m pytest -q -W error`：5 passed，无警告（含两个现有训练测试）；`ty check scripts src/model` 和新评测文件的 F/UP/TRY004/SIM102/SIM115/SIM117 规则检查通过。
- 实际只评测 dev，未使用 test 选模型：现有 `outputs/siglip-tiny-flickr8k/best.pt` 为 epoch 10，完整候选库为 1000 图片/5000 描述；image_to_text R@1/5/10 为 0.00% / 0.40% / 0.80%，text_to_image 为 0.10% / 0.50% / 1.00%。batch size 16、32 的报告指标相同。
- 另用真实 checkpoint 和数据验证分开编码后的分数，经相同缩放/偏置后与模型原 forward 对齐，真实特征为有限值。指标接近随机检索水平，说明当前 checkpoint 的检索效果较弱；未据此擅自修改算法或断言具体训练原因。
- 按 README 的最新维护约定，将说明写入本文件，不再创建已移走的 AGENTS.md；README、训练产物和项目数据均保持不变。

## 学习版训练脚本：以主流程可读性为先

- 按用户要求将 `scripts/train.py` 简化为 193 行；以下旧训练记录中的严格恢复、防覆盖、原子保存等行为仅供历史追溯，不再是当前入口的保证。
- 当前文件只有图片 collator、参数解析、`train_one_epoch()`、`evaluate()`、`main()`。main 按“配置和设备、数据、模型与优化器、基本恢复、训练验证与保存”五段排列；没有把工程逻辑转移到其他模块。
- 保留 Tiny/Small、Flickr8k/COCO、CPU/CUDA、自动 FP32/BF16/FP16、AdamW 参数分组、训练随机描述/验证固定首条描述、训练 drop_last 和完整验证尾批次。编码器和 SigLIP 损失公式保持不变。
- 删除自定义 YAML Loader 与整套配置 schema 检查、训练设置比对、目录/历史日志保护、RNG 全状态保存恢复、临时文件原子 checkpoint、资源复制、run.json、global_step 和计时信息。改为 yaml.safe_load 和直接 torch.save；请继续使用独立的 tiny.yaml / small.yaml，而不是重复键的 siglip.yaml。
- 只保留训练批次至少包含两张图且确实存在一个 batch 的必要前提，以及训练接口 loss 非 None 的断言；其他无效输入由 Python / PyTorch 的自然错误暴露，不再有统一异常包装。
- 训练与验证分成两个直线循环。训练中明确显示 zero_grad、forward、backward、step；AMP 的 scaler 在 FP32/BF16 时禁用，不再围绕可选 optimizer/scaler 编写同一个复杂循环。
- 输出为 last.pt、best.pt 和仅含 epoch/train_loss/val_loss 的 metrics.jsonl。checkpoint 保留 epoch、模型、优化器、scaler、best_val_loss、config；同目录恢复保留旧 best 指标，新目录恢复则从新阶段重新选择 best，不再搬运文件。
- 基本恢复加载权重、优化器、scaler 和已完成 epoch；不恢复 RNG/样本顺序，不保证与不中断训练逐位一致。允许改 batch_size 等训练参数，不再严格匹配；优化器中的学习率、weight_decay 等设置沿用 checkpoint，恢复时会覆盖新建 optimizer 的命令行初值。
- 同一目录重新运行会覆盖 last/best，metrics 继续追加；保存不再是原子的，写入中断可能损坏该 checkpoint。学习实验请给不同运行使用不同 output_dir。预处理资源仍使用 PROCESSOR_NAME 对应的本地缓存或正常下载，不再复制到输出目录。
- 现有主要命令行参数保留，例如：

```bash
uv run python scripts/train.py \
  --config configs/tiny.yaml --dataset flickr8k \
  --batch_size 8 --num_epochs 10 --output_dir outputs/tiny-flickr8k

# num_epochs 仍是总 epoch 数，不是追加数量。
uv run python scripts/train.py \
  --config configs/tiny.yaml --dataset flickr8k \
  --batch_size 8 --num_epochs 20 --resume outputs/tiny-flickr8k/last.pt
```

- 删除只针对已移除工程契约的测试：精确 RNG 重放、严格配置/优化器设置拒绝和全面非法参数防护。保留两个真实离线训练回归，覆盖双塔参数更新、配置对应的实际模型形状，以及改 batch 后基本恢复的 epoch/Adam 更新进度。
- `uv run --locked python -m pytest -q -W error`：2 passed，无警告；`uv run --locked ty check scripts src/model` 通过；`ruff check --select F,UP,TRY004,SIM102,SIM115,SIM117 scripts/train.py tests/test_train.py` 通过。
- 真实 GPU 小样本：Tiny/Flickr8k BF16，第一轮 batch 2 后以 batch 4 恢复，Adam 更新次数从 2 继续到 3；另用保存了零学习率的 checkpoint 验证恢复到新目录确实加载原权重和优化器设置，权重逐元素不变且 Adam 进度继续增加。
- 真实 Small/COCO、两个数据 worker、CUDA FP16 的训练、验证尾批次和 checkpoint 均通过。仅做小样本 smoke，临时产物已清理，没有启动全量预训练。
- README、配置、模型、数据及依赖保持不变；本次仅简化训练脚本、调整对应测试并记录说明。

## 剩余 Ruff 诊断与 ty 类型表达式配置

- 按最新 `tmp.txt` 修复 `scripts/prepare_data.py` 和 `scripts/train.py` 的 5 处诊断，没有关闭这些 Ruff 规则，也未改写用户的其他格式化设置。
- COCO 的 images/annotations 容器类型错误、非字符串 caption 改为 `TypeError`；CLI 同步捕获 `TypeError`，保留简洁错误信息和旧清单，不因异常类型调整输出 traceback。
- 图片临时文件直接由 `with NamedTemporaryFile(...)` 管理。在打开 ZIP 成员前记录临时路径，关闭文件后才解码校验和原子替换；finally 同时覆盖 ZIP 成员打开失败、CRC 读取失败及图片解码失败的清理。路径赋值使两次上下文获取之间存在必要的顺序，不盲目合并而遗漏 entry 失败清理。
- 训练输出目录保护合并为单个复合条件，保留短路逻辑：新运行或恢复到不同目录时才检查已有 last/best checkpoint；同目录恢复不被误拒绝。
- 用户的“忽略变量 type”按变量/表达式被用于类型注解的 `invalid-type-form` 配置。在现有 `include = ["src/", "scripts/"]` 的 ty overrides.rules 中添加 ignore，保留 invalid-mutable-override 和其他设置；未关闭 invalid-assignment、invalid-argument-type 等规则。此配置不保证动态注解在运行时必然有效。
- `uv run --locked ruff check --select TRY004,SIM102,SIM115,SIM117 scripts`：All checks passed；本次验证范围为报告涉及的四条 Ruff 规则。
- `uv run --locked ty check scripts src/model`：All checks passed；临时独立项目另验证类型表达式诊断只在覆盖范围内忽略，赋值类型错误及范围外的类型表达式错误仍然报告。
- `uv run --locked python -m pytest -q -W error`：6 passed，无警告。
- 临时小 ZIP 运行实际数据准备 CLI：五份清单正常生成，重复运行保留 JPEG 字节和修改时间；两处坏标注分支均抛 TypeError，CLI 非零退出且旧清单不变。真实构造 ZIP header 不一致和 CRC 错误，均保留旧图并清理临时文件。
- 训练入口另验证已有 checkpoint 时拒绝无 resume 覆盖，原字节保留，未创建新的训练资源。所有 smoke 目录已清理，项目数据、依赖、锁文件和 README 未修改。

## PyTorch forward 的 ty 配置

- 按用户要求，在 `pyproject.toml` 添加 `tool.ty.overrides`，仅对 `src/model/**` 和 `scripts/test_fun.py` 忽略 `invalid-mutable-override`，处理 PyTorch 可写 `nn.Module.forward` 属性与子类方法覆盖的诊断冲突。
- 这是一条按文件生效的规则覆盖，不是仅按方法名屏蔽；范围内的其他可变属性覆盖诊断也会被忽略。其他规则和范围外文件保留原检查行为，没有将 PyTorch 导入替换成 Any。
- `uv run --locked ty check scripts src/model`：All checks passed。未修改运行时代码、依赖或锁文件。

## 按 `tmp.txt` 修正脚本类型诊断

- 以用户提供的 30 条诊断为依据，未先重跑检查确认已报告错误。仅修改 `scripts/test_fun.py` 和 `scripts/train.py`，没有改动模型、数据准备、依赖、类型检查规则或 `tmp.txt`。
- PyTorch 将 `nn.Module.forward` 声明为可写的 `Callable[..., Any]` 属性；`test_fun.py` 将原两层计算保留在 `_forward()`，并在构造函数中绑定 `self.forward = self._forward`，避免方法覆盖可写属性的类型冲突。没有扩大 Tensor 参数/返回类型或添加忽略规则。
- DataLoader 不再解包混合类型的普通 `common` 字典，改为分别显式传入参数；两个 loader 仍共享同一个 collator 和 generator，shuffle、drop_last、worker 和 pin_memory 行为保持不变。
- `run_epoch()` 明确 optimizer / GradScaler 的可选类型和标量 Tensor loss 类型；训练缺少 scaler 时明确报错，只有 optimizer 与 scaler 均存在的分支才执行梯度更新。验证仍允许不传两者，FP32/BF16 训练继续使用 disabled GradScaler。
- 恢复相关路径操作直接由 `resume_path is not None` 保护，不再要求类型检查器推断 checkpoint 与路径变量之间的隐含关系；同目录与新目录恢复逻辑保持不变。
- `uv run --locked ty check scripts`：All checks passed，没有 `type: ignore`、宽泛 cast 或禁用诊断。
- `uv run --locked python -m pytest -q -W error`：6 passed，无警告，包含 CPU 连续训练与恢复的一致性。
- 实际执行 `test_fun.py`，绑定的 forward 与原有 Linear/ReLU/Linear 计算逐元素一致，参数梯度有限；缺少 scaler 的训练调用得到明确的前置条件错误。
- 实际 Tiny/Flickr8k CUDA FP16 smoke：train 4 / val 3 / batch 2，第一轮 global_step=2，恢复到新目录后第二轮 global_step=4；验证尾批次、best checkpoint 和预处理资源保留正常。临时训练产物已清理，没有启动全量训练。

## 单卡训练入口：`scripts/train.py`

- 支持真实 Flickr8k / COCO 图文训练与每轮验证；保留用户的编码器结构、零初始化 logit scale/bias 和 `-logsigmoid` 损失公式，没有加载预训练编码器权重。
- 修复原来 `uv run python scripts/train.py` 无法导入项目 `src` 的启动问题；同时支持 `python -m scripts.train`。相对配置、输出、恢复路径均按项目根目录解释。
- `SigLIPModel(config=None, tokenizer=None)` 可接收已有 `tiny.yaml` / `small.yaml` 的整体配置，实际设置层数、隐藏维度、头数、MLP、图像/patch 尺寸、文本长度和投影维度；无配置时保留原默认模型。只加载 tokenizer 与 image processor 资源，不加载 Google 模型权重。
- 模型内部 token IDs 移至文本编码器所在设备；positive mask 校验精确形状和 bool/0/1 取值，并对齐 logits 的设备及 dtype。未修改损失的数学定义。
- 训练 Dataset 随机选取该图的一条描述；验证 Dataset 通过新增的 keyword-only `random_caption=False` 固定取第一条描述，默认 Dataset 行为不变。
- 图片经官方 SigLIP image processor 做确定性 resize/归一化，文本原样交给模型内 tokenizer；没有重复分词或重复归一化。训练与验证均没有额外随机图像增强。
- Flickr8k 使用 train/dev，COCO 使用 train2017/val2017；不读取 Flickr8k test 进行训练或选模型。每图一条描述、无替换采样，所以同一批次使用单位矩阵作 positive mask。
- AdamW 固定学习率，二维及以上参数使用指定 weight decay，一维参数（含归一化、偏置、logit 参数）不衰减。当前不使用 LR scheduler、梯度累积或多卡分布式训练。
- `--precision auto`：CPU 为 FP32，CUDA 支持 BF16 时使用 BF16，否则使用 FP16；也可显式选择。FP16 使用 GradScaler，并且只有真实执行优化器更新才增加 global_step。
- 训练 `drop_last=True`，避免最后不足 batch 的负样本数变化；训练数据不足一个 batch 或 batch_size 小于 2 会明确报错。验证保留最后一个不完整 batch。
- 日志中的 val_loss 为固定首条描述、批内配对损失的逐图加权平均，不是全数据集检索 Recall@K；负样本数量受 batch_size 影响，不应直接跨 batch_size 比较损失。
- `configs/siglip.yaml` 目前存在重复顶层键；训练配置读取拒绝重复 YAML 键，避免无声覆盖。请使用两个独立的 tiny/small 配置，不修改用户原配置文件。

### 训练命令

```bash
# 小样本试跑，不修改清单或原图；输出与正式训练分开。
uv run python scripts/train.py \
  --config configs/tiny.yaml --dataset flickr8k \
  --batch_size 4 --num_epochs 1 --train_limit 16 --val_limit 8 \
  --output_dir outputs/smoke-tiny

# 完整 Flickr8k，默认 auto 选择 CUDA 和可用的混合精度。
uv run python scripts/train.py \
  --config configs/tiny.yaml --dataset flickr8k \
  --batch_size 8 --num_epochs 10 --num_workers 2 \
  --output_dir outputs/tiny-flickr8k

# COCO / Small，同一套训练代码；按实际显存调整 batch_size。
uv run python scripts/train.py \
  --config configs/small.yaml --dataset coco \
  --batch_size 8 --num_epochs 10 --num_workers 2 \
  --output_dir outputs/small-coco

# 接着训练至总计 20 个 epoch，不是额外运行 20 个。
uv run python scripts/train.py \
  --config configs/tiny.yaml --dataset flickr8k \
  --batch_size 8 --num_epochs 20 --num_workers 2 \
  --resume outputs/tiny-flickr8k/last.pt
```

- 默认 `--lr 1e-4`、`--weight_decay 0.01`、`--seed 42`、`--num_workers 0`、`--batch_size 32`；输出目录不指定时为 `outputs/<config.name>-<dataset>`。`--device cpu --precision fp32` 可显式使用 CPU。
- `--train_limit` / `--val_limit` 为 0 表示全量；正数只限制本次 DataLoader 使用的数据，不改写数据准备清单。小样本 checkpoint 不能通过直接去掉 limit 续训为全量；正式训练应使用新输出目录从头开始，避免误认为恢复了同一训练过程。
- 输出 `last.pt`、按最低 val_loss 保存的 `best.pt`、`metrics.jsonl`、`run.json`、`tokenizer/`、`image_processor/`。首次运行需要相应 Hugging Face 预处理资源已缓存或网络可用。
- checkpoint 包含模型、AdamW、GradScaler、已完成 epoch、真实 global_step、最佳 val_loss、模型配置、训练设置、Python/Torch/CUDA RNG 和 DataLoader generator 状态。每轮验证成功后原子写入，临时文件在失败时清理。
- 恢复发生在完整 epoch 边界；中途退出只能从上一个已保存 epoch 恢复，不能恢复到任意 batch。模型配置、dataset、batch_size、lr、weight_decay、worker 数、seed、数据限制和实际 precision 必须匹配；允许增大总 epoch 数。CUDA 不承诺跨硬件/内核的逐位一致性。
- 无 `--resume` 时拒绝覆盖已有 checkpoint；从较旧 best checkpoint 恢复到含较新 metrics 的同一目录也会拒绝，可选择新输出目录。恢复到新目录时携带原 best 权重和预处理资源；请保留原运行目录的 tokenizer、image_processor、best.pt。

### 训练实际验证

- `uv run --locked python -m pytest -q -W error`：6 passed，无警告。永久测试使用本地生成的真实 tokenizer、image processor、小 JPEG 和小模型，无外网及模型/优化器 mock；验证两个编码器权重更新、CPU epoch 边界恢复与连续训练的参数/优化器/非计时指标完全一致，以及不匹配恢复和非法 batch 的保护。
- 实际 RTX 4060 Laptop GPU：Tiny/Flickr8k，真实训练/验证各 4 张、batch 2，BF16 训练完成两个 optimizer steps，随后恢复到第二轮，global_step 连续到 4；两个编码器权重与 logit scale/bias 均继续更新。该配置实际参数量 17,385,986。
- 同一 Tiny checkpoint 恢复到新输出目录并完成第三轮，global_step 到 6，预处理资源和 best checkpoint 正常保留。
- 实际 Small/COCO：CUDA FP32、两个数据 worker、train_limit 5 / val_limit 3 / batch 2，训练实际使用 4 张并完成两个更新，验证完整包含 3 张尾批次；参数量 46,546,946。
- 实际 Tiny/Flickr8k CUDA FP16：train 8 / val 4 / batch 2，完成 4 个更新，GradScaler 状态保存正确，模型权重有限。
- 上述仅为小样本功能 smoke，不是正式预训练或效果评估；临时输出和 checkpoint 已清理，没有改写项目数据或启动全量长训练。
- README、模型配置、encoder.py、transformer.py、依赖和锁文件保持不变；详细训练用法仅记录在本文件。

## 编码器合并：统一到 `src/model/encoder.py`

- `encoder.py` 只定义 `VisionEncoder` 和 `TextEncoder`；共享的 `SelfAttention`、`TransformerBlock` 继续位于 `transformer.py`，未修改其实现。
- 原 `PatchEmbedding` 的卷积在 `VisionEncoder.__init__()` 中注册为 `patch_projection`，计算过程收进 `_embed_patches()`；原 `AttentionPooling` 的 query、Attention、LayerNorm、MLP 分别注册为 `pool_query`、`pool_attention`、`pool_norm`、`pool_mlp`，计算过程收进 `_pool_tokens()`。
- 保留两个编码器的构造参数、默认值、初始化顺序、输入检查和计算逻辑；保留视觉 `forward_features()`、MAP 残差，以及文本末位置池化和输出投影。
- `siglip.py`、`src/loss.py`、`scripts/test_model.py` 的导入统一改为 `from src.model.encoder import TextEncoder, VisionEncoder`。旧 `vision_encoder.py`、`text_encoder.py` 已移除，没有旧模块转发或兼容别名。
- 视觉 `state_dict` 键名变化：`patch_embedding.proj.*` → `patch_projection.*`，`pool.query` → `pool_query`，`pool.attention.*` → `pool_attention.*`，`pool.norm.*` → `pool_norm.*`，`pool.mlp.*` → `pool_mlp.*`；旧视觉 checkpoint 不可假定能够直接严格加载。文本参数键名不变。
- 使用临时基线验证四组配置：视觉与文本各包含一个小配置和一个 192 维、两层的常规配置。相同参数、输入下，前向输出、所有参数梯度、视觉输入梯度、一次 SGD 更新后的参数与输出均逐元素完全一致；视觉 token 特征也完全一致。参数对象在重复前向中保持注册稳定，CPU `float64` 转换后仍可运行；未验证 CUDA。
- 本次只做组织重构；`siglip.py` 仍通过 Hugging Face 构建双塔，`src/loss.py` 的待实现内容保持原样，未代替用户完成后续算法。`README.md` 保持不变。
- 通过 `runpy.run_module("scripts.test_model", run_name="__main__")` 实际执行原有编码器烟测，前向及反向检查通过；`src.loss` 和 `src.model.siglip` 的迁移后导入通过。临时等价性基线已清理，没有新增永久测试或验证脚手架。

## 2026-10-30：统一处理本地五个 ZIP

- 新入口为 `scripts/prepare_data.py`，只读取 `data/` 中的本地压缩包，不访问网络、不下载数据、不修改依赖。下方 2026-10-29 的旧入口记录仅用于历史追溯，当前使用本节命令。
- 输入：`Flickr8k_Dataset.zip`、`Flickr8k_text.zip`、`train2017.zip`、`val2017.zip`、`annotations_trainval2017.zip`。
- Flickr8k 使用原始 token captions 与 train/dev/test 划分，保留每图五条描述；当前压缩包元数据为 6000/1000/1000 张。划分之外的额外图片和标注不进入清单。
- COCO 使用 captions 标注，按原始数字 image ID 排序，保留每图全部有效描述；默认包含完整的 118287 张 train 和 5000 张 val，不再默认截取十万张。实例分割、关键点等标注不解压。
- 图片保留原始文件名与 JPEG 字节；Flickr8k 输出到 `data/flickr8k/{train,dev,test}/images/`，COCO 输出到 `data/coco/{train,val}2017/`。
- 清单写入 `data/manifests/flickr8k_{train,dev,test}.jsonl` 和 `data/manifests/coco_{train,val}2017.jsonl`，字段为 `source`、`split`、`image_id`、`image_path`、`captions`。`image_path` 相对于项目根目录。
- `image_id` 在各分割内从零连续编号，如 `flickr8k_train_000000`、`coco2017_train_000000`；原图文件名用于追溯。读取方式仍为 `ImageCaptionDataset(manifest, root=项目根目录)`，不需要修改 `src/data.py`。

### 当前使用方式

```bash
# 完整处理两个数据集；默认 --dataset all、--limit 0。
uv run python scripts/prepare_data.py

# 只处理一个数据集，不改动另一个数据集的清单。
uv run python scripts/prepare_data.py --dataset flickr8k
uv run python scripts/prepare_data.py --dataset coco

# 每个分割只准备前 10 张图片，不解压未选中的图片。
uv run python scripts/prepare_data.py --limit 10
```

- 项目根目录由脚本位置确定，不依赖启动时的工作目录。
- `--limit` 成功运行后会替换所选分割的清单，日志会明确警告；完整训练前必须重新运行不带 `--limit` 的命令。限量运行仍检查完整标注、划分与图片成员引用，但不会解压并验证未选中的 JPEG 内容。
- 重跑时按解压大小与 CRC32 检查已有图片，匹配时复用、不重写，并检查图片可解码；不匹配时从 ZIP 读取、检查 CRC 与图片解码后原子替换。保留原始 ZIP。
- 仅提取需要的图片成员，不使用无条件 `extractall()`；拒绝标注中的路径穿越、ZIP 图片符号链接，以及逃出数据目录的输出路径。
- 所选分割全部处理成功后才发布清单；元数据、缺图、CRC 或解码失败时不替换旧清单，并清理临时文件。失败前已经解出的有效图片会保留供下次复用。

### 当前实际验证

- 五个真实 ZIP：`uv run --locked python scripts/prepare_data.py --limit 3` 成功，五个分割各准备 3 张图片。
- `uv run --locked python -m pytest -q -W error`：13 passed，无警告；覆盖全量/限量、独立数据集、连续编号、原始字节、重复运行、同大小 CRC 不符的本地图像修复、缺图/空描述/坏图/重复划分/不安全路径/坏 ZIP 的失败保护。
- 从项目外运行同一命令再次处理：15 张图片全部通过校验并复用，大小与修改时间不变。
- 五份实际小样本清单均通过现有 `ImageCaptionDataset` 和 `DataLoader` 读取，图片为 RGB，文本对应本条记录，连续编号全局唯一。
- 临时小 ZIP 运行实际 CLI 的默认全量路径：五份清单、COCO 数字排序、每图全部描述（包含六条描述的样本）均正确；未提取无关标注和额外的路径穿越成员。
- 临时小 ZIP 在最后一个分割制造损坏 JPEG：命令非零退出并指出具体成员，五份原清单及原有图片均保留，临时文件已清理。
- 当前真实数据产物仅为每分割三张的 smoke 样本，尚未运行真实数据的全量解压；正式训练前执行本节第一条完整命令。
- `README.md` 已包含正确的新入口命令，按文档维护要求保持不变；详细参数和调试记录仅写在本文件。

## 2026-10-29

- 环境：`uv 0.10.12`、Python 项目 `.venv`、`datasets 4.8.5`；`uv run --locked` 可正常使用现有锁文件，没有修改依赖。
- `scripts/prepare_flickr8k.py`：真实执行 `--limit 1 --splits train` 复现 `AttributeError: 'IterableDataset' object has no attribute 'select'`。流式数据集也不支持 `len()`，所以不传 `--limit` 时原来的结束计数同样有问题。
- Flickr8k 修复：限量使用 `take()`，按实际迭代记录数打印结果；沿用现有 `scripts/check_data.py` 的临时清单方案，成功完成后再替换清单，下载失败时保留旧清单。没有修改 `check_data.py`。
- `scripts/download_coco.sh`：原来的 `https://images.cocodataset.org/...` 在当前网络下复现 TLS 主机名不匹配，`curl` 退出码 60。
- COCO 修复：改用同一官方 bucket 的 `https://s3.amazonaws.com/images.cocodataset.org/...`，保留 TLS 校验和断点续传，下载后校验 ZIP，校验失败不解压。
- `scripts/prepare_coco.py` 只读取已解压的数据并生成清单，不下载数据；无需修改。

### 验证记录

- `uv run --locked python -m pytest -q`：6 passed，包含新增 Flickr8k 流式限量、全量、限量超过数据总量，以及流中断/空数据不覆盖旧清单的回归用例。
- `bash -n scripts/download_coco.sh`：通过。
- `uv run --no-sync python scripts/prepare_flickr8k.py --limit 1 --splits train dev test`：三个分割各下载 1 张真实图片，每张含 5 条 captions；随后通过 `ImageCaptionDataset` 读取图片与文本。这些小样本保留在项目 `data/` 下。
- COCO 三个远程 ZIP 分别请求前 1024 字节，均返回 HTTP 206 和正确的 Content-Range。未下载完整 COCO。
- 临时目录运行实际 COCO shell 脚本：本地 HTTP Range 服务提供小 ZIP；从 31 字节残片续传、完整 ZIP 校验、解压和重复运行跳过下载均通过。临时文件已清理。
- 临时 COCO 图像与标注运行实际 `prepare_coco.py --train-limit 1`：训练采样固定，验证集保持独立，生成路径和 5 条 captions 正确，图片可读；临时文件已清理。

### 使用方式

在项目根目录运行：

```bash
# 小样本检查；--limit 是每个分割的上限。
uv run python scripts/prepare_flickr8k.py --limit 10
# 完整 Flickr8k：train 6000 / dev 1000 / test 1000。
uv run python scripts/prepare_flickr8k.py

# COCO 必须先下载并解压，再生成清单。
bash scripts/download_coco.sh
uv run python scripts/prepare_coco.py
```

- Flickr8k 每次成功运行会替换所选分割的清单；小样本检查后，训练前应重新运行不带 `--limit` 的命令。
- COCO 默认清单选取 100000 张训练图片与全部 5000 张验证图片；`--train-limit 0` 可选择全部训练图片。限量参数不减少 ZIP 下载量。
- COCO 压缩包合计约 20.4 GB（十进制），脚本会同时保留 ZIP 和解压数据，需额外预留解压空间。
- 按 `READEME.md` 中的要求，仅在本文件记录调试说明，没有修改该文档。