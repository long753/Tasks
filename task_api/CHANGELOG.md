# 实现与调试记录

## 当前关系协议：几何规则 + 稀疏语义提取

- `config/relations.yaml` 保存关系定义和可调提示词。配置直接读取，不增加配置校验框架；`on` 等 YAML 布尔值名称仍须加引号。
- `left_of`、`right_of` 按检测框中心在本地计算，不发送给 Qwen。每个无序对象对只保存一条成立关系：默认统一为 left_of；单独选择 right_of 时反向表达。中心相同不记录横向关系。
- Qwen 每个场景最多请求一次，只提取有明确图像证据的语义关系，默认是 on、holding。不再生成全量对象对问题、q1 编号或否定/未知回答；不确定的候选项直接省略。
- 未返回只表示没有提取出该关系，不能解释为已证明 no。输出用 `mode: sparse_positive` 标记此约定。
- 每条关系字段为 `subject_id,predicate,object_id,source,reason`，其中 source 为 geometry 或 qwen；不再有关系级 status。场景级 status 只表示 ok/no_pairs。
- 保留模型输出的必要检查：合法的当前整数 ID、不同主体/客体、允许的语义关系、非空理由和不重复的有向三元组。异常说明具体错误，不填充虚构关系、不自动重试。
- `analyze_scene()` 是图片、视频和单图 CLI 共同使用的入口，先计算几何关系，再按需执行一次语义提取。对象不足两个或没有配置语义关系时不请求 API。
- `save_relations()` 保存关系 JSON 和同字段的 CSV；API 调用数、几何/语义数量与实际模型、耗时、用量单独记录，不再记录旧查询数或批次数。

### 单图关系入口

```bash
uv run python scripts/relations.py \
  --detections outputs/image_main/img01_laptop_cup/objects.json \
  --relations left_of on holding \
  --output outputs/relations
```

默认分析整幅图。`--relations` 可筛选关系类型；旧的 `--pair` 已移除。`--dry-run` 只保存配置、最终提示词和请求说明，不发 API 请求，也不伪造最终语义结果。

关系模型从 `.env` 的 `QWEN_MODEL` 读取，`--model` 可覆盖；已有终端环境变量优先于 `.env`。纯几何分析无需 API 密钥。若代理变量使用不支持的 socks://，可用 `env -u ALL_PROXY -u all_proxy` 临时移除这两个变量。

## 图片批处理入口

`scripts/main_image.py` 按编号处理 img01—img10，排除 test.png。检测参数和路径约定不变，每张图通过 `analyze_scene()` 分析一次，不再有 `--batch-size`。

```bash
uv run python scripts/main_image.py --images 1 \
  --yolo-model models/yolo26n.pt \
  --output outputs/sparse_relations_test/image
```

默认输出仍为 `outputs/image_main/`，每图包含 original.png、detected.png、objects.json、request.json、relations.json、relations.csv；根目录包含 config.yaml 和 summary.csv。不生成 prompts/、responses/。

- `--model`/`QWEN_MODEL` 控制关系模型；`--yolo-model` 控制检测权重。原有默认检测权重、置信度和输入选择没有因关系协议升级而改变。
- summary.csv 记录 image、object_count、geometric_relation_count、semantic_relation_count、relation_count、api_call_count、status，不再统计 yes/no/unknown、查询数和批次数。
- 标注沿用黑底白字、单次绘制；只有已检测对象参与关系分析。空检测结果正常保存，API 调用数为 0。
- 当前实跑仅处理图片 1，检测到 3 个对象，保存 3 条几何关系，语义提取为空，实际请求 1 次；未重新处理其他图片。旧输出目录中的历史结果不会自动迁移。

## 视频检测与 ByteTrack 跟踪

`scripts/main_video.py` 现在分两阶段执行：逐帧 YOLO + ByteTrack 跟踪并按视频时间保存快照；跟踪结束后读取 `.env` 调用 Qwen 分析快照中的关系。不生成提示词或原始响应目录。

```bash
# 默认完整处理 video01；不会自动处理其他视频
uv run python scripts/main_video.py

# 只处理第一个视频的前 75 帧
uv run python scripts/main_video.py \
  --source data/videos/video01_book_place.mp4 \
  --max-frames 75

# 摄像头入口；需要实际摄像头和桌面环境
uv run python scripts/main_video.py --source 0 --show
```

- 默认使用 `models/yolo26n.pt`，可通过 `--model` 更换检测权重；Qwen 模型使用 `.env` 的 `QWEN_MODEL`，可用 `--relation-model` 临时覆盖。
- 模型只加载一次，OpenCV 逐帧读取，调用 `model.track(frame, tracker="bytetrack.yaml", persist=True)`，读取 `result.boxes.id`。`persist=True` 保留相邻帧的轨迹，编号不由 `enumerate()` 产生。
- 每次运行只打开一个视频源并新建模型/跟踪器；ID 只在当前视频内有意义，可能跳号，不保证离开画面再出现仍沿用原 ID。
- 当前默认检测置信度为 0.4，可通过 `--conf` 调整；使用 Ultralytics 自带 `bytetrack.yaml`，没有另写跟踪算法或自动调参。
- 复用 `detect.draw_object()` 绘制黑底白字标签，绘图时把轨迹 ID 传给 `object_id`，保存的数据字段仍为 `track_id`；不叠加 `result.plot()`。
- 只保存当前帧已确认且匹配到的轨迹；没有跟踪对象时仍写一帧记录，`objects` 为空，不编造 ID。
- 默认不打开窗口，适合无桌面环境；`--show` 打开预览，按 q 正常结束并保存汇总。摄像头预览速度受采集、推理和显示共同影响，不承诺所有设备都实时。
- 默认用输入帧率保存视频；无法读取帧率时需显式提供 `--fps`。该参数只控制输出播放帧率，不提高推理速度。

输出位于 `outputs/video_main/<视频名>/`，摄像头使用 `camera_<编号>`：

```text
outputs/video_main/video01_book_place/
  tracked.mp4
  tracks.jsonl
  summary.json
  config.yaml
  samples/sample_000/
    original.png
    detected.png
    objects.json
    relations.json
    relations.csv
```

- `tracked.mp4` 是带跟踪 ID、类别及置信度的画面，不包含原音频；重跑同一来源会覆盖这三个文件。
- `tracks.jsonl` 每行对应一帧，含从 1 开始的 `frame_index`、`timestamp_seconds` 和对象列表。文件时间戳来自输入视频的解码时间；摄像头使用本次处理开始后的采集时刻。对象包含 `track_id`、类别、置信度和原图坐标下的 `bbox_xyxy`。
- `summary.json` 记录帧数、画面尺寸、输入/输出 FPS、输出时长、处理耗时、平均处理 FPS、轨迹数和各轨迹的首末帧、出现帧数。轨迹摘要的 `class_name` 是首次出现时的类别，逐帧类别以 JSONL 为准。
- 处理 FPS 仅衡量第一阶段的读帧、检测跟踪、绘制、采样图片、视频和 JSONL 写入，不包含模型加载及第二阶段 Qwen 分析；轨迹数量不等于真实物体数量，类别正确性也不由 ByteTrack 保证。

### 早期视频跟踪验证记录

- 本次仅使用 `video01_book_place.mp4` 的开头，主验证处理前 75 帧；没有运行 video02、video03 或完整视频。
- 输出视频可完整解码 75 帧，720×1366、25 FPS、3 秒；JSONL 恰有 75 行，时间戳为 0 至 2.96 秒、间隔 0.04 秒，视频和汇总计数一致。
- 观察到 7 条轨迹，ID 1、3 在全部 75 帧持续出现；已检查每帧无重复 ID，轨迹首末帧与出现次数一致，并查看标注视频预览。
- 本次平均处理速度约 32.08 FPS，高于源视频的 25 FPS，仅代表当前环境和这段短片。
- 当前 YOLO 权重存在误分类和类别跳变：同一跟踪区域出现 book/knife/laptop、dining table/cake/pizza 等类别变化。未用类别平滑掩盖检测结果，也不能据此断言 ID 对应的类别一定正确。
- 另用同一视频前 5 帧、置信度 1.0 验证空轨迹分支：逐帧对象列表为空、无虚构 ID；显式输出 20 FPS 时，输入时间戳仍保持 25 FPS 的间隔。临时输出已清理。
- `--max-frames 0` 在打开视频前报错。摄像头和桌面窗口预览入口已实现，但本次没有进行硬件或窗口测试。

## ty 与 Ruff 检查修正

- 按现有 `pyproject.toml` 配置格式化 `scripts/` 下四个脚本，没有新增忽略规则、关闭诊断或修改业务参数。
- `main_video.py` 显式区分整数摄像头编号与字符串视频路径，避免将 `int | str` 直接传给 `Path()`。
- 使用 OpenCV 已声明且运行可用的 `cv2.VideoWriter.fourcc()`，替换旧的模块级名称。
- 用 `next(iter(results))` 取得本帧首个跟踪结果，兼容 Ultralytics 声明的列表和迭代器返回类型，不强制转换或复制整个结果列表。
- 为轨迹统计字典补充 `TrackHistory` 类型，明确 `frames_seen` 等计数为整数；不改变实际 JSON 字段或轨迹更新逻辑。
- 已通过 `uv run ty check`、`uv run ruff check` 和 `uv run ruff format --check scripts`。
- 冒烟验证使用图片 1 和视频 1 前 25 帧：检测图与 JSON 正常；关系 dry-run 生成指定查询；视频可完整解码，时间戳、轨迹首末帧及计数一致；图片批处理的空检测分支正常结束。没有调用 Qwen、运行全部素材或覆盖现有输出，临时产物已清理。

## 按时间采样与已有快照重分析

```bash
# 常规入口：逐帧跟踪，每秒采样，然后提取稀疏关系
uv run python scripts/main_video.py \
  --source data/videos/video01_book_place.mp4 \
  --relation-interval 1.0

# 已有快照时直接重做关系分析，不重复运行 YOLO/ByteTrack
uv run python scripts/main_video.py \
  --relations-only outputs/video_main/video01_book_place
```

- `track_video()` 保持逐帧跟踪，从视频时间 0 秒开始采样，不依据处理速度或输出 FPS。只有采样帧在绘图前复制原图；大时间跳跃不补造历史快照。
- `samples/sample_XXX/` 按采样序号命名，objects.json 和 relations.json 均记录计划时间、实际视频时间和从 1 开始的帧号。保留 track_id，关系接口使用相同值的 object_id，不重新编号。
- 正常入口在跟踪结束后执行关系分析；摄像头模式同样在停止采集后才分析，不是异步实时语义判断。
- `--relations-only` 读取指定目录 summary.json 的 samples 清单，不扫描历史遗留目录；只更新关系 JSON/CSV、配置快照和汇总，不改写视频、逐帧记录、对象信息或图片。
- `--model` 仍是 YOLO 权重；关系模型优先级为 `--relation-model` > 当前环境 QWEN_MODEL > .env。可使用 --config、--max-tokens（默认 4096）、--timeout（默认 120）；旧 --batch-size 已移除。
- 视频和图片共用同一稀疏协议。每个有效采样点最多一次语义请求；summary.json 汇总 relation_mode、relation_api_call_count、geometric_relation_count、semantic_relation_count、relation_count、relation_sample_count 和 relation_model。
- 不足两个对象时仍保存快照和空关系，api_call_count 为 0；未出现的对象和未返回的关系都不代表否定事实。

### 本次迁移验证

- 已直接重分析 video01 的 10 个既有快照，成功完成 9 次实际 Qwen 请求，保存 17 条几何关系和 2 条语义关系。没有重跑完整视频检测。
- 原跟踪视频、tracks.jsonl、10 份对象信息及 20 张原图/编号图共 32 个文件的 SHA-256 均未改变；每个采样点的 ID、时间、帧号和 JSON/CSV 一致。
- 0 秒处 3 个对象的新结果只有 3 条几何关系和 1 条 holding，语义请求 1 次；不再为 24 项全量问题生成回答。
- 单图 CLI 的纯 right_of 模式无需 API 密钥，语义 dry-run 不产生最终语义结果；同一视频前 5 帧与图片 1 的空检测分支，在不可达 API 地址下也正常完成，确认未请求模型。
- tests/test_relations.py 的 8 个回归测试覆盖几何方向、去除逆关系副本、同中心边界、纯几何零请求、稀疏空列表、非法 ID/关系/状态、重复三元组及旧编号映射拒绝。
- `uv run ty check`、`uv run ruff check`、`uv run ruff format --check scripts tests` 与 unittest 均通过。临时验证输出已清理；实际图片 1 结果保存在 outputs/sparse_relations_test/image，视频关系结果更新在原视频输出目录。

