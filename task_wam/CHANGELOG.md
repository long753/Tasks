# 变更记录

## 2026-11-21：迁入单层 WAM 基线

- 从 `task_jepa` 迁入 `src/jepa_wam`、FourRoom 仿真依赖和训练、批量评估、单任务录像入口，保留模型、训练目标与评估协议。
- 独立配置放在 `config/fourroom.yaml`。使用 Python 3.11、`uv` 锁文件和 CUDA 12.8 版 PyTorch；不依赖旧工程或 Conda 环境运行。
- 复制现有 WAM `last.pt`，从 FourRoom LeWM 离线导出 ViT 初始化权重；未复制数据和历史输出。`.gitignore` 原样复制，`README.md` 留空。
- 按要求未执行训练、评估、冒烟测试或 Ruff/ty 检查。
