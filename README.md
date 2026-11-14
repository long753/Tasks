# Learning Tasks

This repository brings together my learning tasks and experiments including:

- [task_api](task_api/): Detect and track objects in images and videos, then use a vision-language API to analyze their relationships.
- [task_cv](task_cv/): Compare object detection models on crack detection and explore Slot Attention for object-centric visual learning.
- [task_jepa](task_jepa/): Train and evaluate H-JEPA, HWM, and LeWM world models, with experiments on adaptive temporal hierarchies and hierarchical planning.
- [task_siglip](task_siglip/): Train and evaluate SigLIP models for image–text alignment and retrieval.
- [task_smolval](task_smolval/): Fine-tune SmolVLA for a Panda robot's cube-to-tray task in MuJoCo and compare performance before and after fine-tuning.

## Getting started

Each task has its own environment, dependencies, and run instructions; see its README before installing or running it. `task_jepa` is included as a regular directory, not a Git submodule.

Datasets, model checkpoints, experiment outputs, and local credentials are excluded from version control. Prepare the required data and models locally as described in each task.