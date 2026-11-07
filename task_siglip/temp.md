#import "@preview/quill-assignment:0.1.0": *

#show: assignment.with(
  title: "Embodied Artificial Intelligence\nfrom Scratch",
  subtitle: "Foundations, Architectures, Scaling, and Systems for Generalist Robot Policies",
  course: "EAI from Scratch",
  assignment: "Assignment",
  student: "Morethan",
  student-id: "2026-123456",
  instructor: "Dr. Mystery",
  department: "Department of Bed No.1",
  university: "Dormitory University",
  date: datetime.today(),
  theme: "nord-light",
  cover-page: true,
  cover-style: "geometric",
  toc: true,
  font: "Libertinus Serif",
  code-font: "DejaVu Sans Mono",
  margin: (x: 40pt, top: 50pt, bottom: 50pt)
)

#pagebreak()
= Syntax Examples

Here is normal content.

== Multilevel titles
=== Multilevel titles
==== Multilevel titles
===== Multilevel titles
====== Multilevel titles
======= Multilevel titles

Not recommended to use *more than 3 levels* of headings.

Answer or notes here. This is an inline formular $E=m c^2$. In Typst, the only warpper for formulas is the *signle* dollar sign.

== Machine Learning Foundations

Self-attention computes a weighted representation of every token by comparing queries and keys. #cite(<vaswani2017attention>, form: "prose") argue that removing recurrence lets every position be processed in parallel, which is the core efficiency gain of the Transformer. The residual connections that keep deep stacks trainable were established earlier in computer vision @he2016resnet, and the attention layer plays the same gradient-stabilizing role here.

Self-attention computes a weighted representation of every token by comparing queries and keys, entirely in parallel across positions. Because no recurrent state is carried through time, training parallelizes over the sequence length, which is the core efficiency argument made by the Transformer paper @vaswani2017attention. The residual connections that keep deep stacks trainable were established earlier in computer vision @he2016resnet, and the attention layer plays the same gradient-stabilizing role here.

A thorough treatment of the surrounding building blocks (embeddings, layer normalization, optimization) can be found in @goodfellow2016dl.

#question(title: "Gradient Derivation for Multi-Layer Perceptrons")[
  Consider a multi-layer perceptron (MLP) with activation function $sigma(z)$ and loss function $L(y, hat(y))$. Derive the parameter update rule for the weight matrix $W^((l))$ at layer $l$ using the chain rule.
]

#answer[
  The forward pass for layer $l$ is defined by:
  $ z^((l)) = W^((l)) a^((l-1)) + b^((l)), quad a^((l)) = sigma(z^((l))) $

  By applying the multivariate chain rule, the error term $delta^((l)) = (partial L) / (partial z^((l)))$ is computed recursively:
  $ delta^((l)) = ((W^((l+1)))^T delta^((l+1))) circle.tiny sigma'(z^((l))) $

  Consequently, the gradient with respect to the weight matrix $W^((l))$ is given by:
  $ (partial L) / (partial W^((l))) = delta^((l)) (a^((l-1)))^T $

  #note(type: "tip", title: "Numerical Stability")[
    When computing gradients for deep networks, use log-sum-exp stabilization to avoid floating-point overflow during softmax calculations.
  ]
]

== Optimization & Implementation

#question(title: "PyTorch Residual Layer")[
  Write a modular PyTorch module implementing a residual dense block with layer normalization and dropout.
]

#answer[
  ```python
  import torch
  import torch.nn as nn

  class ResidualBlock(nn.Module):
      """Residual dense block with LayerNorm and Dropout."""
      def __init__(self, dim: int, dropout: float = 0.1):
          super().__init__()
          self.fc1 = nn.Linear(dim, dim * 4)
          self.act = nn.GELU()
          self.fc2 = nn.Linear(dim * 4, dim)
          self.norm = nn.LayerNorm(dim)
          self.drop = nn.Dropout(dropout)

      def forward(self, x: torch.Tensor) -> torch.Tensor:
          residual = x
          x = self.drop(self.act(self.fc1(self.norm(x))))
          x = self.drop(self.fc2(x))
          return x + residual
  ```

  #note(type: "info", title: "Architecture Note")[
    LayerNorm is applied prior to activation (pre-LN configuration) to promote smooth gradient propagation in ultra-deep networks.
  ]
]

== System Configuration & Key Properties

#question(title: "Hyperparameter Specifications")[
  Summarize the hyperparameter setup and evaluation environment for the model training runs.
]

#answer[
  #kv-table(
    "Model Architecture",
    "Transformer-Encoder (12 Layers)",
    "Optimizer",
    "AdamW (weight decay = 0.01)",
    "Learning Rate",
    "3e-4 with Cosine Annealing",
    "Batch Size",
    "256 across 4 GPUs",
    "Precision",
    "Mixed Precision (FP16)",
  )
]

== Empirical Results & Benchmarks

#question(title: "Optimizer Performance Comparison")[
  Compare the empirical convergence speed and final accuracy across standard optimizers evaluated on ImageNet-1k.
]

#answer[
  #table(
    columns: (auto, auto, auto, auto),
    align: center + horizon,
    [*Optimizer*], [*Learning Rate*], [*Top-1 Accuracy*], [*Train Time*],
    [SGD + Momentum], [0.1], [76.4%], [12.4 hrs],
    [AdamW], [3e-4], [79.2%], [10.1 hrs],
    [Lion], [1e-4], [79.6%], [8.5 hrs],
  )

  #note(type: "warning", title: "Hyperparameter Sensitivity")[
    The Lion optimizer is notably sensitive to weight decay settings; excessive decay can lead to early convergence plateaus.
  ]
]

#pagebreak()
#include "chapters/overview.typ"

#pagebreak()
#include "chapters/robotic_control_and_simulation.typ"

#pagebreak()
#include "chapters/model_arch.typ"

#pagebreak()
#include "chapters/offline_learning.typ"

#pagebreak()
#include "chapters/data_engineering.typ"

#pagebreak()
#include "chapters/scaling_law.typ"

#pagebreak()
#include "chapters/systems.typ"

#pagebreak()
#include "chapters/online_learning.typ"

#pagebreak()
#bibliography("refs.bib", title: "References", style: "ieee")

这是 handout.typ 里面包括了基础的模板语法，handout.pdf 文件整体遵循：基础理论内容的讲解，穿插问题、代码实现

---

#import "@preview/quill-assignment:0.1.0": *

= Overview

这是 chapters/overview.typ 文件，是整个课程的开篇

---

#import "@preview/quill-assignment:0.1.0": *

= Robotic Control and Simulation

== 1. Introduction: From Actions to Physical Motion

== 2. Robot Mechanics and Rigid-Body Dynamics
=== 2.1 Generalized Coordinates and Degrees of Freedom
=== 2.2 Equations of Motion
=== 2.3 Numerical Integration
=== 2.4 Collision and Contact Dynamics

== 3. Spatial Geometry and Coordinate Representations
=== 3.1 Rotation Matrices and SO(3)
=== 3.2 Homogeneous Transformations and SE(3)
=== 3.3 Coordinate Frames and Transformation Pipelines
=== 3.4 Euler Angles and Gimbal Lock
=== 3.5 Unit Quaternions and Double Cover
=== 3.6 Continuous 6D Rotation Representations

== 4. Robot Kinematics
=== 4.1 Manipulator Configuration and Joint Models
=== 4.2 Forward Kinematics
=== 4.3 Geometric and Analytical Jacobians
=== 4.4 Singularities and Manipulability

== 5. Inverse Kinematics
=== 5.1 Analytical Inverse Kinematics
=== 5.2 Jacobian Pseudoinverse
=== 5.3 Damped Least-Squares IK
=== 5.4 Redundancy Resolution and Null-space Control

== 6. Robot Control
=== 6.1 Joint-space Position and Velocity Control
=== 6.2 Cartesian Motion Control
=== 6.3 Operational Space Control
=== 6.4 Joint and Cartesian Impedance Control

== 7. Contact Dynamics and Robotic Grasping
=== 7.1 Collision Detection
=== 7.2 Contact Constraints and Friction
=== 7.3 Grasp Stability

== 8. Building a Simulation Environment
=== 8.1 MuJoCo Model and Simulation Pipeline
=== 8.2 State and Observation Extraction
=== 8.3 Camera Rendering and Coordinate Calibration
=== 8.4 Cartesian Delta Action Space
=== 8.5 Gymnasium Environment Interface

== 9. From-Scratch Implementation
=== 9.1 SE(3) Transformation Utilities
=== 9.2 Forward and Inverse Kinematics
=== 9.3 Custom Cartesian Controller
=== 9.4 Pick-and-Place Simulation Sandbox
=== 9.5 Evaluation and Reproducibility

== 10. High-level Robotics Utilities
=== 10.1 Pinocchio
=== 10.2 PyTorch3D
=== 10.3 MuJoCo Playground and ManiSkill3

== 11. Further Reading
=== 11.1 Classical Robotics
=== 11.2 Modern Robot Simulation
=== 11.3 Differentiable Physics and Geometric Learning

这是 chapters/robotic_control_and_simulation.typ 文件，包括了你之前提到的相关章节内容

---

╭─ morethan@myarch  ~/GitHub/eai-from-scratch
╰─❯ tree
.
├── CHANGELOG.md
├── handout
│   ├── assets
│   │   └── neural_net.jpg
│   ├── chapters
│   │   ├── data_engineering.typ
│   │   ├── model_arch.typ
│   │   ├── offline_learning.typ
│   │   ├── online_learning.typ
│   │   ├── overview.typ
│   │   ├── robotic_control_and_simulation.typ
│   │   ├── scaling_law.typ
│   │   └── systems.typ
│   ├── handout.pdf
│   ├── handout.typ
│   └── refs.bib
├── justfile
├── mise.toml
├── pyproject.toml
├── README.md
├── scripts
│   └── release_student.py
├── src
│   └── eai336
│       ├── c1_robotic_control
│       │   └── __init__.py
│       ├── c2_architectures
│       │   └── __init__.py
│       ├── c3_offline_policy_learning
│       │   └── __init__.py
│       ├── c4_data
│       │   └── __init__.py
│       ├── c5_scaling_laws
│       │   └── __init__.py
│       ├── c6_systems
│       │   └── __init__.py
│       ├── c7_online_policy_learning
│       │   └── __init__.py
│       └── __init__.py
├── tests
│   ├── __init__.py
│   └── test_sanity.py
├── tinymist.lock
└── uv.lock

15 directories, 30 files

这是我目前的目录结构，主要的代码放在 src 中保证能相互导入，然后 tests 就是主要的测试文件夹目录，`release_student.py` 用于将src中除了 `__init__.py` 以外的文件全部清空然后发布学生分支

---

现在假设你是一个 Stanford University 的教授，请你按照我之前拟定的课程大纲，同时借鉴 CS336 课程测试驱动的风格结合 Embodied AI 的具体实际帮我写一下 chapters/overview.typ 这个内容请你仔细思考之后谨慎落笔，为整个课程定主基调，整体风格扎实但不老旧；然后写一下 chapters/robotic_control_and_simulation.typ 的内容，可以的话 python 实验代码也可以配备上；其他章节留空就行，注意不要忘记了 ref.bib 文件，并且不要频繁分段分点，遵循自然分段的写作风格，语言使用英文，逻辑清晰，语言风格质朴平实，不要给学生堆叠一堆黑话；总之就是一句话，你是一个资深教授，你怎么写这个教材才能够让学生构建正确的心智模型，获得扎实基础、掌握领域最新动向的同时又不会负担过重
