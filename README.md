<div id="top" align="center">

<p align="center">
  <img src="assets/images/repo/logo.jpg" alt="NexusAD Logo" width="200">
</p>

# **NexusAD**  
*Multimodal Perception and Comprehension of Corner Cases in Autonomous Driving*

Authors <a name="authors"></a>
Mengjingcheng Mo, Jingxin Wang, Like Wang, Haosheng Chen, Changjun Gu, Jiaxu Leng, Xinbo Gao
Chongqing University of Posts and Telecommunications

**注意：目前代码还在更新中，敬请期待更多功能和改进。**

`ECCV 2024 Autonomous Driving Workshop` **Corner Case Scene Understanding** [Leaderboard](https://eccv2024.autonomousdriving.com)

</div>

---

<div id="top" align="center">

[![Project Page](https://img.shields.io/badge/Project%20Page-8A2BE2)](https://opendrivelab.com/DriveLM/)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](#license)
[![arXiv](https://img.shields.io/badge/arXiv-2312.14150-b31b1b.svg)](https://openreview.net/forum?id=example-link)
[![Latest Release](https://img.shields.io/badge/Latest%20release-v1.1-yellow)](#getting-started)

</div>

---

## 项目亮点 <a name="highlight"></a>

- 🔥 **NexusAD** 提出了一种基于 **InternVL-2.0** 的多模态感知与理解框架，通过对 **CODA-LM** 数据集的精细调优，显著提升了对复杂场景的检测、深度估计与推理能力。
- 🏁 **NexusAD** 参与了 [**`ECCV 2024 Autonomous Driving Workshop`**](https://eccv2024.autonomousdriving.com)，专注于极端驾驶场景中的多模态场景理解任务。

<p align="center">
  <img src="assets/images/repo/nexusad_architecture.jpg" alt="NexusAD Architecture" width="600">
</p>

---

## 最新动态 <a name="news"></a>

- **2024/09/08**: NexusAD 在 ECCV 2024 提交，并取得 68.97 分的成绩。
- **2024/06/01**: NexusAD 团队发布了最新版本的代码和训练数据。

<p align="right">(<a href="#top">回到顶部</a>)</p>

---

## 目录

1. [项目亮点](#highlight)
2. [快速开始](#getting-started)
3. [模型架构](#model-architecture)
4. [实验结果](#results)
5. [贡献指南](#contributing)
6. [许可与引用](#license)

---

## 快速开始 <a name="getting-started"></a>

请按照以下步骤开始使用 NexusAD：

1. **克隆仓库**：
   ```bash
   git clone https://github.com/OpenVisualLab/NexusAD.git
   cd NexusAD
   ```

2. **安装依赖**：
   ```bash
   pip install -r requirements.txt
   ```

3. **下载 [CODA-LM 数据集](https://example.com/coda-lm-dataset)** 并将其放置在指定目录中。

4. **下载 [LoRA 权重](https://example.com/lora-weights)** 并放置在 `weights/` 目录下。

5. **运行模型**：
   ```bash
   python preprocess.py --data_path <path-to-CODA-LM>
   python train.py --config config.json
   python evaluate.py --data_path <path-to-evaluation-set>
   ```

<p align="right">(<a href="#top">回到顶部</a>)</p>

---

## 模型架构 <a name="model-architecture"></a>

NexusAD 模型架构由以下几部分组成：

1. **初步视觉感知**：使用 **Grounding DINO** 进行对象检测，使用 **DepthAnything v2** 进行深度估计，将空间信息转换为易于理解的结构化文本。
   
2. **场景感知增强检索生成**：使用 **Retrieval-Augmented Generation (RAG)** 技术，检索和选择相关样本以增强对复杂驾驶场景的理解。

3. **驾驶提示优化**：通过 **Chain-of-Thought (CoT)** 提示，生成基于上下文的结构化驾驶建议。

4. **精细调优**：我们使用 **LoRA** 技术进行高效的参数微调，以优化性能并节省计算资源。

<p align="right">(<a href="#top">回到顶部</a>)</p>

---

## 实验结果 <a name="results"></a>

在 ECCV 2024 的角落案例理解任务中，NexusAD 的表现超过了基准模型，取得了 **68.97** 的最终得分：

| 模型                | 一般感知   | 区域感知   | 驾驶建议   | 最终得分  |
|--------------------|------------|------------|------------|----------|
| GPT-4V             | 57.50      | 56.26      | 63.30      | 59.02    |
| CODA-VLM           | 55.04      | 77.68      | 58.14      | 63.62    |
| InternVL-2.0-26B   | 43.39      | 64.91      | 48.04      | 52.11    |
| **NexusAD (Ours)** | **57.58**  | **84.31**  | **65.02**  | **68.97**|

<p align="right">(<a href="#top">回到顶部</a>)</p>

---

## 贡献指南 <a name="contributing"></a>

我们欢迎任何形式的贡献！请查看 [CONTRIBUTING.md](CONTRIBUTING.md) 了解如何参与。

<p align="right">(<a href="#top">回到顶部</a>)</p>

---

## 许可与引用 <a name="license"></a>

本项目根据 [MIT 许可](./LICENSE) 发布。如果该项目对你的研究有帮助，请引用以下内容：

```BibTeX
@article{mo2024nexusad,
  title={NexusAD: Multimodal Perception and Comprehension of Corner Cases in Autonomous Driving},
  author={Mo, Mengjingcheng and Wang, Jingxin and Wang, Like and Chen, Haosheng and Gu, Changjun and Leng,Jiaxu and Gao, Xinbo},
  journal={ECCV 2024 Autonomous Driving Workshop Abstract Paper},
  year={2024}
}
```

<p align="right">(<a href="#top">回到顶部</a>)</p>

