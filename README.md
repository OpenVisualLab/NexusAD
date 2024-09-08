# NexusAD: Multimodal Perception and Comprehension of Corner Cases in Autonomous Driving

**Authors**:  
Mengjingcheng Mo, Jingxin Wang, Like Wang, Haosheng Chen, Changjun Gu, Jiaxu Leng\*, Xinbo Gao\*  
Chongqing University of Posts and Telecommunications  
\*Corresponding authors: Jiaxu Leng, Xinbo Gao

---

Welcome to the official repository for **NexusAD**, our approach for addressing the **Corner Case Scene Understanding** track at the ECCV 2024 Autonomous Driving Workshop.

## Quick Links

- [LoRA Weights](https://example.com/lora-weights) 
- [Poster](https://example.com/nexusad-poster)
- [PPT Presentation](https://example.com/nexusad-presentation)
- [OpenReview Paper](https://openreview.net/forum?id=example-link)
- [ECCV 2024 Workshop](https://eccv2024.autonomousdriving.com)
- [Corner Case Scene Understanding Track](https://eccv2024.autonomousdriving.com/track)

## Table of Contents
- [NexusAD: Multimodal Perception and Comprehension of Corner Cases in Autonomous Driving](#nexusad-multimodal-perception-and-comprehension-of-corner-cases-in-autonomous-driving)
  - [Quick Links](#quick-links)
  - [Table of Contents](#table-of-contents)
  - [Project Overview](#project-overview)
  - [Features](#features)
  - [Installation](#installation)
  - [Usage](#usage)
  - [Model Architecture](#model-architecture)
  - [Results](#results)
  - [Supplementary Materials](#supplementary-materials)
  - [Contributing](#contributing)
  - [License](#license)

## Project Overview

NexusAD is our approach for addressing corner cases in autonomous driving by leveraging **multimodal large-scale language models (MLLMs)**. Using **InternVL-2.0** as the foundation and the **CODA-LM** dataset, we aim to improve spatial understanding, object detection, and reasoning for autonomous driving in complex and uncommon scenarios.

The challenge of corner cases in autonomous driving remains largely unexplored. NexusAD utilizes advanced techniques like **scene-aware retrieval**, **depth estimation**, and **chain-of-thought reasoning** to improve the model's understanding and decision-making capabilities in these difficult scenarios.

With a final score of **68.97** on the ECCV 2024 leaderboard, our approach surpasses baseline models and improves the performance of tasks like **general perception**, **region perception**, and **driving suggestions**.

## Features

- **Multimodal Perception**: Combines visual and textual data for a more comprehensive understanding of traffic scenes.
- **Fine-Tuning on CODA-LM Dataset**: Trained on a specialized dataset focused on corner cases in autonomous driving.
- **Contextual Scene Understanding**: Enhances perception by integrating scene-aware retrieval to provide contextually relevant information.
- **Driving Recommendations**: Generates detailed driving suggestions based on a deep understanding of traffic environments.

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/OpenVisualLab/NexusAD.git
    cd NexusAD
    ```

2. Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```

3. Download the [CODA-LM dataset](https://example.com/coda-lm-dataset) and place it in the appropriate directory as per the instructions.

4. Download the [LoRA Weights](https://example.com/lora-weights) and place them in the `weights/` directory.

## Usage

To run the NexusAD model:

1. **Preprocess the dataset**:
    ```bash
    python preprocess.py --data_path <path-to-CODA-LM>
    ```

2. **Train the model**:
    ```bash
    python train.py --config config.json
    ```

3. **Evaluate the model** on corner case scenarios:
    ```bash
    python evaluate.py --data_path <path-to-evaluation-set>
    ```

## Model Architecture

The model consists of four primary stages:

1. **Preliminary Visual Perception**: We use **Grounding DINO** for object detection and **DepthAnything v2** for accurate depth estimation, converting spatial information from the images into structured textual formats that are easier for large language models to understand.
   
2. **Scene-aware Retrieval-Augmented Generation**: We employ a custom **Retrieval-Augmented Generation (RAG)** technique to enhance understanding of complex driving scenes by selecting contextually relevant samples for retrieval.
   
3. **Driving Prompt Optimization**: Using **Chain-of-Thought (CoT) prompting**, we guide the model through intermediate reasoning steps to generate structured, context-aware driving suggestions based on traffic rules and scene understanding.

4. **Fine-Tuning**: We applied **LoRA** (Low-Rank Adaptation) for parameter-efficient fine-tuning, optimizing model performance while conserving GPU memory and training time.

![NexusAD Architecture](https://example.com/architecture-diagram.png)

## Results

On the ECCV 2024 competition leaderboard, NexusAD achieved a **final score of 68.97**, surpassing other baseline models in the tasks of **General Perception (GP)**, **Region Perception (RP)**, and **Driving Suggestions (DS)**.

| Model              | General Perception | Region Perception | Driving Suggestion | Final Score |
|--------------------|-------------------|------------------|-------------------|-------------|
| GPT-4V             | 57.50             | 56.26            | 63.30             | 59.02       |
| CODA-VLM           | 55.04             | 77.68            | 58.14             | 63.62       |
| InternVL-2.0-26B   | 43.39             | 64.91            | 48.04             | 52.11       |
| **NexusAD (Ours)** | **57.58**         | **84.31**        | **65.02**         | **68.97**   |

NexusAD improved by **14.19 points** on **General Perception**, **19.40 points** on **Region Perception**, and **16.98 points** on **Driving Suggestions** over the baseline.

## Supplementary Materials
For more detailed information about our approach, methodology, and experiments, please refer to the following:

- [Technical Report](https://example.com/technical-report)
- [Supplementary Materials](https://example.com/supplementary-materials)

## Contributing

We welcome contributions! Please check the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
