# NexusAD: Multimodal Perception and Comprehension of Corner Cases in Autonomous Driving

Welcome to the official repository for **NexusAD**, our approach for addressing the **Corner Case Scene Understanding** track at the ECCV 2024 Autonomous Driving Workshop.

## Project Overview

NexusAD is designed to improve multimodal perception and comprehension in **corner cases**—the most challenging and uncommon scenarios in autonomous driving. By leveraging **large vision-language models (LVLMs)** and the **CODA-LM dataset**, we aim to enhance autonomous vehicle safety through improved object detection, depth estimation, and reasoning in complex traffic environments.

Key components of NexusAD include:
- **InternVL-2.0 Model**: A state-of-the-art vision-language model used as the backbone for scene understanding.
- **Scene-Aware Retrieval-Augmented Generation**: A method for retrieving contextually relevant samples to improve perception in corner cases.
- **Driving Prompt Optimization**: Using chain-of-thought reasoning to generate structured, accurate driving suggestions.

The complete report can be found in the repository's documentation.

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

3. Download the CODA-LM dataset and place it in the appropriate directory as per the instructions.

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

1. **Preliminary Visual Perception**: Extracts spatial and depth information from images using InternVL-2.0 and object detection techniques.
2. **Scene-aware Retrieval-Augmented Generation**: Utilizes a custom retrieval system to enhance understanding of complex driving scenes.
3. **Driving Prompt Optimization**: Generates context-aware and structured outputs for driving suggestions.
4. **Fine-Tuning**: Conducted with parameter-efficient methods to optimize the model's performance.

## Results

On the ECCV 2024 competition leaderboard, NexusAD achieved a **final score of 68.97**, significantly improving on tasks like **general perception** and **driving suggestions** when compared to baseline models.

| Model              | General Perception | Region Perception | Driving Suggestion | Final Score |
|--------------------|-------------------|------------------|-------------------|-------------|
| GPT-4V             | 57.50             | 56.26            | 63.30             | 59.02       |
| CODA-VLM           | 55.04             | 77.68            | 58.14             | 63.62       |
| InternVL-2.0-26B   | 43.39             | 64.91            | 48.04             | 52.11       |
| **NexusAD (Ours)** | **57.58**         | **84.31**        | **65.02**         | **68.97**   |

## Contributing

We welcome contributions! Please check the [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines on how to contribute to this project.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
