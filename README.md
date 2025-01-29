# CurupiraIA: Hate Speech Detection ML Model 🇧🇷🛡️

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.10-blue?style=for-the-badge&logo=python" alt="Python"/>
  <img src="https://img.shields.io/badge/PyTorch-2.0-orange?style=for-the-badge&logo=pytorch" alt="PyTorch"/>
  <img src="https://img.shields.io/badge/HuggingFace_Transformers-4.30-yellow?style=for-the-badge" alt="Transformers"/>
</p>

> <img src="https://img.shields.io/badge/Status-Active_Development-red?style=for-the-badge" alt="Status"/>
## Table of Contents

- [Project Description](#project-description-)
- [Current Features](#current-features-)
- [Installation](#installation-)
- [Dataset](#dataset-)
- [Model Architecture](#model-architecture-)
- [Training](#training-)
- [Contributors](#contributors-)
- [License](#license-)

## Project Description 🔍

Portuguese-language hate speech detection model using BERT architecture, currently in development phase. Key aspects:

- Fine-tuning neuralmind/bert-base-portuguese-cased
- Binary classification (hate speech detection)
- Focus on Brazilian social media text patterns
- Experimental phase with HateBR dataset

## Current Features ✅

- Data preprocessing pipeline
- BERT model fine-tuning setup
- Basic evaluation metrics
- Hugging Face integration
- Experiment tracking (W&B)

## Installation ⚙️

### Prerequisites
- Python 3.10+
- CUDA-enabled GPU (recommended)

```bash
git clone https://github.com/yourusername/curupira-ml.git
cd curupira-ml
pip install -r requirements.txt
```

## Dataset 📊

### **HateBR Dataset**
```python
from datasets import load_dataset

dataset = load_dataset("ruanchaves/hatebr")

Source: Hugging Face Datasets
7,000 annotated comments
Features:
Text: Raw comment text
Label: Binary classification (0=normal, 1=hate)
```

## Model Architecture 🧠

### **Base Configuration**
```python
from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

```
## Fine-Tuning Phase 🎯

**Current Status:**  
```plaintext
[2025] Model in active fine-tuning stage
```
### **We're currently optimizing the BERT-large model through:**

- Progressive learning rate decay
- Class imbalance mitigation techniques
- Early stopping implementation
- Validation metric monitoring (F1-score focus)


## Training 🔥
### **Traning Command**

```python
trainer.train()
```
## Contributors :octocat:

| [<img src="https://avatars.githubusercontent.com/u/142019936?v=4" width=115><br><sub>Kim Gomes</sub>](https://github.com/barbiedeti) |   
| :---: |

## Licença 

<img src="http://img.shields.io/static/v1?label=License&message=MIT&color=green&style=for-the-badge"/>

Copyright :copyright: 2025 - Curupira AI
