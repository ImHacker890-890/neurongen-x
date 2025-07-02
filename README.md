# NeuroGen-X: Self-Learning Text Generation Model

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)

NeuroGen-X is an advanced self-learning neural network for text generation, capable of producing stories, fairy tales, documents, and other text formats. The model combines Transformer architecture with reinforcement learning for continuous improvement.

## Features

- 🧠 Hybrid Transformer architecture with RL components
- 📚 Learns from raw text without pre-trained weights
- ✍️ Generates coherent text in various styles
- 🔄 Continuous self-improvement through reinforcement learning
- ⚙️ Custom BPE tokenizer trained from scratch

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ImHacker890-890/neurongen-x.git
cd neurongen-x
```
2. Install dependencies:
```bash
pip install -r requirements.txt
```
## Quick Start
Preprocess the data:
```bash
python training/preprocess.py
```
Train the model:
```bash
python training/train.py
```
Generate text:
```bash
python app.py --prompt "Жил-был старик "
```
## Requirements
Python 3.8+

PyTorch 2.0+
## Acknowledgments
•The Transformer architecture (Vaswani et al.)

•Hugging Face for inspiration on tokenizer implementation

•PyTorch community for excellent documentation
