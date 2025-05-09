# 🖼️ CIFAR-10 Image Classifier (v1.0 - Simple CNN)

This repository implements a simple CNN-based image classifier trained on the **CIFAR-10** dataset using **PyTorch**.

> 🎯 Final test accuracy with this version: **81.43%**

---

## 🚀 Features

- ✅ Custom Convolutional Neural Network (3 conv layers)
- ✅ Batch Normalization & Dropout for regularization
- ✅ TensorBoard for training visualization
- ✅ Early stopping & best model checkpointing
- ✅ Reproducible training with fixed seeds
- ✅ Compatible with Apple Silicon (MPS), CUDA, or CPU

---

## 📁 Project Structure

```
.
├── data/               # CIFAR-10 data (downloaded automatically)
├── main.py             # Training/evaluation pipeline
├── model.py            # CNN model definition
├── dataset.py          # Data loaders and transforms
├── train.py            # Training loop with early stopping
├── evaluate.py         # Evaluation function
├── config.py           # Device configuration
├── utils.py            # Seed setting and helpers
├── best_model.pth      # Best model weights (optional)
├── runs/               # TensorBoard logs
└── README.md
```

---

## 🧪 Quickstart

```bash
# Clone and create virtual env
git clone https://github.com/SabaMG/cifar10-cnn-pytorch.git
cd cifar10-cnn-pytorch
python3 -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# Install requirements
pip install -r requirements.txt

# Train the model
python main.py

# Visualize logs (optional)
tensorboard --logdir=runs
```

---

## 🧾 Results

| Epochs | Accuracy | Loss    |
|--------|----------|---------|
| 30     | **81.43%**   | ~796    |

Training configuration:
- CNN (3 conv layers + 2 FC layers)
- Dropout 0.3
- Batch Normalization
- AdamW optimizer
- StepLR scheduler
- Early stopping

---

## 🛠️ Dependencies

- Python 3.11+
- torch
- torchvision
- matplotlib *(optional for debug or plotting)*
- tensorboard *(optional for logs visualization)*

---

## 📌 Notes

- This is version **v1.0** with a simple CNN architecture.
- Future versions will explore:
  - ✅ Deeper CNNs (v1.1)
  - ✅ ResNet-18 (v2.0)
  - ✅ WideResNet-28-10 (v3.0)

---

## 🪪 License

MIT License