# 🧠 CIFAR-10 Image Classification (ResNet-18 + PyTorch)

This project implements a deep learning pipeline to classify CIFAR-10 images using PyTorch, progressively improving from a basic CNN to a ResNet-18 backbone.

> 🚀 Current version: **v3.1**
> 🎯 Final test accuracy: **92.47%** (ResNet-18 + stronger augmentation + advanced regularization)

---

## 🚀 Features

- ✅ ResNet-18 backbone for high-accuracy image classification
- ✅ Modular codebase: model, data, training, evaluation, config
- ✅ Training reproducibility with fixed random seeds
- ✅ `AdamW` optimizer with **CosineAnnealingLR** scheduler
- ✅ Label Smoothing via `CrossEntropyLoss(label_smoothing=0.1)`
- ✅ **Dropout(0.1)** regularization added after FC layer
- ✅ **Advanced Data Augmentation**: crop, flip, color jitter, rotation
- ✅ Best model saving during training (`best_model.pth`)
- ✅ TensorBoard logging: loss, accuracy, learning rate, weight histograms
- ✅ MPS support for Apple Silicon (macOS)

---

## 🧪 Quickstart

### 1. Clone and create virtual environment

```bash
git clone https://github.com/SabaMG/cifar10-cnn-pytorch.git
cd cifar10-cnn-pytorch
python3 -m venv .venv
source .venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the model

```bash
python main.py
```

The best model will be saved automatically as `best_model.pth`.

### 4. Launch TensorBoard (optional)

```bash
tensorboard --logdir=runs
```

---

## 🧠 Model Versions

| Version | Architecture     | Description                                           | Accuracy   |
|---------|------------------|-------------------------------------------------------|------------|
| v1.0    | Basic CNN (3 conv) | Lightweight, easy to train                            | 81.43%     |
| v2.0    | Deep CNN (6 conv) | Larger capacity, dropout, batchnorm                   | 87.43%     |
| v3.0    | ResNet-18         | Deeper pretrained-style model, stable                 | 90.99%     |
| v3.1    | ResNet-18 + Aug   | Strong augmentation, dropout, Cosine LR, smoothing    | **92.47%** |

---

## 📁 Project Structure

```
.
├── dataset.py           # CIFAR-10 loading and transforms
├── model.py             # ResNet-18 model definition
├── train.py             # Training loop (TensorBoard, checkpoint, early stop)
├── evaluate.py          # Evaluation logic
├── main.py              # Main entrypoint
├── utils.py             # Helpers (set_seed, etc.)
├── config.py            # Device config
├── best_model.pth       # Best model (autogenerated)
├── runs/                # TensorBoard logs
├── requirements.txt     # Dependencies
└── README.md
```

---

## 📌 Notes

- Trained on Apple Silicon (MPS backend), supports CPU/GPU as well
- `train.py` supports early stopping, checkpointing, and logging by default
- Model performance is logged with TensorBoard after every epoch
- Strong regularization makes this model more robust and generalizable

---

## 🪪 License

MIT License