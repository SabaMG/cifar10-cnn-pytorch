# 📘 Changelog

All notable changes to this project will be documented in this file.  
This project follows [Semantic Versioning](https://semver.org/).

---

## [v1.0] - 2024-05-09

### 🎉 Added
- Initial CNN model for CIFAR-10 classification (3 convolutional layers + 2 FC).
- Support for **Dropout**, **Batch Normalization** and **AdamW optimizer**.
- Custom training loop with:
  - Early stopping
  - Best model checkpointing
  - Learning rate scheduling
- **TensorBoard** integration for loss, accuracy, weights, and image samples.
- Configuration file to set training parameters (device, path).
- Reproducibility: fixed random seed support.
- Modularized architecture with `main.py`, `model.py`, `dataset.py`, `train.py`, `evaluate.py`, etc.

### 📊 Performance
- ✅ Final test accuracy: **81.43%**
- ✅ Model: CNN with 3 convolutional layers, kernel 3×3, ReLU, MaxPooling.

---

## 🔜 Planned (v2.0+)
- Deeper CNN (4+ conv layers).
- ResNet-18 backbone integration.
- WideResNet-28-10 for state-of-the-art performance.
- Improved data augmentation (Cutout, ColorJitter, etc.).
- Learning rate warmup and cosine decay.

---

> Maintained by [SabaMG](https://github.com/SabaMG)