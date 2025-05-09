# 📄 Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v2.0] - 2025-05-09

### Added
- Deep CNN architecture with **6 convolutional layers** for improved performance.
- Improved data augmentation: random horizontal flip + random crop with padding.
- Layer normalization retained with BatchNorm after each convolution.
- Dropout (0.3) applied before fully connected layer for regularization.
- TensorBoard logs extended to include the deeper model training.
- Updated `README.md` with new architecture, logs and final performance.

### Changed
- `model.py`: upgraded architecture to 6 conv layers.
- `dataset.py`: updated transforms to stronger augmentations.
- `main.py`: maintained training script with same optimizer and scheduler.

### 📊 Performance
- ✅ Final test accuracy: **87.43%**
- ✅ Model: Deep CNN (6 conv layers, ReLU, MaxPooling, BatchNorm, Dropout)

---

## [v1.0] - 2025-05-09

### Added
- Initial training pipeline for CIFAR-10 classification using a basic CNN.
- Modular project structure with `model.py`, `train.py`, `dataset.py`, `evaluate.py`, and `main.py`.
- Support for `AdamW` optimizer and learning rate scheduling (`StepLR`).
- Label smoothing loss with `nn.CrossEntropyLoss(label_smoothing=0.1)`.
- TensorBoard logging: loss, accuracy, learning rate, weight histograms.
- Best model saving to `best_model.pth` and automatic loading after training.
- MPS support for Apple Silicon (`config.device`).
- Early stopping mechanism.
- Seed reproducibility and multi-seed support.
- Project README.md with usage instructions.
- `.gitignore` and `requirements.txt` prepared for clean environments.

### 📊 Performance
- ✅ Final test accuracy: **81.43%**
- ✅ Model: CNN with 3 convolutional layers, kernel 3×3, ReLU, MaxPooling

---

## 🔜 Planned (v3.0+)
- ResNet-18 backbone integration.
- WideResNet-28-10 for state-of-the-art performance.
- Improved data augmentation (Cutout, ColorJitter, etc.).
- Learning rate warmup and cosine decay.

---

> Maintained by [SabaMG](https://github.com/SabaMG)