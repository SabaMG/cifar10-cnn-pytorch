# 📄 Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),  
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [v3.1] - 2025-05-10

### Added
- Enhanced **data augmentation**: RandomCrop, RandomHorizontalFlip, ColorJitter, RandomRotation.
- Switched scheduler from `StepLR` ➜ **CosineAnnealingLR** for smoother learning rate decay.
- Added **Dropout(0.1)** after the fully connected layer in ResNet-18 for better regularization.
- Maintained **label smoothing** with `CrossEntropyLoss(label_smoothing=0.1)`.
- TensorBoard logs continue to track training with all new features.
- Updated `README.md` and performance logs accordingly.

### Changed
- `dataset.py`: stronger augmentation transforms added.
- `train.py`: replaced `StepLR` with `CosineAnnealingLR`.
- `model.py`: slight change to ResNet FC to include dropout.
- `main.py`: no structural change, but training includes new optimizer/scheduler settings.

### 📊 Performance
- ✅ Final test accuracy: **92.47%**
- ✅ Model: ResNet-18 with advanced augmentation, CosineAnnealingLR, dropout

---

## [v3.0] - 2025-05-10

### Added
- 📦 Integration of **ResNet-18** architecture as the model backbone.
- `torchvision.models.resnet18(pretrained=False)` used with modified input/output layers.
- Compatibility with CIFAR-10 (RGB input, 10-class output).
- TensorBoard logs support for ResNet-18 training.
- Model performance saved and evaluated as `best_model.pth`.

### Changed
- `model.py`: switched from custom CNN to ResNet-18 backbone.
- `main.py`: updated model call, training and evaluation unchanged.
- `README.md`: added version comparison, model details, and final results.

### 📊 Performance
- ✅ Final test accuracy: **90.99%**
- ✅ Model: ResNet-18 (modified for CIFAR-10, trained from scratch)

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

## 🔜 Planned (v4.0)
- WideResNet-28-10 for state-of-the-art performance.
- Cutout regularization in augmentation.
- Learning rate warmup and cosine decay.
- Automated inference script from image file.
- Multi-model comparison utility with CLI.

---

> Maintained by [SabaMG](https://github.com/SabaMG)