# 🧠 CIFAR-10 Deep CNN Classifier (v2.0)

This version improves upon the baseline CNN by significantly deepening the architecture (6 convolutional layers) to achieve higher accuracy on the CIFAR-10 dataset.

> 🎯 Final test accuracy: **87.43%** (deep CNN + regularization + data augmentation)

---

## 🚀 Features (v2.0)

- ✅ Deep CNN architecture (6 convolutional layers)
- ✅ Batch Normalization & Dropout (0.3)  
- ✅ Data Augmentation: horizontal flip & random crop  
- ✅ Best model saving (based on validation accuracy)  
- ✅ MPS (Apple Silicon) / CUDA support  
- ✅ TensorBoard logging  
- ✅ Training reproducibility with fixed seeds  
- ✅ Early stopping (patience 12 epochs)

---

## 🧾 Results

| Epochs | Accuracy | Loss     |
|--------|----------|----------|
| 30     | **87.43%**   | ~768     |

- Dataset: CIFAR-10 (50k train / 10k test)
- Optimizer: AdamW (lr=0.0005, weight_decay=1e-4)
- Scheduler: StepLR (step_size=10, gamma=0.1)
- Input normalization: CIFAR-10 statistics

---

## 📁 Project Structure

```
.
├── main.py             # Launcher with fixed seed
├── model.py            # CNN 6-layer architecture
├── dataset.py          # Augmented CIFAR-10 loader
├── train.py            # Training loop
├── evaluate.py         # Evaluation helper
├── config.py           # Device configuration
├── utils.py            # Seeding, checkpointing
├── runs/               # TensorBoard logs
├── best_model.pth      # Trained weights
├── README.md
├── CHANGELOG.md
└── requirements.txt
```

---

## 🧪 Quickstart

```bash
git clone https://github.com/SabaMG/cifar10-cnn-pytorch.git
cd cifar10-cnn-pytorch
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python main.py
```

Open TensorBoard:

```bash
tensorboard --logdir=runs
```

---

## 📌 Notes

- This version focuses on deeper architectures.
- Early stopping is enabled (max 30 epochs with patience of 12).
- You can adjust the architecture in `model.py` and augmentations in `dataset.py`.

---

## 🪪 License

MIT License