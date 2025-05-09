from dataset import get_data_loaders
from model import ResNet18
from train import train
from evaluate import evaluate
from utils import set_seed
import torch.nn as nn
import torch
from config import device

if __name__ == "__main__":
    print(f"Using device: {device}")
    set_seed(42)

    train_loader, test_loader = get_data_loaders()

    model = ResNet18().to(device)

    # ✅ AdamW optimizer with classic weight decay for ResNet
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0005, weight_decay=5e-4)

    # ✅ Loss with label smoothing for regularization
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best_model_path = "best_model.pth"

    # ✅ Training with improved scheduler + logging dir
    train(model, train_loader, test_loader, optimizer, loss_fn,
          save_path=best_model_path,
          log_dir="resnet_v3.1_train")

    # ✅ Load and evaluate best model
    model.load_state_dict(torch.load(best_model_path))
    print("\n📦 Loaded best saved model for final evaluation:")
    evaluate(model, test_loader)