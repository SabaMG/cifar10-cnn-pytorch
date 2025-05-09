# model.py

import torch.nn as nn
from torchvision.models import resnet18

class ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        # Charger ResNet-18 sans poids pré-entraînés
        self.backbone = resnet18(weights=None)

        # Adapter le premier conv pour CIFAR-10 (3x32x32 images)
        self.backbone.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.backbone.maxpool = nn.Identity()  # Enlever maxpool trop agressif pour 32x32
        self.backbone.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        return self.backbone(x)