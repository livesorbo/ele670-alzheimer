import torch
import torch.nn as nn
import torchvision.models as models

class ResNet18MRI(nn.Module):
    def __init__(self, in_channels=3, num_classes=2, pretrained=True, freeze_until=5):
        """
        freeze_until: how many layers to freeze (0 = none, 5 = freeze up to layer4)
        """
        super().__init__()
        base = models.resnet18(weights=models.ResNet18_Weights.DEFAULT if pretrained else None)

        # Modify first conv layer if channel count differs
        if in_channels != 3:
            old_conv = base.conv1
            base.conv1 = nn.Conv2d(in_channels, old_conv.out_channels,
                                   kernel_size=old_conv.kernel_size,
                                   stride=old_conv.stride,
                                   padding=old_conv.padding,
                                   bias=old_conv.bias)
            nn.init.kaiming_normal_(base.conv1.weight, mode="fan_out", nonlinearity="relu")

        # Freeze earlier layers progressively
        children = list(base.children())
        if freeze_until > 0:
            for layer in children[:freeze_until]:
                for param in layer.parameters():
                    param.requires_grad = False

        num_ftrs = base.fc.in_features
        base.fc = nn.Linear(num_ftrs, num_classes)
        self.model = base

    def forward(self, x):
        return self.model(x)
