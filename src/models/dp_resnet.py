import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_GROUPS = 16

# Drawn from: https://github.com/fastestimator/fastestimator/blob/r1.2/fastestimator/architecture/pytorch/resnet9.py/#L22-L89

# Replace batch norm with group norm
class DPResNet9(nn.Module):
    def __init__(self, num_channels, num_classes):
        super().__init__()
        self.conv0 = nn.Conv2d(num_channels, 64, 3, padding=(1, 1))
        self.conv0_gn = nn.GroupNorm(NUM_GROUPS, 64)
        self.conv1 = nn.Conv2d(64, 128, 3, padding=(1, 1))
        self.conv1_gn = nn.GroupNorm(NUM_GROUPS, 128)
        self.residual1 = Residual(128)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=(1, 1))
        self.conv2_gn = nn.GroupNorm(NUM_GROUPS, 256)
        self.residual2 = Residual(256)
        self.conv3 = nn.Conv2d(256, 512, 3, padding=(1, 1))
        self.conv3_gn = nn.GroupNorm(NUM_GROUPS, 512)
        self.residual3 = Residual(512)
        self.fc1 = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # prep layer
        x = self.conv0(x)
        x = self.conv0_gn(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        # layer 1
        x = self.conv1(x)
        x = F.max_pool2d(x, 2)
        x = self.conv1_gn(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = x + self.residual1(x)
        # layer 2
        x = self.conv2(x)
        x = F.max_pool2d(x, 2)
        x = self.conv2_gn(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = x + self.residual2(x)
        # layer 3
        x = self.conv3(x)
        x = F.max_pool2d(x, 2)
        x = self.conv3_gn(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = x + self.residual3(x)
        # layer 4
        x = nn.AdaptiveMaxPool2d((1, 1))(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

class Residual(nn.Module):
    """A two-layer unit for ResNet9. The output size is the same as input.
    Args:
        channel: Number of input channels.
    """
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=(1, 1))
        self.conv1_gn = nn.GroupNorm(NUM_GROUPS, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=(1, 1))
        self.conv2_gn = nn.GroupNorm(NUM_GROUPS, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.conv1_gn(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        x = self.conv2(x)
        x = self.conv2_gn(x)
        x = F.leaky_relu(x, negative_slope=0.1)
        return x