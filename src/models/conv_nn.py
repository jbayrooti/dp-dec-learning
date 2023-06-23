import torch.nn as nn

# Referenced: https://github.com/javieryu/nn_distributed_training

class ConvNet(nn.Module):
    def __init__(self, num_channels=1, num_filters=3, kernel_size=5, linear_width=64, num_classes=10):
        super().__init__()
        conv_out_width = 32 - (kernel_size - 1)
        pool_out_width = int(conv_out_width / 2)
        fc1_indim = num_filters * (pool_out_width ** 2)

        self.seq = nn.Sequential(
            nn.Conv2d(num_channels, num_filters, kernel_size, 1),
            nn.Mish(inplace=True),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(fc1_indim, linear_width),
            nn.Mish(inplace=True),
            nn.Linear(linear_width, num_classes),
        )

    def forward(self, x):
        return self.seq(x)