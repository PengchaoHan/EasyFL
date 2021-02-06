import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import torch.nn as nn


class ModelCNNEmnist(nn.Module):
    def __init__(self):
        super(ModelCNNEmnist, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      ),

            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=32,
                      out_channels=32,
                      kernel_size=5,
                      stride=1,
                      padding=2,
                      ),
            nn.ReLU(),
            nn.LocalResponseNorm(4, alpha=0.001 / 9.0, beta=0.75, k=1),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(7 * 7 * 32, 256)
        self.fc2 = nn.Linear(256, 62)

    def forward(self, x, out_activation=False):
        conv1_ = self.conv1(x)
        conv2_ = self.conv2(conv1_)
        fc_ = conv2_.view(-1, 32*7*7)
        fc1_ = self.fc1(fc_).clamp(min=0)  # Achieve relu using clamp
        output = self.fc2(fc1_)
        if out_activation:
            return output, conv1_, conv2_
        else:
            return output





