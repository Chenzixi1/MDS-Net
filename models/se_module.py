from torch import nn
import torch


# ESANet
class SqueezeAndExcitation(nn.Module):
    def __init__(self, channel,
                 reduction=16, activation=nn.ReLU(inplace=True)):
        super(SqueezeAndExcitation, self).__init__()
        self.fc = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            activation,
            nn.Conv2d(channel // reduction, channel, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        weighting = nn.functional.adaptive_avg_pool2d(x, 1)
        weighting = self.fc(weighting)
        y = x * weighting
        return y