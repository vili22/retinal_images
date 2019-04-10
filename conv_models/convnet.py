import torch.nn as nn
import torch.nn.functional as F


class ConvNet(nn.Module):
    def __init__(self, in_channels, out_channels1, out_channels2, out_dens):
        super(ConvNet, self).__init__()
        self.out_channels2 = out_channels2
        self.conv1 = nn.Conv2d(in_channels, out_channels1, kernel_size=5, stride=1, padding=0)
        self.batchn1 = nn.BatchNorm2d(out_channels1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(out_channels1, out_channels2, kernel_size=5, stride=1, padding=0)
        self.batchn2 = nn.BatchNorm2d(out_channels2)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc1 = nn.Linear(out_channels2 * 5 * 5, out_dens)
        self.drop1 = nn.Dropout()
        self.fc2 = nn.Linear(out_dens, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.batchn1(x)
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.batchn2(x)
        x = self.pool2(x)
        x = x.view(-1, self.out_channels2 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = self.drop1(x)
        x = self.fc2(x)
        return self.sigmoid(x)
