import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(12544, 100)
        self.fc2 = nn.Linear(100, 10)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        # x = F.max_pool2d(x, 2)

        x = self.conv2(x)
        x = self.relu(x)
        # x = F.max_pool2d(x, 2)

        x = self.conv3(x)
        x = self.relu(x)
        # x = F.max_pool2d(x, 2)

        x = self.conv4(x)
        x = self.relu(x)
        x = F.max_pool2d(x, 2)

        x = x.view(-1, 12544)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
