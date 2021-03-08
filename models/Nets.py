import torch
import torch.nn as nn
import torch.nn.functional as F


class NetFull(nn.Module):
    def __init__(self, n=10, nb_piece=7):
        super(NetFull, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
        self.relu2 = nn.ReLU(inplace=True)

        output = n*n*nb_piece
        self.fc1 = nn.Linear(64*(n-3)*(n-3),2048)
        self.relu3 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(2048,output)


    def forward(self, x):
        
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.relu3(x)

        return F.relu(self.fc2(x))

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class NetLinesConv(nn.Module):
    def __init__(self, n=10):
        super(NetLinesConv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv_ver = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(1, n))
        self.relu2 = nn.ReLU(inplace=True)
        self.conv_hor = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=(n, 1))
        self.relu3 = nn.ReLU(inplace=True)

        output = 1      # Only the q-value of the transition grid
        self.fc1 = nn.Linear(32*2*n+32*n*n, 512)
        self.relu4 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(512,32)
        self.relu5 = nn.ReLU(inplace=True)

        self.fc3 = nn.Linear(32, output)

        self._create_weights()

        
    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0.01)

    def forward(self, x):
        
        x = self.conv1(x)
        y = self.relu1(x)
        hor = self.conv_hor(y[:, :16, :, :])
        hor = self.relu2(hor)
        hor = hor.view(-1, self.num_flat_features(hor))
        ver = self.conv_ver(y[:, 16:, :, :])
        ver = self.relu3(ver)
        ver = ver.view(-1, self.num_flat_features(ver))
        y = y.view(-1, self.num_flat_features(y))


        x = torch.cat(tuple([hor, ver, y]), dim=1)
        # x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.relu4(x)

        x = self.fc2(x)
        x = self.relu5(x)

        return F.relu(self.fc3(x))

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

class NetTransitions(nn.Module):
    def __init__(self, n=10):
        super(NetTransitions, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2)
        self.relu2 = nn.ReLU(inplace=True)

        output = 1      # Only the q-value of the transition grid
        self.fc1 = nn.Linear(64*(n-3)*(n-3), 512)
        self.relu3 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(512,32)
        self.relu4 = nn.ReLU(inplace=True)

        self.fc3 = nn.Linear(32, output)

        self._create_weights()

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)
        x = self.relu4(x)

        return self.fc3(x)

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0.01)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


class NetTransitionsWtConv(nn.Module):
    def __init__(self, n=10):
        super(NetTransitionsWtConv, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=2)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1)
        self.relu2 = nn.ReLU(inplace=True)

        output = 1      # Only the q-value of the transition grid
        self.fc1 = nn.Linear(64*(n-1)*(n-1), 512)
        self.relu3 = nn.ReLU(inplace=True)

        self.fc2 = nn.Linear(512,32)
        self.relu4 = nn.ReLU(inplace=True)

        self.fc3 = nn.Linear(32, output)

        self._create_weights()

    def forward(self, x):
        
        x = self.conv1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.relu2(x)

        x = x.view(-1, self.num_flat_features(x))
        x = self.fc1(x)
        x = self.relu3(x)

        x = self.fc2(x)
        x = self.relu4(x)

        return self.fc3(x)

    def _create_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.uniform_(m.weight, -0.01, 0.01)
                nn.init.constant_(m.bias, 0.01)

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features