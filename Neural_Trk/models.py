import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


class CNNLoc(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=5, kernel_size=(60, 17))
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(5, 2)

    def forward(self, inputs):
        x = F.relu(self.conv1(inputs))
        x = self.avg(x
                     )
        x = x.squeeze(-1).squeeze(-1)
        out = self.fc(x)
        return out


class CNNSim(nn.Module):
    def __init__(self):
        super().__init__()
        # self.conv1 = nn.Conv1d(in_channels=61, out_channels=60, kernel_size=3, padding=1)
        # self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2)
        # self.conv2 = nn.Conv1d(in_channels=60, out_channels=2, kernel_size=3, padding=1)
        # self.fc1 = nn.Linear(126, 30)
        # self.fc2 = nn.Linear(30, 30)
        # self.fc3 = nn.Linear(30, 15)
        # self.fc4 = nn.Linear(15, 1)
        # self.bn1 = nn.BatchNorm1d(2)
        # self.bn2 = nn.BatchNorm1d(30)
        # self.bn3 = nn.BatchNorm1d(30)
        # self.bn4 = nn.BatchNorm1d(15)
        # self.dr1 = nn.Dropout(0.2)
        # self.dr2 = nn.Dropout(0.2)
        # self.dr3 = nn.Dropout(0.5)
        #---------------------------------------#
        self.conv1 = nn.Conv1d(in_channels=61, out_channels=60, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=1)
        self.conv2 = nn.Conv1d(in_channels=60, out_channels=60, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2)
        self.conv3 = nn.Conv1d(in_channels=60, out_channels=2, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(124, 30)
        self.fc2 = nn.Linear(30, 30)
        self.fc3 = nn.Linear(30, 15)
        self.fc4 = nn.Linear(15, 1)
        self.bn1 = nn.BatchNorm1d(2)
        self.bn2 = nn.BatchNorm1d(30)
        self.bn3 = nn.BatchNorm1d(30)
        self.bn4 = nn.BatchNorm1d(15)
        self.dr1 = nn.Dropout(0.2)
        self.dr2 = nn.Dropout(0.2)
        self.dr3 = nn.Dropout(0.5)
        #--------------------------------------#
        # self.conv1 = nn.Conv1d(in_channels=61, out_channels=60, kernel_size=3, padding=1)
        # self.pool1 = nn.MaxPool1d(kernel_size=3, stride=1)
        # self.conv2 = nn.Conv1d(in_channels=60, out_channels=60, kernel_size=3, padding=1)
        # self.pool2 = nn.MaxPool1d(kernel_size=3, stride=1)
        # self.conv3 = nn.Conv1d(in_channels=60, out_channels=60, kernel_size=3, padding=1)
        # self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2)
        # self.conv4 = nn.Conv1d(in_channels=60, out_channels=2, kernel_size=3, padding=1)
        # self.fc1 = nn.Linear(122, 30)
        # self.fc2 = nn.Linear(30, 30)
        # self.fc3 = nn.Linear(30, 15)
        # self.fc4 = nn.Linear(15, 1)
        # self.bn1 = nn.BatchNorm1d(2)
        # self.bn2 = nn.BatchNorm1d(30)
        # self.bn3 = nn.BatchNorm1d(30)
        # self.bn4 = nn.BatchNorm1d(15)
        # self.dr1 = nn.Dropout(0.2)
        # self.dr2 = nn.Dropout(0.2)
        # self.dr3 = nn.Dropout(0.5)

    def forward(self, inputs):
        # x = self.pool1(F.relu(self.conv1(inputs)))
        # x = F.relu(self.bn1(self.conv2(x)))
        # x = x.flatten(1)
        # x = F.relu(self.dr1(self.bn2(self.fc1(x))))
        # x = F.relu(self.dr2(self.bn3(self.fc2(x))))
        # x = F.relu(self.dr3(self.bn4(self.fc3(x))))
        # out = self.fc4(x)
        # return F.sigmoid(out)
        #----------------------------------------
        x = self.pool1(F.relu(self.conv1(inputs)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.bn1(self.conv3(x)))
        x = x.flatten(1)
        x = F.relu(self.dr1(self.bn2(self.fc1(x))))
        x = F.relu(self.dr2(self.bn3(self.fc2(x))))
        x = F.relu(self.dr3(self.bn4(self.fc3(x))))
        out = self.fc4(x)
        return F.sigmoid(out)
        #--------------------------------------------
        # x = self.pool1(F.relu(self.conv1(inputs)))
        # x = self.pool2(F.relu(self.conv2(x)))
        # x = self.pool3(F.relu(self.conv3(x)))
        # x = F.relu(self.bn1(self.conv4(x)))
        # x = x.flatten(1)
        # x = F.relu(self.dr1(self.bn2(self.fc1(x))))
        # x = F.relu(self.dr2(self.bn3(self.fc2(x))))
        # x = F.relu(self.dr3(self.bn4(self.fc3(x))))
        # out = self.fc4(x)
        # return F.sigmoid(out)


if __name__ == "__main__":
    model = CNNSim()
    print(summary(model, (61, 100), device="cpu"))

