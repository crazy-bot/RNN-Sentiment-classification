import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self,in_channels,num_actions,alpha):
        super(DQN,self).__init__()
        # input 84*84*4
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=32,kernel_size=8,stride=4,padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        # input 20*20*32
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=4, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        # input 9*9*64
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=2)
        self.bn3 = nn.BatchNorm2d(128)
        # output 3*3*128

        self.fc1 = nn.Linear(in_features=3*3*128,out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_actions)

        self.relu = nn.ReLU()

        self.net = nn.Sequential(self.conv1,self.bn1,self.relu,self.conv2,self.bn2,self.relu,self.conv3,self.bn3,self.relu)

        self.optim = optim.RMSprop(self.parameters(),lr=alpha)
        self.loss = nn.MSELoss()
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, X):
        # stack of frames and send the variable to device
        X = torch.Tensor(X).to(self.device)
        # reshape the tensor into channel*height*width to feed in conv layer
        X = X.view(-1,1,84,84) #185*95
        X = self.net(X)
        X = X.view(X.size(0),-1)
        X = self.relu(self.fc1(X))

        # stack of frames passed as input. so the output will also be a matrix where rows = no of frames and cols = possible actions
        actions = self.fc2(X)
        return actions



# x = torch.randn(4, 4)
# x= x.view(-1)
# print(x)

