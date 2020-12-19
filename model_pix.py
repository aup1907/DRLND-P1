import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_shape, action_size, seed, fc1_units=64, fc2_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(QNetwork, self).__init__()
        """self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32,kernel_size=8,stride=4, padding=0)
        self.conv2 = nn.Conv2d(in_channels=32,out_channels=64,kernel_size=4,stride=2, padding=0)
        self.conv3 = nn.Conv2d(64,64, kernel_size=3, stride=1)"""
        
        nfilters = [128, 128*2, 128*2]
        self.seed = torch.manual_seed(seed)
        self.conv1 = nn.Conv3d(3, nfilters[0], kernel_size=(1, 3, 3), stride=(1,3,3))
        self.bn1 = nn.BatchNorm3d(nfilters[0])
        self.conv2 = nn.Conv3d(nfilters[0], nfilters[1], kernel_size=(1, 3, 3), stride=(1,3,3))
        self.bn2 = nn.BatchNorm3d(nfilters[1])
        self.conv3 = nn.Conv3d(nfilters[1], nfilters[2], kernel_size=(4, 3, 3), stride=(1,3,3))
        self.bn3 = nn.BatchNorm3d(nfilters[2])
        
        
        n_size = self._get_conv_out_size(state_shape)
        fc = [n_size, 1024]
        print('---init----')
        
        self.fc4 = nn.Linear(fc[0], fc[1])
        self.fc5 = nn.Linear(fc[1], action_size)
        
        #self.fc4 = nn.Linear(380, 19)  
        #self.fc5 = nn.Linear(1805, action_size)


    def forward(self, state):
        """Build a network that maps state -> action values.
        ts = torch.squeeze(state,0)
        x = F.relu(self.conv1(ts))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        return x"""
        ts = torch.squeeze(state,0)
        x = self._cnn(ts)
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x
    
    def _get_conv_out_size(self, shape):
        x = torch.rand(shape)
        print(x.shape)
        x = self._cnn(x)
        n_size = x.data.view(1, -1).size(1)
        print('Convolution output size:', n_size)
        return n_size
    
    def _cnn(self, x):
        #print(self.conv1.weight.shape)
        #print(x.shape)
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = x.view(x.size(0), -1)
        return x 

    
       