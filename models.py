## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        # 224
        self.conv1 = nn.Conv2d(1, 32, 5)
        # 224 - 5 + 1 = 220 --> OUT: (32, 220, 220)
        self.pool1 = nn.MaxPool2d(2, 2)
        # (220 - 2) / 2 +1 = 110 --> OUT: (32, 110, 110)
        self.drop1 = nn.Dropout(p = 0.1)

        # 110
        self.conv2 = nn.Conv2d(32, 64, 3)
        # 110 - 3 +1 = 108 --> OUT: (64, 108, 108)
        self.pool2 = nn.MaxPool2d(2, 2)
        # (108 - 2) / 2 +1 = 54 --> OUT: (64, 54, 54)
        self.drop2 = nn.Dropout(p = 0.2)

        # 54
        self.conv3 = nn.Conv2d(64, 128, 3)
        # 54 - 3 +1 = 52 --> OUT: (128, 52, 52)
        self.pool3 = nn.MaxPool2d(2, 2)
        # (52 - 2) / 2 + 1 = 26 --> OUT: (128, 26, 26)
        self.drop3 = nn.Dropout(p = 0.3)

        # 26
        self.conv4 = nn.Conv2d(128, 256, 3)
        # 26 - 3 + 1 = 24 --> OUT: (256, 24, 24)
        self.pool4 = nn.MaxPool2d(2, 2)
        # (24 - 2) / 2 +1 = 12 --> OUT: (256, 12, 12)
        self.drop4 = nn.Dropout(p = 0.4)
        
        # First number should be 256 * 12 * 12 but that fails and I am confused.
        self.fc1 = nn.Linear(12, 1024)
        self.fc2 = nn.Linear(1024, 136)

        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        x = self.drop1(self.pool1(F.relu(self.conv1(x))));
        x = self.drop2(self.pool2(F.relu(self.conv2(x))));
        x = self.drop3(self.pool3(F.relu(self.conv3(x))));
        x = self.drop4(self.pool4(F.relu(self.conv4(x))));
        x = self.fc1(x);
        x = self.fc2(x);
        
        
        # a modified x, having gone through all the layers of your model, should be returned
        return x
