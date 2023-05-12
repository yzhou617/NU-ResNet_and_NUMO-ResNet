
from torchvision.models import resnet18
import torch.nn as nn
import torch 



class ResNet_18_grayscale_mat(nn.Module):
    
    def __init__(self, input_num_channels=4):
        super(ResNet_18_grayscale_mat, self).__init__()
        self.resnet_18 = resnet18(pretrained=False)
        self.resnet_18.conv1 = nn.Conv2d(in_channels=input_num_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.input_num_features = self.resnet_18.fc.in_features
        self.resnet_18.fc = nn.Linear(in_features=self.input_num_features, out_features=2)
        
    
    def forward(self, x):
        x = self.resnet_18(x)
        return x


# The following example is for testing the ResNet_18_grayscale_mat architecture
'''
NN_model = ResNet_18_grayscale_mat()
input = torch.randn((20,4,410,410))
result = NN_model(input)
print(result.shape)
'''