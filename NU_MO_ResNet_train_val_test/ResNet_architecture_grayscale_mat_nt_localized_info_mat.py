
from torchvision.models import resnet18
import torch
import torch.nn as nn 



class ResNet_18_pair_grayscale_mat_nt_localized_info_mat(nn.Module):
    
    def __init__(self, nt_info_mat_input_num_channels=1, grayscale_mat_input_num_channels=4):
        super(ResNet_18_pair_grayscale_mat_nt_localized_info_mat, self).__init__()
        self.resnet_18_color_mat = resnet18(pretrained=False)
        self.resnet_18_color_mat.conv1 = nn.Conv2d(in_channels=grayscale_mat_input_num_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet_18_color_mat_modified = nn.Sequential(*list(self.resnet_18_color_mat.children())[:-1], nn.Flatten())
        self.resnet_18_nt_localized_mat = resnet18(pretrained=False)
        self.resnet_18_nt_localized_mat.conv1 = nn.Conv2d(in_channels=nt_info_mat_input_num_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)
        self.resnet_18_nt_localized_mat_modified = nn.Sequential(*list(self.resnet_18_nt_localized_mat.children())[:-1], nn.Flatten())
        self.linear_layer_1 = nn.Linear(in_features=1024, out_features=100)
        self.linear_layer_2 = nn.Linear(in_features=100, out_features=100)
        self.linear_layer_3 = nn.Linear(in_features=100, out_features=2)
    
    def forward(self, color_mat, nt_localized_info_mat):
        x_1 = self.resnet_18_color_mat_modified(color_mat)
        x_2 = self.resnet_18_nt_localized_mat_modified(nt_localized_info_mat)
        x = torch.cat((x_1, x_2), dim=1)
        x = self.linear_layer_1(x)
        x = self.linear_layer_2(x)
        x = self.linear_layer_3(x)
        return x






'''
NN_model = ResNet_18_pair_grayscale_mat_nt_localized_info_mat()
input_1 = torch.randn((20,4,410,410))
input_2 = torch.randn((20,1,410,18))
result = NN_model(input_1, input_2)
print(result)
print(result.shape)
'''

  
    
    
    
    
    
    
    
    
    
    
    
    
    
    