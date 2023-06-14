import torch
import torch.nn as nn

class Thrifty(nn.Module):
    def __init__(self, hidden_dim, depth, downsampling, num_classes):
        super().__init__()

        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding = 1, bias = False)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding = 1, bias = False)

        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.depth = depth
        self.downsampling = downsampling

        self.reduction = 1
        for i in downsampling:
            self.reduction *= 2

        bn1 = []
        bn2 = []
        subfilters1 = []
        subfilters2 = []
        reduction = self.reduction
        for i in range(depth):
            if i in self.downsampling:
                reduction = reduction // 2
            bn1.append(nn.BatchNorm2d(hidden_dim // reduction))
            bn2.append(nn.BatchNorm2d(hidden_dim // reduction))
            subfilters1.append(torch.randperm(hidden_dim)[:hidden_dim // reduction])
            subfilters2.append(torch.randperm(hidden_dim)[:hidden_dim // reduction])

            
        self.bn1 = nn.ModuleList(bn1)
        self.bn2 = nn.ModuleList(bn2)
        self.subfilters1 = subfilters1
        self.subfilters2 = subfilters2
            
        self.hidden_dim = hidden_dim

        self.embed = torch.nn.Sequential(
            nn.Conv2d(3, hidden_dim // self.reduction, 3, padding = 1, bias = False),
            nn.BatchNorm2d(hidden_dim // self.reduction),
            nn.ReLU()
        )


    def forward(self, x):
        y = self.embed(x)

        reduction = self.reduction
        
        for i in range(self.depth):
#            z = self.conv1(y)
            if i in self.downsampling:
                new_reduction = reduction // 2
            else:
                new_reduction = reduction
            convweight1 = self.conv1.weight[self.subfilters1[i]].permute(1,0,2,3)[self.subfilters1[i]].permute(1,0,2,3)
            z = torch.nn.functional.conv2d(y, weight=convweight1[:,:self.hidden_dim // reduction], padding=1)
            z = self.bn1[i](z)
            z = torch.relu(z)
#            z = self.conv2(z)
            convweight2 = self.conv2.weight[self.subfilters2[i]].permute(1,0,2,3)[self.subfilters2[i]].permute(1,0,2,3)
            z = torch.nn.functional.conv2d(z, weight=convweight2, padding=1)
            z = self.bn2[i](z)
            if i in self.downsampling:
                size = y.shape[1]
                y = torch.nn.functional.pad(y, (0, 0, 0, 0, 0, size))
            y = y + z
            y = torch.relu(y)
            if i+1 in self.downsampling:
                y = nn.MaxPool2d(2,2)(y)
            reduction = new_reduction

        y = y.mean(-1).mean(-1)

        return self.classifier(y)

def thrifty18(num_classes, large_input, width):
    return Thrifty(width, 8, [2, 4, 6], num_classes)

def thrifty50(num_classes, large_input, width):
    return Thrifty(width, 16, [3, 7, 13], num_classes)

def thrifty20(num_classes, large_input, width):
    return Thrifty(width, 9, [3, 6], num_classes)

def thrifty56(num_classes, large_input, width):
    return Thrifty(width, 27, [9, 18], num_classes)
    
def resnet34(num_classes, large_input, width):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, large_input, width)

def resnet101(num_classes, large_input, width):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, large_input, width)

def resnet152(num_classes, large_input, width):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, large_input, width)

def resnet32(num_classes, large_input, width):
    return ResNet(BasicBlock, [5, 5, 5], num_classes, large_input, width)

def resnet44(num_classes, large_input, width):
    return ResNet(BasicBlock, [7, 7, 7], num_classes, large_input, width)

def resnet110(num_classes, large_input, width):
    return ResNet(BasicBlock, [18, 18, 18], num_classes, large_input, width)

def resnet1202(num_classes, large_input, width):
    return ResNet(BasicBlock, [200, 200, 200], num_classes, large_input, width)
