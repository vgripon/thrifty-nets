import torch
import torch.nn as nn

class Thrifty(nn.Module):
    def __init__(self, hidden_dim, depth, downsampling, num_classes):
        super().__init__()

        self.embed = torch.nn.Sequential(
            nn.Conv2d(3, hidden_dim, 3, padding = 1, bias = False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU()
        )


        self.conv1 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding = 1, bias = False)
        self.conv2 = nn.Conv2d(hidden_dim, hidden_dim, 3, padding = 1, bias = False)

        self.bn1 = nn.ModuleList([nn.BatchNorm2d(hidden_dim) for _ in range(depth)])
        self.bn2 = nn.ModuleList([nn.BatchNorm2d(hidden_dim) for _ in range(depth)])

        self.classifier = nn.Linear(hidden_dim, num_classes)
        self.depth = depth
        self.downsampling = downsampling

    def forward(self, x):
        y = self.embed(x)

        for i in range(self.depth):
            z = self.conv1(y)
            z = self.bn1[i](z)
            z = torch.relu(z)
            z = self.conv2(z)
            z = self.bn2[i](z)
            y = y + z
            y = torch.relu(y)
            if i+1 in self.downsampling:
                y = nn.MaxPool2d(2,2)(y)

        y = y.mean(-1).mean(-1)

        return self.classifier(y)

def thrifty(num_classes, large_input, width):
    return Thrifty(width, 8, [2, 4, 6], num_classes)

def resnet18(num_classes, large_input, width):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes, large_input, width)

def resnet34(num_classes, large_input, width):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes, large_input, width)

def resnet50(num_classes, large_input, width):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes, large_input, width)

def resnet101(num_classes, large_input, width):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes, large_input, width)

def resnet152(num_classes, large_input, width):
    return ResNet(Bottleneck, [3, 8, 36, 3], num_classes, large_input, width)

def resnet20(num_classes, large_input, width):
    return ResNet(BasicBlock, [3, 3, 3], num_classes, large_input, width)

def resnet32(num_classes, large_input, width):
    return ResNet(BasicBlock, [5, 5, 5], num_classes, large_input, width)

def resnet44(num_classes, large_input, width):
    return ResNet(BasicBlock, [7, 7, 7], num_classes, large_input, width)

def resnet56(num_classes, large_input, width):
    return ResNet(BasicBlock, [9, 9, 9], num_classes, large_input, width)

def resnet110(num_classes, large_input, width):
    return ResNet(BasicBlock, [18, 18, 18], num_classes, large_input, width)

def resnet1202(num_classes, large_input, width):
    return ResNet(BasicBlock, [200, 200, 200], num_classes, large_input, width)
