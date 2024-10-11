import torch
import torch.nn as nn

# Arch1: Modified VGG-style convolutional layers for 28x28x1 input, n_params = 648,974
# Arch2: Modified ResNet model with BasicBlock (lesser blocks), n_params = 909,290.
# Arch3: Modified DenseNet model with Bottleneck and Transition layers, n_params = 100k
# Arch4: MobileNetV2-like model with DepthwiseSeparableConv layers, n_params = 149,566

class Arch1(nn.Module):
    def __init__(self, num_classes=62):
        super(Arch1, self).__init__()
        
        # Modified VGG-style convolutional layers for 28x28x1 input
        self.features = nn.Sequential(
            # Conv Layer 1 (Input: 28x28x1, Output: 28x28x64)
            nn.Conv2d(1, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Conv Layer 2 (Output: 28x28x64)
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 14x14x64
            
            # Conv Layer 3 (Output: 14x14x128)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            # Conv Layer 4 (Output: 14x14x128)
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: 7x7x128
        )
        
        # Fully connected layer without hidden layers
        # Flatten the features from 7x7x128 to 6272 before feeding into the output layer
        self.classifier = nn.Linear(7 * 7 * 128, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.classifier(x)
        return x
    
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = torch.relu(out)
        return out

class ResNetMod(nn.Module):
    def __init__(self, block, num_blocks, num_classes=62):
        super(ResNetMod, self).__init__()
        self.in_planes = 64

        # Adjust input conv layer for 28x28x1 input instead of 224x224x3
        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # ResNet Layers
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        # The last two layers (residual blocks) are discarded as per your request.
        # So, we stop here, no layer3, layer4

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Adapted to small input size
        self.fc = nn.Linear(128 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Instantiate the modified ResNet model
def Arch2(num_classes=62):
    return ResNetMod(BasicBlock, [2, 2], num_classes)

class Bottleneck(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(Bottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, 4 * growth_rate, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(4 * growth_rate)
        self.conv2 = nn.Conv2d(4 * growth_rate, growth_rate, kernel_size=3, padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(torch.relu(self.bn1(x)))
        out = self.conv2(torch.relu(self.bn2(out)))
        out = torch.cat([x, out], 1)
        return out

class Transition(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Transition, self).__init__()
        self.bn = nn.BatchNorm2d(in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x):
        out = self.conv(torch.relu(self.bn(x)))
        out = self.pool(out)
        return out

class DenseNetMod(nn.Module):
    def __init__(self, num_classes=62, growth_rate=24, block_layers=[6, 6]):
        super(DenseNetMod, self).__init__()
        self.growth_rate = growth_rate
        num_planes = 2 * growth_rate  # Starting number of planes
        
        # Initial convolution layer
        self.conv1 = nn.Conv2d(1, num_planes, kernel_size=3, padding=1, bias=False)
        
        # Dense Block 1
        self.block1 = self._make_dense_layers(Bottleneck, num_planes, block_layers[0])
        num_planes += block_layers[0] * growth_rate
        self.trans1 = Transition(num_planes, num_planes // 2)
        num_planes = num_planes // 2
        
        # Dense Block 2
        self.block2 = self._make_dense_layers(Bottleneck, num_planes, block_layers[1])
        num_planes += block_layers[1] * growth_rate
        self.trans2 = Transition(num_planes, num_planes // 2)
        num_planes = num_planes // 2
        
        # Global average pooling and fully connected layer
        self.bn = nn.BatchNorm2d(num_planes)
        self.fc = nn.Linear(num_planes, num_classes)

    def _make_dense_layers(self, block, in_channels, nblock):
        layers = []
        for i in range(nblock):
            layers.append(block(in_channels, self.growth_rate))
            in_channels += self.growth_rate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.trans1(out)
        out = self.block2(out)
        out = self.trans2(out)
        out = torch.relu(self.bn(out))
        out = torch.nn.functional.adaptive_avg_pool2d(out, (1, 1))
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

# Instantiate the modified ResNet model
def Arch3(num_classes=62):
    return DenseNetMod(num_classes=num_classes, growth_rate=12, block_layers=[6, 6])

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        out = torch.relu(self.bn1(self.depthwise(x)))
        out = self.bn2(self.pointwise(out))
        return out

class Arch4(nn.Module):
    def __init__(self, num_classes=62):
        super(Arch4, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)

        self.dw_conv1 = DepthwiseSeparableConv(32, 64)
        self.dw_conv2 = DepthwiseSeparableConv(64, 128, stride=2)
        self.dw_conv3 = DepthwiseSeparableConv(128, 128)
        self.dw_conv4 = DepthwiseSeparableConv(128, 256, stride=2)
        self.dw_conv5 = DepthwiseSeparableConv(256, 256)

        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(256, num_classes)

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.dw_conv1(out)
        out = self.dw_conv2(out)
        out = self.dw_conv3(out)
        out = self.dw_conv4(out)
        out = self.dw_conv5(out)
        out = self.global_avg_pool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out

if __name__ == "__main__":

    input_image = torch.rand(1, 1, 28, 28)
    model = Arch1(num_classes=62)
    output = model(input_image)
    print(output.shape)
    model = Arch2(num_classes=62)
    output = model(input_image)
    print(output.shape)
    model = Arch3(num_classes=62)
    output = model(input_image)
    print(output.shape)
    model = Arch4(num_classes=62)
    output = model(input_image)
    print(output.shape)
    

