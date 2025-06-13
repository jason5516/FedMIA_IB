'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from .ib_layers import InformationBottleneck


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = nn.GroupNorm(32, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = nn.GroupNorm(32, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                #nn.BatchNorm2d(self.expansion*planes)
                nn.GroupNorm(32, self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class BasicBlockwithIB(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockwithIB, self).__init__()
        
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(planes)
        self.bn1 = nn.GroupNorm(32, planes)
        # IB 
        self.ib = InformationBottleneck(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        #self.bn2 = nn.BatchNorm2d(planes)
        self.bn2 = nn.GroupNorm(32, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                #nn.BatchNorm2d(self.expansion*planes)
                nn.GroupNorm(32, self.expansion*planes)
            )

    def forward(self, x):
        out = self.bn1(self.conv1(x))
        # ib
        z = self.ib(out)

        out = F.relu(z)
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(64)
        self.bn1 =  nn.GroupNorm(32, 64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.shape[-1])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

class ResNetwithIB(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetwithIB, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(64)
        self.bn1 =  nn.GroupNorm(32, 64)
        self.layer1, self.kl_list1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2, self.kl_list2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3, self.kl_list3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4, self.kl_list4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        ib_list = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        for layer_block in layers:
            ib_list.append(layer_block.ib)

        return nn.Sequential(*layers), ib_list

    def get_kl_loss(self, kl_list):
        kl_loss = 0
        for kl in kl_list:
            kl_loss += kl.kld
        return kl_loss

    def forward(self, x):
        tot_kl = 0
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.shape[-1])
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        tot_kl += self.get_kl_loss(self.kl_list1)
        tot_kl += self.get_kl_loss(self.kl_list2)
        tot_kl += self.get_kl_loss(self.kl_list3)
        tot_kl += self.get_kl_loss(self.kl_list4)
        
        return out, tot_kl

class ResNetwithIB_layer(nn.Module):
    def __init__(self, block_list, num_blocks, num_classes=10):
        super(ResNetwithIB_layer, self).__init__()
        self.in_planes = 64
        self.kl_list = []

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3,
                               stride=1, padding=1, bias=False)
        #self.bn1 = nn.BatchNorm2d(64)
        self.bn1 =  nn.GroupNorm(32, 64)
        
        
        self.layer1 = self._make_layer(block_list[0], 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block_list[1], 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block_list[2], 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block_list[3], 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block_list[3].expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        for layer_block in layers:
            try:
                self.kl_list.append(layer_block.ib)
            except:
                continue

        return nn.Sequential(*layers)

    def get_kl_loss(self, kl_list):
        kl_loss = 0
        for kl in kl_list:
            kl_loss += kl.kld
        return kl_loss

    def forward(self, x):
        tot_kl = 0
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, out.shape[-1])
        out = out.view(out.size(0), -1)
        out = self.linear(out)

        tot_kl = self.get_kl_loss(self.kl_list)
        
        return out, tot_kl
    
def ResNet18_IB_Block(num_classes):
    model = ResNetwithIB(BasicBlockwithIB, [2, 2, 2, 2], num_classes=num_classes)
    for name, param in model.named_parameters():
        if 'prior_z_logD' in name:
            param.requires_grad = False
    return model

def ResNet18_IB_layer(num_classes, ib_layer_pos):
    block_list = [BasicBlock for i in range(4)]
    block_list[ib_layer_pos] = BasicBlockwithIB

    model = ResNetwithIB_layer(block_list, [2, 2, 2, 2], num_classes=num_classes)
    for name, param in model.named_parameters():
        if 'prior_z_logD' in name:
            param.requires_grad = False
    return model


def ResNet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2],num_classes=num_classes)


def ResNet34():
    return ResNet(BasicBlock, [3, 4, 6, 3])


def ResNet50():
    return ResNet(Bottleneck, [3, 4, 6, 3])


def ResNet101():
    return ResNet(Bottleneck, [3, 4, 23, 3])


def ResNet152():
    return ResNet(Bottleneck, [3, 8, 36, 3])


def test():
    net = ResNet18()
    y = net(torch.randn(1, 3, 32, 32))
    print(y.size())

# test()