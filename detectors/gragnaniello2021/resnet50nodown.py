# -*- coding: utf-8 -*-
# %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
#
# Copyright (c) 2019 Image Processing Research Group of University Federico II of Naples ('GRIP-UNINA').
# All rights reserved.
# This work should only be used for nonprofit purposes.
#
# By downloading and/or using any of these files, you implicitly agree to all the
# terms of the license, as specified in the document LICENSE.md
# (included in this package) and online at
# http://www.grip.unina.it/download/LICENSE_OPEN.txt
#

import torch
import torch.nn as nn
import torchvision.transforms as transforms
LIMIT_SIZE  = 1536
LIMIT_SLIDE = 1024

class ChannelLinear(nn.Linear):

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(ChannelLinear, self).__init__(in_features, out_features, bias)

    def forward(self, x):
        out_shape = [x.shape[0], x.shape[2], x.shape[3], self.out_features]
        x = x.permute(0,2,3,1).reshape(-1,self.in_features)
        x = x.matmul(self.weight.t())
        if self.bias is not None:
            x = x + self.bias[None,:]
        x = x.view(out_shape).permute(0,3,1,2)
        return x

    
def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = conv1x1(inplanes, planes)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = conv1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1, stride0=2):
        super(ResNet, self).__init__()
        self.inplanes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=stride0, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=stride0, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.num_features = 512 * block.expansion
        self.fc = ChannelLinear(self.num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # transform form Pillow
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                             std=[0.229, 0.224, 0.225])])



    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def change_output(self, num_classes):
        self.fc = ChannelLinear(self.num_features, num_classes)
        torch.nn.init.normal_(self.fc.weight.data, 0.0, 0.02)
        return self
        
    def change_input(self, num_inputs):
        data = self.conv1.weight.data
        old_num_inputs = int(data.shape[1])
        if num_inputs>old_num_inputs:
            times = num_inputs//old_num_inputs
            if (times*old_num_inputs)<num_inputs:
                times = times+1
            data = data.repeat(1,times,1,1) / times
        elif num_inputs==old_num_inputs:
            return self
        
        data = data[:,:num_inputs,:,:]
        print(self.conv1.weight.data.shape, '->', data.shape)
        self.conv1.weight.data = data
        
        return self
    
    def feature(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x
    
    def forward(self, x):
        x = self.feature(x)
        x = self.avgpool(x)
        x = self.fc(x)

        return x
            
    def apply(self, pil):
        device = self.conv1.weight.device
        if (pil.size[0]>LIMIT_SIZE) and (pil.size[1]>LIMIT_SIZE):
            import numpy as np
            print('err:', pil.size)
            with torch.no_grad():
                img = self.transform(pil)
                list_logit  = list()
                list_weight = list()
                for index0 in range(0, img.shape[-2], LIMIT_SLIDE):
                    for index1 in range(0, img.shape[-1], LIMIT_SLIDE):
                        clip = img[..., index0:min(index0+LIMIT_SLIDE,  img.shape[-2]),
                                        index1:min(index1+LIMIT_SLIDE,  img.shape[-1])]
                        logit  = torch.squeeze(self(clip.to(device)[None,:,:,:])).cpu().numpy()
                        weight = clip.shape[-2] * clip.shape[-1]
                        list_logit.append(logit)
                        list_weight.append(weight)
            
            logit = np.mean(np.asarray(list_logit) * np.asarray(list_weight)) / np.mean(list_weight)
        else:
            with torch.no_grad():
                logit = torch.squeeze(self(self.transform(pil).to(device)[None,:,:,:])).cpu().numpy()
        
        return logit

def resnet50nodown(device, filename, num_classes=1):
    """Constructs a ResNet-50 nodown model.
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, stride0=1)
    model.load_state_dict(torch.load(filename, map_location=torch.device('cpu'))['model'])
    model = model.to(device).eval()
    return model


