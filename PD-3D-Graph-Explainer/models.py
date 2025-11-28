# models.py (updated)
import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock3D(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, norm='batch', num_groups=8):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        if norm == 'group':
            self.bn1 = nn.GroupNorm(num_groups, planes)
        else:
            self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        if norm == 'group':
            self.bn2 = nn.GroupNorm(num_groups, planes)
        else:
            self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = None
        if stride != 1 or in_planes != planes:
            if norm == 'group':
                self.downsample = nn.Sequential(
                    nn.Conv3d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.GroupNorm(num_groups, planes)
                )
            else:
                self.downsample = nn.Sequential(
                    nn.Conv3d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                    nn.BatchNorm3d(planes)
                )
    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(identity)
        out += identity
        out = F.relu(out)
        return out

class ResNet3D_withSliceHead(nn.Module):
    """3D ResNet encoder + patient-level head + slice-level head on top of last feature map."""
    def __init__(self, in_channels=3, base_channels=16, block=BasicBlock3D, layers=[2,2,2,2], num_classes=1, norm='group', num_groups=8, dropout=0.0):
        super().__init__()
        self.in_planes = base_channels
        self.norm = norm
        self.num_groups = num_groups
        self.conv1 = nn.Conv3d(in_channels, base_channels, kernel_size=7, stride=(1,2,2), padding=3, bias=False)
        if norm == 'group':
            self.bn1 = nn.GroupNorm(num_groups, base_channels)
        else:
            self.bn1 = nn.BatchNorm3d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(block, base_channels*2, layers[1], stride=(1,2,2))
        self.layer3 = self._make_layer(block, base_channels*4, layers[2], stride=(1,2,2))
        self.layer4 = self._make_layer(block, base_channels*8, layers[3], stride=1)
        self.avgpool = nn.AdaptiveAvgPool3d((1,1,1))
        self.fc = nn.Linear(base_channels*8*block.expansion, num_classes)
        self.dropout = nn.Dropout(dropout) if dropout and dropout>0 else None
    def _make_layer(self, block, planes, blocks, stride=1):
        strides = [stride] + [1]*(blocks-1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, stride=s, norm=self.norm, num_groups=self.num_groups))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x, return_feat_map=False):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        feat_map = self.layer4(x)
        pooled = self.avgpool(feat_map)
        pooled = pooled.view(pooled.size(0), -1)
        if self.dropout is not None:
            pooled = self.dropout(pooled)
        logits = self.fc(pooled).squeeze(-1)
        outputs = (logits,)
        if return_feat_map:
            outputs += (feat_map,)
        if len(outputs) == 1:
            return outputs[0]
        return outputs

ResNet3D = ResNet3D_withSliceHead
