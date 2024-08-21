import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from robustbench.model_zoo.architectures.wide_resnet import WideResNet


class Adapter(nn.Module):
    def __init__(self, c_in, reduction=4):
        super(Adapter, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(c_in, c_in // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c_in // reduction, c_in, bias=False),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.fc(x)
        return x
    

class WideResNetWithAdapter(WideResNet):
    def __init__(self, depth=28, num_classes=10, widen_factor=10, sub_block1=False, dropRate=0.0, bias_last=True, reduction=4):
        super(WideResNetWithAdapter, self).__init__(depth, num_classes, widen_factor, sub_block1, dropRate, bias_last)
        self.adapter = Adapter(self.nChannels, reduction)  

    def forward(self, x, adapter_ratio=1.0):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)

        adapter_out = self.adapter(out)
        out = adapter_ratio * adapter_out + (1 - adapter_ratio) * out

        return self.fc(out)


class WideResNetWithDropout(WideResNet):
    def __init__(self, depth=28, num_classes=10, widen_factor=10, sub_block1=False, dropRate=0.0, bias_last=True):
        super(WideResNetWithDropout, self).__init__(depth, num_classes, widen_factor, sub_block1, dropRate, bias_last)

    def forward(self, x, dropout_rate=0.0):
        out = self.conv1(x)
        out = self.block1(out)
        out = F.dropout(out, p=dropout_rate, training=True) # let dropout always work despite the model mode
        out = self.block2(out)
        out = F.dropout(out, p=dropout_rate, training=True)
        out = self.block3(out)
        out = F.dropout(out, p=dropout_rate, training=True)
        out = self.relu(self.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.nChannels)
        return self.fc(out)
