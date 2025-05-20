from __future__ import print_function, division
from collections import OrderedDict
import os
import torch
import torch.nn as nn
import torch.nn.functional as F





class DFL(nn.Module):
    def __init__(self, use_cuda=True, in_channels=256, out_channels=256, num_branch=4):  # 4,10,2
        super(DFL, self).__init__()
        assert in_channels % 4 == 0
        self.key_channel = in_channels // 4
        self.padding = 1
        self.kernel_size = 3
        self.in_planes = in_channels
        self.num_branch = num_branch
        self.query = nn.Sequential(nn.Conv2d(in_channels+2, in_channels*2, kernel_size=1, padding=0),
                                   nn.ReLU(True),
                                   nn.Conv2d(in_channels*2, self.key_channel, kernel_size=1, padding=0),
                                   nn.ReLU(True))


        self.memory_key = nn.Conv2d(self.key_channel, self.num_branch, kernel_size=1, padding=0, bias=False)
        self.memory_value = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1) for _ in range(self.num_branch)])

        if use_cuda:
            self.memory_key = self.memory_key.cuda()
            self.memory_value = self.memory_value.cuda()
            self.query = self.query.cuda()

    def dynamic_conv2d(self, x, aggregate_weight, aggregate_bias):
        batch_size, in_planes, height, width = x.size()
        x = x.reshape(1, -1, height, width)

        output = F.conv2d(x, weight=aggregate_weight, bias=aggregate_bias, padding=self.padding, groups=batch_size)
        output = output.reshape(batch_size, -1, output.size(-2), output.size(-1))
        return output

    def forward(self, x, y,scale,flag):
        batch_size, in_planes, height, width = x.size()
        if flag:
            scale_h = torch.ones(1, 1).to(x.device) * scale[0]
            scale_w = torch.ones(1, 1).to(x.device) * scale[1]
            scale_info = torch.cat([scale_h,scale_w],dim=1).repeat(batch_size, 1).unsqueeze(-1).unsqueeze(-1)
        else:
            scale = scale.reshape(batch_size, 1)
            scale_info = scale.repeat(1,2).unsqueeze(-1).unsqueeze(-1)


        x = x + self.memory_value[0](x)
        y = y + self.memory_value[0](y)

        global_x = F.adaptive_avg_pool2d(x, (1, 1))
        global_y = F.adaptive_avg_pool2d(y, (1, 1))

        query = self.query(torch.cat([(global_x + global_y),scale_info],dim=1))


        memory_key = self.memory_key.weight  # K, C//32
        memory_key = memory_key.view(-1, self.num_branch)  # C//32, K
        softmax_attention = torch.mm(query[:, :, 0, 0], memory_key)  # B, H, W, K

        softmax_attention = F.softmax(softmax_attention / 20, dim=1).view(x.size(0), -1)


        weight = self.memory_value[0].weight.unsqueeze(0)
        for i in range(1, self.num_branch):
            weight = torch.cat((weight, self.memory_value[i].weight.unsqueeze(0)), dim=0)
        weight = weight.view(self.num_branch, -1)

        # 
        aggregate_weight = torch.mm(softmax_attention, weight).view(-1, self.in_planes, self.kernel_size,
                                                                    self.kernel_size)

        bias = self.memory_value[0].bias.unsqueeze(0)
        for i in range(1, self.num_branch):
            bias = torch.cat((bias, self.memory_value[i].bias.unsqueeze(0)), dim=0)
        bias = bias.view(self.num_branch, -1)
        # bias = self.memory_value.bias.view(self.num_branch, -1)
        aggregate_bias = torch.mm(softmax_attention, bias).view(-1)

        x = self.dynamic_conv2d(x, aggregate_weight, aggregate_bias) + x
        y = self.dynamic_conv2d(y, aggregate_weight, aggregate_bias) + y
        return x, y


class attention2d(nn.Module):
    def __init__(self, in_planes, ratios, K, temperature, init_weight=True):
        super(attention2d, self).__init__()
        assert temperature % 3 == 1
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        if in_planes != 3:
            hidden_planes = int(in_planes * ratios) + 1
        else:
            hidden_planes = K
        self.fc1 = nn.Conv2d(in_planes, hidden_planes, 1, bias=False)
        # self.bn = nn.BatchNorm2d(hidden_planes)
        self.fc2 = nn.Conv2d(hidden_planes, K, 1, bias=True)
        self.temperature = temperature
        if init_weight:
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def updata_temperature(self):
        if self.temperature != 1:
            self.temperature -= 3
            print('Change temperature to:', str(self.temperature))

    def forward(self, x, y):
        x = self.avgpool(x)
        y = self.avgpool(y)
        x = self.fc1(x + y)
        x = F.relu(x)
        x = self.fc2(x).view(x.size(0), -1)
        return F.softmax(x / self.temperature, 1)






def dynamic_conv_blcok():
    return DFL(True,in_channels=64, out_channels=64, num_branch=4)


if __name__ == "__main__":
    m = DFL(True, 64, 64, 2).cuda()
    x = torch.randn(8, 64, 64, 64).cuda()
    y = torch.randn(8, 64, 64, 64).cuda()
    x1, y1 = m(x, y)
    print(x1.shape)
    print(y1.shape)
    y2, x2 = m(y, x)
    print(torch.norm(x2 - x1))
    print(torch.norm(y2 - y1))
