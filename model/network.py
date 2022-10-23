import torch
import torch.nn as nn
import torch.nn.functional as F
from model.Res2Net_v1b import res2net50_v1b


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)  # 输入两个通道，一个是maxpool 一个是avgpool的
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)  # 对池化完的数据cat 然后进行卷积
        return self.sigmoid(x)


class CEM(nn.Module):
    def __init__(self, in_c, out_c):
        super(CEM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c[0], out_c[0], kernel_size=3, stride=4, dilation=3, padding=2),
            nn.BatchNorm2d(out_c[0]),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c[1], out_c[1], kernel_size=3, stride=2, dilation=2, padding=2),
            nn.BatchNorm2d(out_c[1]),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_c[2], out_c[2], kernel_size=3, stride=1, dilation=1, padding=1),
            nn.BatchNorm2d(out_c[2]),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_c[2], out_c[2], kernel_size=1, stride=1),
            nn.BatchNorm2d(out_c[2]),
            nn.ReLU()
        )
        self.conv_ = nn.Sequential(
            nn.Conv2d(sum(out_c) - out_c[3], out_c[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_c[3]),
            nn.ReLU()
        )

    def forward(self, x1, x2, x3):
        x1_ = self.conv1(x1)
        x2_ = self.conv2(x2)
        x3_ = self.conv3(x3)
        x3 = self.conv(x3)
        out = self.conv_(torch.cat((x3_, x2_, x1_), dim=1))

        return x3 + out


class CEM_2(nn.Module):
    def __init__(self, in_c, out_c):
        super(CEM_2, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c[0], out_c[0], kernel_size=3, stride=2, dilation=2, padding=2),
            nn.BatchNorm2d(out_c[0]),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c[1], out_c[1], kernel_size=3, stride=2, dilation=2, padding=2),
            nn.BatchNorm2d(out_c[1]),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_c[2], out_c[2], kernel_size=3, stride=1, dilation=1, padding=1),
            nn.BatchNorm2d(out_c[2]),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_c[2], out_c[2], kernel_size=1, stride=1),
            nn.BatchNorm2d(out_c[2]),
            nn.ReLU()
        )
        self.conv_ = nn.Sequential(
            nn.Conv2d(sum(out_c) - out_c[3], out_c[3], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_c[3]),
            nn.ReLU()
        )

    def forward(self, x1, x2, x3):
        x1_ = self.conv1(x1)
        x2_ = self.conv2(x2)
        x3_ = self.conv3(x3)
        x3 = self.conv(x3)
        out = self.conv_(torch.cat((x3_, x2_, x1_), dim=1))

        return x3 + out


class CEM_1(nn.Module):
    def __init__(self, in_c, out_c):
        super(CEM_1, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c[0], out_c[0], kernel_size=3, stride=1, dilation=1, padding=1),
            nn.BatchNorm2d(out_c[0]),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c[1], out_c[1], kernel_size=3, stride=1, dilation=1, padding=1),
            nn.BatchNorm2d(out_c[1]),
            nn.ReLU()
        )
        self.conv = nn.Sequential(
            nn.Conv2d(in_c[1], out_c[1], kernel_size=1, stride=1),
            nn.BatchNorm2d(out_c[1]),
            nn.ReLU()
        )
        self.conv_ = nn.Sequential(
            nn.Conv2d(out_c[0] + out_c[1], out_c[2], kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_c[2]),
            nn.ReLU()
        )

    def forward(self, x1, x2):
        x1_ = self.conv1(x1)
        x2_ = self.conv2(x2)
        x3 = self.conv(x2)
        out = self.conv_(torch.cat((x1_, x2_), dim=1))

        return x3 + out


class CFAM(nn.Module):
    def __init__(self, in_c, out_c):
        super(CFAM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_c[0], out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_c[0], out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_c[1], out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
        self.conv4 = nn.Sequential(
            nn.Conv2d(out_c * 2, out_c, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )

    def forward(self, h_feat, l_feat):
        l_feat = F.interpolate(l_feat, scale_factor=2, mode="bilinear")
        l1 = self.conv1(l_feat)
        l2 = self.conv2(l_feat)

        h1 = self.conv3(h_feat)
        h2 = self.conv4(torch.cat((l2 + h1, l1), dim=1))

        return l1 + h2


class C_S_attn(nn.Module):
    def __init__(self, in_c):
        super(C_S_attn, self).__init__()
        self.c_attn = ChannelAttention(in_c)
        self.s_attn = SpatialAttention()

    def forward(self, x):
        out1 = self.c_attn(x)
        x = out1 * x
        out2 = self.s_attn(x)
        x = out2 * x

        return x


class CIM(nn.Module):
    def __init__(self, backbone_pre=True):
        super(CIM, self).__init__()

        self.res = res2net50_v1b(pretrained=backbone_pre)
        self.conv1 = nn.Sequential(
            nn.Conv2d(1024, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 1, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(1),
            nn.ReLU()
        )

        self.cem1 = CEM_1([64, 256], [64, 256, 256])
        self.cem2 = CEM_2([64, 256, 512], [64, 256, 512, 512])
        self.cem3 = CEM([256, 512, 1024], [256, 512, 1024, 1024])
        self.cem4 = CEM([512, 1024, 2048], [512, 1024, 2048, 2048])

        self.cfam1 = CFAM([512, 256], 256)
        self.cfam2 = CFAM([1024, 512], 512)
        self.cfam3 = CFAM([2048, 1024], 1024)

        self.c_s_attn1 = C_S_attn(64)
        self.c_s_attn2 = C_S_attn(256)
        self.c_s_attn3 = C_S_attn(512)
        self.c_s_attn4 = C_S_attn(1024)
        self.c_s_attn5 = C_S_attn(2048)

    def forward(self, x):
        x = self.res.conv1(x)
        x = self.res.bn1(x)
        x = self.res.relu(x)
        x1 = self.res.maxpool(x)
        x2 = self.res.layer1(x1)
        x3 = self.res.layer2(x2)
        x4 = self.res.layer3(x3)
        x5 = self.res.layer4(x4)

        x1 = self.c_s_attn1(x1)
        x2 = self.c_s_attn2(x2)
        x3 = self.c_s_attn3(x3)
        x4 = self.c_s_attn4(x4)
        x5 = self.c_s_attn5(x5)

        out_1 = self.cem1(x1, x2)
        out_2 = self.cem2(x1, x2, x3)
        out_3 = self.cem3(x2, x3, x4)
        out_4 = self.cem4(x3, x4, x5)

        out1 = self.cfam3(out_3, out_4)
        out2 = self.cfam2(out_2, out_3)
        out3 = self.cfam1(out_1, out_2)

        out1 = self.conv1(out1)
        out2 = self.conv2(out2)
        out3 = self.conv3(out3)

        out1 = F.interpolate(out1, scale_factor=16, mode='bilinear')
        out2 = F.interpolate(out2, scale_factor=8, mode='bilinear')
        out3 = F.interpolate(out3, scale_factor=4, mode='bilinear')

        return out1, out2, out3


if __name__ == '__main__':
    input = torch.rand(size=(1, 3, 352, 352)).cuda()
    model = CIM().cuda()
    # summary(model,input_size=(1,3,352,352))
    out1, out2, out3 = model(input)
    print(out1.shape, out2.shape, out3.shape)
