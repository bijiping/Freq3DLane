import torch
import torchvision as tv
from thop import profile
from torch import nn
from z_my.singleheadatt import SHSA
from z_my.FreqFusion import FreqFusion

def naive_init_module(mod):
    for m in mod.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
    return mod


class InstanceEmbedding_offset_y_z(nn.Module):
    def __init__(self, ci, co=1):
        super(InstanceEmbedding_offset_y_z, self).__init__()
        self.neck_new = nn.Sequential(
            # SELayer(ci),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, ci, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ci),
            nn.ReLU(),
        )

        self.ms_new = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )

        self.m_offset_new = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )

        self.m_z = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )

        self.me_new = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, co, 3, 1, 1, bias=True)
        )

        naive_init_module(self.ms_new)
        naive_init_module(self.me_new)
        naive_init_module(self.m_offset_new)
        naive_init_module(self.m_z)
        naive_init_module(self.neck_new)

    def forward(self, x):
        feat = self.neck_new(x)
        return self.ms_new(feat), self.me_new(feat), self.m_offset_new(feat), self.m_z(feat)


class InstanceEmbedding(nn.Module):
    def __init__(self, ci, co=1):
        super(InstanceEmbedding, self).__init__()
        self.neck = nn.Sequential(
            # SELayer(ci),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, ci, 3, 1, 1, bias=False),
            nn.BatchNorm2d(ci),
            nn.ReLU(),
        )

        self.ms = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, 1, 3, 1, 1, bias=True)
        )

        self.me = nn.Sequential(
            # nn.Dropout2d(0.2),
            nn.Conv2d(ci, 128, 3, 1, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.Conv2d(128, 64, 3, 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.Conv2d(64, co, 3, 1, 1, bias=True)
        )

        naive_init_module(self.ms)
        naive_init_module(self.me)
        naive_init_module(self.neck)

    def forward(self, x):
        feat = self.neck(x)
        return self.ms(feat), self.me(feat)


class LaneHeadResidual_Instance_with_offset_z(nn.Module):
    def __init__(self, output_size, input_channel=256):
        super(LaneHeadResidual_Instance_with_offset_z, self).__init__()

        self.bev_up_new = nn.Sequential(
            nn.Upsample(scale_factor=2),  #
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(input_channel, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 128, 3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                ),
                downsample=nn.Conv2d(input_channel, 128, 1),
            ),
            nn.Upsample(size=output_size),  #
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    # nn.ReLU(),
                ),
                downsample=nn.Conv2d(128, 64, 1),
            ),
        )
        self.head = InstanceEmbedding_offset_y_z(64, 2)
        naive_init_module(self.head)
        naive_init_module(self.bev_up_new)

    def forward(self, bev_x):
        bev_feat = self.bev_up_new(bev_x)
        return self.head(bev_feat)


class LaneHeadResidual_Instance(nn.Module):
    def __init__(self, output_size, input_channel=256):
        super(LaneHeadResidual_Instance, self).__init__()

        self.bev_up = nn.Sequential(
            nn.Upsample(scale_factor=2),  # 60x 24
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(input_channel, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 128, 3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
                ),
                downsample=nn.Conv2d(input_channel, 128, 1),
            ),
            nn.Upsample(scale_factor=2),  # 120 x 48
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1, bias=False),
                    nn.BatchNorm2d(64),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(64, 32, 3, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                    # nn.ReLU(),
                ),
                downsample=nn.Conv2d(128, 32, 1),
            ),

            nn.Upsample(size=output_size),  # 300 x 120
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(32, 16, 3, padding=1, bias=False),
                    nn.BatchNorm2d(16),
                    nn.ReLU(),
                    nn.Dropout2d(p=0.2),
                    nn.Conv2d(16, 32, 3, padding=1, bias=False),
                    nn.BatchNorm2d(32),
                )
            ),
        )

        self.head = InstanceEmbedding(32, 2)
        naive_init_module(self.head)
        naive_init_module(self.bev_up)

    def forward(self, bev_x):
        bev_feat = self.bev_up(bev_x)
        return self.head(bev_feat)


class FCTransform_(nn.Module):
    def __init__(self, image_featmap_size, space_featmap_size):
        super(FCTransform_, self).__init__()
        ic, ih, iw = image_featmap_size  # (256, 16, 16)
        sc, sh, sw = space_featmap_size  # (128, 16, 32)
        self.image_featmap_size = image_featmap_size
        self.space_featmap_size = space_featmap_size
        self.fc_transform = nn.Sequential(
            nn.Linear(ih * iw, sh * sw),
            nn.ReLU(),
            nn.Linear(sh * sw, sh * sw),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=ic, out_channels=sc, kernel_size=1 * 1, stride=1, bias=False),
            nn.BatchNorm2d(sc),
            nn.ReLU(), )
        self.residual = Residual(
            module=nn.Sequential(
                nn.Conv2d(in_channels=sc, out_channels=sc, kernel_size=3, padding=1, stride=1, bias=False),
                nn.BatchNorm2d(sc),
            ))

    def forward(self, x):
        x = x.view(list(x.size()[:2]) + [self.image_featmap_size[1] * self.image_featmap_size[2], ])  # 这个 B,V,C,H*W
        bev_view = self.fc_transform(x)  # 拿出一个视角
        bev_view = bev_view.view(list(bev_view.size()[:2]) + [self.space_featmap_size[1], self.space_featmap_size[2]])
        bev_view = self.conv1(bev_view)
        bev_view = self.residual(bev_view)
        return bev_view


class Residual(nn.Module):
    def __init__(self, module, downsample=None):
        super(Residual, self).__init__()
        self.module = module
        self.downsample = downsample
        self.relu = nn.ReLU()

    def forward(self, x):
        identity = x
        out = self.module(x)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)

class ChannelAttention(nn.Module):
    def __init__(self, in_planes, out_planes, ratio=4, flag=True):
        super(ChannelAttention, self).__init__()
        # 自适应平均池化，将输入特征图变为1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 自适应最大池化，将输入特征图变为1x1
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 卷积层1，将输入通道数降维
        self.conv1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        # ReLU激活函数
        self.relu = nn.ReLU()
        # 卷积层2，将通道数恢复
        self.conv2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0)

        # 标志位，决定输出是否与输入相乘
        self.flag = flag
        # Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

        # 初始化卷积层权重
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)

    def forward(self, x):
        # 通过平均池化和两层卷积计算通道注意力
        avg_out = self.conv2(self.relu(self.conv1(self.avg_pool(x))))
        # 通过最大池化和两层卷积计算通道注意力
        max_out = self.conv2(self.relu(self.conv1(self.max_pool(x))))
        # 将平均池化和最大池化的结果相加
        out = avg_out + max_out
        # 根据flag决定是否与输入相乘并返回结果
        out = self.sigmoid(out) * x if self.flag else self.sigmoid(out)
        out = self.relu(self.conv3(out))

        return out


class ChannelAttention0(nn.Module):
    def __init__(self, in_planes, out_planes, ratio=4, flag=True):
        super(ChannelAttention0, self).__init__()
        # 自适应平均池化，将输入特征图变为1x1
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # 自适应最大池化，将输入特征图变为1x1
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        # 卷积层1，将输入通道数降维
        self.conv1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        # ReLU激活函数
        self.relu = nn.ReLU()
        # 卷积层2，将通道数恢复
        self.conv2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0)

        # 标志位，决定输出是否与输入相乘
        self.flag = flag
        # Sigmoid激活函数
        self.sigmoid = nn.Sigmoid()

        # 初始化卷积层权重
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)

    def forward(self, x):
        # 通过平均池化和两层卷积计算通道注意力
        avg_out = self.avg_pool(x)
        avg_out = self.conv1(avg_out)
        avg_out = self.relu(avg_out)
        avg_out = self.conv2(avg_out)

        # 通过最大池化和两层卷积计算通道注意力
        max_out = self.conv2(self.relu(self.conv1(self.max_pool(x))))
        # 将平均池化和最大池化的结果相加
        out = avg_out + max_out
        # 根据flag决定是否与输入相乘并返回结果
        aAA = self.sigmoid(out)
        out = self.sigmoid(out) * x if self.flag else self.sigmoid(out)
        out = out + x
        out = self.relu(self.conv3(out))

        return out


# model
# ResNet34 骨干网络 (self.bb)，在 ImageNet 上进行预训练。
# 一个下采样层 (self.down)，用于减小特征图的空间维度。
# 两个全连接变换层 (self.s32transformer 和 self.s64transformer)，将 ResNet 骨干网络的特征图转换为 BEV 表示。
# 车道线检测头 (self.lane_head)，以 BEV 表示作为输入，输出表示检测到的车道线的张量。
# 可选的 2D 图像车道线检测头 (self.lane_head_2d)，以 ResNet 骨干网络的输出作为输入，输出表示原始图像中检测到的车道线的张量。
class BEV_LaneDet(nn.Module):  # BEV-LaneDet
    def __init__(self, bev_shape, output_2d_shape, train=True):
        super(BEV_LaneDet, self).__init__()

        self.bb1 = nn.Sequential(*list(tv.models.resnet34(pretrained=True).children())[:-4])

        self.bb2 = nn.Sequential(*list(tv.models.resnet34(pretrained=True).children())[-4])

        self.bb3 = nn.Sequential(*list(tv.models.resnet34(pretrained=True).children())[-3])

        self.down = naive_init_module(
            Residual(
                module=nn.Sequential(
                    nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),  # S64
                    nn.BatchNorm2d(1024),
                    nn.ReLU(),
                    nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
                    nn.BatchNorm2d(1024)

                ),
                downsample=nn.Conv2d(512, 1024, kernel_size=3, stride=2, padding=1),
            )
        )

        self.att2 = ChannelAttention(256, 512)
        self.att3 = ChannelAttention(512, 512)
        self.att4 = ChannelAttention(1024, 512)

        self.ff1 = FreqFusion(hr_channels=512, lr_channels=512)
        self.ff2 = FreqFusion(hr_channels=512, lr_channels=512)


        self.s16transformer = FCTransform_((512, 36, 64), (256, 25, 10))
        self.s32transformer = FCTransform_((512, 18, 32), (256, 25, 10))  # FCTransform_ 是一个自定义模块，通常用于特征转换。参数 (1024, 9, 16) 和 (256, 25, 5) 可能分别代表输入和输出的形状。这个模块会将特征图从一个形状转换为另一个形状。
        self.s64transformer = FCTransform_((512, 9, 16), (256, 25, 10))


        self.lane_head = LaneHeadResidual_Instance_with_offset_z(bev_shape, input_channel=512)
        self.satt16 = SHSA(256, 16, 128)
        self.satt32 = SHSA(256, 16, 128)
        self.satt64 = SHSA(256, 16, 128)

        self.lane_head = LaneHeadResidual_Instance_with_offset_z(bev_shape, input_channel=512+256)
        self.is_train = train
        if self.is_train:
            self.lane_head_2d = LaneHeadResidual_Instance(output_2d_shape, input_channel=512)


    def forward(self, img):
        img_s8 = self.bb1(img)  # [batch, 512, 72, 128] 图像从(576,1024)缩小8倍
        img_s16 = self.bb2(img_s8)  # torch.Size([2, 256, 36, 64])
        img_s32 = self.bb3(img_s16)  # torch.Size([2, 512, 18, 32])
        img_s64 = self.down(img_s32)  # torch.Size([batch, 1024, 9, 16])


        s_16 = self.att2(img_s16)
        s_32 = self.att3(img_s32)
        s_64 = self.att4(img_s64)

        x2, x3, x4 = s_16, s_32, s_64

        y4 = x4
        _, x3, y4_up = self.ff1(hr_feat=x3, lr_feat=y4)
        y3 = x3 + y4_up
        _, x2, y3_up = self.ff2(hr_feat=x2, lr_feat=y3)
        y2 = x2 + y3_up

        fs_16, fs_32 = y2, y3

        bev_16 = self.s16transformer(fs_16)
        bev_32 = self.s32transformer(fs_32)
        bev_64 = self.s64transformer(s_64)


        bev_16 = self.satt16(bev_16)
        bev_32 = self.satt32(bev_32)
        bev_64 = self.satt64(bev_64)

        bev = torch.cat([bev_64, bev_32, bev_16], dim=1)
        if self.is_train:
            return self.lane_head(bev), self.lane_head_2d(img_s32)
        else:
            return self.lane_head(bev)


if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    output_2d_shape = (144, 256)

    ''' BEV range '''
    x_range = (3, 103)
    y_range = (-12, 12)
    meter_per_pixel = 0.5  # grid size
    bev_shape = (int((x_range[1] - x_range[0]) / meter_per_pixel), int((y_range[1] - y_range[0]) / meter_per_pixel))

    model = BEV_LaneDet(bev_shape=bev_shape, output_2d_shape=output_2d_shape, train=False).to(device)

    x = torch.randn(1, 3, 576, 1024).to(device)

    y = model(x)
    print(y[0][0].shape)
    # print(y)

    #
    #
    #
    flops, params = profile(model.to(device), inputs=(x,))
    print("flops:", flops)
    print("params:", params)
    # net = ChannelAttention0(48, 256)
    # x = torch.randn(1, 48, 72, 128)
    # y = net(x)
    # print(y.shape)

