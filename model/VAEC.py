import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


class WideResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, drop_rate=0.0):
        super(WideResBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(drop_rate)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)

    def forward(self, x):
        out = self.bn1(x)
        out = self.relu1(out)
        out = self.conv1(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = out + self.shortcut(x)
        return out


class StageBlock(nn.Module):
    def __init__(self, WideResBlock_num, in_channels, out_channels, stride, drop_rate=0.0):
        super(StageBlock, self).__init__()
        self.stage_layer = self._make_layer(WideResBlock_num, in_channels, out_channels, stride, drop_rate)

    @staticmethod
    def _make_layer(WideResBlock_num, in_channels, out_channels, stride, drop_rate):
        layers = []
        for i in range(int(WideResBlock_num)):
            if i == 0:
                layers.append(WideResBlock(in_channels, out_channels, stride, drop_rate))
            else:
                layers.append(WideResBlock(out_channels, out_channels, 1, drop_rate))
        return nn.Sequential(*layers)

    def forward(self, x):
        return self.stage_layer(x)


class WideResNet(nn.Module):
    def __init__(self, dataset, depth, num_classes, widen_factor=1, drop_rate=0.5):
        super(WideResNet, self).__init__()
        stage_channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        assert ((depth - 4) % 6 == 0)  # WideResnet的网络深度为6n+4  一个卷积+3个Stage
        n = (depth - 4) / 6  # 每个Stage中分n块，每块中有2个卷积层，每个stage中有一个shortcut层

        # 首先需要有一个卷积层 channels->16
        self.conv1 = nn.Conv2d(3, stage_channels[0], kernel_size=3, stride=1, padding=1)
        # 1st block
        self.block1 = StageBlock(n, stage_channels[0], stage_channels[1], 1, drop_rate)
        # 2nd block
        self.block2 = StageBlock(n, stage_channels[1], stage_channels[2], 2, drop_rate)
        # 3rd block
        self.block3 = StageBlock(n, stage_channels[2], stage_channels[3], 2, drop_rate)
        # global average pooling and classifier
        self.bn1 = nn.BatchNorm2d(stage_channels[3], momentum=0.9)
        self.relu = nn.ReLU(inplace=True)
        self.global_avgpool = nn.AvgPool2d(8)
        if dataset == "cifar10":
            self.fc_input = stage_channels[3] * 1
        elif dataset == "miniimagenet":
            self.fc_input = stage_channels[3] * 56 * 56
        else:
            assert False, "需要指定数据集，CIFAR10或者Imagenet"
        self.fc = nn.Linear(self.fc_input, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.block1(out)
        out = self.block2(out)
        out = self.block3(out)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.global_avgpool(out)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResBlock, self).__init__()
        self.direct_connect = nn.Sequential(
            nn.LeakyReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=1, stride=1, padding=0),
        )
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.shortcut(x) + self.direct_connect(x)


class VAEC_cifar(nn.Module):
    def __init__(self, fm_channel, z_dim, with_classifier=True, widen_factor=1, drop_rate=0.5):
        """
        :param fm_channel: feature map 的 channel数, 残差网络之后的特征图的维度，h or w
        :param z_dim: 隐变量的维度
        :param with_classifier: 是否包含分类器，进行联合训练
        """
        super(VAEC_cifar, self).__init__()
        self.fm_channel = fm_channel
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, self.fm_channel // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.fm_channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fm_channel // 2, self.fm_channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.fm_channel),
            nn.ReLU(inplace=True),
            ResBlock(self.fm_channel, self.fm_channel),
            nn.BatchNorm2d(self.fm_channel),
            ResBlock(self.fm_channel, self.fm_channel),
        )
        self.fc2mu = nn.Linear(self.fm_channel * 8 * 8, self.z_dim)
        self.fc2logvar = nn.Linear(self.fm_channel * 8 * 8, self.z_dim)
        self.fcjudge = nn.Linear(self.z_dim, 10)
        self.fcz2reconfeature = nn.Linear(self.z_dim, self.fm_channel * 8 * 8)

        self.decoder = nn.Sequential(
            ResBlock(self.fm_channel, self.fm_channel),
            nn.BatchNorm2d(self.fm_channel),
            ResBlock(self.fm_channel, self.fm_channel),
            nn.BatchNorm2d(self.fm_channel),

            nn.ConvTranspose2d(self.fm_channel, self.fm_channel // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.fm_channel // 2),
            nn.LeakyReLU(inplace=True),
            nn.ConvTranspose2d(self.fm_channel // 2, 3, kernel_size=4, stride=2, padding=1, bias=False),
        )
        self.tanh = nn.Tanh()
        self.final_bn = nn.BatchNorm2d(3)

        self.with_classifier = with_classifier
        if self.with_classifier:
            self.classifier = WideResNet("cifar10", 28, 10, widen_factor=widen_factor, drop_rate=drop_rate)

    def encode(self, x):
        fm = self.encoder(x)
        fm = fm.view(fm.size(0), self.fm_channel * 8 * 8)
        return fm, self.fc2mu(fm), self.fc2logvar(fm)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def decode(self, z_sampling):
        reconfeature = self.fcz2reconfeature(z_sampling)
        temp = reconfeature.view(reconfeature.size(0), self.fm_channel, 8, 8)
        temp = self.decoder(temp)
        temp = self.tanh(temp)
        temp = self.final_bn(temp)
        return temp

    def forward(self, x, only_classfier):
        if not only_classfier:
            _, mu, logvar = self.encode(x)
            z_sampling = self.reparameterize(mu, logvar)  # + noise* torch.randn(mu.size()).cuda()
            judge_result = self.fcjudge(z_sampling)
            recon_image = self.decode(z_sampling)

            if self.with_classifier:
                classify_result = self.classifier(x - recon_image)  # ###
                return classify_result, z_sampling, judge_result, recon_image, mu, logvar
            else:
                return recon_image
        else:
            """
            在这里输入是直接输入到分类器中的
            """
            classify_result = self.classifier(x)
            return classify_result


class VAEC_Encoder_cifar(nn.Module):
    def __init__(self, fm_channel, z_dim):
        """
        :param fm_channel: feature map 的 channel数, 残差网络之后的特征图的维度，h or w
        :param z_dim: 隐变量的维度
        """
        super(VAEC_Encoder_cifar, self).__init__()
        self.fm_channel = fm_channel
        self.z_dim = z_dim

        self.encoder = nn.Sequential(
            nn.Conv2d(3, self.fm_channel // 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.fm_channel // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.fm_channel // 2, self.fm_channel, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(self.fm_channel),
            nn.ReLU(inplace=True),
            ResBlock(self.fm_channel, self.fm_channel),
            nn.BatchNorm2d(self.fm_channel),
            ResBlock(self.fm_channel, self.fm_channel),
        )
        self.fc2mu = nn.Linear(self.fm_channel * 8 * 8, self.z_dim)
        self.fc2logvar = nn.Linear(self.fm_channel * 8 * 8, self.z_dim)

    def encode(self, x):
        fm = self.encoder(x)
        fm = fm.view(fm.size(0), self.fm_channel * 8 * 8)
        return fm, self.fc2mu(fm), self.fc2logvar(fm)

    def reparameterize(self, mu, logvar):
        if self.training:
            std = logvar.mul(0.5).exp_()
            eps = std.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        _, mu, logvar = self.encode(x)
        z_sampling = self.reparameterize(mu, logvar)  # + noise* torch.randn(mu.size()).cuda()
        return z_sampling


