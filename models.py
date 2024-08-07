import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(self, max_words, emb_size, hid_size, dropout=0.2):
        super().__init__()
        self.max_words = max_words
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.dropout = dropout
        self.Embedding = nn.Embedding(self.max_words, self.emb_size)
        self.my_LSTM = nn.LSTM(self.emb_size, self.hid_size, num_layers=1,
                            batch_first=True, bidirectional=True)  # 2层双向LSTM
        self.dp = nn.Dropout(self.dropout)
        self.fc1 = nn.Linear(self.hid_size * 2, self.hid_size)
        self.fc2 = nn.Linear(self.hid_size, self.hid_size)

        self.fc3 = nn.Linear(self.hid_size, 2)
        pass

    def forward(self, x):
        x = self.Embedding(x)
        x = self.dp(x)
        x, _ = self.my_LSTM(x)
        x = self.dp(x)
        x = F.relu(self.fc1(x))
        h = F.avg_pool2d(x, (x.shape[1], 1)).squeeze()
        # print("h ",h.shape)
        ch = self.fc2(h)
        # print("ch ",ch.shape)
        out = self.fc3(ch)  # [bs, 2]
        return ch.reshape(-1,128),h.reshape(-1,128),out.reshape(-1,2)  # [bs, 2]
    pass
#-----------------------------------------------------------------------------------------------------------------------
class MyLinearLayer(torch.nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.weights = torch.nn.Parameter(torch.randn(out_features, in_features))
        self.bias = torch.nn.Parameter(torch.zeros(out_features))
        pass

    def forward(self, x):
        return torch.mm(x, self.weights.t()) + self.bias
        pass
    pass



class CNN_mnist(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1=torch.nn.Conv2d(1,10,kernel_size=5)
        self.conv2=torch.nn.Conv2d(10,20,kernel_size=5)
        self.dropout=torch.nn.Dropout()
        self.fc1=torch.nn.Linear(320,256)
        self.fc2=torch.nn.Linear(256,128)
        self.fc3=torch.nn.Linear(128,10)

        pass
    def forward(self,x):
        h=F.relu(F.max_pool2d(self.conv1(x),2))
        h=F.relu(F.max_pool2d(self.dropout(self.conv2(h)),2))
        h = h.view(-1, h.shape[1] * h.shape[2] * h.shape[3])
        ch=F.relu(self.fc1(h))
        ch=self.fc2(ch)
        x=self.fc3(ch)
        return ch,h,x
        pass
    pass



class CNN_fashion_mnist(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=torch.nn.Sequential(
            torch.nn.Conv2d(1,16,kernel_size=5,padding=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.layer2=torch.nn.Sequential(
            torch.nn.Conv2d(16,32,kernel_size=5,padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.fc1=torch.nn.Linear(1568,256)
        self.fc2=torch.nn.Linear(256,128)
        self.fc3=torch.nn.Linear(128,10)

        pass
    def forward(self,x):
        h=self.layer1(x)
        h=self.layer2(h)
        h=h.view(h.size(0),-1)
        # print("h ",h.shape )
        ch=F.relu(self.fc1(h))
        ch=self.fc2(ch)
        x=self.fc3(ch)
        return ch,h,x
        pass
    pass



class CNN_emnist(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1=torch.nn.Sequential(
            torch.nn.Conv2d(1,16,kernel_size=5,padding=2),
            torch.nn.BatchNorm2d(16),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.layer2=torch.nn.Sequential(
            torch.nn.Conv2d(16,32,kernel_size=5,padding=2),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2)
        )
        self.fc1=torch.nn.Linear(7*7*32,256)
        self.fc2=torch.nn.Linear(256,128)
        self.fc3=torch.nn.Linear(128,62)

        pass
    def forward(self,x):
        h=self.layer1(x)
        h=self.layer2(h)
        h=h.view(h.size(0),-1)
        ch = F.relu(self.fc1(h))
        ch= self.fc2(ch)
        x=self.fc3(ch)
        return ch,h,x
        pass
    pass



class CNN_cifar10(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 6, 5)
        self.relu =torch.nn.ReLU()
        self.pool = torch.nn.MaxPool2d(2, 2)
        self.conv2 = torch.nn.Conv2d(6, 16, 5)

        self.fc1 = torch.nn.Linear(16*5*5, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 10)

    def forward(self, x):

        h = self.pool(self.relu(self.conv1(x)))
        h = self.pool(self.relu(self.conv2(h)))
        h= h.view(-1, 16 * 5 * 5)

        ch = F.relu(self.fc1(h))
        ch = self.fc2(ch)
        x = self.fc3(ch)
        return ch,h,x

#-----------------------------------------------------------------------------------------------------------------------
class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.block_conv = torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            torch.nn.BatchNorm2d(out_channels),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            torch.nn.BatchNorm2d(out_channels)
        )

        self.shortcut = torch.nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = torch.nn.Sequential(
                torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                torch.nn.BatchNorm2d(out_channels)
            )
            pass
        pass

    def forward(self, x):
        out1 = self.block_conv(x)
        out2 = self.shortcut(x) + out1
        out2 = F.relu(out2)
        return out2
    pass


class ResNet18(torch.nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(64)
        self.relu = torch.nn.ReLU(inplace=True)
        self.layer1 = self.layer(64, 64, 2)
        self.layer2 = self.layer(64, 128, 2, stride=2)
        self.layer3 = self.layer(128, 256, 2, stride=2)
        self.layer4 = self.layer(256, 512, 2, stride=2)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = torch.nn.Linear(512, 256)
        self.fc2 = torch.nn.Linear(256, 128)

        self.fc3 = torch.nn.Linear(128, num_classes)

        pass
    def layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels))
        return torch.nn.Sequential(*layers)

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu(h)
        h = self.layer1(h)
        h = self.layer2(h)
        h = self.layer3(h)
        h = self.layer4(h)
        h = self.avg_pool(h)
        h = torch.flatten(h, 1)
       #x = self.fc(h)
        ch=F.relu(self.fc1(h))
        ch=self.fc2(ch)
        x=self.fc3(ch)
        return ch,h,x
    pass





#ShuffleNet------------------------------------------------------------------------------------------------------------------------------------------------------------------


class ShuffleBlock(nn.Module):
    def __init__(self, groups):
        super(ShuffleBlock, self).__init__()
        self.groups = groups

    def forward(self, x):
        '''Channel shuffle: [N,C,H,W] -> [N,g,C/g,H,W] -> [N,C/g,g,H,w] -> [N,C,H,W]'''
        N, C, H, W = x.size()
        g = self.groups
        return x.view(N, g, C // g, H, W).permute(0, 2, 1, 3, 4).reshape(N, C, H, W)


class Bottleneck(nn.Module):
    def __init__(self, in_planes, out_planes, stride, groups):
        super(Bottleneck, self).__init__()
        self.stride = stride

        mid_planes = out_planes // 4
        g = 1 if in_planes == 24 else groups
        self.conv1 = nn.Conv2d(in_planes, mid_planes, kernel_size=1, groups=g, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_planes)
        self.shuffle1 = ShuffleBlock(groups=g)
        self.conv2 = nn.Conv2d(mid_planes, mid_planes, kernel_size=3, stride=stride, padding=1, groups=mid_planes,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(mid_planes)
        self.conv3 = nn.Conv2d(mid_planes, out_planes, kernel_size=1, groups=groups, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 2:
            self.shortcut = nn.Sequential(nn.AvgPool2d(3, stride=2, padding=1))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.shuffle1(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        res = self.shortcut(x)
        out = F.relu(torch.cat([out, res], 1)) if self.stride == 2 else F.relu(out + res)
        return out


class ShuffleNet(nn.Module):
    def __init__(self, num_classes=10):
        super(ShuffleNet, self).__init__()
        cfg = {'out_planes': [200, 400, 800], 'num_blocks': [4, 8, 4], 'groups': 2}

        out_planes = cfg['out_planes']
        num_blocks = cfg['num_blocks']
        groups = cfg['groups']

        self.conv1 = nn.Conv2d(3, 24, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(24)
        self.in_planes = 24
        self.layer1 = self._make_layer(out_planes[0], num_blocks[0], groups)
        self.layer2 = self._make_layer(out_planes[1], num_blocks[1], groups)
        self.layer3 = self._make_layer(out_planes[2], num_blocks[2], groups)
        # self.classification_layer = nn.Linear(out_planes[2], num_classes)

        self.fc1 = nn.Linear(out_planes[2], 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)


    def _make_layer(self, out_planes, num_blocks, groups):
        layers = []
        for i in range(num_blocks):
            stride = 2 if i == 0 else 1
            cat_planes = self.in_planes if i == 0 else 0
            layers.append(Bottleneck(self.in_planes, out_planes - cat_planes, stride=stride, groups=groups))
            self.in_planes = out_planes
        return nn.Sequential(*layers)


    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 4)
        h = out.view(out.size(0), -1)
        # print("[]",feature.shape)
        #out = self.classification_layer(feature)
        # return feature,feature,out

        ch=F.relu(self.fc1(h))
        ch=self.fc2(ch)
        x=self.fc3(ch)
        return ch,h,x




# MOBILENET-------------------------------------------------------------------------------------------------------------------------

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride, norm_layer):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = norm_layer(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = norm_layer(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = norm_layer(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                norm_layer(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)

    def __init__(self, num_classes=10, norm_layer=nn.BatchNorm2d,shrink=1):
        super(MobileNetV2, self).__init__()
        self.norm_layer = norm_layer
        self.cfg = [(1,  16//shrink, 1, 1),
                   (6,  24//shrink, 2, 1),
                   (6,  32//shrink, 3, 2),
                   (6,  64//shrink, 4, 2),
                   (6,  96//shrink, 3, 1),
                   (6, 160//shrink, 3, 2),
                   (6, 320//shrink, 1, 1)]


        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = self.norm_layer(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(self.cfg[-1][1], 1280//shrink, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = self.norm_layer(1280//shrink)

        # print("[1280//shrink] = ",1280//shrink)
        self.fc1 = nn.Linear(1280//shrink, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, num_classes)

        # self.classification_layer = nn.Linear(1280//shrink, num_classes)
        pass

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride, self.norm_layer))
                in_planes = out_planes
        return nn.Sequential(*layers)


    def extract_features(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return out


    def forward(self, x):
        h = self.extract_features(x)
        # feature = self.classification_layer(feature)

        ch=F.relu(self.fc1(h))
        ch=self.fc2(ch)
        x=self.fc3(ch)
        return ch,h,x
       # return out
    pass





# def mobilenetv2(num_classes=10):
#     return MobileNetV2(norm_layer=nn.BatchNorm2d, shrink=2, num_classes=num_classes)

