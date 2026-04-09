import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import cv2 as cv
import torch.utils.data as Data

bn_momentum = 0.1  # BN层的momentum
torch.cuda.manual_seed(1)  # 设置随机种子

# 编码器
class Encoder(nn.Module):
    def __init__(self, input_channels):
        super(Encoder, self).__init__()
        vgg16 = models.vgg16_bn(weights=None)
        #state_dict = torch.load('vgg16_bn-6c64b313.pth')
        #vgg16.load_state_dict(state_dict)
        self.enco1 = vgg16.features[0:6]
        self.enco2 = vgg16.features[7:13]
        self.enco3 = vgg16.features[14:23]
        self.enco4 = vgg16.features[24:33]
        self.enco5 = vgg16.features[34:43]

    def forward(self, x):
        id = []

        x = self.enco1(x)
        x, id1 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)  # 保留最大值的位置
        id.append(id1)
        x = self.enco2(x)
        x, id2 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        id.append(id2)
        x = self.enco3(x)
        x, id3 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        id.append(id3)
        x = self.enco4(x)
        x, id4 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        id.append(id4)
        x = self.enco5(x)
        x, id5 = F.max_pool2d(x, kernel_size=2, stride=2, return_indices=True)
        id.append(id5)

        return x, id


# 编码器+解码器
class SegNet(nn.Module):
    def __init__(self, input_channels=3, output_channels=1):
        super(SegNet, self).__init__()

        self.encoder = Encoder(input_channels)

        self.deco1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU()
        )
        self.deco2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(512, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU()
        )
        self.deco3 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU()
        )
        self.deco4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU()
        )
        self.deco5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64, momentum=bn_momentum),
            nn.ReLU(),
            nn.Conv2d(64, output_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x):
        x, id = self.encoder(x)

        x = F.max_unpool2d(x, id[4], kernel_size=2, stride=2)
        x = self.deco1(x)
        x = F.max_unpool2d(x, id[3], kernel_size=2, stride=2)
        x = self.deco2(x)
        x = F.max_unpool2d(x, id[2], kernel_size=2, stride=2)
        x = self.deco3(x)
        x = F.max_unpool2d(x, id[1], kernel_size=2, stride=2)
        x = self.deco4(x)
        x = F.max_unpool2d(x, id[0], kernel_size=2, stride=2)
        x = self.deco5(x)

        return F.sigmoid(x)


class MyDataset(Data.Dataset):
    def __init__(self, txt_path):
        super(MyDataset, self).__init__()

        paths = open(txt_path, "r")

        image_label = []
        for line in paths:
            line.rstrip("\n")
            line.lstrip("\n")
            path = line.split()
            image_label.append((path[0], path[1]))

        self.image_label = image_label

    def __getitem__(self, item):
        image, label = self.image_label[item]

        image = cv.imread(image)
        image = cv.resize(image, (224, 224))
        image = image/255.0  # 归一化输入
        image = torch.Tensor(image)
        image = image.permute(2, 0, 1)  # 将图片的维度转换成网络输入的维度（channel, width, height）

        label = cv.imread(label, 0)
        label = cv.resize(label, (224, 224))
        label = torch.Tensor(label)

        return image, label

    def __len__(self):
        return  len(self.image_label)