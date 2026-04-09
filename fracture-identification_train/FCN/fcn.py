import torch.nn as nn
import torch.nn.functional as F

class FCN8s(nn.Module):
    def __init__(self, n_class=1):
        super(FCN8s, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.block4 = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.block5 = nn.Sequential(
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.block6 = nn.Sequential(
            nn.Conv2d(512, 4096, 7, padding=3),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True)
        )
        self.block7 = nn.Sequential(
            nn.Conv2d(4096, 4096, 1),
            nn.BatchNorm2d(4096),
            nn.ReLU(inplace=True)
        )
        self.upscore_pool5 = nn.Sequential(
            nn.Conv2d(4096, n_class, 1),
            nn.ConvTranspose2d(n_class, n_class, 4, 2, padding=1, bias=False)
        )
        self.score_pool4 = nn.ConvTranspose2d(512, n_class, 1, bias=False)
        self.score_pool3 = nn.ConvTranspose2d(256, n_class, 1, bias=False)
        self.upscore_pool4 = nn.ConvTranspose2d(n_class, n_class, 4, 2, padding=1, bias=False)
        self.upscore_pool = nn.ConvTranspose2d(n_class, n_class, 16, 8, padding=4, bias=False)
 
    def forward(self, x):
        x1 = self.block1(x)
        x2 = self.block2(x1)
        x3 = self.block3(x2)
        x4 = self.block4(x3)
        x5 = self.block5(x4)
        x6 = self.block6(x5)
        x7 = self.block7(x6)
        pool5 = self.upscore_pool5(x7)
        pool4 = self.score_pool4(x4)
        pool3 = self.score_pool3(x3)
        pool4 = (pool4 + pool5)
        pool4 = self.upscore_pool4(pool4)
        pool = (pool3 + pool4)
        pool = self.upscore_pool(pool)
        
        return F.sigmoid(pool)