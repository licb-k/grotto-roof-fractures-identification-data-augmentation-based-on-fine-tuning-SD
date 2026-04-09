import torch
from torch.utils.data import Dataset

import cv2
import numpy as np
import os

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:  #np.random.random() 是一个NumPy函数，用于生成一个0到1之间的随机浮点数
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.array(hue_shift).astype(np.uint8)
        h += hue_shift #色相（hue）偏移的范围，范围从-180到180
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)  #饱和度（saturation）的随机调整
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift) #亮度（value）的调整
        image = cv2.merge((h, s, v))
        #image = cv2.merge((s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)

    return image

def randomShiftScaleRotate(image, label,
                           shift_limit=(-0.0, 0.0),
                           scale_limit=(-0.0, 0.0),
                           rotate_limit=(-30.0, 30.0), 
                           aspect_limit=(-0.0, 0.0),
                           borderMode=cv2.BORDER_CONSTANT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape

        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(
                                        0, 0,
                                        0,))
        label = cv2.warpPerspective(label, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                   borderValue=(
                                       0, 0,
                                       0,))

    return image, label

def randomHorizontalFlip(image, label, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        label = cv2.flip(label, 1)

    return image, label

def randomVerticleFlip(image, label, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        label = cv2.flip(label, 0)

    return image, label

def randomRotate90(image, label, u=0.5):
    if np.random.random() < u:
        image=np.rot90(image)
        label=np.rot90(label)

    return image, label

def default_loader(id, root):
    image_path = root
    # 根据image_path生成label_path
    label_path = root.replace('image', 'label')
    image = cv2.imread(os.path.join(image_path,'{}.png').format(id))
    label = cv2.imread(os.path.join(label_path,'{}.png').format(id), cv2.IMREAD_GRAYSCALE)

    image = randomHueSaturationValue(image,
                                   hue_shift_limit=(-30, 30),
                                   sat_shift_limit=(-5, 5),
                                   val_shift_limit=(-15, 15))
   
    image, label = randomShiftScaleRotate(image, label,
                                       shift_limit=(-0.1, 0.1),
                                       scale_limit=(-0.1, 0.1),
                                       aspect_limit=(-0.1, 0.1),
                                       rotate_limit=(-30.0, 30.0))
   
    image, label = randomHorizontalFlip(image, label)
    image, label = randomVerticleFlip(image, label)
    image, label = randomRotate90(image, label)
    
    label = np.expand_dims(label, axis=2) #增加到三个维度
    #将通道维度（原始图像是 H x W x C，其中 H 为高度，W 为宽度，C 为通道数）重新排列为 C x H x W 的形式。
    #这是为了将图像数据的通道维度与神经网络的输入匹配。
    #* 3.2 - 1.6 对图像进行进一步的线性变换，将像素值重新缩放到 -1.6 到 1.6 之间
    image = np.array(image, np.float32).transpose(2,0,1)/255.0 * 3.2 - 1.6#
    label = np.array(label, np.float32).transpose(2,0,1)/255.0
    label[label>=0.5] = 1
    label[label<=0.5] = 0
    #label = abs(label-1)
    return image, label

class ImageFolder(Dataset):

    def __init__(self, trainlist, root , transform=True):
        self.ids = trainlist
        self.loader = default_loader
        self.root = root
        self.transform = transform

    def __getitem__(self, index):
        
        id = self.ids[index]
        image, label = self.loader(id, self.root)
        #label = label.copy()
        image = torch.Tensor(image)
        label = torch.Tensor(label)
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        return image, label

    def __len__(self):
        return len(self.ids)