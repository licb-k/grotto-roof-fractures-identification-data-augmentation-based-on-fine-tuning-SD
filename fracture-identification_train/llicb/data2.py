import os
import cv2
import numpy as np
import torch
import re
from torch.utils.data import Dataset

def save_image(image, label, stage, id, output_dir):
    image_filename = os.path.join(output_dir, f'{id}_{stage}.png')
    label_filename = os.path.join(output_dir, f'{id}_{stage}_label.png')
    cv2.imwrite(image_filename, image)
    cv2.imwrite(label_filename, label)

def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),
                             sat_shift_limit=(-255, 255),
                             val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1]+1)
        hue_shift = np.array(hue_shift).astype(np.uint8)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
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
                                    borderValue=(0, 0, 0,))
        label = cv2.warpPerspective(label, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,
                                    borderValue=(0, 0, 0,))
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
        image = np.rot90(image)
        label = np.rot90(label)
    return image, label

def default_loader(id, root, output_dir):
    image_path = root
    label_path = root.replace('image', 'label')
    image = cv2.imread(os.path.join(image_path, '{}.png'.format(id)))
    label = cv2.imread(os.path.join(label_path, '{}.png'.format(id)), cv2.IMREAD_GRAYSCALE)

    save_image(image, label, 'original', id, output_dir)

    image = randomHueSaturationValue(image,
                                     hue_shift_limit=(-30, 30),
                                     sat_shift_limit=(-5, 5),
                                     val_shift_limit=(-15, 15))
    save_image(image, label, 'hue_saturation_value', id, output_dir)

    image, label = randomShiftScaleRotate(image, label,
                                          shift_limit=(-0.1, 0.1),
                                          scale_limit=(-0.1, 0.1),
                                          aspect_limit=(-0.1, 0.1),
                                          rotate_limit=(-0.0, 0.0))
    save_image(image, label, 'shift_scale_rotate', id, output_dir)

    image, label = randomHorizontalFlip(image, label)
    save_image(image, label, 'horizontal_flip', id, output_dir)

    image, label = randomVerticleFlip(image, label)
    save_image(image, label, 'vertical_flip', id, output_dir)

    image, label = randomRotate90(image, label)
    save_image(image, label, 'rotate90', id, output_dir)
    
    label = np.expand_dims(label, axis=2)
    image = np.array(image, np.float32).transpose(2, 0, 1) / 255.0 * 3.2 - 1.6
    label = np.array(label, np.float32).transpose(2, 0, 1) / 255.0
    label[label >= 0.5] = 1
    label[label <= 0.5] = 0
    return image, label

class ImageFolder(Dataset):

    def __init__(self, trainlist, root, output_dir, transform=None):
        self.ids = trainlist
        self.loader = default_loader
        self.root = root
        self.transform = transform
        self.output_dir = output_dir

    def __getitem__(self, index):
        id = self.ids[index]
        image, label = self.loader(id, self.root, self.output_dir)
        image = torch.Tensor(image)
        label = torch.Tensor(label)
        if self.transform:
            image = self.transform(image)
            label = self.transform(label)
        return image, label

    def __len__(self):
        return len(self.ids)
    
def main():
    # 图像和标签的根目录
    root = r"C:\Users\lcb\Desktop\1111222\grotto-roof-dataset\pred\image"
    # 保存增强后的图像的目录
    output_dir = r"C:\Users\lcb\Desktop\1"
    # 创建保存目录
    os.makedirs(output_dir, exist_ok=True)
    # 示例数据列表
    folderlist = [filename for filename in os.listdir(root) if filename.endswith('.png')]
    trainlist = [re.match(r'\d+', item).group() for item in folderlist]

    # 创建数据集
    train_dataset = ImageFolder(trainlist, root, output_dir)

    # 处理并保存所有图像
    for i in range(len(train_dataset)):
        image, label = train_dataset[i]
        print(f"Processed image {i+1}/{len(train_dataset)}")

if __name__ == "__main__":
    main()