import cv2
import numpy as np
import os

def preprocess_image(image):
    # 将图像二值化，大于0.5的为1，小于等于0.5的为0
    _, binary_image = cv2.threshold(image, 0.5, 1, cv2.THRESH_BINARY)
    return binary_image.astype(np.uint8)

def calculate_iou(label, pred):
    intersection = np.logical_and(label, pred)
    union = np.logical_or(label, pred)

    # 处理分母为零的情况
    if np.sum(union) == 0:
        iou = 1
    else:
        iou = np.sum(intersection) / np.sum(union)
    return iou

def IoU(label_path, pred_path):
    label_files = os.listdir(label_path)
    pred_files = os.listdir(pred_path)

    assert sorted(label_files) == sorted(pred_files)

    total_iou = 0

    for file_name in label_files:
        label_image = cv2.imread(os.path.join(label_path, file_name), cv2.IMREAD_GRAYSCALE)
        pred_image = cv2.imread(os.path.join(pred_path, file_name), cv2.IMREAD_GRAYSCALE)

        assert label_image.shape == pred_image.shape == (640, 640)

        # 预处理图像
        label_binary = preprocess_image(label_image)
        pred_binary = preprocess_image(pred_image)

        # 计算IoU并累加
        total_iou += calculate_iou(label_binary, pred_binary)

    # 计算平均IoU
    miou = total_iou / len(label_files) * 100

    return f"{miou:.2f}%"
