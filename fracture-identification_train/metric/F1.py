import cv2
import numpy as np
import os
from sklearn.metrics import precision_score, recall_score

def preprocess_image(image):
    # 将图像二值化，大于0.5的为1，小于等于0.5的为0
    _, binary_image = cv2.threshold(image, 0.5, 1, cv2.THRESH_BINARY)
    return binary_image.astype(np.uint8)

def calculate_f1(label, pred, smooth=1e-5):
    label_flat = label.flatten()
    pred_flat = pred.flatten()
    
    precision = precision_score(label_flat, pred_flat, zero_division=1)
    recall = recall_score(label_flat, pred_flat, zero_division=1)
    
    f1 = (2 * precision * recall) / (precision + recall + smooth)
    return f1

def F1(label_path, pred_path, smooth=1e-5):
    label_files = os.listdir(label_path)
    pred_files = os.listdir(pred_path)

    assert sorted(label_files) == sorted(pred_files)

    total_f1 = 0

    for file_name in label_files:
        label_image = cv2.imread(os.path.join(label_path, file_name), cv2.IMREAD_GRAYSCALE)
        pred_image = cv2.imread(os.path.join(pred_path, file_name), cv2.IMREAD_GRAYSCALE)

        assert label_image.shape == pred_image.shape == (640, 640)

        # 预处理图像
        label_binary = preprocess_image(label_image)
        pred_binary = preprocess_image(pred_image)

        # 计算F1并累加
        total_f1 += calculate_f1(label_binary, pred_binary, smooth)

    # 计算平均F1
    mean_f1 = total_f1 / len(label_files)

    return mean_f1