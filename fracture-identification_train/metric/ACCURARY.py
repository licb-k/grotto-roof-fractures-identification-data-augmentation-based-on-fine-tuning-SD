import cv2
import numpy as np
import os

def preprocess_image(image):
    # 将图像二值化，大于0.5的为1，小于等于0.5的为0
    _, binary_image = cv2.threshold(image, 0.5, 1, cv2.THRESH_BINARY)
    return binary_image.astype(np.uint8)

def Accuracy(label_path, pred_path):
    label_files = os.listdir(label_path)
    pred_files = os.listdir(pred_path)

    assert sorted(label_files) == sorted(pred_files)

    total_pixels = 0
    correct_pixels = 0

    for file_name in label_files:
        label_image = cv2.imread(os.path.join(label_path, file_name), cv2.IMREAD_GRAYSCALE)
        pred_image = cv2.imread(os.path.join(pred_path, file_name), cv2.IMREAD_GRAYSCALE)

        assert label_image.shape == pred_image.shape == (640, 640)

        # 预处理图像
        label_binary = preprocess_image(label_image)
        pred_binary = preprocess_image(pred_image)

        total_pixels += label_binary.size
        correct_pixels += np.sum(label_binary == pred_binary)

    accuracy = correct_pixels / total_pixels * 100

    return f"{accuracy:.2f}%"
