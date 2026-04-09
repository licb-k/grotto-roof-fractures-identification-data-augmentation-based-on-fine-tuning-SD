import os
import cv2
import numpy as np
from sklearn.metrics import accuracy_score

def preprocess_image(image):
    _, binary_image = cv2.threshold(image, 0.5, 1, cv2.THRESH_BINARY)
    return binary_image.astype(np.uint8)

def calculate_mpa(y_true, y_pred):
    y_true_flat = y_true.flatten()
    y_pred_flat = y_pred.flatten()
    pixel_accuracy = accuracy_score(y_true_flat, y_pred_flat)
    return pixel_accuracy

def MPA(label_folder, pred_folder, num_classes=1):
    classwise_accuracy = []

    label_files = os.listdir(label_folder)
    pred_files = os.listdir(pred_folder)

    assert sorted(label_files) == sorted(pred_files)

    for file_name in label_files:
        label_image = cv2.imread(os.path.join(label_folder, file_name), cv2.IMREAD_GRAYSCALE)
        pred_image = cv2.imread(os.path.join(pred_folder, file_name), cv2.IMREAD_GRAYSCALE)

        assert label_image.shape == pred_image.shape == (640, 640)

        label_binary = preprocess_image(label_image)
        pred_binary = preprocess_image(pred_image)

        for class_label in range(num_classes):
            class_true = (label_binary == class_label).astype(int)  # 修改此处
            class_pred = (pred_binary == class_label).astype(int)  # 修改此处

            class_accuracy = calculate_mpa(class_true, class_pred)
            classwise_accuracy.append(class_accuracy)

        mean_pixel_accuracy = np.mean(classwise_accuracy)

    return f'{mean_pixel_accuracy * 100:.2f}%'
