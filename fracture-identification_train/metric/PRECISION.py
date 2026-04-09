import os
import cv2
import numpy as np

def calculate_precision_recall(label, pred, threshold=0.5):
    # 二值化处理
    label_binary = (label > threshold).astype(int)
    pred_binary = (pred > threshold).astype(int)

    # 计算混淆矩阵
    true_positives = np.sum(np.logical_and(label_binary == 1, pred_binary == 1))
    false_positives = np.sum(np.logical_and(label_binary == 0, pred_binary == 1))

    # 计算查准率
    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0.0

    return precision

def Precision(label_folder, pred_folder, threshold=0.5):
    label_files = os.listdir(label_folder)
    pred_files = os.listdir(pred_folder)

    assert sorted(label_files) == sorted(pred_files)

    total_precision = 0

    for file_name in label_files:
        # 读取图像
        label_image = cv2.imread(os.path.join(label_folder, file_name), cv2.IMREAD_GRAYSCALE)
        pred_image = cv2.imread(os.path.join(pred_folder, file_name), cv2.IMREAD_GRAYSCALE)

        # 计算查准率和查全率
        precision = calculate_precision_recall(label_image, pred_image, threshold)

        # 累加总体的查准率和查全率
        total_precision += precision

    # 计算平均查准率和平均查全率
    average_precision = total_precision / len(label_files) * 100

    return f"{average_precision:.2f}%"
