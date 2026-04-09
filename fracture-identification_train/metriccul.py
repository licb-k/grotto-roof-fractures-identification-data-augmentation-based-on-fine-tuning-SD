import cv2
import numpy as np
from skimage.morphology import skeletonize
import os

def adjust_fracture_thickness(image_path, output_path):
    # Step 1: Read the image
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Check if the image is loaded properly
    if image is None:
        raise ValueError(f"Image not found at the path: {image_path}")

    # Step 2: Binarize the image (ensure black=0 and white=255)
    _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Step 3: Invert the image for skeletonization (fractures become 1, background 0)
    inverted_binary = binary_image // 255  # Scale to 0 and 1

    # Step 4: Perform skeletonization
    skeleton = skeletonize(inverted_binary).astype(np.uint8) * 255  # Convert back to 0-255

    # Step 5: Dilate the skeleton to a thickness of 3 pixels
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    thickened = cv2.dilate(skeleton, kernel, iterations=1)

    # Step 6: Save the output image
    cv2.imwrite(output_path, thickened)

def process_all_folders(input_root, output_root):
    # Walk through the input directory structure
    for root, _, files in os.walk(input_root):
        # Compute relative path from input root
        relative_path = os.path.relpath(root, input_root)

        # Create corresponding output directory
        output_folder = os.path.join(output_root, relative_path)
        os.makedirs(output_folder, exist_ok=True)

        # Process each file in the current directory
        for file_name in files:
            input_path = os.path.join(root, file_name)
            output_path = os.path.join(output_folder, file_name)

            # Process only image files
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff')):
                try:
                    adjust_fracture_thickness(input_path, output_path)
                    print(f"Processed and saved: {output_path}")
                except ValueError as e:
                    print(e)

def preprocess_image(image):
    return (image > 0).astype(np.uint8)

def calculate_metrics_for_folders(test_label_root, processed_data_root):
    for folder_name in os.listdir(processed_data_root):
        folder_path = os.path.join(processed_data_root, folder_name)
        if not os.path.isdir(folder_path):
            continue

        total_tp = 0
        total_fp = 0
        total_fn = 0
        total_intersection = 0
        total_union = 0
        total_label = 0
        total_pred = 0
        file_count = 0

        for file_name in os.listdir(folder_path):
            processed_image_path = os.path.join(folder_path, file_name)
            label_image_path = os.path.join(test_label_root, file_name)

            if not os.path.exists(label_image_path):
                print(f"Label image missing for: {file_name}")
                continue

            label_image = cv2.imread(label_image_path, cv2.IMREAD_GRAYSCALE)
            processed_image = cv2.imread(processed_image_path, cv2.IMREAD_GRAYSCALE)

            if label_image is None or processed_image is None:
                print(f"Error loading images for: {file_name}")
                continue

            if label_image.shape != processed_image.shape:
                print(f"Image shapes do not match for: {file_name}")
                continue

            # Preprocess images
            label_binary = preprocess_image(label_image)
            processed_binary = preprocess_image(processed_image)

            # Calculate values for metrics
            intersection = np.logical_and(label_binary, processed_binary)
            union = np.logical_or(label_binary, processed_binary)

            true_positives = np.sum(intersection)
            false_positives = np.sum(np.logical_and(label_binary == 0, processed_binary == 1))
            false_negatives = np.sum(np.logical_and(label_binary == 1, processed_binary == 0))

            total_tp += true_positives
            total_fp += false_positives
            total_fn += false_negatives
            total_intersection += np.sum(intersection)
            total_union += np.sum(union)
            total_label += np.sum(label_binary)
            total_pred += np.sum(processed_binary)

            file_count += 1

        if file_count > 0:
            smooth = 1e-5
            mean_precision = total_tp / (total_tp + total_fp + smooth)
            mean_recall = total_tp / (total_tp + total_fn + smooth)
            mean_f1 = (2 * mean_precision * mean_recall + smooth) / (mean_precision + mean_recall + smooth)
            mean_dice = (2 * total_intersection + smooth) / (total_label + total_pred + smooth)
            mean_iou = (total_intersection + smooth) / (total_union + smooth)

            print(f"Metrics for folder {folder_name}: ")
            print(f"  Mean Dice: {mean_dice * 100:.2f}%")
            print(f"  Mean IoU: {mean_iou * 100:.2f}%")
            print(f"  Mean Precision: {mean_precision * 100:.2f}%")
            print(f"  Mean Recall: {mean_recall * 100:.2f}%")
            print(f"  Mean F1 Score: {mean_f1 * 100:.2f}%")

# Provide the root input and output folder paths
input_root = r"C:\Users\lcb\Desktop\data_con\test_kejian"
output_root = r"C:\Users\lcb\Desktop\4.30test\test"
test_label_root = r"C:\Users\lcb\Desktop\4.30test\test\label"
# Process all folders
#process_all_folders(input_root, output_root)

# Calculate metrics for all processed folders
calculate_metrics_for_folders(test_label_root, output_root)