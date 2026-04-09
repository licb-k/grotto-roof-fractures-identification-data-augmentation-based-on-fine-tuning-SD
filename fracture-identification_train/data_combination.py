import os
import random
import shutil
import pandas as pd
from datetime import datetime

# 文件夹路径
image_origin_folder = r"D:\lcb\data_con\oringin_data\image"
label_origin_folder = r"D:\lcb\data_con\oringin_data\label"
image_generate_folder = r"D:\lcb\data_con\generates_data2\image"
label_generate_folder = r"D:\lcb\data_con\generates_data2\label"
image_output_folder = r"C:\Users\lcb\Desktop\4.30test\data6\image"
label_output_folder = r"C:\Users\lcb\Desktop\4.30test\data6\label"
output_excel_folder = r"C:\Users\lcb\Desktop\4.30test\data6\output_excel1"

# 获取原始图像和标签、生成图像和标签的文件列表
image_origin_files = os.listdir(image_origin_folder)
label_origin_files = os.listdir(label_origin_folder)
image_generate_files = os.listdir(image_generate_folder)
label_generate_files = os.listdir(label_generate_folder)

# 确保文件夹路径存在
os.makedirs(image_output_folder, exist_ok=True)
os.makedirs(label_output_folder, exist_ok=True)
os.makedirs(output_excel_folder, exist_ok=True)  # 创建Excel保存文件夹
# 清空文件夹
def clear_folder(folder_path):
    for file in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

# 将组合的文件名记录到 Excel 文件中
def write_to_excel(data, combination_type):
    # 获取当前时间并格式化为字符串
    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建文件名，包含当前时间和组合方式
    excel_filename = f"{current_time}_组合方式{combination_type}.xlsx"
    
    # 完整的保存路径（指定文件夹）
    excel_filepath = os.path.join(output_excel_folder, excel_filename)
    
    # 创建DataFrame并写入Excel
    df = pd.DataFrame(data, columns=["Image", "Label"])
    df.to_excel(excel_filepath, index=False)

# 随机选择图像和标签，并确保它们的文件名一一对应
def select_images_and_labels(num_origin, num_generate):
    # 从原始图像和标签文件中选取，确保图像和标签文件名一致
    selected_image_origin = random.sample(image_origin_files, num_origin)
    selected_label_origin = [img.replace(".jpg", ".png") for img in selected_image_origin]  # 假设标签的扩展名是 .png

    # 选取生成图像和标签，确保图像和标签文件名一致
    selected_image_generate = random.sample(image_generate_files, num_generate)
    selected_label_generate = [img.replace(".jpg", ".png") for img in selected_image_generate]  # 假设标签的扩展名是 .png

    return selected_image_origin, selected_label_origin, selected_image_generate, selected_label_generate

# 拷贝文件到输出文件夹
def copy_files_to_output(selected_image_origin, selected_label_origin, selected_image_generate, selected_label_generate):
    # 创建组合文件名列表
    combined_image_files = []
    combined_label_files = []

    # 拷贝原始图像和标签
    for image_file, label_file in zip(selected_image_origin, selected_label_origin):
        shutil.copy(os.path.join(image_origin_folder, image_file), os.path.join(image_output_folder, image_file))
        shutil.copy(os.path.join(label_origin_folder, label_file), os.path.join(label_output_folder, label_file))
        combined_image_files.append(image_file)
        combined_label_files.append(label_file)
    
    # 拷贝生成图像和标签
    for image_file, label_file in zip(selected_image_generate, selected_label_generate):
        shutil.copy(os.path.join(image_generate_folder, image_file), os.path.join(image_output_folder, image_file))
        shutil.copy(os.path.join(label_generate_folder, label_file), os.path.join(label_output_folder, label_file))
        combined_image_files.append(image_file)
        combined_label_files.append(label_file)

    return combined_image_files, combined_label_files

# 主函数
def main():
    # 定义组合方式
    combinations = [
        (1000, 0),   # 组合方式1
        (800, 200),  # 组合方式2
        (600, 400),  # 组合方式3
        (400, 600),  # 组合方式4
        (200, 800),  # 组合方式5
        (0, 1000),   # 组合方式6
    ]
    
    all_combinations = []

    # 清空文件夹
    clear_folder(image_output_folder)
    clear_folder(label_output_folder)

    # 随机选择一个组合方式
    combination_type = 6  # 1 到 6 之间的随机数
    num_origin, num_generate = combinations[combination_type - 1]

    # 选择图像和标签
    selected_image_origin, selected_label_origin, selected_image_generate, selected_label_generate = select_images_and_labels(num_origin, num_generate)

    # 拷贝到输出文件夹
    combined_image_files, combined_label_files = copy_files_to_output(selected_image_origin, selected_label_origin, selected_image_generate, selected_label_generate)

    # 将组合的图像和标签记录到 Excel 文件中
    for image_file, label_file in zip(combined_image_files, combined_label_files):
        all_combinations.append([image_file, label_file])
    
    write_to_excel(all_combinations, combination_type)

    print(f"已完成组合: {num_origin} 原始图像 和 {num_generate} 生成图像, 文件已保存为 Excel。")

if __name__ == "__main__":
    main()