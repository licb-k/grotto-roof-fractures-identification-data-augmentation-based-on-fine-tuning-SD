import os
import shutil
import pandas as pd

# 定义文件夹路径
image_folder = r"D:\lcb\data_con\combinations_all\image"
label_folder = r"D:\lcb\data_con\combinations_all\label"
image1_folder = "D:/lcb/data_con/combinations/Combination_3_1/image"
label1_folder = "D:/lcb/data_con/combinations/Combination_3_1/label"

# 确保目标文件夹存在
os.makedirs(image1_folder, exist_ok=True)
os.makedirs(label1_folder, exist_ok=True)

# 读取 CSV 或 Excel 表格
file_path = r"C:\Users\lcb\Desktop\seg_result\or600_gen400\0104\20250104_180912_组合方式3.xlsx" # 这里改成你的文件路径（可以是 CSV 或 XLSX）
df = pd.read_excel(file_path)  # 如果是 CSV，改为 pd.read_csv(file_path)

# 遍历表格中的文件名
for index, row in df.iterrows():
    image_name = row["Image"]
    label_name = row["Label"]

    # 构建源文件路径
    image_src = os.path.join(image_folder, image_name)
    label_src = os.path.join(label_folder, label_name)

    # 构建目标文件路径
    image_dest = os.path.join(image1_folder, image_name)
    label_dest = os.path.join(label1_folder, label_name)

    # 复制文件
    if os.path.exists(image_src):
        shutil.copy(image_src, image_dest)
    else:
        print(f"警告: {image_src} 不存在！")

    if os.path.exists(label_src):
        shutil.copy(label_src, label_dest)
    else:
        print(f"警告: {label_src} 不存在！")

print("文件复制完成！")