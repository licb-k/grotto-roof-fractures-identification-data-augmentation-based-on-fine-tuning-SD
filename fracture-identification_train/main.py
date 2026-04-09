import sys
import tkinter as tk
from tkinter import filedialog, messagebox
import threading
import torch
import cv2
import os
import re
import numpy as np
import matplotlib.pyplot as plt
from openpyxl import Workbook
from torchvision import transforms
from tkinter import font

from time import time
from datetime import datetime

from llicb.framework import MyFrame
from llicb.framework1 import MyFrame1
from llicb.data1 import ImageFolder
from llicb.predict import TTAFrame

from torch.utils.data import DataLoader
#from D_LinkNet.dlinknet import DinkNet34_less_pool, DinkNet34, DinkNet50, DinkNet101, LinkNet34
from FCN.fcn import FCN8s
#from PSPNet.pspnet import Pspnet
from SegNet.SegNet import SegNet
#from Deeplabv_3.deeplabv3 import deeplabv3_resnet50, deeplabv3_resnet101
#from ThreeUNet.dinknet_attention_2 import DinkNet34_Attention
#from ThreeUNet.vgguNet18_attention_2 import vgguNet18_Attention
from UNet.UNet.models.UNet import UNet
from UNet.UNet_2Plus.models.UNet_2Plus import UNet_2Plus

from loss.dice_bce import DiceLoss, BceLoss, Dice_BceLoss
from loss.focal import FocalLoss
from loss.iou import IoULoss
from loss.tversky import TverskyLoss
from loss.generalized_dice import GeneralizedDiceLoss
from loss.dice_focal import DiceFocalLoss

from metric.ACCURARY import Accuracy
from metric.DICE import Dice
from metric.F1 import F1
from metric.IOU import IoU
from metric.MPA import MPA
from metric.PRECISION import Precision
from metric.RECALL import Recall

from PIL import Image, ImageTk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from tqdm import tqdm

try:
    import pyi_splash
    pyi_splash.close()
except ImportError:
    pass

class SemanticSegmentationGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("语义分割GUI")
        self.root.geometry("2000x1300")
        self.root.state('zoomed')

        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=2)
        self.root.grid_columnconfigure(1, weight=2)
        self.root.grid_columnconfigure(2, weight=2)

        # 左边框架
        self.left_frame = tk.Frame(root, bd=1, relief=tk.RIDGE)
        self.left_frame.grid(row=0, column=0, sticky="nsew")
        self.left_frame.grid_propagate(False)  

        # 中间框架，分为上下两部分
        self.mid_frame = tk.Frame(self.root)
        self.mid_frame.grid(row=0, column=1, sticky="nsew")

        # 右边框架，分为上下两部分
        self.right_frame = tk.Frame(self.root)
        self.right_frame.grid(row=0, column=2, sticky="nsew")
       
        # 创建中间 Scrollbar
        self.scrollbar_mid = tk.Scrollbar(self.mid_frame, orient=tk.VERTICAL)
        self.scrollbar_mid.pack(side=tk.RIGHT, fill=tk.Y, anchor=tk.E)
        
        # 中间分为上下两部分
        self.upper_mid_frame = tk.Frame(self.mid_frame)
        self.upper_mid_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.lower_mid_frame = tk.Frame(self.mid_frame)
        self.lower_mid_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # 添加中间上部 Canvas 的标题
        self.upper_mid_label = tk.Label(self.upper_mid_frame, text="Image", height=1, font=("Microsoft YaHei", 20, "bold"))
        self.upper_mid_label.pack(side=tk.TOP, fill=tk.X)       
        # 创建中间上部 Canvas
        self.upper_mid_canvas = tk.Canvas(self.upper_mid_frame, bg="white")
        self.upper_mid_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # 添加中间下部 Canvas 的标题
        self.lower_mid_label = tk.Label(self.lower_mid_frame, text="Label", height=1, font=("Microsoft YaHei", 20, "bold"))
        self.lower_mid_label.pack(side=tk.TOP, fill=tk.X)

        # 创建中间下部 Canvas
        self.lower_mid_canvas = tk.Canvas(self.lower_mid_frame, bg="white")
        self.lower_mid_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.upper_mid_canvas.configure(yscrollcommand=self.scrollbar_mid.set)
        self.lower_mid_canvas.configure(yscrollcommand=self.scrollbar_mid.set)
        
        # 右边分为上下两部分
        self.upper_right_frame = tk.Frame(self.right_frame)
        self.upper_right_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.lower_right_frame = tk.Frame(self.right_frame)
        self.lower_right_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        # 右边添加上部 Canvas 的标题
        self.upper_right_label = tk.Label(self.upper_right_frame, text="Processing", height=1, font=("Microsoft YaHei", 20, "bold"))
        self.upper_right_label.pack(side=tk.TOP, fill=tk.X)       
        # 右边创建上部 Canvas
        self.upper_right_canvas = tk.Canvas(self.upper_right_frame, bg="white")
        self.upper_right_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)  
        self.upper_right_canvas.pack_propagate(False)        
        # 右边添加下部 Canvas 的标题
        self.lower_label = tk.Label(self.lower_right_frame, text="Prediction", height=1, font=("Microsoft YaHei", 20, "bold"))
        self.lower_label.pack(side=tk.TOP, fill=tk.X)
        # 右边创建下部 Canvas
        self.lower_right_canvas = tk.Canvas(self.lower_right_frame, bg="white")
        self.lower_right_canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.lower_right_canvas.configure(yscrollcommand=self.scrollbar_mid.set)
        self.scrollbar_mid.config(command=self._sync_scroll)

        self.upper_mid_canvas.bind('<MouseWheel>', self._on_mouse_wheel)
        self.lower_mid_canvas.bind('<MouseWheel>', self._on_mouse_wheel)
        self.lower_right_canvas.bind('<MouseWheel>', self._on_mouse_wheel)
        
        self.left_frame.grid_rowconfigure(0, weight=4)
        self.left_frame.grid_rowconfigure(1, weight=2)
        self.left_frame.grid_rowconfigure(2, weight=1)
        self.left_frame.grid_rowconfigure(3, weight=2)
        self.left_frame.grid_rowconfigure(4, weight=2)
        self.left_frame.grid_rowconfigure(5, weight=1)
        self.left_frame.grid_rowconfigure(6, weight=2)
        self.left_frame.grid_rowconfigure(7, weight=2)
        self.left_frame.grid_rowconfigure(8, weight=2)
        self.left_frame.grid_rowconfigure(9, weight=2)
        self.left_frame.grid_rowconfigure(10, weight=2)
        self.left_frame.grid_rowconfigure(11, weight=2)
        self.left_frame.grid_rowconfigure(12, weight=2)
        self.left_frame.grid_rowconfigure(13, weight=2)
        self.left_frame.grid_rowconfigure(14, weight=2)
        self.left_frame.grid_rowconfigure(15, weight=2)
        self.left_frame.grid_rowconfigure(16, weight=1)
        self.left_frame.grid_rowconfigure(17, weight=2)
        self.left_frame.grid_rowconfigure(18, weight=2)
        self.left_frame.grid_rowconfigure(19, weight=2) 
        self.left_frame.grid_rowconfigure(20, weight=2)   
        self.left_frame.grid_rowconfigure(21, weight=2) 
        self.left_frame.grid_rowconfigure(22, weight=2) 
        self.left_frame.grid_rowconfigure(23, weight=2) 
        self.left_frame.grid_rowconfigure(24, weight=2) 
        self.left_frame.grid_rowconfigure(25, weight=2) 
        self.left_frame.grid_rowconfigure(26, weight=2)       
        self.left_frame.grid_columnconfigure(0, weight=1)
        self.left_frame.grid_columnconfigure(1, weight=1)
        # 训练集文件夹
        self.train_data_label = tk.Label(self.left_frame, text="训练集文件夹:", font=("Microsoft YaHei", 14, "bold"))
        self.train_data_label.grid(row=1, column=0, padx=10, pady=3, sticky="nsew")
        self.train_data_button = tk.Button(self.left_frame, text="浏览", command=self.select_train_data, font=("Microsoft YaHei", 12, "bold"))
        self.train_data_button.grid(row=1, column=1, padx=10, pady=3, sticky="ew")
        # train文件夹目录
        self.train_data_label_selected = tk.Label(self.left_frame, text="", font=("Times New Roman", 8, "normal"), wraplength=330)
        self.train_data_label_selected.grid(row=2, columnspan=2, padx=10, pady=1, sticky="ew")
        # 选择模型
        self.model_label = tk.Label(self.left_frame, text="选择模型:", font=("Microsoft YaHei", 14, "bold"))
        self.model_label.grid(row=3, column=0, padx=10, pady=3, sticky="ew")
        self.model_var = tk.StringVar(root)
        self.model_var.set("UNet")  # 默认值
        self.model_menu = tk.OptionMenu(self.left_frame, self.model_var, "DinkNet34_less_pool", "DinkNet34",
                                        "DinkNet50", "DinkNet101", "LinkNet34", "FCN8s", "Pspnet", "SegNet",
                                        "DinkNet34_Attention", "vgguNet18_Attention", "UNet", "UNet_2Plus", "SegNet", "deeplabv3_resnet50", "deeplabv3_resnet101")
        self.model_menu.grid(row=3, column=1, padx=10, pady=3, sticky="ew")
        self.new_font = font.Font(family="Times New Roman", size=12, weight="bold")#左边字体
        self.model_menu.config(font=self.new_font)
        self.model_menus = {
            #"DinkNet34_less_pool": DinkNet34_less_pool,
            #"DinkNet34": DinkNet34,
            #"DinkNet50": DinkNet50,
            #"DinkNet101": DinkNet101,
            #"LinkNet34": LinkNet34,
            "FCN8s": FCN8s,
            #"Pspnet": Pspnet,
            #"DinkNet34_Attention": DinkNet34_Attention,
            #"vgguNet18_Attention": vgguNet18_Attention,
            "UNet": UNet,
            #"UNet_2Plus": UNet_2Plus,
            "SegNet": SegNet,
            #"deeplabv3_resnet50": deeplabv3_resnet50,
            #"deeplabv3_resnet101": deeplabv3_resnet101 
        }
        # 选择预训练权重
        self.weights_path = None
        self.weights_label = tk.Label(self.left_frame, text="选择模型权重:", font=("Microsoft YaHei", 14, "bold"))
        self.weights_label.grid(row=4, column=0, padx=10, pady=3, sticky="ew")
        self.weights_button = tk.Button(self.left_frame, text="浏览", command=self.select_weights, font=("Microsoft YaHei", 12, "bold"))
        self.weights_button.grid(row=4, column=1, padx=10, pady=3, sticky="ew")
        # weights文件夹目录
        self.weights_data_label_selected = tk.Label(self.left_frame, text="", font=("Times New Roman", 8, "normal"), wraplength=330)
        self.weights_data_label_selected.grid(row=5, columnspan=2, padx=10, pady=1, sticky="ew")       
        # 选择损失函数
        self.loss_label = tk.Label(self.left_frame, text="选择损失函数:", font=("Microsoft YaHei", 14, "bold"))
        self.loss_label.grid(row=6, column=0, padx=10, pady=3, sticky="ew")
        self.loss_var = tk.StringVar(root)
        self.loss_var.set("Dice_BceLoss")  # 默认值
        self.loss_menu = tk.OptionMenu(self.left_frame, self.loss_var, "CrossEntropyLoss", "FocalLoss",
                                       "DiceLoss", "BceLoss", "IoULoss", "TverskyLoss", "GeneralizedDiceLoss", "Dice_BceLoss", "DiceFocalLoss")
        self.loss_menu.grid(row=6, column=1, padx=10, pady=3, sticky="ew")
        self.loss_menu.config(font=self.new_font)
        self.loss_functions = {
            "CrossEntropyLoss": torch.nn.CrossEntropyLoss,
            "FocalLoss": FocalLoss,
            "DiceLoss": DiceLoss,
            "BceLoss": BceLoss,
            "IoULoss": IoULoss,
            "TverskyLoss": TverskyLoss,
            "GeneralizedDiceLoss": GeneralizedDiceLoss,
            "Dice_BceLoss": Dice_BceLoss,
            "FocalDiceLoss": DiceFocalLoss
        }
        # 选择优化器
        self.optimizer_label = tk.Label(self.left_frame, text="选择优化器:", font=("Microsoft YaHei", 14, "bold"))
        self.optimizer_label.grid(row=7, column=0, padx=10, pady=3, sticky="ew")
        self.optimizer_var = tk.StringVar(root)
        self.optimizer_var.set("Adam")  # 默认值
        self.optimizer_menu = tk.OptionMenu(self.left_frame, self.optimizer_var, "Adam", "SGD", "RMSprop")
        self.optimizer_menu.config(font=self.new_font)
        self.optimizer_menu.grid(row=7, column=1, padx=10, pady=3, sticky="ew")
        # 清空预览窗口
        self.clear_button = tk.Button(self.left_frame, text="清空预览窗口", command=self.clear_image_previews, font=("Microsoft YaHei", 14, "bold"))
        self.clear_button.grid(row=8, columnspan=2, padx=10, pady=3)
        # 训练图像大小
        self.shape_label = tk.Label(self.left_frame, text="训练图像大小shape:", font=("Microsoft YaHei", 12, "bold"))
        self.shape_label.grid(row=9, column=0, padx=10, pady=1, sticky="ew")
        self.shape_var = tk.StringVar(value="640")
        self.shape_entry = tk.Entry(self.left_frame, textvariable=self.shape_var, justify="center", font=self.new_font)
        self.shape_entry.grid(row=9, column=1, padx=10, pady=1, sticky="ew")
        # 批次大小
        self.batchsize_label = tk.Label(self.left_frame, text="批次大小batchsize:", font=("Microsoft YaHei", 12, "bold"))
        self.batchsize_label.grid(row=10, column=0, padx=10, pady=1, sticky="ew")
        self.batchsize_var = tk.StringVar(value="2")
        self.batchsize_entry = tk.Entry(self.left_frame, textvariable=self.batchsize_var, justify="center", font=self.new_font)
        self.batchsize_entry.grid(row=10, column=1, padx=10, pady=1, sticky="ew")
        # 学习率Ir
        self.lr_label = tk.Label(self.left_frame, text="学习率Ir:", font=("Microsoft YaHei", 12, "bold"))
        self.lr_label.grid(row=11, column=0, padx=10, pady=1, sticky="ew")
        self.lr_var = tk.StringVar(value="1e-4")
        self.lr_entry = tk.Entry(self.left_frame, textvariable=self.lr_var, justify="center", font=self.new_font)
        self.lr_entry.grid(row=11, column=1, padx=10, pady=1, sticky="ew")
        # 训练策略
        self.max_no_optim_label = tk.Label(self.left_frame, text="训练早停no_optim次数:", font=("Microsoft YaHei", 12, "bold"))
        self.max_no_optim_label.grid(row=12, column=0, padx=10, pady=1, sticky="ew")
        self.max_no_optim_var = tk.StringVar(value="6")
        self.max_no_optim_entry = tk.Entry(self.left_frame, textvariable=self.max_no_optim_var, justify="center", font=self.new_font)
        self.max_no_optim_entry.grid(row=12, column=1, padx=10, pady=1, sticky="ew") 
        self.Iradjust_no_optim_label = tk.Label(self.left_frame, text="Ir更新no_optim次数:", font=("Microsoft YaHei", 12, "bold"))
        self.Iradjust_no_optim_label.grid(row=13, column=0, padx=10, pady=1, sticky="ew")
        self.Iradjust_no_optim_var = tk.StringVar(value="3")
        self.Iradjust_no_optim_entry = tk.Entry(self.left_frame, textvariable=self.Iradjust_no_optim_var, justify="center", font=self.new_font)
        self.Iradjust_no_optim_entry.grid(row=13, column=1, padx=10, pady=1, sticky="ew")
        self.min_Ir_label = tk.Label(self.left_frame, text="Ir更新最小值:", font=("Microsoft YaHei", 12, "bold"))
        self.min_Ir_label.grid(row=14, column=0, padx=10, pady=1, sticky="ew")
        self.min_Ir_var = tk.StringVar(value="5e-7")
        self.min_Ir_entry = tk.Entry(self.left_frame, textvariable=self.min_Ir_var, justify="center", font=self.new_font)
        self.min_Ir_entry.grid(row=14, column=1, padx=10, pady=1, sticky="ew") 
        self.Ir_update_label = tk.Label(self.left_frame, text="Ir更新降低倍数:", font=("Microsoft YaHei", 12, "bold"))
        self.Ir_update_label.grid(row=15, column=0, padx=10, pady=1, sticky="ew")
        self.Ir_update_var = tk.StringVar(value="5")
        self.Ir_update_entry = tk.Entry(self.left_frame, textvariable=self.Ir_update_var, justify="center", font=self.new_font)
        self.Ir_update_entry.grid(row=15, column=1, padx=10, pady=1, sticky="ew")                      
        # 总训练迭代次数
        self.epochs_label = tk.Label(self.left_frame, text="总训练epoches:", font=("Microsoft YaHei", 12, "bold"))
        self.epochs_label.grid(row=16, column=0, padx=10, pady=3, sticky="ew")
        self.epochs_var = tk.StringVar(value="300")
        self.epochs_entry = tk.Entry(self.left_frame, textvariable=self.epochs_var, justify="center", font=self.new_font)
        self.epochs_entry.grid(row=16, column=1, padx=10, pady=3, sticky="ew")
        # 开始训练
        self.train_button = tk.Button(self.left_frame, text="开始训练", command=self.start_training, font=("Microsoft YaHei", 14, "bold"))
        self.train_button.grid(row=17, column=0, padx=10, pady=3, sticky="we")  
        #停止训练
        self.stop_button = tk.Button(self.left_frame, text="停止训练", command=self.stop_training, font=("Microsoft YaHei", 14, "bold"))
        self.stop_button.grid(row=17, column=1, padx=10, pady=3, sticky="we")
        # 训练进度
        self.progress_label = tk.Label(self.left_frame, text="训练进度:", font=("Microsoft YaHei", 14, "bold"))
        self.progress_label.grid(row=18, column=0, padx=10, pady=3, sticky="ew")
        self.progress = tk.StringVar()
        self.progress.set("0%")
        self.progress_bar = tk.Label(self.left_frame, textvariable=self.progress)
        self.progress_bar.grid(row=18, column=1, padx=10, pady=3, sticky="we") 
        # 选择测试集
        self.predict_data_label = tk.Label(self.left_frame, text="测试集文件夹:", font=("Microsoft YaHei", 14, "bold"))
        self.predict_data_label.grid(row=19, column=0, padx=10, pady=3, sticky="nsew")
        self.predict_data_button = tk.Button(self.left_frame, text="浏览", command=self.select_predict_data, font=("Microsoft YaHei", 12, "bold"))
        self.predict_data_button.grid(row=19, column=1, padx=10, pady=3, sticky="ew")
        # val文件夹目录
        self.predict_data_label_selected = tk.Label(self.left_frame, text="", font=("Times New Roman", 8, "normal"), wraplength=350)
        self.predict_data_label_selected.grid(row=20, columnspan=2, padx=10, pady=1, sticky="ew")
        # 开始预测
        self.predict_button = tk.Button(self.left_frame, text="开始预测", command=self.start_predicting, font=("Microsoft YaHei", 14, "bold"))
        self.predict_button.grid(row=21, column=0, padx=10, pady=3, sticky="we")  
        # 停止预测
        self.stop_button = tk.Button(self.left_frame, text="停止预测", command=self.stop_predicting, font=("Microsoft YaHei", 14, "bold"))
        self.stop_button.grid(row=21, column=1, padx=10, pady=3, sticky="we")     
        # 选择性能指标
        self.val_fun_label = tk.Label(self.left_frame, text="选择性能指标:", font=("Microsoft YaHei", 14, "bold"))
        self.val_fun_label.grid(row=22, column=0, padx=10, pady=3, sticky="ew")
        self.val_fun_var = tk.StringVar(root)
        self.val_fun_var.set("Dice")  # 默认值
        self.val_fun_menu = tk.OptionMenu(self.left_frame, self.val_fun_var, "Accuracy", "Dice", "F1", "IoU", "MPA", "Precision", "Recall", )
        self.val_fun_menu.grid(row=22, column=1, padx=10, pady=3, sticky="ew")
        self.val_fun_menu.config(font=self.new_font)
        self.val_functions = {
            "Accuracy": Accuracy,
            "Dice": Dice,
            "F1": F1,
            "IoU": IoU,
            "MPA": MPA,
            "Precision": Precision,
            "Recall": Recall,            
        }
        # 测试模型性能
        self.eval_button = tk.Button(self.left_frame, text="评估模型", command=self.evaluate_model, font=("Microsoft YaHei", 14, "bold"))
        self.eval_button.grid(row=23, columnspan=2, padx=10, pady=3, sticky="we")

        # 在 19 上添加 Text 小部件
        self.text_widget = tk.Text(self.left_frame, bg="white", wrap=tk.WORD, height=5, width=5)
        self.text_widget.grid(row=24, columnspan=2, rowspan=3, padx=10, pady=3, sticky="nswe")

        self.new_font1 = font.Font(family="SimSun", size=12, weight="bold")#右上字体
        self.text_widget.configure(font=self.new_font1)  

        self.training_thread = None
        self.predicting_thread = None
        self.stop_training_flag = threading.Event()
        self.stop_predicting_flag = threading.Event()

    def select_train_data(self):
        self.train_data_folder = filedialog.askdirectory()
        if not self.train_data_folder:  # 检查用户是否选择了文件夹
            return  # 如果未选择文件夹，则直接返回
        if os.path.basename(self.train_data_folder) != "image":
            messagebox.showerror("错误", "请选择名为 'image' 的文件夹")
            return
        print(f"选择的训练数据文件夹: {self.train_data_folder}")
        self.train_data_label_selected.config(text=f"{self.train_data_folder}")
        self.update_image_previews()

    def select_weights(self):
        self.weights_path = filedialog.askopenfilename(
            filetypes=[("权重文件", "*.th")]
        )
        if not self.weights_path:  # 检查用户是否选择了文件夹
            self.weights_data_label_selected.config(text=f"")
            self.weights_path =  None
            return  # 如果未选择文件夹，则直接返回
        print(f"选择的权重文件: {self.weights_path}")
        self.weights_data_label_selected.config(text=f"{self.weights_path}")

    def update_image_previews(self):
        self.image_refs_upper = []  # 重置图像引用列表
        self.label_refs_lower = []  # 重置图像引用列表
        self.display_all_images(self.upper_mid_canvas, self.train_data_folder, self.image_refs_upper)
    
        label_data_folder = os.path.join(os.path.dirname(self.train_data_folder), 'label')
        if os.path.exists(label_data_folder):
            self.display_all_images(self.lower_mid_canvas, label_data_folder, self.label_refs_lower)
        else:
            print("标签数据文件夹不存在")

    def display_all_images(self, canvas, folder_path, image_refs):
        # 清除之前的内容
        canvas.delete("all")
        
        image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
        x_offset = 10
        y_offset = 10
        container_width = self.lower_right_frame.winfo_width()
        max_width = container_width - 10
        images_width = max_width // 3 - 10
        for image_file in image_files:
            image_path = os.path.join(folder_path, image_file)
            photo = self.display_image(canvas, image_path, x_offset, y_offset, images_width)
            image_refs.append(photo)  # 存储图片引用
            x_offset += (images_width +10)  # 假设每张图片宽约为200，并留有间隙
            if x_offset + images_width > max_width:
                x_offset = 10
                y_offset += (images_width +10)

        # 调整 Canvas 的滚动区域
        canvas.config(scrollregion=(0, 0, max_width, y_offset + images_width + 10))

    def display_image(self, canvas, image_path, x_offset, y_offset, image_width):
        image = Image.open(image_path)
        image = image.resize((image_width, image_width))  # 调整图像大小为缩略图
        photo = ImageTk.PhotoImage(image)
        canvas.create_image(x_offset, y_offset, anchor=tk.NW, image=photo)
        return photo  # 返回图片引用
    
    def _sync_scroll(self, *args):
        self.upper_mid_canvas.yview(*args)
        self.lower_mid_canvas.yview(*args)
        self.lower_right_canvas.yview(*args)

    def _on_mouse_wheel(self, event):
        self.upper_mid_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.lower_mid_canvas.yview_scroll(int(-1*(event.delta/120)), "units")
        self.lower_right_canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def clear_image_previews(self):
        has_images = False  # 用于跟踪是否有图片存在
        if hasattr(self, 'image_refs_upper') and self.image_refs_upper:
            self.clear_canvas(self.upper_mid_canvas, self.image_refs_upper)
            self.image_refs_upper = []  # 重置图像引用列表
            has_images = True
        if hasattr(self, 'label_refs_lower') and self.label_refs_lower:
            self.clear_canvas(self.lower_mid_canvas, self.label_refs_lower)
            self.label_refs_lower = []  # 重置图像引用列表
            has_images = True
        if hasattr(self, 'predict_refs_lower') and self.predict_refs_lower:
            self.clear_canvas(self.lower_right_canvas, self.predict_refs_lower)
            self.predict_refs_lower = []  # 重置图像引用列表
            has_images = True
        if self.upper_right_canvas.winfo_children():
            for widget in self.upper_right_canvas.winfo_children():
                widget.destroy()
            has_images = True
        if not has_images:
            messagebox.showinfo("预览窗格", "没有图片")

    def clear_canvas(self, canvas, image_refs):
        canvas.delete("all")
        del image_refs[:]
   
    def get_loss_function(self, loss_name):
        try:
            return self.loss_functions[loss_name]
        except KeyError:
            raise ValueError(f"未知的损失函数: {loss_name}")

    def start_training(self):
        if self.training_thread is None or not self.training_thread.is_alive():
            self.create_progress_label()
            self.training_thread = threading.Thread(target=self.training_process)
            self.stop_training_flag.clear()
            self.training_thread.start()
        else:
            messagebox.showwarning("警告", "训练正在进行中")

    def create_progress_label(self):
        self.progress_label = tk.Label(self.left_frame, text="训练进度:", font=("Microsoft YaHei", 14, "bold"))
        self.progress_label.grid(row=18, column=0, padx=10, pady=5, sticky="ew")
        self.progress = tk.StringVar()
        self.progress.set("0%")
        self.progress_bar = tk.Label(self.left_frame, textvariable=self.progress)
        self.progress_bar.grid(row=18, column=1, padx=10, pady=5, sticky="we")

    def stop_training(self):
        if self.training_thread is not None and self.training_thread.is_alive():
            if messagebox.askyesno("停止训练", "确定要停止训练吗？"):
                self.stop_training_flag.set()
        else:
            messagebox.showwarning("警告", "当前没有正在进行的训练")

    def training_process(self):    
        if not hasattr(self, 'train_data_folder'):
            messagebox.showerror("错误", "请先选择训练图像文件夹")
            return  
        train_data_folder = self.train_data_folder
        model_name = self.model_var.get()
        weights_path = self.weights_path
        loss_name = self.loss_var.get()
        optimizer_name = self.optimizer_var.get()
        SHAPE1 = int(self.shape_entry.get())
        batch_size = int(self.batchsize_entry.get())
        lr = float(self.lr_entry.get())
        max_no_optim = int(self.max_no_optim_entry.get())
        Iradjust_no_optim = int(self.Iradjust_no_optim_entry.get())
        min_Ir = float(self.min_Ir_entry.get())
        Ir_update = int(self.Ir_update_entry.get())
        epochs = int(self.epochs_entry.get())
        self.TIME = datetime.now()  # 获取当前时间并保存在 TIME 变量中

        print(f"使用模型: {model_name}, 损失函数: {loss_name}, 优化器: {optimizer_name}, 批次大小: {batch_size}, 学习率: {lr}, 总迭代次数: {epochs}")
        folderlist = [filename for filename in os.listdir(train_data_folder) if filename.endswith('.png')]
        trainlist = [re.match(r'\d+', item).group() for item in folderlist]
        model, Manual = self.load_model(model_name)  # 加载模型
        loss_fn = self.get_loss_function(loss_name)  # 加载损失函数
        optimizer = self.load_optimizer(optimizer_name)  # 加载优化器
        print(Manual)
        if Manual is True:
            solver = MyFrame1(Manual=Manual, net=model(aux=True, num_classes=1), optimizer=optimizer, loss=loss_fn, lr=lr) 
        else:
            solver = MyFrame(net=model, optimizer=optimizer, loss=loss_fn, lr=lr)

        if weights_path is not None and weights_path != "":
            solver.load(weights_path)
        batchsize = torch.cuda.device_count() * batch_size

        # 加载数据
        transform = transforms.Compose([
            transforms.Resize((640, 640)),  # 调整图像大小
            # 转换为张量
            # 添加其他数据增强和预处理操作
        ])
        train_dataset = ImageFolder(trainlist, train_data_folder, transform=transform)
        train_loader = DataLoader(train_dataset, batch_size=batchsize, shuffle=True, num_workers=0)
        SHAPE = (SHAPE1, SHAPE1)
        self.NAME = model_name + "_" + self.TIME.strftime("%Y_%m_%d_%H_%M")
        # 开始训练
        # 检查并创建logs文件夹
        logs_folder = 'logs'
        if not os.path.exists(logs_folder):
            os.makedirs(logs_folder)
        self.train_losses_excel = []
        weights_folder = 'weights'
        if not os.path.exists(weights_folder):
            os.makedirs(weights_folder)        
        mylog = open(os.path.join(logs_folder, self.NAME + '.log'), 'w')
        tic = time()
        no_optim = 0
        total_epoch = epochs
        train_epoch_best_loss = epochs
        for epoch in range(1, total_epoch + 1):

            data_loader_iter = iter(train_loader)
            train_epoch_loss = 0

            for image, label in data_loader_iter:
                if self.stop_training_flag.is_set():
                    print('stopped by user.')
                    messagebox.showinfo("训练停止", "训练已停止")
                    break
                # print(image.shape)
                solver.set_input(image, label)
                train_loss = solver.optimize()
                # print("loss:", train_loss)
                train_epoch_loss += train_loss
            if self.stop_training_flag.is_set():
                break            
            train_epoch_loss /= len(train_loader)
            print('********', file=mylog)
            print('epoch:', epoch, '    time:', int(time() - tic), file=mylog)
            print('train_loss:', train_epoch_loss, file=mylog)
            print('SHAPE:', SHAPE, file=mylog)
            print('********')
            print('epoch:', epoch, '    time:', int(time() - tic))
            print('train_loss:', train_epoch_loss)
            print('SHAPE:', SHAPE)
            self.update_progress(epoch, total_epoch, train_epoch_loss)
            if train_epoch_loss >= train_epoch_best_loss:
                no_optim += 1
            else:
                no_optim = 0
                train_epoch_best_loss = train_epoch_loss
                solver.save('weights/' + self.NAME + '.th')
            if no_optim > max_no_optim:
                print('early stop at %d epoch' % epoch, file=mylog)
                print('early stop at %d epoch' % epoch)
                break
            if no_optim > Iradjust_no_optim:
                if solver.old_lr < min_Ir:
                    break
                solver.load('weights/' + self.NAME + '.th')
                solver.update_lr(Ir_update, factor=True, mylog=mylog)
            mylog.flush()
            # 检查是否在主线程
            #self.load_logs_and_plot(logs_folder)
    
        print('Finish!', file=mylog)
        print('Finish!')
        mylog.close()
        trainlossfolder_path = 'trainlossexcel'
        if not os.path.exists(trainlossfolder_path):
            os.makedirs(trainlossfolder_path)
        # 创建Excel文件
        file_path = os.path.join(trainlossfolder_path, self.NAME + '.xlsx')
        wb = Workbook()
        ws = wb.active
        ws.title = 'Train Losses'
        # 写入数据
        for i, loss in enumerate(self.train_losses_excel, start=1):
            ws.cell(row=i, column=1, value=loss)
        # 保存文件
        wb.save(file_path)
        print(f"Train losses saved to {file_path}")  
    
    def is_main_thread(self):  # ✅ 这样才能用 self 调用
        return threading.current_thread() == threading.main_thread()
    
    def load_logs_and_plot(self, log_path):
        self.train_losses_excel = []
        self.train_losses_excel = self.extract_train_loss(log_path)
        self.plot_train_loss()
        
    def extract_train_loss(self, log_path):
        with open(os.path.join(log_path, self.NAME + '.log'),'r') as file:
            for line in file:
                match = re.search(r'train_loss:\s*([0-9.]+)', line)
                if match:
                    self.train_losses_excel.append(float(match.group(1)))
        return self.train_losses_excel

    def plot_train_loss(self):
        fig, ax = plt.subplots()
        ax.plot(range(1, len(self.train_losses_excel) + 1), self.train_losses_excel, label='Train Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Train Loss Over Epochs')
        ax.legend()
        plt.close('all')
        fig.tight_layout()
        # 清空现有的图像
        for widget in self.upper_right_canvas.winfo_children():
            widget.destroy()
        canvas = FigureCanvasTkAgg(fig, master=self.upper_right_canvas)
        canvas.draw()
        canvas_widget = canvas.get_tk_widget()
        # 设置固定大小，例如宽度为600，高度为400
        canvas_widget.configure(width=1000, height=600)
        canvas_widget.pack(side=tk.TOP)
    
    def update_progress(self, current_epoch, total_epochs, loss):
        progress_percentage = (current_epoch / total_epochs) * 100
        self.progress.set(f"{progress_percentage:.2f}% (Epoch: {current_epoch}/{total_epochs}, Loss: {loss:.4f})")
        self.progress_bar.update_idletasks()  # 强制更新界面

    def load_model(self, model_name):
        try:
            model = self.model_menus[model_name]
            print(model_name)
            if model_name == "deeplabv3_resnet50" or model_name == "deeplabv3_resnet101":
                manual = True  # 获取用户设置的辅助分类器学习率倍率
            else:
                manual = None
            print(manual)   
            return model, manual
        except KeyError:
            raise ValueError(f"未知的模型: {model_name}")

    def load_optimizer(self, optimizer_name):
        if optimizer_name == "Adam":
            # 替换为你的UNet模型加载代码
            optimizer = torch.optim.Adam
        elif optimizer_name == "SGD":
            # 替换为你的SegNet模型加载代码
            optimizer = torch.optim.SGD
        elif optimizer_name == "RMSprop":
            optimizer = torch.optim.RMSprop
        return optimizer

    def update_predict_previews(self):
        self.predict_refs_lower = []  # 重置图像引用列表

        self.predictions_data_folder = os.path.join(os.path.dirname(self.predict_data_folder), self.foldername)
        if not os.path.exists(self.predictions_data_folder):
            os.makedirs(self.predictions_data_folder)
        if os.path.exists(self.predictions_data_folder):
            self.display_all_images(self.lower_right_canvas, self.predictions_data_folder, self.predict_refs_lower)
        else:
            print("预测数据文件夹不存在")    

    def select_predict_data(self):
        self.predict_data_folder = filedialog.askdirectory()
        if not self.predict_data_folder:  # 检查用户是否选择了文件夹
            return  # 如果未选择文件夹，则直接返回
        if os.path.basename(self.predict_data_folder) != "image":
            messagebox.showerror("错误", "请选择名为 'image' 的文件夹")
            return
        print(f"选择的测试数据文件夹: {self.predict_data_folder}")
        self.predict_data_label_selected.config(text=f"{self.predict_data_folder}")
        self.image_refs_upper = []  # 重置图像引用列表
        self.label_refs_lower = []  # 重置图像引用列表
        self.predict_refs_lower = []

        self.display_all_images(self.upper_mid_canvas, self.predict_data_folder, self.image_refs_upper)
        self.predlabel_data_folder = os.path.join(os.path.dirname(self.predict_data_folder), 'label')
        if os.path.exists(self.predlabel_data_folder):
            self.display_all_images(self.lower_mid_canvas, self.predlabel_data_folder, self.label_refs_lower)
        else:
            print("预测label文件夹不存在") 
        self.predictions_data_folder = os.path.join(os.path.dirname(self.predict_data_folder), 'predictions')
        if os.path.exists(self.predictions_data_folder):
            self.display_all_images(self.lower_right_canvas, self.predictions_data_folder, self.predict_refs_lower)
        else:
            print("预测predictions文件夹不存在") 

    def start_predicting(self):
        if self.predicting_thread is None or not self.predicting_thread.is_alive():
            self.predicting_thread = threading.Thread(target=self.predict_images)
            self.stop_predicting_flag.clear()
            self.predicting_thread.start()
        else:
            messagebox.showwarning("警告", "预测正在进行中")

    def stop_predicting(self):
        if self.predicting_thread is not None and self.predicting_thread.is_alive():
            if messagebox.askyesno("停止预测", "确定要停止预测吗？"):
                self.stop_predicting_flag.set()
        else:
            messagebox.showwarning("警告", "当前没有正在进行的预测")

    def predict_images(self):
        if not hasattr(self, 'predict_data_folder'):
            messagebox.showerror("错误", "请先选择测试图像文件夹")
            return
        
        model_name = self.model_var.get()
        weights_path = self.weights_path
        batch_size = int(self.batchsize_entry.get())
        net, Manual = self.load_model(model_name)
        print(Manual)
        if Manual is True:
            tta_model = TTAFrame(Manual=Manual, net=net(aux=True, num_classes=1), batchsize=batch_size) 
        else:
            tta_model = TTAFrame(Manual=Manual, net=net, batchsize=batch_size) 
        
        if weights_path is None or weights_path == "":
            messagebox.showerror("错误", "请先选择权重")
            return
        tta_model.load(weights_path)
            # 获取当前时间并格式化
        current_time = datetime.now().strftime("%Y_%m_%d_%H_%M")
        self.foldername = 'predictions_{model_name}_{current_time}'
        # 根据模型名称和时间来生成output_folder
        output_folder = os.path.join(os.path.dirname(self.predict_data_folder), f'predictions_{model_name}_{current_time}')
        os.makedirs(output_folder, exist_ok=True)

        image_files = sorted(os.listdir(self.predict_data_folder))

        for image_file in tqdm(image_files):
            image_path = os.path.join(self.predict_data_folder, image_file)
            print("Processing image", image_path)
            mask = tta_model.test_one_img_from_path(image_path)
            h = 4
            mask[mask > h] = 255
            mask[mask <= h] = 0
            mask = np.concatenate([mask[:, :, None], mask[:, :, None], mask[:, :, None]], axis=2)

            # 保存处理后的图像
            output_name = os.path.splitext(image_file)[0] + ".png"  # 在原始预测名称后添加 "_pred" 后缀
            output_path = os.path.join(output_folder, output_name)
            cv2.imwrite(output_path, mask.astype(np.uint8))
            self.update_predict_previews()
            if self.stop_predicting_flag.is_set():
                print('stopped by user.')
                messagebox.showinfo("预测停止", "预测已停止")
                break

        messagebox.showinfo("完成", "预测完成！结果保存在 " + output_folder)

    def get_val_function(self, val_fun):
        try:
            return self.val_functions[val_fun]
        except KeyError:
            raise ValueError(f"未知的性能指标函数: {val_fun}")
    
    def evaluate_model(self):
        if not hasattr(self, 'predictions_data_folder'):
            messagebox.showerror("错误", "请先选择预测文件夹")
            return
        metric_name = self.val_fun_var.get()
        metric_fun = self.get_val_function(metric_name)
        metric_value = metric_fun(self.predlabel_data_folder, self.predictions_data_folder)
        result_string = (
            f"{self.weights_path}\n"
            f"标签文件夹: {self.predlabel_data_folder}\n"
            f"预测文件夹: {self.predictions_data_folder}\n"
            f"{metric_name}: {metric_value}\n"
        )
        print(result_string)
        os.makedirs('evaluation', exist_ok=True)# 确保evaluation文件夹存在
        with open('evaluation/evaluation.txt', 'a') as file:# 将结果追加到evaluation.txt中
            file.write(result_string + '\n')

class TextRedirector:
    def __init__(self, text_widget):
        self.text_widget = text_widget

    def write(self, message):
        self.text_widget.insert(tk.END, message)
        self.text_widget.see(tk.END)  # 确保最新的消息在视图中可见

    def flush(self):
        pass

if __name__ == "__main__":
    root = tk.Tk()
    app = SemanticSegmentationGUI(root)

    # 重定向标准输出和标准错误输出到 Text 小部件
    sys.stdout = TextRedirector(app.text_widget)
    sys.stderr = TextRedirector(app.text_widget)

    root.mainloop()

    # 恢复标准输出和标准错误输出
    sys.stdout = sys.__stdout__
    sys.stderr = sys.__stderr__