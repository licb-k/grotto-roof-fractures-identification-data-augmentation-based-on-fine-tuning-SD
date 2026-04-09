import torch
import torch.nn as nn
import cv2
import numpy as np

class MyFrame1():
    def __init__(self, Manual, net, optimizer, loss, lr=2e-4, evalmode=False):
        # DeepLabV3模型会被送到cuda，并使用多卡训练
        self.net = net.cuda() if Manual else net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=list(range(torch.cuda.device_count())))
        
        # 设置不同学习率
        params_to_optimize = [
            {"params": [p for p in self.net.module.backbone.parameters() if p.requires_grad]},  # 主干网络
            {"params": [p for p in self.net.module.classifier.parameters() if p.requires_grad]}  # 主分类器
        ]

        # 处理辅助分类器 aux_classifier
        if hasattr(self.net.module, "aux_classifier") and self.net.module.aux_classifier is not None:
            params_to_optimize.append({
                "params": [p for p in self.net.module.aux_classifier.parameters() if p.requires_grad],
                "lr": lr * 8  # 设置辅助分类器的学习率为 10 倍
            })

        # 使用传入的优化器类实例化优化器
        self.optimizer = optimizer(params_to_optimize, lr=lr)
        
        #self.optimizer = optimizer(params=self.net.parameters(), lr=lr)
        self.loss = loss()
        self.old_lr = lr
        # 设置为评估模式
        if evalmode:
            for module in self.net.modules():
                if isinstance(module, nn.BatchNorm2d):
                    module.eval()

    def set_input(self, image_batch, label_batch=None, image_id=None):
        # 设置输入数据
        self.image = image_batch
        self.label = label_batch
        self.image_id = image_id

    def test_one_image(self, image):
        # 对一张图片进行测试，返回二值化标签
        pred = self.net.forward(image)['out']
        pred = torch.argmax(pred, dim=1).cpu().numpy()
        return pred

    def test_batch(self):
        # 对一个batch进行测试
        self.forward(volatile=True)
        with torch.no_grad():
            pred = self.net.forward(self.image)['out']
            pred = torch.argmax(pred, dim=1).cpu().numpy()
        return pred, self.image_id

    def forward(self, volatile=False):
        # 设置前向传播时使用的输入数据
        self.image = self.image.cuda().float()
        if self.label is not None:
            self.label = self.label.cuda().float()

    def optimize(self):
        # 计算损失并进行反向传播
        self.forward()
        self.optimizer.zero_grad()

        # 在DeepLabV3中，模型的输出包含了一个字典
        pred = self.net.forward(self.image)
        losses = {}
        for name, x in pred.items():
            x = torch.sigmoid(x)
            #print(f"{name}: label min={self.label.min().item()}, max={self.label.max().item()}")
            #print(f"{name}: pred min={x.min().item()}, max={x.max().item()}")    
            losses[name] = self.loss(self.label, x)
        if len(losses) == 1:
            loss = losses['out']
        else:
            loss = losses['out'] + 0.5 * losses['aux']
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, path):
        # 保存模型的状态字典
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        # 加载模型的状态字典
        self.net.load_state_dict(torch.load(path))

    def update_lr(self, new_lr, mylog, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print(f'update learning rate: {self.old_lr} -> {new_lr}', file=mylog)
        print(f'update learning rate: {self.old_lr} -> {new_lr}')
        self.old_lr = new_lr

