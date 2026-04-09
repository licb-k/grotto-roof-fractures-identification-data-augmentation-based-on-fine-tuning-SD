import torch
import torch.nn as nn
import cv2
import numpy as np

class MyFrame():
    def __init__(self, net, optimizer, loss, lr=2e-4, evalmode=False):
        self.net = net().cuda()
        self.net = torch.nn.DataParallel(self.net, device_ids=list(range(torch.cuda.device_count())))
        self.optimizer = optimizer(params=self.net.parameters(), lr=lr)
        self.loss = loss()
        self.old_lr = lr
        if evalmode:
            for i in self.net.modules():
                if isinstance(i, nn.BatchNorm2d):
                    i.eval()

    def set_input(self, image_batch, label_batch=None, image_id=None):
        self.image = image_batch
        self.label = label_batch
        self.image_id = image_id

    def test_one_image(self, image):
        pred = self.net.forward(image)

        pred[pred > 0.5] = 1
        pred[pred <= 0.5] = 0

        label = pred.squeeze().cpu().data.numpy()
        return label
    def test_batch(self):
        self.forward(volatile=True)
        label = self.net.forward(self.image).cpu().data.numpy().squeeze(1)
        label[label > 0.5] = 1
        label[label <= 0.5] = 0

        return label, self.image_id

    def test_one_image_from_path(self, path):
        image = cv2.imread(path)
        image = np.array(image, np.float32) / 255.0 * 3.2 - 1.6
        image = torch.Tensor(image).cuda()

        label = self.net.forward(image).squeeze().cpu().data.numpy()
        label[label > 0.5] = 1
        label[label <= 0.5] = 0

        return label

    def forward(self, volatile=False):
        self.image = self.image.cuda().float()
        if self.label is not None:
            self.label = self.label.cuda().float()

    def optimize(self):
        self.forward()
        self.optimizer.zero_grad()
        pred = self.net.forward(self.image)
        #print(pred.shape)
        loss = self.loss(self.label, pred)
        loss.backward()
        self.optimizer.step()
        return loss.item()

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))

    def update_lr(self, new_lr, mylog, factor=False):
        if factor:
            new_lr = self.old_lr / new_lr
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = new_lr

        print('update learning rate: %f -> %f' % (self.old_lr, new_lr), file=mylog)
        print('update learning rate: %f -> %f' % (self.old_lr, new_lr))
        self.old_lr = new_lr

