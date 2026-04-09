import torch

def GeneralizedDiceLoss(target, pred, epsilon=1e-6):

    # 计算每个类别的目标值总和
    wei = torch.sum(target, dim=[0, 2, 3])  # 形状: (n_class,)
    # 计算每个类别的平方和的倒数
    wei = 1 / (wei**2 + epsilon)
    # 计算每个类别的加权交集
    intersection = torch.sum(wei * torch.sum(pred * target, dim=[0, 2, 3]))
    # 计算每个类别的加权并集
    union = torch.sum(wei * torch.sum(pred + target, dim=[0, 2, 3]))
    # 计算加权Dice损失
    gldice_loss = 1 - (2. * intersection) / (union + epsilon)
    return gldice_loss