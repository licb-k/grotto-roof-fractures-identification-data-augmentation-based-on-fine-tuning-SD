import torch.nn as nn

class TverskyLoss(nn.Module):
    def __init__(self, beta=0.7, weights=None):
        super(TverskyLoss, self).__init__()
        self.beta = beta
        self.alpha = 1.0 - beta
        self.weights = weights

    def forward(self, targets, inputs):
        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        # Calculate true positives, false positives, and false negatives
        tp = (inputs * targets).sum()
        fp = ((1 - targets) * inputs).sum()
        fn = (targets * (1 - inputs)).sum()
        
        tversky_index = tp / (tp + self.alpha * fp + self.beta * fn)
        loss = 1 - tversky_index
        
        return loss