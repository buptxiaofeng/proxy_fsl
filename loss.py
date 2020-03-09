import torch
import torch.nn as nn

#class CenterLoss(nn.Module):
#    """Center loss.
#
#    Reference:
#    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
#
#   Args:
#        num_classes (int): number of classes.
#       feat_dim (int): feature dimension.
#    """
#    def __init__(self, num_classes=5, feat_dim=1600, use_gpu=True):
#        super(CenterLoss, self).__init__()
#        self.num_classes = num_classes
#        self.feat_dim = feat_dim
#        self.use_gpu = use_gpu
#
#        if self.use_gpu:
#            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
#        else:
#            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
#
#    def forward(self, x, labels):
#        """
#        Args:
#            x: feature matrix with shape (batch_size, feat_dim).
#            labels: ground truth labels with shape (batch_size).
#        """
#        batch_size = x.size(0)
#        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
#                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
#        distmat.addmm_(1, -2, x, self.centers.t())
#
#        classes = torch.arange(self.num_classes).long()
#        if self.use_gpu: classes = classes.cuda()
#        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
#        mask = labels.eq(classes.expand(batch_size, self.num_classes))
#
#        dist = distmat * mask.float()
#        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size
#
#        return loss
class ContrastLoss(nn.Module):
    def __init__(self, margin = 1):
        super(ContrastLoss, self).__init__()
        self.margin  = margin

    def forward(self, d, y):
        d = d.flatten()
        y = y.flatten()
        amount = d.shape[0]

        mdist = self.margin - d
        dist = torch.clamp(mdist, min=0.0)
        loss = y * torch.pow(d, 2) + (1 - y) * torch.pow(dist, 2)
        loss = torch.sum(loss) / 2.0 / amount

        return loss

class MMD2Loss(nn.Module):
    def __init__(self):
        super(MMD2Loss, self).__init__()

    def forward(self, x1, x2):
        x1 = torch.reshape(x1, (x1.shape[0], x1.shape[1] * x1.shape[2] * x1.shape[3]))
        x2 = torch.reshape(x2, (x2.shape[0], x2.shape[1] * x2.shape[2] * x2.shape[3]))
        loss = 0.0
        delta = x1.float().mean(0) - x2.float().mean(0)
        loss = torch.sum(delta * delta)
        return loss

class CenterLoss(nn.Module):
    def __init__(self):
        super(CenterLoss, self).__init__()
        self.mse = nn.MSELoss(reduction='sum')

    def forward(self, x, center):
        loss = 0
        for i in range(x.shape[0]):
            for j in range(x.shape[1]):
                tmp1 = x[i,j,...].squeeze().flatten()
                tmp2 = center[i, ...].squeeze().flatten()
                loss += self.mse(tmp1, tmp2) / tmp1.shape[0]

        return loss / (x.shape[0] * x.shape[1])
