import torch
import torch.nn as nn


class CenterLoss(nn.Module):
    """Center loss.

    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())

        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = torch.pow(dist.clamp(min=1e-12, max=1e+12).sum(), 0.5) / batch_size
        # loss = dist.clamp(min=1e-12, max=1e+12).mean()

        return loss

class AD_CenterLoss(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(AD_CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
            self.aux_centers = nn.Parameter(torch.randn(1, self.feat_dim).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))
            self.aux_centers = nn.Parameter(torch.randn(1, self.feat_dim))

    def forward(self, x, labels, aux_emb):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        aux_size = aux_emb.size(0)

        ## Distance
        # x <-> Center
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        # Calculating aux center
        auxmat = torch.pow(aux_emb, 2).sum(dim=1, keepdim=True).expand(aux_size, 1) + \
                  torch.pow(self.aux_centers, 2).sum(dim=1, keepdim=True).expand(1, aux_size).t()
        auxmat.addmm_(1, -2, aux_emb, self.aux_centers.t())

        # x <-> aux_center
        dist_aux_mat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, 1) + \
                  torch.pow(self.aux_centers, 2).sum(dim=1, keepdim=True).expand(1, batch_size).t()
        dist_aux_mat.addmm_(1, -2, x, self.aux_centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()

        loss = (torch.pow(dist.clamp(min=1e-12, max=1e+12).sum(), 0.5) / batch_size) + \
               (torch.pow(auxmat.clamp(min=1e-12, max=1e+12).sum(), 0.5) / aux_size) - \
               (torch.pow(dist_aux_mat.clamp(min=1e-12, max=1e+12).sum(), 0.5) / batch_size)

        return loss