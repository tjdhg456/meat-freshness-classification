import torch
import torch.nn as nn

import torch
import torch.nn as nn
from torch.autograd import Variable


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

    def forward(self, x, labels, imp=None):
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
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

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

        # loss = torch.min(dist.clamp(min=1e-12, max=1e+12).sum() / batch_size + \
        #        auxmat.clamp(min=1e-12, max=1e+12).sum() / aux_size - \
        #        dist_aux_mat.clamp(min=1e-12, max=1e+12).sum() / batch_size, 0)

        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size + \
               auxmat.clamp(min=1e-12, max=1e+12).sum() / aux_size

        return loss


################################################################
## Triplet related loss
################################################################
def pdist(A, squared=False, eps=1e-4):
    prod = torch.mm(A, A.t())
    norm = prod.diag().unsqueeze(1).expand_as(prod)
    res = (norm + norm.t() - 2 * prod).clamp(min=0)
    return res if squared else (res + eps).sqrt() + eps

class TripletCenterLoss(nn.Module):
    def __init__(self, margin=0, num_classes=10, use_gpu=True):
        super(TripletCenterLoss, self).__init__()
        self.margin = margin

        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.use_gpu = use_gpu
        if self.use_gpu == True:
            self.ranking_loss = self.ranking_loss.cuda()
            self.centers = nn.Parameter(torch.randn(num_classes, 2).cuda())
        else:
            self.centers = nn.Parameter(torch.randn(num_classes, 2))

    def forward(self, inputs, targets, emb=None):
        if emb is not None:
            inputs = torch.cat([inputs, emb], dim=0)
            emb_target = (torch.ones([emb.size(0)]) * 2).long()
            if self.use_gpu == True:
                emb_target = emb_target.cuda()
            targets = torch.cat([targets, emb_target], dim=0)

        batch_size = inputs.size(0)
        targets_expand = targets.view(batch_size, 1).expand(batch_size, inputs.size(1))
        centers_batch = self.centers.gather(0, targets_expand)  # centers batch

        # compute pairwise distances between input features and corresponding centers
        centers_batch_bz = torch.stack([centers_batch] * batch_size)
        inputs_bz = torch.stack([inputs] * batch_size).transpose(0, 1)
        dist = torch.sum((centers_batch_bz - inputs_bz) ** 2, 2).squeeze()
        dist = dist.clamp(min=1e-12).sqrt()  # for numerical stability

        # for each anchor, find the hardest positive and negative
        mask = targets.expand(batch_size, batch_size).eq(targets.expand(batch_size, batch_size).t())
        dist_ap, dist_an = [], []
        for i in range(batch_size):  # for each sample, we compute distance
            dist_ap.append(dist[i][mask[i]].max())  # mask[i]: positive samples of sample i
            dist_an.append(dist[i][mask[i] == 0].min())  # mask[i]==0: negative samples of sample i
        dist_ap = torch.stack(dist_ap, dim=0)
        dist_an = torch.stack(dist_an, dim=0)

        # generate a new label y
        # compute ranking hinge loss
        y = dist_an.data.new()
        y.resize_as_(dist_an.data)
        y.fill_(1)
        y = Variable(y)
        # y_i = 1, means dist_an > dist_ap + margin will casuse loss be zero
        loss = self.ranking_loss(dist_an, dist_ap, y)

        prec = (dist_an.data > dist_ap.data).sum() * 1. / y.size(0)  # normalize data by batch size
        return loss