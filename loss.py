import torch
import torch.nn as nn
from torch.autograd import Variable as V
import cupy as cp
from cucim.core.operations import morphology
import cv2
import numpy as np
from torch import nn, Tensor
import torch
import torch.nn.functional as F


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())


class SoftSkeletonize(torch.nn.Module):

    def __init__(self, num_iter=50):  # 40

        super(SoftSkeletonize, self).__init__()
        self.num_iter = num_iter

    def soft_erode(self, img):

        if len(img.shape) == 4:
            p1 = -F.max_pool2d(-img, (3, 1), (1, 1), (1, 0))
            p2 = -F.max_pool2d(-img, (1, 3), (1, 1), (0, 1))
            return torch.min(p1, p2)
        elif len(img.shape) == 5:
            p1 = -F.max_pool3d(-img, (3, 1, 1), (1, 1, 1), (1, 0, 0))
            p2 = -F.max_pool3d(-img, (1, 3, 1), (1, 1, 1), (0, 1, 0))
            p3 = -F.max_pool3d(-img, (1, 1, 3), (1, 1, 1), (0, 0, 1))
            return torch.min(torch.min(p1, p2), p3)

    def soft_dilate(self, img):

        if len(img.shape) == 4:
            return F.max_pool2d(img, (3, 3), (1, 1), (1, 1))
        elif len(img.shape) == 5:
            return F.max_pool3d(img, (3, 3, 3), (1, 1, 1), (1, 1, 1))

    def soft_open(self, img):

        return self.soft_dilate(self.soft_erode(img))

    def soft_skel(self, img):

        img1 = self.soft_open(img)
        skel = F.relu(img - img1)

        for j in range(self.num_iter):
            img = self.soft_erode(img)
            img1 = self.soft_open(img)
            delta = F.relu(img - img1)
            skel = skel + F.relu(delta - skel * delta)

        return skel

    def forward(self, img):

        return self.soft_skel(img)


class dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 0.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        return a + b


class c_dice_bce_loss(nn.Module):
    def __init__(self, batch=True):
        super(c_dice_bce_loss, self).__init__()
        self.batch = batch
        self.bce_loss = nn.BCELoss()

        # Initialize the CLossTopo loss
        self.topo_loss = CLossTopo(beta=1)
        self.gama = 0.03

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 1e-5  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_true, y_pred):
        # a = self.bce_loss(y_pred, y_true)
        # b = self.soft_dice_loss(y_true, y_pred)
        # Compute Topological Loss using CLossTopo
        topo_loss = self.topo_loss(y_pred, y_true)
        a = self.bce_loss(y_pred, y_true)
        b = self.soft_dice_loss(y_true, y_pred)
        loss = (1 - self.gama) * ( a + b) + self.gama * topo_loss
        return loss


class CLossTopo(nn.Module):
    def __init__(self, batch_dice: bool = False, do_bg: bool = False, smooth: float = 1e-5, ddp: bool = False, iter_=50,
                 beta=1,batch=True):
        """
        Calculates the Closs term that addresses the topological critical pixels of the predicted image and ground truth image
        Args:
            pred:   The likelihood pytorch tensor for neural networks.
            gt:   The groundtruth of pytorch tensor.
        Returns:
            ClossTopo:   The Closs value belonging to the topological term (tensor)
        """
        super(CLossTopo, self).__init__()
        self.batch = batch
        self.do_bg = do_bg
        self.batch_dice = batch_dice
        self.smooth = smooth
        self.ddp = ddp
        self.soft_skeletonize = SoftSkeletonize(num_iter=iter_)
        self.beta = beta  # for F1 score weighting, recall weighted beta times more important, optional use eg. for CLoss(dice)

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 1e-5  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()


    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss


    def forward(self, x, y, loss_mask=None):
        """
        Input shape: (batchsize,number_of_classes,z_dim,y_dim,x_dim)
        x: pred      number_of_classes is 2 in binary case, includes background
        y: gt       number_of_classes is 1 in binary case
        """

        x_logits = x  # presumed example shape 2,2,200,1024,1024

        # pred_binary = torch.argmax(x, dim=1)  # pred shape = 2,200,1024,1024 contains pred idx analog gt
        # pred_binary = pred_binary.unsqueeze(1)  # 2,1,200,1024,1024, binary
        pred_binary = (x > 0.5).float()  # 将概率大于0.5的像素值设为1，否则为0

        with torch.no_grad():
            if x.ndim != y.ndim:
                y = y.view((y.shape[0], 1, *y.shape[1:]))

        # # this one MUST be outside the with torch.no_grad(): context. Otherwise no gradients for you
        # if not self.do_bg:
        #     x = x[:, 1:]  # shape 2,1,200,1024,1024

        # skelet gt-pred
        with torch.no_grad():
            skel_gt = self.soft_skeletonize(
                y)  # float of binary tensor, otherwise error for boolean values in soft_skeletonize
        diff_skel_gt_minus_pred = skel_gt - pred_binary
        diff_skel_gt_minus_pred[diff_skel_gt_minus_pred < 0] = 0  # 2,1,200,1024,1024,

        # skelet gt-pred
        with torch.no_grad():
            skel_pred = self.soft_skeletonize(x)
        diff_skel_pred_minus_gt = skel_pred - y
        diff_skel_pred_minus_gt[diff_skel_pred_minus_gt < 0] = 0  # 2,1,200,1024,1024,

        # well predicted for gt&pred skelet
        well_pred_gt_skel = skel_gt - diff_skel_gt_minus_pred
        well_pred_gt_skel[well_pred_gt_skel < 0] = 0
        well_preds_pred_skel = skel_pred - diff_skel_pred_minus_gt
        well_preds_pred_skel[well_pred_gt_skel < 0] = 0

        # find context pxl around crit pxls in gt&pred skel
        cp_diff_skel_gt_minus_pred_gt_skel = cp.asarray(diff_skel_gt_minus_pred.cpu().detach().numpy())
        cp_well_pred_skel_gt_skel = cp.asarray(well_pred_gt_skel.cpu().detach().numpy())

        cp_diff_skel_pred_minus_gt_pred_skel = cp.asarray(diff_skel_pred_minus_gt.cpu().detach().numpy())
        cp_well_pred_skel_pred_skel = cp.asarray(well_preds_pred_skel.cpu().detach().numpy())

        # prepare empty storage
        cp_dist_crit_pxls_gt_skel = cp.empty_like(cp_diff_skel_gt_minus_pred_gt_skel)
        cp_dist_already_good_pxls_gt_skel = cp.empty_like(cp_well_pred_skel_gt_skel)

        cp_dist_crit_pxls_pred_skel = cp.empty_like(cp_diff_skel_pred_minus_gt_pred_skel)
        cp_dist_already_good_pxls_pred_skel = cp.empty_like(cp_well_pred_skel_pred_skel)

        for i in range(x.shape[0]):
            # distance transforms gt skel
            cp_dist_crit_pxls_gt_skel[i, 0] = morphology.distance_transform_edt(
                1 - cp_diff_skel_gt_minus_pred_gt_skel[i, 0])  # morph return shape: 200,1024,1024
            cp_dist_already_good_pxls_gt_skel[i, 0] = morphology.distance_transform_edt(
                1 - cp_well_pred_skel_gt_skel[i, 0])  # morph return shape: 200,1024,1024

            # distance transforms pred skel
            cp_dist_crit_pxls_pred_skel[i, 0] = morphology.distance_transform_edt(
                1 - cp_diff_skel_pred_minus_gt_pred_skel[i, 0])  # morph return shape: 200,1024,1024
            cp_dist_already_good_pxls_pred_skel[i, 0] = morphology.distance_transform_edt(
                1 - cp_well_pred_skel_pred_skel[i, 0])  # morph return shape: 200,1024,1024

        # compare distances to find context pxl around gt&pred skeletons
        all_pxl_nearest_critical_pxls_gt_skel = cp.where(cp_dist_already_good_pxls_gt_skel > cp_dist_crit_pxls_gt_skel,
                                                         1, 0)
        all_pxl_nearest_critical_pxls_pred_skel = cp.where(
            cp_dist_already_good_pxls_pred_skel > cp_dist_crit_pxls_pred_skel, 1, 0)

        cp_y = cp.asarray(y.cpu().detach().numpy())
        cp_gt_focused_around_crit_pxls_gt_skel = all_pxl_nearest_critical_pxls_gt_skel * cp_y

        # where foreground class dominant
        cp_pred_binary = cp.asarray(pred_binary.cpu().detach().numpy())
        cp_pred_focused_around_crit_pxls_pred_skel = all_pxl_nearest_critical_pxls_pred_skel * cp_pred_binary

        # convert to torch
        torch_gt_focused_around_crit_pxls_gt_skel = torch.from_numpy(
            cp_gt_focused_around_crit_pxls_gt_skel.get()).cuda()
        torch_pred_focused_around_crit_pxls_pred_skel = torch.from_numpy(
            cp_pred_focused_around_crit_pxls_pred_skel.get()).cuda()

        # # 统计 torch_gt_focused_around_crit_pxls_gt_skel 的信息
        # print(f"torch_gt_focused_around_crit_pxls_gt_skel 形状: {torch_gt_focused_around_crit_pxls_gt_skel.shape}")
        # print(f"torch_gt_focused_around_crit_pxls_gt_skel 数据类型: {torch_gt_focused_around_crit_pxls_gt_skel.dtype}")
        # print(
        #     f"torch_gt_focused_around_crit_pxls_gt_skel 中值为 1 的数量: {torch.sum(torch_gt_focused_around_crit_pxls_gt_skel == 1).item()}")
        # print(
        #     f"torch_gt_focused_around_crit_pxls_gt_skel 中值为 0 的数量: {torch.sum(torch_gt_focused_around_crit_pxls_gt_skel == 0).item()}")
        #
        # # 统计 torch_pred_focused_around_crit_pxls_pred_skel 的信息
        # print(
        #     f"torch_pred_focused_around_crit_pxls_pred_skel 形状: {torch_pred_focused_around_crit_pxls_pred_skel.shape}")
        # print(
        #     f"torch_pred_focused_around_crit_pxls_pred_skel 数据类型: {torch_pred_focused_around_crit_pxls_pred_skel.dtype}")
        # print(
        #     f"torch_pred_focused_around_crit_pxls_pred_skel 中值为 1 的数量: {torch.sum(torch_pred_focused_around_crit_pxls_pred_skel == 1).item()}")
        # print(
        #     f"torch_pred_focused_around_crit_pxls_pred_skel 中值为 0 的数量: {torch.sum(torch_pred_focused_around_crit_pxls_pred_skel == 0).item()}")

        # # Dice loss (soft dice calculation, similar to cldice but returns -dc analogue to MemoryEfficientSoftDiceLoss of nnunet)
        # tprec = (torch.sum(torch.multiply(torch_pred_focused_around_crit_pxls_pred_skel, y)) + self.smooth) / (
        #         torch.sum(torch_pred_focused_around_crit_pxls_pred_skel) + self.smooth)
        # tsens = (torch.sum(torch.multiply(torch_gt_focused_around_crit_pxls_gt_skel, x)) + self.smooth) / (
        #         torch.sum(torch_gt_focused_around_crit_pxls_gt_skel) + self.smooth)
        # beta_2 = self.beta ** 2
        # dice_result = -(1 + beta_2) * (tprec * tsens) / ((beta_2 * tprec) + tsens)

        # CE loss
        ce_loss = nn.BCELoss()
        all_crit_pxls = torch.logical_or(torch_pred_focused_around_crit_pxls_pred_skel,
                                         torch_gt_focused_around_crit_pxls_gt_skel)  # 2,1,200,1024,1024; could restrain to just gt pxls
        x_part = x_logits * all_crit_pxls  # shape x_logits is (2,2,200,1024,1024)*all_crit_pxls...(2,1,200,1024,1024)=(2,2,200,1024,1024) because shape broadcasting
        y_part = y * all_crit_pxls

        # # 统计 x_part 中值为 1 的元素数量
        # x_part_ones_count = torch.sum(x_part == 1)
        #
        # # 统计 y_part 中值为 1 的元素数量
        # y_part_ones_count = torch.sum(y_part == 1)

        # # 输出结果
        # print(f"x_part 中值为 1 的数量: {x_part_ones_count.item()}")
        # print(f"y_part 中值为 1 的数量: {y_part_ones_count.item()}")

        ce_result = ce_loss(x_part, y_part)
        # x_part 和 y_part 已经计算出来
        # 调用 soft_dice_loss 来计算损失
        dice_loss = self.soft_dice_loss(y_part, x_part)
        return  dice_loss + ce_result

