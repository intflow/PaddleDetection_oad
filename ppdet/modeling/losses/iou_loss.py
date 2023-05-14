# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import math
import paddle

from ppdet.core.workspace import register, serializable
from ..bbox_utils import bbox_iou
from copy import deepcopy

__all__ = ['IouLoss', 'GIoULoss', 'DIouLoss', 'SIoULoss', 'GDLoss_v1']
@register
@serializable
class GDLoss_v1(object):
    """Gaussian based loss.

    Args:
        loss_type (str):  Type of loss.
        fun (str, optional): The function applied to distance.
            Defaults to 'log1p'.
        tau (float, optional): Defaults to 1.0.
        reduction (str, optional): The reduction method of the
            loss. Defaults to 'mean'.
        loss_weight (float, optional): The weight of loss. Defaults to 1.0.

    Returns:
        loss (paddle.Tensor)
    """
    def __init__(self,
                 fun='sqrt',
                 tau=1.0,
                 reduction='mean',
                 loss_weight=1.0,
                 **kwargs):
        super(GDLoss_v1, self).__init__()
        assert reduction in ['none', 'sum', 'mean']
        assert fun in ['log1p', 'sqrt', '']
        self.loss = self.gwd_loss
        self.preprocess = self.xy_wh_r_2_xy_sigma
        self.fun = fun
        self.tau = tau
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.kwargs = kwargs

    def xy_wh_r_2_xy_sigma(self, xywhr):
        """Convert oriented bounding box to 2-D Gaussian distribution.

        Args:
            xywhr (paddle.Tensor): rbboxes with shape (N, 5).

        Returns:
            xy (paddle.Tensor): center point of 2-D Gaussian distribution
                with shape (N, 2).
            sigma (paddle.Tensor): covariance matrix of 2-D Gaussian distribution
                with shape (N, 2, 2).
        """
        _shape = xywhr.shape
        assert _shape[-1] == 5
        xy = xywhr[..., :2]
        wh = xywhr[..., 2:4].clip(min=1e-4, max=1e4).reshape([-1, 2])
        r = xywhr[..., 4]
        cos_r = paddle.cos(r)
        sin_r = paddle.sin(r)
        R = paddle.stack((cos_r, -sin_r, sin_r, cos_r), axis=-1).reshape([-1, 2, 2])
        S = 0.5 * paddle.nn.functional.diag_embed(wh)

        sigma = R.bmm(S.square()).bmm(R.transpose([0, 2, 1])).reshape(_shape[:-1] + [2, 2])

        return xy, sigma

    def gwd_loss(self, pred, target, fun='log1p', tau=2.0):
        """Gaussian Wasserstein distance loss.

        Args:
            pred (paddle.Tensor): Predicted bboxes.
            target (paddle.Tensor): Corresponding gt bboxes.
            fun (str): The function applied to distance. Defaults to 'log1p'.
            tau (float): Defaults to 1.0.

        Returns:
            loss (paddle.Tensor)
        """
        mu_p, sigma_p = pred
        mu_t, sigma_t = target

        xy_distance = (mu_p - mu_t).square().sum(axis=-1)

        whr_distance = sigma_p.diagonal(axis1=-2, axis2=-1).sum(axis=-1)
        whr_distance = whr_distance + sigma_t.diagonal(
            axis1=-2, axis2=-1).sum(axis=-1)

        _t_tr = (sigma_p.bmm(sigma_t)).diagonal(axis1=-2, axis2=-1).sum(axis=-1)
        _t_det_sqrt = (paddle.linalg.det(sigma_p) * paddle.linalg.det(sigma_t)).clip(1e-7).sqrt()
        whr_distance += (-2) * (_t_tr + 2 * _t_det_sqrt).clip(1e-7).sqrt()

        dis = xy_distance + whr_distance
        gwd_dis = dis.clip(min=1e-6)

        if fun == 'sqrt':
            loss = 1 - 1 / (tau + paddle.sqrt(gwd_dis))
        elif fun == 'log1p':
            loss = 1 - 1 / (tau + paddle.log1p(gwd_dis))
        else:
            scale = 2 * (_t_det_sqrt.sqrt().sqrt()).clamp(1e-7)
            loss = paddle.log1p(paddle.sqrt(gwd_dis) / scale)
        return loss


    def bcd_loss(self, pred, target, fun='log1p', tau=1.0):
        """Bhatacharyya distance loss.

        Args:
            pred (paddle.Tensor): Predicted bboxes.
            target (paddle.Tensor): Corresponding gt bboxes.
            fun (str): The function applied to distance. Defaults to 'log1p'.
            tau (float): Defaults to 1.0.

        Returns:
            loss (paddle.Tensor)
        """
        mu_p, sigma_p = pred
        mu_t, sigma_t = target

        mu_p = mu_p.reshape(-1, 2)
        mu_t = mu_t.reshape(-1, 2)
        sigma_p = sigma_p.reshape(-1, 2, 2)
        sigma_t = sigma_t.reshape(-1, 2, 2)

        delta = (mu_p - mu_t).unsqueeze(-1)
        sigma = 0.5 * (sigma_p + sigma_t)
        sigma_inv = paddle.inverse(sigma)

        term1 = paddle.log(
            paddle.det(sigma) /
            (paddle.sqrt(paddle.det(sigma_t.matmul(sigma_p))))).reshape(-1, 1)
        term2 = delta.transpose(-1, -2).matmul(sigma_inv).matmul(delta).squeeze(-1)
        dis = 0.5 * term1 + 0.125 * term2
        bcd_dis = dis.clamp(min=1e-6)

        if fun == 'sqrt':
            loss = 1 - 1 / (tau + paddle.sqrt(bcd_dis))
        elif fun == 'log1p':
            loss = 1 - 1 / (tau + paddle.log1p(bcd_dis))
        else:
            loss = 1 - 1 / (tau + bcd_dis)
        return loss


    def kld_loss(self, pred, target, fun='log1p', tau=1.0):
        """Kullback-Leibler Divergence loss.

        Args:
            pred (paddle.Tensor): Predicted bboxes.
            target (paddle.Tensor): Corresponding gt bboxes.
            fun (str): The function applied to distance. Defaults to 'log1p'.
            tau (float): Defaults to 1.0.

        Returns:
            loss (paddle.Tensor)
        """
        mu_p, sigma_p = pred
        mu_t, sigma_t = target

        mu_p = mu_p.reshape(-1, 2)
        mu_t = mu_t.reshape(-1, 2)
        sigma_p = sigma_p.reshape(-1, 2, 2)
        sigma_t = sigma_t.reshape(-1, 2, 2)

        delta = (mu_p - mu_t).unsqueeze(-1)
        sigma_t_inv = paddle.inverse(sigma_t)
        term1 = delta.transpose(-1,
                                -2).matmul(sigma_t_inv).matmul(delta).squeeze(-1)
        term2 = paddle.diagonal(
            sigma_t_inv.matmul(sigma_p),
            axis1=-2, axis2=-1).sum(axis=-1, keepaxis=True) + \
            paddle.log(paddle.det(sigma_t) / paddle.det(sigma_p)).reshape(-1, 1)
        dis = term1 + term2 - 2
        kl_dis = dis.clamp(min=1e-6)

        if fun == 'sqrt':
            kl_loss = 1 - 1 / (tau + paddle.sqrt(kl_dis))
        else:
            kl_loss = 1 - 1 / (tau + paddle.log1p(kl_dis))
        return kl_loss

    def __call__(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (paddle.Tensor): Predicted convexes.
            target (paddle.Tensor): Corresponding gt convexes.
            weight (paddle.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
               override the original reduction method of the loss.
               Defaults to None.
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not paddle.any(weight > 0)) and (
                reduction != 'none'):
            return (pred * weight).sum()
        if weight is not None and weight.dim() > 1:
            assert weight.shape == pred.shape
            weight = weight.mean(-1)
        _kwargs = deepcopy(self.kwargs)
        _kwargs.update(kwargs)

        ##mask = (weight > 0).detach()
        ##pred = pred[mask]
        ##target = target[mask]
        pred = self.preprocess(pred)
        target = self.preprocess(target)

        return self.loss(
            pred, target, fun=self.fun, tau=self.tau, **
            _kwargs) * self.loss_weight

@register
@serializable
class IouLoss(object):
    """
    iou loss, see https://arxiv.org/abs/1908.03851
    loss = 1.0 - iou * iou
    Args:
        loss_weight (float): iou loss weight, default is 2.5
        max_height (int): max height of input to support random shape input
        max_width (int): max width of input to support random shape input
        ciou_term (bool): whether to add ciou_term
        loss_square (bool): whether to square the iou term
    """

    def __init__(self,
                 loss_weight=2.5,
                 giou=False,
                 diou=False,
                 ciou=False,
                 loss_square=True):
        self.loss_weight = loss_weight
        self.giou = giou
        self.diou = diou
        self.ciou = ciou
        self.loss_square = loss_square

    def __call__(self, pbox, gbox):
        iou = bbox_iou(
            pbox, gbox, giou=self.giou, diou=self.diou, ciou=self.ciou)
        if self.loss_square:
            loss_iou = 1 - iou * iou
        else:
            loss_iou = 1 - iou

        loss_iou = loss_iou * self.loss_weight
        return loss_iou


@register
@serializable
class GIoULoss(object):
    """
    Generalized Intersection over Union, see https://arxiv.org/abs/1902.09630
    Args:
        loss_weight (float): giou loss weight, default as 1
        eps (float): epsilon to avoid divide by zero, default as 1e-10
        reduction (string): Options are "none", "mean" and "sum". default as none
    """

    def __init__(self, loss_weight=1., eps=1e-10, reduction='none'):
        self.loss_weight = loss_weight
        self.eps = eps
        assert reduction in ('none', 'mean', 'sum')
        self.reduction = reduction

    def bbox_overlap(self, box1, box2, eps=1e-10):
        """calculate the iou of box1 and box2
        Args:
            box1 (Tensor): box1 with the shape (..., 4)
            box2 (Tensor): box1 with the shape (..., 4)
            eps (float): epsilon to avoid divide by zero
        Return:
            iou (Tensor): iou of box1 and box2
            overlap (Tensor): overlap of box1 and box2
            union (Tensor): union of box1 and box2
        """
        x1, y1, x2, y2 = box1
        x1g, y1g, x2g, y2g = box2

        xkis1 = paddle.maximum(x1, x1g)
        ykis1 = paddle.maximum(y1, y1g)
        xkis2 = paddle.minimum(x2, x2g)
        ykis2 = paddle.minimum(y2, y2g)
        w_inter = (xkis2 - xkis1).clip(0)
        h_inter = (ykis2 - ykis1).clip(0)
        overlap = w_inter * h_inter

        area1 = (x2 - x1) * (y2 - y1)
        area2 = (x2g - x1g) * (y2g - y1g)
        union = area1 + area2 - overlap + eps
        iou = overlap / union

        return iou, overlap, union

    def __call__(self, pbox, gbox, iou_weight=1., loc_reweight=None):
        x1, y1, x2, y2 = paddle.split(pbox, num_or_sections=4, axis=-1)
        x1g, y1g, x2g, y2g = paddle.split(gbox, num_or_sections=4, axis=-1)
        box1 = [x1, y1, x2, y2]
        box2 = [x1g, y1g, x2g, y2g]
        iou, overlap, union = self.bbox_overlap(box1, box2, self.eps)
        xc1 = paddle.minimum(x1, x1g)
        yc1 = paddle.minimum(y1, y1g)
        xc2 = paddle.maximum(x2, x2g)
        yc2 = paddle.maximum(y2, y2g)

        area_c = (xc2 - xc1) * (yc2 - yc1) + self.eps
        miou = iou - ((area_c - union) / area_c)
        if loc_reweight is not None:
            loc_reweight = paddle.reshape(loc_reweight, shape=(-1, 1))
            loc_thresh = 0.9
            giou = 1 - (1 - loc_thresh
                        ) * miou - loc_thresh * miou * loc_reweight
        else:
            giou = 1 - miou
        if self.reduction == 'none':
            loss = giou
        elif self.reduction == 'sum':
            loss = paddle.sum(giou * iou_weight)
        else:
            loss = paddle.mean(giou * iou_weight)
        return loss * self.loss_weight


@register
@serializable
class DIouLoss(GIoULoss):
    """
    Distance-IoU Loss, see https://arxiv.org/abs/1911.08287
    Args:
        loss_weight (float): giou loss weight, default as 1
        eps (float): epsilon to avoid divide by zero, default as 1e-10
        use_complete_iou_loss (bool): whether to use complete iou loss
    """

    def __init__(self, loss_weight=1., eps=1e-10, use_complete_iou_loss=True):
        super(DIouLoss, self).__init__(loss_weight=loss_weight, eps=eps)
        self.use_complete_iou_loss = use_complete_iou_loss

    def __call__(self, pbox, gbox, iou_weight=1.):
        x1, y1, x2, y2 = paddle.split(pbox, num_or_sections=4, axis=-1)
        x1g, y1g, x2g, y2g = paddle.split(gbox, num_or_sections=4, axis=-1)
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1

        cxg = (x1g + x2g) / 2
        cyg = (y1g + y2g) / 2
        wg = x2g - x1g
        hg = y2g - y1g

        x2 = paddle.maximum(x1, x2)
        y2 = paddle.maximum(y1, y2)

        # A and B
        xkis1 = paddle.maximum(x1, x1g)
        ykis1 = paddle.maximum(y1, y1g)
        xkis2 = paddle.minimum(x2, x2g)
        ykis2 = paddle.minimum(y2, y2g)

        # A or B
        xc1 = paddle.minimum(x1, x1g)
        yc1 = paddle.minimum(y1, y1g)
        xc2 = paddle.maximum(x2, x2g)
        yc2 = paddle.maximum(y2, y2g)

        intsctk = (xkis2 - xkis1) * (ykis2 - ykis1)
        intsctk = intsctk * paddle.greater_than(
            xkis2, xkis1) * paddle.greater_than(ykis2, ykis1)
        unionk = (x2 - x1) * (y2 - y1) + (x2g - x1g) * (y2g - y1g
                                                        ) - intsctk + self.eps
        iouk = intsctk / unionk

        # DIOU term
        dist_intersection = (cx - cxg) * (cx - cxg) + (cy - cyg) * (cy - cyg)
        dist_union = (xc2 - xc1) * (xc2 - xc1) + (yc2 - yc1) * (yc2 - yc1)
        diou_term = (dist_intersection + self.eps) / (dist_union + self.eps)

        # CIOU term
        ciou_term = 0
        if self.use_complete_iou_loss:
            ar_gt = wg / hg
            ar_pred = w / h
            arctan = paddle.atan(ar_gt) - paddle.atan(ar_pred)
            ar_loss = 4. / np.pi / np.pi * arctan * arctan
            alpha = ar_loss / (1 - iouk + ar_loss + self.eps)
            alpha.stop_gradient = True
            ciou_term = alpha * ar_loss

        diou = paddle.mean((1 - iouk + ciou_term + diou_term) * iou_weight)

        return diou * self.loss_weight


@register
@serializable
class SIoULoss(GIoULoss):
    """
    see https://arxiv.org/pdf/2205.12740.pdf 
    Args:
        loss_weight (float): siou loss weight, default as 1
        eps (float): epsilon to avoid divide by zero, default as 1e-10
        theta (float): default as 4
        reduction (str): Options are "none", "mean" and "sum". default as none
    """

    def __init__(self, loss_weight=1., eps=1e-10, theta=4., reduction='none'):
        super(SIoULoss, self).__init__(loss_weight=loss_weight, eps=eps)
        self.loss_weight = loss_weight
        self.eps = eps
        self.theta = theta
        self.reduction = reduction

    def __call__(self, pbox, gbox):
        x1, y1, x2, y2 = paddle.split(pbox, num_or_sections=4, axis=-1)
        x1g, y1g, x2g, y2g = paddle.split(gbox, num_or_sections=4, axis=-1)

        box1 = [x1, y1, x2, y2]
        box2 = [x1g, y1g, x2g, y2g]
        iou = bbox_iou(box1, box2)

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1 + self.eps
        h = y2 - y1 + self.eps

        cxg = (x1g + x2g) / 2
        cyg = (y1g + y2g) / 2
        wg = x2g - x1g + self.eps
        hg = y2g - y1g + self.eps

        x2 = paddle.maximum(x1, x2)
        y2 = paddle.maximum(y1, y2)

        # A or B
        xc1 = paddle.minimum(x1, x1g)
        yc1 = paddle.minimum(y1, y1g)
        xc2 = paddle.maximum(x2, x2g)
        yc2 = paddle.maximum(y2, y2g)

        cw_out = xc2 - xc1
        ch_out = yc2 - yc1

        ch = paddle.maximum(cy, cyg) - paddle.minimum(cy, cyg)
        cw = paddle.maximum(cx, cxg) - paddle.minimum(cx, cxg)

        # angle cost
        dist_intersection = paddle.sqrt((cx - cxg)**2 + (cy - cyg)**2)
        sin_angle_alpha = ch / dist_intersection
        sin_angle_beta = cw / dist_intersection
        thred = paddle.pow(paddle.to_tensor(2), 0.5) / 2
        thred.stop_gradient = True
        sin_alpha = paddle.where(sin_angle_alpha > thred, sin_angle_beta,
                                 sin_angle_alpha)
        angle_cost = paddle.cos(paddle.asin(sin_alpha) * 2 - math.pi / 2)

        # distance cost
        gamma = 2 - angle_cost
        # gamma.stop_gradient = True
        beta_x = ((cxg - cx) / cw_out)**2
        beta_y = ((cyg - cy) / ch_out)**2
        dist_cost = 1 - paddle.exp(-gamma * beta_x) + 1 - paddle.exp(-gamma *
                                                                     beta_y)

        # shape cost
        omega_w = paddle.abs(w - wg) / paddle.maximum(w, wg)
        omega_h = paddle.abs(hg - h) / paddle.maximum(h, hg)
        omega = (1 - paddle.exp(-omega_w))**self.theta + (
            1 - paddle.exp(-omega_h))**self.theta
        siou_loss = 1 - iou + (omega + dist_cost) / 2

        if self.reduction == 'mean':
            siou_loss = paddle.mean(siou_loss)
        elif self.reduction == 'sum':
            siou_loss = paddle.sum(siou_loss)

        return siou_loss * self.loss_weight
