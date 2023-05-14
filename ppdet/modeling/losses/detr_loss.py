# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

import paddle
import paddle.nn as nn
import paddle.nn.functional as F
from ppdet.core.workspace import register
from .iou_loss import GIoULoss, GDLoss_v1
from ..transformers import bbox_cxcywh_to_xyxy, sigmoid_focal_loss, varifocal_loss_with_logits
from ..bbox_utils import bbox_iou
from ext_op import rbox_iou, matched_rbox_iou

__all__ = ['DETRLoss', 'DINOLoss', 'DETRLoss_oad', 'DINOLoss_oad', 'DINOLoss_oadkpt']


@register
class DETRLoss(nn.Layer):
    __shared__ = ['num_classes', 'use_focal_loss']
    __inject__ = ['matcher']

    def __init__(self,
                 num_classes=80,
                 matcher='HungarianMatcher',
                 loss_coeff={
                     'class': 1,
                     'bbox': 5,
                     'giou': 2,
                     'no_object': 0.1,
                     'mask': 1,
                     'dice': 1
                 },
                 aux_loss=True,
                 use_focal_loss=False,
                 use_vfl=False,
                 use_uni_match=False,
                 uni_match_ind=0):
        r"""
        Args:
            num_classes (int): The number of classes.
            matcher (HungarianMatcher): It computes an assignment between the targets
                and the predictions of the network.
            loss_coeff (dict): The coefficient of loss.
            aux_loss (bool): If 'aux_loss = True', loss at each decoder layer are to be used.
            use_focal_loss (bool): Use focal loss or not.
        """
        super(DETRLoss, self).__init__()

        self.num_classes = num_classes
        self.matcher = matcher
        self.loss_coeff = loss_coeff
        self.aux_loss = aux_loss
        self.use_focal_loss = use_focal_loss
        self.use_vfl = use_vfl
        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind

        if not self.use_focal_loss:
            self.loss_coeff['class'] = paddle.full([num_classes + 1],
                                                   loss_coeff['class'])
            self.loss_coeff['class'][-1] = loss_coeff['no_object']
        self.giou_loss = GIoULoss()

    def _get_loss_class(self,
                        logits,
                        gt_class,
                        match_indices,
                        bg_index,
                        num_gts,
                        postfix="",
                        iou_score=None):
        # logits: [b, query, num_classes], gt_class: list[[n, 1]]
        name_class = "loss_class" + postfix

        target_label = paddle.full(logits.shape[:2], bg_index, dtype='int64')
        bs, num_query_objects = target_label.shape
        num_gt = sum(len(a) for a in gt_class)
        if num_gt > 0:
            index, updates = self._get_index_updates(num_query_objects,
                                                     gt_class, match_indices)
            target_label = paddle.scatter(
                target_label.reshape([-1, 1]), index, updates.astype('int64'))
            target_label = target_label.reshape([bs, num_query_objects])
        if self.use_focal_loss:
            target_label = F.one_hot(target_label,
                                     self.num_classes + 1)[..., :-1]
            if iou_score is not None and self.use_vfl:
                target_score = paddle.zeros([bs, num_query_objects])
                if num_gt > 0:
                    target_score = paddle.scatter(
                        target_score.reshape([-1, 1]), index, iou_score)
                target_score = target_score.reshape(
                    [bs, num_query_objects, 1]) * target_label
                loss_ = self.loss_coeff['class'] * varifocal_loss_with_logits(
                    logits, target_score, target_label,
                    num_gts / num_query_objects)
            else:
                loss_ = self.loss_coeff['class'] * sigmoid_focal_loss(
                    logits, target_label, num_gts / num_query_objects)
        else:
            loss_ = F.cross_entropy(
                logits, target_label, weight=self.loss_coeff['class'])
        return {name_class: loss_}

    def _get_loss_bbox(self, boxes, gt_bbox, match_indices, num_gts,
                       postfix=""):
        # boxes: [b, query, 4], gt_bbox: list[[n, 4]]
        name_bbox = "loss_bbox" + postfix
        name_giou = "loss_giou" + postfix

        loss = dict()
        if sum(len(a) for a in gt_bbox) == 0:
            loss[name_bbox] = paddle.to_tensor([0.])
            loss[name_giou] = paddle.to_tensor([0.])
            return loss

        src_bbox, target_bbox = self._get_src_target_assign(boxes, gt_bbox,
                                                            match_indices)
        loss[name_bbox] = self.loss_coeff['bbox'] * F.l1_loss(
            src_bbox, target_bbox, reduction='sum') / num_gts
        loss[name_giou] = self.giou_loss(
            bbox_cxcywh_to_xyxy(src_bbox), bbox_cxcywh_to_xyxy(target_bbox))
        loss[name_giou] = loss[name_giou].sum() / num_gts
        loss[name_giou] = self.loss_coeff['giou'] * loss[name_giou]
        return loss

    def _get_loss_mask(self, masks, gt_mask, match_indices, num_gts,
                       postfix=""):
        # masks: [b, query, h, w], gt_mask: list[[n, H, W]]
        name_mask = "loss_mask" + postfix
        name_dice = "loss_dice" + postfix

        loss = dict()
        if sum(len(a) for a in gt_mask) == 0:
            loss[name_mask] = paddle.to_tensor([0.])
            loss[name_dice] = paddle.to_tensor([0.])
            return loss

        src_masks, target_masks = self._get_src_target_assign(masks, gt_mask,
                                                              match_indices)
        src_masks = F.interpolate(
            src_masks.unsqueeze(0),
            size=target_masks.shape[-2:],
            mode="bilinear")[0]
        loss[name_mask] = self.loss_coeff['mask'] * F.sigmoid_focal_loss(
            src_masks,
            target_masks,
            paddle.to_tensor(
                [num_gts], dtype='float32'))
        loss[name_dice] = self.loss_coeff['dice'] * self._dice_loss(
            src_masks, target_masks, num_gts)
        return loss

    def _dice_loss(self, inputs, targets, num_gts):
        inputs = F.sigmoid(inputs)
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_gts

    def _get_loss_aux(self,
                      boxes,
                      logits,
                      gt_bbox,
                      gt_class,
                      bg_index,
                      num_gts,
                      dn_match_indices=None,
                      postfix="",
                      masks=None,
                      gt_mask=None):
        loss_class = []
        loss_bbox, loss_giou = [], []
        loss_mask, loss_dice = [], []
        if dn_match_indices is not None:
            match_indices = dn_match_indices
        elif self.use_uni_match:
            match_indices = self.matcher(
                boxes[self.uni_match_ind],
                logits[self.uni_match_ind],
                gt_bbox,
                gt_class,
                masks=masks[self.uni_match_ind] if masks is not None else None,
                gt_mask=gt_mask)
        for i, (aux_boxes, aux_logits) in enumerate(zip(boxes, logits)):
            aux_masks = masks[i] if masks is not None else None
            if not self.use_uni_match and dn_match_indices is None:
                match_indices = self.matcher(
                    aux_boxes,
                    aux_logits,
                    gt_bbox,
                    gt_class,
                    masks=aux_masks,
                    gt_mask=gt_mask)
            if self.use_vfl:
                if sum(len(a) for a in gt_bbox) > 0:
                    src_bbox, target_bbox = self._get_src_target_assign(
                        aux_boxes.detach(), gt_bbox, match_indices)
                    iou_score = bbox_iou(
                        bbox_cxcywh_to_xyxy(src_bbox).split(4, -1),
                        bbox_cxcywh_to_xyxy(target_bbox).split(4, -1))
                else:
                    iou_score = None
            else:
                iou_score = None
            loss_class.append(
                self._get_loss_class(aux_logits, gt_class, match_indices,
                                     bg_index, num_gts, postfix, iou_score)[
                                         'loss_class' + postfix])
            loss_ = self._get_loss_bbox(aux_boxes, gt_bbox, match_indices,
                                        num_gts, postfix)
            loss_bbox.append(loss_['loss_bbox' + postfix])
            loss_giou.append(loss_['loss_giou' + postfix])
            if masks is not None and gt_mask is not None:
                loss_ = self._get_loss_mask(aux_masks, gt_mask, match_indices,
                                            num_gts, postfix)
                loss_mask.append(loss_['loss_mask' + postfix])
                loss_dice.append(loss_['loss_dice' + postfix])
        loss = {
            "loss_class_aux" + postfix: paddle.add_n(loss_class),
            "loss_bbox_aux" + postfix: paddle.add_n(loss_bbox),
            "loss_giou_aux" + postfix: paddle.add_n(loss_giou)
        }
        if masks is not None and gt_mask is not None:
            loss["loss_mask_aux" + postfix] = paddle.add_n(loss_mask)
            loss["loss_dice_aux" + postfix] = paddle.add_n(loss_dice)
        return loss

    def _get_index_updates(self, num_query_objects, target, match_indices):
        batch_idx = paddle.concat([
            paddle.full_like(src, i) for i, (src, _) in enumerate(match_indices)
        ])
        src_idx = paddle.concat([src for (src, _) in match_indices])
        src_idx += (batch_idx * num_query_objects)
        target_assign = paddle.concat([
            paddle.gather(
                t, dst, axis=0) for t, (_, dst) in zip(target, match_indices)
        ])
        return src_idx, target_assign

    def _get_src_target_assign(self, src, target, match_indices):
        src_assign = paddle.concat([
            paddle.gather(
                t, I, axis=0) if len(I) > 0 else paddle.zeros([0, t.shape[-1]])
            for t, (I, _) in zip(src, match_indices)
        ])
        target_assign = paddle.concat([
            paddle.gather(
                t, J, axis=0) if len(J) > 0 else paddle.zeros([0, t.shape[-1]])
            for t, (_, J) in zip(target, match_indices)
        ])
        return src_assign, target_assign

    def _get_num_gts(self, targets, dtype="float32"):
        num_gts = sum(len(a) for a in targets)
        num_gts = paddle.to_tensor([num_gts], dtype=dtype)
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.all_reduce(num_gts)
            num_gts /= paddle.distributed.get_world_size()
        num_gts = paddle.clip(num_gts, min=1.)
        return num_gts

    def _get_prediction_loss(self,
                             boxes,
                             logits,
                             gt_bbox,
                             gt_class,
                             masks=None,
                             gt_mask=None,
                             postfix="",
                             dn_match_indices=None,
                             num_gts=1):
        if dn_match_indices is None:
            match_indices = self.matcher(
                boxes, logits, gt_bbox, gt_class, masks=masks, gt_mask=gt_mask)
        else:
            match_indices = dn_match_indices

        if self.use_vfl:
            if sum(len(a) for a in gt_bbox) > 0:
                src_bbox, target_bbox = self._get_src_target_assign(
                    boxes.detach(), gt_bbox, match_indices)
                iou_score = bbox_iou(
                    bbox_cxcywh_to_xyxy(src_bbox).split(4, -1),
                    bbox_cxcywh_to_xyxy(target_bbox).split(4, -1))
            else:
                iou_score = None
        else:
            iou_score = None

        loss = dict()
        loss.update(
            self._get_loss_class(logits, gt_class, match_indices,
                                 self.num_classes, num_gts, postfix, iou_score))
        loss.update(
            self._get_loss_bbox(boxes, gt_bbox, match_indices, num_gts,
                                postfix))
        if masks is not None and gt_mask is not None:
            loss.update(
                self._get_loss_mask(masks, gt_mask, match_indices, num_gts,
                                    postfix))
        return loss

    def forward(self,
                boxes,
                logits,
                gt_bbox,
                gt_class,
                masks=None,
                gt_mask=None,
                postfix="",
                **kwargs):
        r"""
        Args:
            boxes (Tensor): [l, b, query, 4]
            logits (Tensor): [l, b, query, num_classes]
            gt_bbox (List(Tensor)): list[[n, 4]]
            gt_class (List(Tensor)): list[[n, 1]]
            masks (Tensor, optional): [l, b, query, h, w]
            gt_mask (List(Tensor), optional): list[[n, H, W]]
            postfix (str): postfix of loss name
        """

        dn_match_indices = kwargs.get("dn_match_indices", None)
        num_gts = kwargs.get("num_gts", None)
        if num_gts is None:
            num_gts = self._get_num_gts(gt_class)

        total_loss = self._get_prediction_loss(
            boxes[-1],
            logits[-1],
            gt_bbox,
            gt_class,
            masks=masks[-1] if masks is not None else None,
            gt_mask=gt_mask,
            postfix=postfix,
            dn_match_indices=dn_match_indices,
            num_gts=num_gts)

        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(
                    boxes[:-1],
                    logits[:-1],
                    gt_bbox,
                    gt_class,
                    self.num_classes,
                    num_gts,
                    dn_match_indices,
                    postfix,
                    masks=masks[:-1] if masks is not None else None,
                    gt_mask=gt_mask))

        return total_loss


@register
class DINOLoss(DETRLoss):
    def forward(self,
                boxes,
                logits,
                gt_bbox,
                gt_class,
                masks=None,
                gt_mask=None,
                postfix="",
                dn_out_bboxes=None,
                dn_out_logits=None,
                dn_meta=None,
                **kwargs):
        num_gts = self._get_num_gts(gt_class)
        total_loss = super(DINOLoss, self).forward(
            boxes, logits, gt_bbox, gt_class, num_gts=num_gts)

        if dn_meta is not None:
            dn_positive_idx, dn_num_group = \
                dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
            assert len(gt_class) == len(dn_positive_idx)

            # denoising match indices
            dn_match_indices = self.get_dn_match_indices(
                gt_class, dn_positive_idx, dn_num_group)

            # compute denoising training loss
            num_gts *= dn_num_group
            dn_loss = super(DINOLoss, self).forward(
                dn_out_bboxes,
                dn_out_logits,
                gt_bbox,
                gt_class,
                postfix="_dn",
                dn_match_indices=dn_match_indices,
                num_gts=num_gts)
            total_loss.update(dn_loss)
        else:
            total_loss.update(
                {k + '_dn': paddle.to_tensor([0.])
                 for k in total_loss.keys()})

        return total_loss

    @staticmethod
    def get_dn_match_indices(labels, dn_positive_idx, dn_num_group):
        dn_match_indices = []
        for i in range(len(labels)):
            num_gt = len(labels[i])
            if num_gt > 0:
                gt_idx = paddle.arange(end=num_gt, dtype="int64")
                gt_idx = gt_idx.tile([dn_num_group])
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((paddle.zeros(
                    [0], dtype="int64"), paddle.zeros(
                        [0], dtype="int64")))
        return dn_match_indices


@register
class MaskDINOLoss(DETRLoss):
    __shared__ = ['num_classes', 'use_focal_loss', 'num_sample_points']
    __inject__ = ['matcher']

    def __init__(self,
                 num_classes=80,
                 matcher='HungarianMatcher',
                 loss_coeff={
                     'class': 4,
                     'bbox': 5,
                     'giou': 2,
                     'mask': 5,
                     'dice': 5
                 },
                 aux_loss=True,
                 use_focal_loss=False,
                 num_sample_points=12544,
                 oversample_ratio=3.0,
                 important_sample_ratio=0.75):
        super(MaskDINOLoss, self).__init__(num_classes, matcher, loss_coeff,
                                           aux_loss, use_focal_loss)
        assert oversample_ratio >= 1
        assert important_sample_ratio <= 1 and important_sample_ratio >= 0

        self.num_sample_points = num_sample_points
        self.oversample_ratio = oversample_ratio
        self.important_sample_ratio = important_sample_ratio
        self.num_oversample_points = int(num_sample_points * oversample_ratio)
        self.num_important_points = int(num_sample_points *
                                        important_sample_ratio)
        self.num_random_points = num_sample_points - self.num_important_points

    def forward(self,
                boxes,
                logits,
                gt_bbox,
                gt_class,
                masks=None,
                gt_mask=None,
                postfix="",
                dn_out_bboxes=None,
                dn_out_logits=None,
                dn_out_masks=None,
                dn_meta=None,
                **kwargs):
        num_gts = self._get_num_gts(gt_class)
        total_loss = super(MaskDINOLoss, self).forward(
            boxes,
            logits,
            gt_bbox,
            gt_class,
            masks=masks,
            gt_mask=gt_mask,
            num_gts=num_gts)

        if dn_meta is not None:
            dn_positive_idx, dn_num_group = \
                dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
            assert len(gt_class) == len(dn_positive_idx)

            # denoising match indices
            dn_match_indices = DINOLoss.get_dn_match_indices(
                gt_class, dn_positive_idx, dn_num_group)

            # compute denoising training loss
            num_gts *= dn_num_group
            dn_loss = super(MaskDINOLoss, self).forward(
                dn_out_bboxes,
                dn_out_logits,
                gt_bbox,
                gt_class,
                masks=dn_out_masks,
                gt_mask=gt_mask,
                postfix="_dn",
                dn_match_indices=dn_match_indices,
                num_gts=num_gts)
            total_loss.update(dn_loss)
        else:
            total_loss.update(
                {k + '_dn': paddle.to_tensor([0.])
                 for k in total_loss.keys()})

        return total_loss

    def _get_loss_mask(self, masks, gt_mask, match_indices, num_gts,
                       postfix=""):
        # masks: [b, query, h, w], gt_mask: list[[n, H, W]]
        name_mask = "loss_mask" + postfix
        name_dice = "loss_dice" + postfix

        loss = dict()
        if sum(len(a) for a in gt_mask) == 0:
            loss[name_mask] = paddle.to_tensor([0.])
            loss[name_dice] = paddle.to_tensor([0.])
            return loss

        src_masks, target_masks = self._get_src_target_assign(masks, gt_mask,
                                                              match_indices)
        # sample points
        sample_points = self._get_point_coords_by_uncertainty(src_masks)
        sample_points = 2.0 * sample_points.unsqueeze(1) - 1.0

        src_masks = F.grid_sample(
            src_masks.unsqueeze(1), sample_points,
            align_corners=False).squeeze([1, 2])

        target_masks = F.grid_sample(
            target_masks.unsqueeze(1), sample_points,
            align_corners=False).squeeze([1, 2]).detach()

        loss[name_mask] = self.loss_coeff[
            'mask'] * F.binary_cross_entropy_with_logits(
                src_masks, target_masks,
                reduction='none').mean(1).sum() / num_gts
        loss[name_dice] = self.loss_coeff['dice'] * self._dice_loss(
            src_masks, target_masks, num_gts)
        return loss

    def _get_point_coords_by_uncertainty(self, masks):
        # Sample points based on their uncertainty.
        masks = masks.detach()
        num_masks = masks.shape[0]
        sample_points = paddle.rand(
            [num_masks, 1, self.num_oversample_points, 2])

        out_mask = F.grid_sample(
            masks.unsqueeze(1), 2.0 * sample_points - 1.0,
            align_corners=False).squeeze([1, 2])
        out_mask = -paddle.abs(out_mask)

        _, topk_ind = paddle.topk(out_mask, self.num_important_points, axis=1)
        batch_ind = paddle.arange(end=num_masks, dtype=topk_ind.dtype)
        batch_ind = batch_ind.unsqueeze(-1).tile([1, self.num_important_points])
        topk_ind = paddle.stack([batch_ind, topk_ind], axis=-1)

        sample_points = paddle.gather_nd(sample_points.squeeze(1), topk_ind)
        if self.num_random_points > 0:
            sample_points = paddle.concat(
                [
                    sample_points,
                    paddle.rand([num_masks, self.num_random_points, 2])
                ],
                axis=1)
        return sample_points


@register
class DETRLoss_oad(nn.Layer):
    __shared__ = ['num_classes', 'use_focal_loss']
    __inject__ = ['matcher']

    def __init__(self,
                 num_classes=80,
                 matcher='HungarianMatcher_oad',
                 loss_coeff={
                     'class': 1,
                     'bbox': 5,
                     'giou': 2,
                     'rads': 1,
                     'no_object': 0.1,
                     'mask': 1,
                     'dice': 1
                 },
                 aux_loss=True,
                 use_focal_loss=False,
                 use_vfl=False,
                 use_uni_match=False,
                 uni_match_ind=0):
        r"""
        Args:
            num_classes (int): The number of classes.
            matcher (HungarianMatcher): It computes an assignment between the targets
                and the predictions of the network.
            loss_coeff (dict): The coefficient of loss.
            aux_loss (bool): If 'aux_loss = True', loss at each decoder layer are to be used.
            use_focal_loss (bool): Use focal loss or not.
        """
        super(DETRLoss_oad, self).__init__()

        self.num_classes = num_classes
        self.matcher = matcher
        self.loss_coeff = loss_coeff
        self.aux_loss = aux_loss
        self.use_focal_loss = use_focal_loss
        self.use_vfl = use_vfl
        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind

        if not self.use_focal_loss:
            self.loss_coeff['class'] = paddle.full([num_classes + 1],
                                                   loss_coeff['class'])
            self.loss_coeff['class'][-1] = loss_coeff['no_object']
        self.giou_loss = GIoULoss()
        self.gwd_loss = GDLoss_v1()

    def _get_loss_class(self,
                        logits,
                        gt_class,
                        match_indices,
                        bg_index,
                        num_gts,
                        postfix="",
                        iou_score=None):
        # logits: [b, query, num_classes], gt_class: list[[n, 1]]
        name_class = "loss_class" + postfix

        target_label = paddle.full(logits.shape[:2], bg_index, dtype='int64')
        bs, num_query_objects = target_label.shape
        num_gt = sum(len(a) for a in gt_class)
        if num_gt > 0:
            index, updates = self._get_index_updates(num_query_objects,
                                                     gt_class, match_indices)
            target_label = paddle.scatter(
                target_label.reshape([-1, 1]), index, updates.astype('int64'))
            target_label = target_label.reshape([bs, num_query_objects])
        if self.use_focal_loss:
            target_label = F.one_hot(target_label,
                                     self.num_classes + 1)[..., :-1]
            if iou_score is not None and self.use_vfl:
                target_score = paddle.zeros([bs, num_query_objects])
                if num_gt > 0:
                    target_score = paddle.scatter(
                        target_score.reshape([-1, 1]), index, iou_score)
                target_score = target_score.reshape(
                    [bs, num_query_objects, 1]) * target_label
                loss_ = self.loss_coeff['class'] * varifocal_loss_with_logits(
                    logits, target_score, target_label,
                    num_gts / num_query_objects)
            else:
                loss_ = self.loss_coeff['class'] * sigmoid_focal_loss(
                    logits, target_label, num_gts / num_query_objects)
        else:
            loss_ = F.cross_entropy(
                logits, target_label, weight=self.loss_coeff['class'])
        return {name_class: loss_}

    def _get_loss_bbox(self, boxes, gt_bbox, match_indices, num_gts,
                       postfix=""):
        # boxes: [b, query, 4], gt_bbox: list[[n, 4]]
        name_bbox = "loss_bbox" + postfix
        name_giou = "loss_giou" + postfix

        loss = dict()
        if sum(len(a) for a in gt_bbox) == 0:
            loss[name_bbox] = paddle.to_tensor([0.])
            loss[name_giou] = paddle.to_tensor([0.])
            return loss

        src_bbox, target_bbox = self._get_src_target_assign(boxes, gt_bbox,
                                                            match_indices)
        loss[name_bbox] = self.loss_coeff['bbox'] * F.l1_loss(
            src_bbox, target_bbox, reduction='sum') / num_gts
        loss[name_giou] = self.giou_loss(
            bbox_cxcywh_to_xyxy(src_bbox), bbox_cxcywh_to_xyxy(target_bbox))
        loss[name_giou] = loss[name_giou].sum() / num_gts
        loss[name_giou] = self.loss_coeff['giou'] * loss[name_giou]
        return loss

    # ANCHOR: Don't use for now, performance drop detected
    def _get_loss_rbbox(self, boxes, gt_bbox, rads, gt_rad, match_indices, num_gts,
                       postfix=""):
        # boxes: [b, query, 4], gt_bbox: list[[n, 4]]
        name_bbox = "loss_bbox" + postfix
        name_giou = "loss_giou" + postfix

        loss = dict()
        if sum(len(a) for a in gt_bbox) == 0:
            loss[name_bbox] = paddle.to_tensor([0.])
            loss[name_giou] = paddle.to_tensor([0.])
            return loss 

        src_bbox, target_bbox = self._get_src_target_assign(boxes, gt_bbox,
                                                            match_indices)
        loss[name_bbox] = self.loss_coeff['bbox'] * F.l1_loss(
            src_bbox, target_bbox, reduction='sum') / num_gts
        # ANCHOR: Replace giou loss with GWD loss for rbboxes
        src_rad, target_rad = self._get_src_target_assign(rads, gt_rad,
                                                            match_indices)
        src_rbbox = paddle.concat([src_bbox, src_rad],axis=-1)
        target_rbbox = paddle.concat([target_bbox, target_rad],axis=-1)
        loss[name_giou] = self.gwd_loss(src_rbbox, target_rbbox)

        loss[name_giou] = loss[name_giou].sum() / num_gts
        loss[name_giou] = self.loss_coeff['giou'] * loss[name_giou]
        return loss

    def _get_loss_rad(self, rads, gt_rad, match_indices, num_gts,
                       postfix=""):
        # rads: [b, query, 1], gt_rad: list[[n, 1]]
        name_rad = "loss_rad" + postfix

        loss = dict()
        if sum(len(a) for a in gt_rad) == 0:
            loss[name_rad] = paddle.to_tensor([0.])
            return loss

        src_rad, target_rad = self._get_src_target_assign(rads, gt_rad,
                                                            match_indices)
        loss[name_rad] = self.loss_coeff['rad'] * \
        (F.smooth_l1_loss(paddle.cos(src_rad), paddle.cos(target_rad), reduction='sum') +
         F.smooth_l1_loss(paddle.sin(src_rad), paddle.sin(target_rad), reduction='sum')) / num_gts
        return loss
    
    def _get_loss_mask(self, masks, gt_mask, match_indices, num_gts,
                       postfix=""):
        # masks: [b, query, h, w], gt_mask: list[[n, H, W]]
        name_mask = "loss_mask" + postfix
        name_dice = "loss_dice" + postfix

        loss = dict()
        if sum(len(a) for a in gt_mask) == 0:
            loss[name_mask] = paddle.to_tensor([0.])
            loss[name_dice] = paddle.to_tensor([0.])
            return loss

        src_masks, target_masks = self._get_src_target_assign(masks, gt_mask,
                                                              match_indices)
        src_masks = F.interpolate(
            src_masks.unsqueeze(0),
            size=target_masks.shape[-2:],
            mode="bilinear")[0]
        loss[name_mask] = self.loss_coeff['mask'] * F.sigmoid_focal_loss(
            src_masks,
            target_masks,
            paddle.to_tensor(
                [num_gts], dtype='float32'))
        loss[name_dice] = self.loss_coeff['dice'] * self._dice_loss(
            src_masks, target_masks, num_gts)
        return loss

    def _dice_loss(self, inputs, targets, num_gts):
        inputs = F.sigmoid(inputs)
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_gts

    def _get_loss_aux(self,
                      boxes,
                      rads,
                      logits,
                      gt_bbox,
                      gt_rad,
                      gt_class,
                      bg_index,
                      num_gts,
                      dn_match_indices=None,
                      postfix="",
                      masks=None,
                      gt_mask=None):
        loss_class = []
        loss_bbox, loss_giou = [], []
        loss_rad = []
        loss_mask, loss_dice = [], []
        if dn_match_indices is not None:
            match_indices = dn_match_indices
        elif self.use_uni_match:
            match_indices = self.matcher(
                boxes[self.uni_match_ind],
                rads[self.uni_match_ind],
                logits[self.uni_match_ind],
                gt_bbox,
                gt_rad,
                gt_class,
                masks=masks[self.uni_match_ind] if masks is not None else None,
                gt_mask=gt_mask)
        for i, (aux_boxes, aux_rads, aux_logits) in enumerate(zip(boxes, rads, logits)):
            aux_masks = masks[i] if masks is not None else None
            if not self.use_uni_match and dn_match_indices is None:
                match_indices = self.matcher(
                    aux_boxes,
                    aux_rads,
                    aux_logits,
                    gt_bbox,
                    gt_rad,
                    gt_class,
                    masks=aux_masks,
                    gt_mask=gt_mask)
            if self.use_vfl:
                if sum(len(a) for a in gt_bbox) > 0:
                    src_bbox, target_bbox = self._get_src_target_assign(
                        aux_boxes.detach(), gt_bbox, match_indices)
                    iou_score = bbox_iou(
                        bbox_cxcywh_to_xyxy(src_bbox).split(4, -1),
                        bbox_cxcywh_to_xyxy(target_bbox).split(4, -1))
                
                    ### ANCHOR: use rbox_iou in ext_op to precisly calculate iou between two rbboxes
                    ##src_rad, target_rad = self._get_src_target_assign(
                    ##    aux_rads.detach(), gt_rad, match_indices)
                    ##src_rbbox = paddle.concat([src_bbox,src_rad],axis=-1) #cx,cy,w,h,rad
                    ##target_rbbox = paddle.concat([target_bbox,target_rad],axis=-1)
                    ##pd_gt_rbox = paddle.to_tensor(src_rbbox, dtype='float32')
                    ##pd_pred_rbox = paddle.to_tensor(target_rbbox, dtype='float32')
                    ##riou_score = matched_rbox_iou(pd_gt_rbox, pd_pred_rbox)
                    ##iou_score = riou_score.unsqueeze(-1)
                else:
                    iou_score = None
            else:
                iou_score = None
            loss_class.append(
                self._get_loss_class(aux_logits, gt_class, match_indices,
                                     bg_index, num_gts, postfix, iou_score)[
                                         'loss_class' + postfix])
            loss_ = self._get_loss_bbox(aux_boxes, gt_bbox, match_indices,
                                        num_gts, postfix)
            ### ANCHOR: rbbox loss, performance bad
            ##loss_ = self._get_loss_rbbox(aux_boxes, gt_bbox, aux_rads, gt_rad, match_indices,
            ##                            num_gts, postfix)
            loss_bbox.append(loss_['loss_bbox' + postfix])
            loss_giou.append(loss_['loss_giou' + postfix])
            loss_rad.append(
                self._get_loss_rad(aux_rads, gt_rad, match_indices,
                                    num_gts, postfix)[
                                         'loss_rad' + postfix])
            if masks is not None and gt_mask is not None:
                loss_ = self._get_loss_mask(aux_masks, gt_mask, match_indices,
                                            num_gts, postfix)
                loss_mask.append(loss_['loss_mask' + postfix])
                loss_dice.append(loss_['loss_dice' + postfix])
        loss = {
            "loss_class_aux" + postfix: paddle.add_n(loss_class),
            "loss_bbox_aux" + postfix: paddle.add_n(loss_bbox),
            "loss_giou_aux" + postfix: paddle.add_n(loss_giou),
            "loss_rad_aux" + postfix: paddle.add_n(loss_rad)
        }
        if masks is not None and gt_mask is not None:
            loss["loss_mask_aux" + postfix] = paddle.add_n(loss_mask)
            loss["loss_dice_aux" + postfix] = paddle.add_n(loss_dice)
        return loss

    def _get_index_updates(self, num_query_objects, target, match_indices):
        batch_idx = paddle.concat([
            paddle.full_like(src, i) for i, (src, _) in enumerate(match_indices)
        ])
        src_idx = paddle.concat([src for (src, _) in match_indices])
        src_idx += (batch_idx * num_query_objects)
        target_assign = paddle.concat([
            paddle.gather(
                t, dst, axis=0) for t, (_, dst) in zip(target, match_indices)
        ])
        return src_idx, target_assign

    def _get_src_target_assign(self, src, target, match_indices):
        src_assign = paddle.concat([
            paddle.gather(
                t, I, axis=0) if len(I) > 0 else paddle.zeros([0, t.shape[-1]])
            for t, (I, _) in zip(src, match_indices)
        ])
        target_assign = paddle.concat([
            paddle.gather(
                t, J, axis=0) if len(J) > 0 else paddle.zeros([0, t.shape[-1]])
            for t, (_, J) in zip(target, match_indices)
        ])
        return src_assign, target_assign

    def _get_num_gts(self, targets, dtype="float32"):
        num_gts = sum(len(a) for a in targets)
        num_gts = paddle.to_tensor([num_gts], dtype=dtype)
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.all_reduce(num_gts)
            num_gts /= paddle.distributed.get_world_size()
        num_gts = paddle.clip(num_gts, min=1.)
        return num_gts

    def _get_prediction_loss(self,
                             boxes,
                             rads,
                             logits,
                             gt_bbox,
                             gt_rad,
                             gt_class,
                             masks=None,
                             gt_mask=None,
                             postfix="",
                             dn_match_indices=None,
                             num_gts=1):
        if dn_match_indices is None:
            match_indices = self.matcher(
                boxes, rads, logits, gt_bbox, gt_rad, gt_class, masks=masks, gt_mask=gt_mask)
        else:
            match_indices = dn_match_indices

        if self.use_vfl:
            if sum(len(a) for a in gt_bbox) > 0:
                src_bbox, target_bbox = self._get_src_target_assign(
                    boxes.detach(), gt_bbox, match_indices)
                iou_score = bbox_iou(
                    bbox_cxcywh_to_xyxy(src_bbox).split(4, -1),
                    bbox_cxcywh_to_xyxy(target_bbox).split(4, -1))
                ### ANCHOR: use rbox_iou in ext_op to precisly calculate iou between two rbboxes
                ##src_rad, target_rad = self._get_src_target_assign(
                ##    rads.detach(), gt_rad, match_indices)
                ##src_rbbox = paddle.concat([src_bbox,src_rad],axis=-1) #cx,cy,w,h,rad
                ##target_rbbox = paddle.concat([target_bbox,target_rad],axis=-1)
                ##pd_gt_rbox = paddle.to_tensor(src_rbbox, dtype='float32')
                ##pd_pred_rbox = paddle.to_tensor(target_rbbox, dtype='float32')
                ##riou_score = matched_rbox_iou(pd_gt_rbox, pd_pred_rbox)
                ##iou_score = riou_score.unsqueeze(-1)
            else:
                iou_score = None
        else:
            iou_score = None

        loss = dict()
        loss.update(
            self._get_loss_class(logits, gt_class, match_indices,
                                 self.num_classes, num_gts, postfix, iou_score))
        ###loss.update(
        ###    self._get_loss_bbox(boxes, gt_bbox, match_indices, num_gts,
        ###                        postfix))
        # ANCHOR: rbbox loss, performance bad
        loss.update(
            self._get_loss_rbbox(boxes, gt_bbox, rads, gt_rad, 
                                 match_indices, num_gts, postfix))
        loss.update(
            self._get_loss_rad(rads, gt_rad, match_indices, num_gts,
                                postfix))
        if masks is not None and gt_mask is not None:
            loss.update(
                self._get_loss_mask(masks, gt_mask, match_indices, num_gts,
                                    postfix))
        return loss

    def forward(self,
                boxes,
                rads,
                logits,
                gt_bbox,
                gt_rad,
                gt_class,
                masks=None,
                gt_mask=None,
                postfix="",
                **kwargs):
        r"""
        Args:
            boxes (Tensor): [l, b, query, 4]
            logits (Tensor): [l, b, query, num_classes]
            gt_bbox (List(Tensor)): list[[n, 4]]
            gt_class (List(Tensor)): list[[n, 1]]
            masks (Tensor, optional): [l, b, query, h, w]
            gt_mask (List(Tensor), optional): list[[n, H, W]]
            postfix (str): postfix of loss name
        """

        dn_match_indices = kwargs.get("dn_match_indices", None)
        num_gts = kwargs.get("num_gts", None)
        if num_gts is None:
            num_gts = self._get_num_gts(gt_class)

        total_loss = self._get_prediction_loss(
            boxes[-1],
            rads[-1],
            logits[-1],
            gt_bbox,
            gt_rad,
            gt_class,
            masks=masks[-1] if masks is not None else None,
            gt_mask=gt_mask,
            postfix=postfix,
            dn_match_indices=dn_match_indices,
            num_gts=num_gts)

        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(
                    boxes[:-1],
                    rads[:-1],
                    logits[:-1],
                    gt_bbox,
                    gt_rad,
                    gt_class,
                    self.num_classes,
                    num_gts,
                    dn_match_indices,
                    postfix,
                    masks=masks[:-1] if masks is not None else None,
                    gt_mask=gt_mask))

        return total_loss

@register
class DINOLoss_oad(DETRLoss_oad):
    def forward(self,
                boxes,
                rads,
                logits,
                gt_bbox,
                gt_rad,
                gt_class,
                masks=None,
                gt_mask=None,
                postfix="",
                dn_out_bboxes=None,
                dn_out_rads=None,
                dn_out_logits=None,
                dn_meta=None,
                **kwargs):
        num_gts = self._get_num_gts(gt_class)
        total_loss = super(DINOLoss_oad, self).forward(
            boxes, rads, logits, gt_bbox, gt_rad, gt_class, num_gts=num_gts)

        if dn_meta is not None:
            dn_positive_idx, dn_num_group = \
                dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
            assert len(gt_class) == len(dn_positive_idx)

            # denoising match indices
            dn_match_indices = self.get_dn_match_indices(
                gt_class, dn_positive_idx, dn_num_group)

            # compute denoising training loss
            num_gts *= dn_num_group
            dn_loss = super(DINOLoss_oad, self).forward(
                dn_out_bboxes,
                dn_out_rads,
                dn_out_logits,
                gt_bbox,
                gt_rad,
                gt_class,
                postfix="_dn",
                dn_match_indices=dn_match_indices,
                num_gts=num_gts)
            total_loss.update(dn_loss)
        else:
            total_loss.update(
                {k + '_dn': paddle.to_tensor([0.])
                 for k in total_loss.keys()})

        return total_loss

    @staticmethod
    def get_dn_match_indices(labels, dn_positive_idx, dn_num_group):
        dn_match_indices = []
        for i in range(len(labels)):
            num_gt = len(labels[i])
            if num_gt > 0:
                gt_idx = paddle.arange(end=num_gt, dtype="int64")
                gt_idx = gt_idx.tile([dn_num_group])
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((paddle.zeros(
                    [0], dtype="int64"), paddle.zeros(
                        [0], dtype="int64")))
        return dn_match_indices
    
# -----------------------------------------------------------------------------------------------

@register
class DETRLoss_oadkpt(nn.Layer):
    __shared__ = ['num_classes', 'use_focal_loss']
    __inject__ = ['matcher']

    def __init__(self,
                 num_classes=80,
                 matcher='HungarianMatcher_oadkpt',
                 loss_coeff={
                     'class': 1,
                     'bbox': 5,
                     'giou': 2,
                     'rads': 1,
                     'no_object': 0.1,
                     'mask': 1,
                     'dice': 1
                 },
                 aux_loss=True,
                 use_focal_loss=False,
                 use_vfl=False,
                 use_uni_match=False,
                 uni_match_ind=0):
        r"""
        Args:
            num_classes (int): The number of classes.
            matcher (HungarianMatcher): It computes an assignment between the targets
                and the predictions of the network.
            loss_coeff (dict): The coefficient of loss.
            aux_loss (bool): If 'aux_loss = True', loss at each decoder layer are to be used.
            use_focal_loss (bool): Use focal loss or not.
        """
        super(DETRLoss_oadkpt, self).__init__()

        self.num_classes = num_classes
        self.matcher = matcher
        self.loss_coeff = loss_coeff
        self.aux_loss = aux_loss
        self.use_focal_loss = use_focal_loss
        self.use_vfl = use_vfl
        self.use_uni_match = use_uni_match
        self.uni_match_ind = uni_match_ind

        if not self.use_focal_loss:
            self.loss_coeff['class'] = paddle.full([num_classes + 1],
                                                   loss_coeff['class'])
            self.loss_coeff['class'][-1] = loss_coeff['no_object']
        self.giou_loss = GIoULoss()

    def _get_loss_class(self,
                        logits,
                        gt_class,
                        match_indices,
                        bg_index,
                        num_gts,
                        postfix="",
                        iou_score=None):
        # logits: [b, query, num_classes], gt_class: list[[n, 1]]
        name_class = "loss_class" + postfix

        target_label = paddle.full(logits.shape[:2], bg_index, dtype='int64')
        bs, num_query_objects = target_label.shape
        num_gt = sum(len(a) for a in gt_class)
        if num_gt > 0:
            index, updates = self._get_index_updates(num_query_objects,
                                                     gt_class, match_indices)
            target_label = paddle.scatter(
                target_label.reshape([-1, 1]), index, updates.astype('int64'))
            target_label = target_label.reshape([bs, num_query_objects])
        if self.use_focal_loss:
            target_label = F.one_hot(target_label,
                                     self.num_classes + 1)[..., :-1]
            if iou_score is not None and self.use_vfl:
                target_score = paddle.zeros([bs, num_query_objects])
                if num_gt > 0:
                    target_score = paddle.scatter(
                        target_score.reshape([-1, 1]), index, iou_score)
                target_score = target_score.reshape(
                    [bs, num_query_objects, 1]) * target_label
                loss_ = self.loss_coeff['class'] * varifocal_loss_with_logits(
                    logits, target_score, target_label,
                    num_gts / num_query_objects)
            else:
                loss_ = self.loss_coeff['class'] * sigmoid_focal_loss(
                    logits, target_label, num_gts / num_query_objects)
        else:
            loss_ = F.cross_entropy(
                logits, target_label, weight=self.loss_coeff['class'])
        return {name_class: loss_}

    def _get_loss_bbox(self, boxes, gt_bbox, match_indices, num_gts,
                       postfix=""):
        # boxes: [b, query, 4], gt_bbox: list[[n, 4]]
        name_bbox = "loss_bbox" + postfix
        name_giou = "loss_giou" + postfix

        loss = dict()
        if sum(len(a) for a in gt_bbox) == 0:
            loss[name_bbox] = paddle.to_tensor([0.])
            loss[name_giou] = paddle.to_tensor([0.])
            return loss

        src_bbox, target_bbox = self._get_src_target_assign(boxes, gt_bbox,
                                                            match_indices)
        loss[name_bbox] = self.loss_coeff['bbox'] * F.l1_loss(
            src_bbox, target_bbox, reduction='sum') / num_gts
        loss[name_giou] = self.giou_loss(
            bbox_cxcywh_to_xyxy(src_bbox), bbox_cxcywh_to_xyxy(target_bbox))
        loss[name_giou] = loss[name_giou].sum() / num_gts
        loss[name_giou] = self.loss_coeff['giou'] * loss[name_giou]
        return loss

    def _get_loss_rad(self, rads, gt_rad, match_indices, num_gts,
                       postfix=""):
        # rads: [b, query, 1], gt_rad: list[[n, 1]]
        name_rad = "loss_rad" + postfix

        loss = dict()
        if sum(len(a) for a in gt_rad) == 0:
            loss[name_rad] = paddle.to_tensor([0.])
            return loss

        src_rad, target_rad = self._get_src_target_assign(rads, gt_rad,
                                                            match_indices)
        loss[name_rad] = self.loss_coeff['rad'] * \
        (F.smooth_l1_loss(paddle.cos(src_rad), paddle.cos(target_rad), reduction='sum') +
         F.smooth_l1_loss(paddle.sin(src_rad), paddle.sin(target_rad), reduction='sum')) / num_gts
        return loss
    
    def _get_loss_kpts(self, kpts, gt_keypoint, boxes, gt_bbox, match_indices, num_gts,
                       postfix=""):
        # kpts: [b, query, 1], gt_keypoint: list[[n, 1]]
        name_kpts = "loss_kpts" + postfix
        
        sigmas = paddle.to_tensor([.5]*int(kpts.shape[-1]/2))

        loss = dict()
        if sum(len(a) for a in gt_keypoint) == 0:
            loss[name_kpts] = paddle.to_tensor([0.])
            return loss

        src_kpts, target_kpts = self._get_src_target_assign(kpts, gt_keypoint, match_indices)
        
        ##src_bbox, target_bbox = self._get_src_target_assign(boxes, gt_bbox, match_indices)
        ##
        ##diff = (src_kpts[:,0::2] - target_kpts[:,0::2]) ** 2 + (src_kpts[:,1::2] - target_kpts[:,1::2]) ** 2
        ##s = paddle.prod(target_bbox[:,-2:], axis=1, keepdim=True)
        ##
        ### OKS Loss 
        ##loss[name_kpts] = self.loss_coeff['kpts'] * (1 - paddle.exp(-diff/(2*(s*sigmas)**2+1e-9))).mean()
        # L1 Loss
        loss[name_kpts] = F.smooth_l1_loss(src_kpts, target_kpts, reduction='sum') / num_gts
        
        # #### example yolov7_oad oks based loss
        # d = (pkpt_x-selected_tkpt[:,0::2])**2 + (pkpt_y-selected_tkpt[:,1::2])**2
        # s = torch.prod(selected_tbox[:,-2:], dim=1, keepdim=True)
        # kpt_loss_factor = (torch.sum(kpt_mask != 0) + torch.sum(kpt_mask == 0))/(torch.sum(kpt_mask != 0) + 1e-9)
        # lkpt += kpt_loss_factor*((1 - torch.exp(-d/(2*(s*sigmas)**2+1e-9)))*kpt_mask).mean()
        
        return loss
    
    def _get_loss_mask(self, masks, gt_mask, match_indices, num_gts,
                       postfix=""):
        # masks: [b, query, h, w], gt_mask: list[[n, H, W]]
        name_mask = "loss_mask" + postfix
        name_dice = "loss_dice" + postfix

        loss = dict()
        if sum(len(a) for a in gt_mask) == 0:
            loss[name_mask] = paddle.to_tensor([0.])
            loss[name_dice] = paddle.to_tensor([0.])
            return loss

        src_masks, target_masks = self._get_src_target_assign(masks, gt_mask,
                                                              match_indices)
        src_masks = F.interpolate(
            src_masks.unsqueeze(0),
            size=target_masks.shape[-2:],
            mode="bilinear")[0]
        loss[name_mask] = self.loss_coeff['mask'] * F.sigmoid_focal_loss(
            src_masks,
            target_masks,
            paddle.to_tensor(
                [num_gts], dtype='float32'))
        loss[name_dice] = self.loss_coeff['dice'] * self._dice_loss(
            src_masks, target_masks, num_gts)
        return loss

    def _dice_loss(self, inputs, targets, num_gts):
        inputs = F.sigmoid(inputs)
        inputs = inputs.flatten(1)
        targets = targets.flatten(1)
        numerator = 2 * (inputs * targets).sum(1)
        denominator = inputs.sum(-1) + targets.sum(-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss.sum() / num_gts

    def _get_loss_aux(self,
                      boxes,
                      rads,
                      kpts,
                      logits,
                      gt_bbox,
                      gt_rad,
                      gt_keypoint,
                      gt_class,
                      bg_index,
                      num_gts,
                      dn_match_indices=None,
                      postfix="",
                      masks=None,
                      gt_mask=None):
        loss_class = []
        loss_bbox, loss_giou = [], []
        loss_rad = []
        loss_kpts = []
        loss_mask, loss_dice = [], []
        if dn_match_indices is not None:
            match_indices = dn_match_indices
        elif self.use_uni_match:
            match_indices = self.matcher(
                boxes[self.uni_match_ind],
                rads[self.uni_match_ind],
                kpts[self.uni_match_ind],
                logits[self.uni_match_ind],
                gt_bbox,
                gt_rad,
                gt_keypoint,
                gt_class,
                masks=masks[self.uni_match_ind] if masks is not None else None,
                gt_mask=gt_mask)
        for i, (aux_boxes, aux_rads, aux_kpts, aux_logits) in enumerate(zip(boxes, rads, kpts, logits)):
            aux_masks = masks[i] if masks is not None else None
            if not self.use_uni_match and dn_match_indices is None:
                match_indices = self.matcher(
                    aux_boxes,
                    aux_rads,
                    aux_kpts,
                    aux_logits,
                    gt_bbox,
                    gt_rad,
                    gt_keypoint,
                    gt_class,
                    masks=aux_masks,
                    gt_mask=gt_mask)
            if self.use_vfl:
                if sum(len(a) for a in gt_bbox) > 0:
                    src_bbox, target_bbox = self._get_src_target_assign(
                        aux_boxes.detach(), gt_bbox, match_indices)
                    iou_score = bbox_iou(
                        bbox_cxcywh_to_xyxy(src_bbox).split(4, -1),
                        bbox_cxcywh_to_xyxy(target_bbox).split(4, -1))
                else:
                    iou_score = None
            else:
                iou_score = None
            loss_class.append(
                self._get_loss_class(aux_logits, gt_class, match_indices,
                                     bg_index, num_gts, postfix, iou_score)[
                                         'loss_class' + postfix])
            loss_ = self._get_loss_bbox(aux_boxes, gt_bbox, match_indices,
                                        num_gts, postfix)
            loss_bbox.append(loss_['loss_bbox' + postfix])
            loss_giou.append(loss_['loss_giou' + postfix])
            loss_rad.append(
                self._get_loss_rad(aux_rads, gt_rad, match_indices,
                                    num_gts, postfix)[
                                         'loss_rad' + postfix])
            loss_kpts.append(
                self._get_loss_kpts(aux_kpts, gt_keypoint, aux_boxes, gt_bbox, match_indices,
                                    num_gts, postfix)[
                                         'loss_kpts' + postfix])
            if masks is not None and gt_mask is not None:
                loss_ = self._get_loss_mask(aux_masks, gt_mask, match_indices,
                                            num_gts, postfix)
                loss_mask.append(loss_['loss_mask' + postfix])
                loss_dice.append(loss_['loss_dice' + postfix])
        loss = {
            "loss_class_aux" + postfix: paddle.add_n(loss_class),
            "loss_bbox_aux" + postfix: paddle.add_n(loss_bbox),
            "loss_giou_aux" + postfix: paddle.add_n(loss_giou),
            "loss_rad_aux" + postfix: paddle.add_n(loss_rad),
            "loss_kpts_aux" + postfix: paddle.add_n(loss_kpts)
        }
        if masks is not None and gt_mask is not None:
            loss["loss_mask_aux" + postfix] = paddle.add_n(loss_mask)
            loss["loss_dice_aux" + postfix] = paddle.add_n(loss_dice)
        return loss

    def _get_index_updates(self, num_query_objects, target, match_indices):
        batch_idx = paddle.concat([
            paddle.full_like(src, i) for i, (src, _) in enumerate(match_indices)
        ])
        src_idx = paddle.concat([src for (src, _) in match_indices])
        src_idx += (batch_idx * num_query_objects)
        target_assign = paddle.concat([
            paddle.gather(
                t, dst, axis=0) for t, (_, dst) in zip(target, match_indices)
        ])
        return src_idx, target_assign

    def _get_src_target_assign(self, src, target, match_indices):
        src_assign = paddle.concat([
            paddle.gather(
                t, I, axis=0) if len(I) > 0 else paddle.zeros([0, t.shape[-1]])
            for t, (I, _) in zip(src, match_indices)
        ])
        target_assign = paddle.concat([
            paddle.gather(
                t, J, axis=0) if len(J) > 0 else paddle.zeros([0, t.shape[-1]])
            for t, (_, J) in zip(target, match_indices)
        ])
        return src_assign, target_assign

    def _get_num_gts(self, targets, dtype="float32"):
        num_gts = sum(len(a) for a in targets)
        num_gts = paddle.to_tensor([num_gts], dtype=dtype)
        if paddle.distributed.get_world_size() > 1:
            paddle.distributed.all_reduce(num_gts)
            num_gts /= paddle.distributed.get_world_size()
        num_gts = paddle.clip(num_gts, min=1.)
        return num_gts

    def _get_prediction_loss(self,
                             boxes,
                             rads,
                             kpts,
                             logits,
                             gt_bbox,
                             gt_rad,
                             gt_keypoint,
                             gt_class,
                             masks=None,
                             gt_mask=None,
                             postfix="",
                             dn_match_indices=None,
                             num_gts=1):
        if dn_match_indices is None:
            match_indices = self.matcher(
                boxes, rads, kpts, logits, gt_bbox, gt_rad, gt_keypoint, gt_class, masks=masks, gt_mask=gt_mask)
        else:
            match_indices = dn_match_indices

        if self.use_vfl:
            if sum(len(a) for a in gt_bbox) > 0:
                src_bbox, target_bbox = self._get_src_target_assign(
                    boxes.detach(), gt_bbox, match_indices)
                iou_score = bbox_iou(
                    bbox_cxcywh_to_xyxy(src_bbox).split(4, -1),
                    bbox_cxcywh_to_xyxy(target_bbox).split(4, -1))
            else:
                iou_score = None
        else:
            iou_score = None

        loss = dict()
        loss.update(
            self._get_loss_class(logits, gt_class, match_indices,
                                 self.num_classes, num_gts, postfix, iou_score))
        loss.update(
            self._get_loss_bbox(boxes, gt_bbox, match_indices, num_gts,
                                postfix))
        loss.update(
            self._get_loss_rad(rads, gt_rad, match_indices, num_gts,
                                postfix))
        loss.update(
            self._get_loss_kpts(kpts, gt_keypoint, boxes, gt_bbox, match_indices, num_gts,
                                postfix))
        if masks is not None and gt_mask is not None:
            loss.update(
                self._get_loss_mask(masks, gt_mask, match_indices, num_gts,
                                    postfix))
        return loss

    def forward(self,
                boxes,
                rads,
                kpts,
                logits,
                gt_bbox,
                gt_rad,
                gt_keypoint,
                gt_class,
                masks=None,
                gt_mask=None,
                postfix="",
                **kwargs):
        r"""
        Args:
            boxes (Tensor): [l, b, query, 4]
            logits (Tensor): [l, b, query, num_classes]
            gt_bbox (List(Tensor)): list[[n, 4]]
            gt_class (List(Tensor)): list[[n, 1]]
            masks (Tensor, optional): [l, b, query, h, w]
            gt_mask (List(Tensor), optional): list[[n, H, W]]
            postfix (str): postfix of loss name
        """

        dn_match_indices = kwargs.get("dn_match_indices", None)
        num_gts = kwargs.get("num_gts", None)
        if num_gts is None:
            num_gts = self._get_num_gts(gt_class)

        total_loss = self._get_prediction_loss(
            boxes[-1],
            rads[-1],
            kpts[-1],
            logits[-1],
            gt_bbox,
            gt_rad,
            gt_keypoint,
            gt_class,
            masks=masks[-1] if masks is not None else None,
            gt_mask=gt_mask,
            postfix=postfix,
            dn_match_indices=dn_match_indices,
            num_gts=num_gts)

        if self.aux_loss:
            total_loss.update(
                self._get_loss_aux(
                    boxes[:-1],
                    rads[:-1],
                    kpts[:-1],
                    logits[:-1],
                    gt_bbox,
                    gt_rad,
                    gt_keypoint,
                    gt_class,
                    self.num_classes,
                    num_gts,
                    dn_match_indices,
                    postfix,
                    masks=masks[:-1] if masks is not None else None,
                    gt_mask=gt_mask))

        return total_loss
    
@register
class DINOLoss_oadkpt(DETRLoss_oadkpt):
    def forward(self,
                boxes,
                rads,
                kpts,
                logits,
                gt_bbox,
                gt_rad,
                gt_keypoint,
                gt_class,
                masks=None,
                gt_mask=None,
                postfix="",
                dn_out_bboxes=None,
                dn_out_rads=None,
                dn_out_kpts=None,
                dn_out_logits=None,
                dn_meta=None,
                **kwargs):
        num_gts = self._get_num_gts(gt_class)
        total_loss = super(DINOLoss_oadkpt, self).forward(
            boxes, rads, kpts, logits, gt_bbox, gt_rad, gt_keypoint, gt_class, num_gts=num_gts)

        if dn_meta is not None:
            dn_positive_idx, dn_num_group = \
                dn_meta["dn_positive_idx"], dn_meta["dn_num_group"]
            assert len(gt_class) == len(dn_positive_idx)

            # denoising match indices
            dn_match_indices = self.get_dn_match_indices(
                gt_class, dn_positive_idx, dn_num_group)

            # compute denoising training loss
            num_gts *= dn_num_group
            dn_loss = super(DINOLoss_oadkpt, self).forward(
                dn_out_bboxes,
                dn_out_rads,
                dn_out_kpts,
                dn_out_logits,
                gt_bbox,
                gt_rad,
                gt_keypoint,
                gt_class,
                postfix="_dn",
                dn_match_indices=dn_match_indices,
                num_gts=num_gts)
            total_loss.update(dn_loss)
        else:
            total_loss.update(
                {k + '_dn': paddle.to_tensor([0.])
                 for k in total_loss.keys()})

        return total_loss

    @staticmethod
    def get_dn_match_indices(labels, dn_positive_idx, dn_num_group):
        dn_match_indices = []
        for i in range(len(labels)):
            num_gt = len(labels[i])
            if num_gt > 0:
                gt_idx = paddle.arange(end=num_gt, dtype="int64")
                gt_idx = gt_idx.tile([dn_num_group])
                assert len(dn_positive_idx[i]) == len(gt_idx)
                dn_match_indices.append((dn_positive_idx[i], gt_idx))
            else:
                dn_match_indices.append((paddle.zeros(
                    [0], dtype="int64"), paddle.zeros(
                        [0], dtype="int64")))
        return dn_match_indices