# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved. 
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

import os
import copy
try:
    from collections.abc import Sequence
except Exception:
    from collections import Sequence
import numpy as np
from ppdet.core.workspace import register, serializable
from .dataset import DetDataset

from ppdet.utils.logger import setup_logger
logger = setup_logger(__name__)

__all__ = ['OADDataSet']


@register
@serializable
class OADDataSet(DetDataset):
    """
    Load dataset with OAD format. (rbbox + keyp)

    Args:
        dataset_dir (str): root directory for dataset.
        image_dir (str): directory for images.
        anno_path (str): coco annotation file path.
        data_fields (list): key name of data dictionary, at least have 'image'.
        sample_num (int): number of samples to load, -1 means all.
        load_crowd (bool): whether to load crowded ground-truth. 
            False as default
        allow_empty (bool): whether to load empty entry. False as default
        empty_ratio (float): the ratio of empty record number to total 
            record's, if empty_ratio is out of [0. ,1.), do not sample the 
            records and use all the empty entries. 1. as default
        repeat (int): repeat times for dataset, use in benchmark.
    """

    def __init__(self,
                 dataset_dir=None,
                 image_dir=None,
                 anno_path=None,
                 data_fields=['image'],
                 sample_num=-1,
                 load_crowd=False,
                 allow_empty=False,
                 empty_ratio=1.,
                 repeat=1):
        super(OADDataSet, self).__init__(
            dataset_dir,
            image_dir,
            anno_path,
            data_fields,
            sample_num,
            repeat=repeat)
        self.load_image_only = False
        self.load_semantic = False
        self.load_crowd = load_crowd
        self.allow_empty = allow_empty
        self.empty_ratio = empty_ratio

    def _sample_empty(self, records, num):
        # if empty_ratio is out of [0. ,1.), do not sample the records
        if self.empty_ratio < 0. or self.empty_ratio >= 1.:
            return records
        import random
        sample_num = min(
            int(num * self.empty_ratio / (1 - self.empty_ratio)), len(records))
        records = random.sample(records, sample_num)
        return records
    def parse_dataset(self):
        anno_path = os.path.join(self.dataset_dir, self.anno_path)
        image_dir = os.path.join(self.dataset_dir, self.image_dir)

        assert anno_path.endswith('.json'), \
            'invalid coco annotation file: ' + anno_path
        from pycocotools.coco import COCO
        coco = COCO(anno_path)
        img_ids = coco.getImgIds()
        img_ids.sort()
        cat_ids = coco.getCatIds()
        records = []
        empty_records = []
        ct = 0
    
        # self.catid2clsid = dict({catid: i for i, catid in enumerate(cat_ids)})
        # self.cname2cid = dict({
        #     coco.loadCats(catid)[0]['name']: clsid
        #     for catid, clsid in self.catid2clsid.items()
        # })
        cls_dict = {cls['name']: cls['id'] for cls in coco.dataset['categories']}
        cls_list_new = [{cls_key: cls_value} for cls_key, cls_value in cls_dict.items()]
        pose_dict = {pose['name']: pose['id'] for pose in coco.dataset['poses']}
        pose_list_new = [{pose_key: pose_value} for pose_key, pose_value in pose_dict.items()]
        combined_dict = {class_key + '_' + pose_key: (len(pose_list_new) * class_value) + pose_value for class_item in cls_list_new for pose_item in pose_list_new for (class_key, class_value), (pose_key, pose_value) in zip(class_item.items(), pose_item.items())}
        self.cname2cid=combined_dict
        self.catid2clsid=dict({catid: i for i, catid in enumerate(combined_dict.values())})
        self.pose_num=len(pose_list_new)
        if 'annotations' not in coco.dataset:
            self.load_image_only = True
            logger.warning('Annotation file: {} does not contains ground truth '
                           'and load image information only.'.format(anno_path))

        for img_id in img_ids:
            img_anno = coco.loadImgs([img_id])[0]
            im_fname = img_anno['file_name']
            im_w = float(img_anno['width'])
            im_h = float(img_anno['height'])

            im_path = os.path.join(image_dir,
                                   im_fname) if image_dir else im_fname
            is_empty = False
            if not os.path.exists(im_path):
                logger.warning('Illegal image file: {}, and it will be '
                               'ignored'.format(im_path))
                continue

            if im_w < 0 or im_h < 0:
                logger.warning('Illegal width: {} or height: {} in annotation, '
                               'and im_id: {} will be ignored'.format(
                                   im_w, im_h, img_id))
                continue

            coco_rec = {
                'im_file': im_path,
                'im_id': np.array([img_id]),
                'h': im_h,
                'w': im_w,
            } if 'image' in self.data_fields else {}

            if not self.load_image_only:
                ins_anno_ids = coco.getAnnIds(
                    imgIds=[img_id], iscrowd=None if self.load_crowd else False)
                instances = coco.loadAnns(ins_anno_ids)

                bboxes = []
                rbboxes = []
                is_rbox_anno = False
                # for inst in instances:
                #     # check gt bbox
                #     if inst.get('ignore', False):
                #         continue
                #     if 'bbox' not in inst.keys():
                #         continue
                #     else:
                #         if not any(np.array(inst['bbox'])):
                #             continue

                #     x1, y1, box_w, box_h = inst['bbox']
                #     x2 = x1 + box_w
                #     y2 = y1 + box_h
                #     eps = 1e-5
                #     if inst['area'] > 0 and x2 - x1 > eps and y2 - y1 > eps:
                #         inst['clean_bbox'] = [
                #             round(float(x), 3) for x in [x1, y1, x2, y2]
                #         ]
                #         bboxes.append(inst)
                #     else:
                #         logger.warning(
                #             'Found an invalid bbox in annotations: im_id: {}, '
                #             'area: {} x1: {}, y1: {}, x2: {}, y2: {}.'.format(
                #                 img_id, float(inst['area']), x1, y1, x2, y2))
                for inst in instances:
                    # check gt bbox
                    if inst.get('ignore', False):
                        continue
                    if 'rbbox' not in inst.keys():
                        continue
                    else:
                        if not any(np.array(inst['rbbox'])):
                            continue

                    x1, y1, box_w, box_h,rad = inst['rbbox']
                    x2 = x1 + box_w
                    y2 = y1 + box_h
                    eps = 1e-5
                    rbboxes.append(inst)
                    # if inst['area'] > 0 and x2 - x1 > eps and y2 - y1 > eps:
                    #     inst['clean_bbox'] = [
                    #         round(float(x), 3) for x in [x1, y1, x2, y2]
                    #     ]
                    #     rbboxes.append(inst)
                    # else:
                    #     logger.warning(
                    #         'Found an invalid bbox in annotations: im_id: {}, '
                    #         'area: {} x1: {}, y1: {}, x2: {}, y2: {}, rad:{}.'.format(
                    #             img_id, float(inst['area']), x1, y1, x2, y2,rad))
                num_bbox = len(rbboxes)
                if num_bbox <= 0 and not self.allow_empty:
                    continue
                elif num_bbox <= 0:
                    is_empty = True

                gt_bbox = np.zeros((num_bbox, 4), dtype=np.float32)
                gt_rad = np.zeros((num_bbox, 1), dtype=np.float32)
                gt_keypoint = np.zeros((num_bbox, 6), dtype=np.float32)
                gt_class = np.zeros((num_bbox, 1), dtype=np.int32)
                gt_pose = np.zeros((num_bbox, 1), dtype=np.int32)
                is_crowd = np.zeros((num_bbox, 1), dtype=np.int32)
                gt_poly = [None] * num_bbox
                gt_track_id = -np.ones((num_bbox, 1), dtype=np.int32)

                has_segmentation = False
                has_track_id = False
                for i, rbox in enumerate(rbboxes):
                    catid = rbox['category_id']
                    poseid = rbox['pose_id']
                    gt_pose[i][0] = poseid
                    class_pose_id=self.pose_num*catid+poseid
                    gt_class[i][0] = self.catid2clsid[class_pose_id]
                    gt_bbox[i, :] = rbox['rbbox'][:4]
                    gt_rad[i, :] = rbox['rbbox'][4]
                    gt_keypoint[i, :] = [rbox['keypoints'][j] for j in range(len(rbox['keypoints'])) if (j+1) % 3 != 0]
                    is_crowd[i][0] = rbox['iscrowd']
                    # check RLE format 
                    if 'segmentation_rbbox' in rbox and rbox['iscrowd'] == 1:
                        gt_poly[i] = [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]]
                    elif 'segmentation_rbbox' in rbox and rbox['segmentation_rbbox']:
                        if not np.array(
                                rbox['segmentation'],
                                dtype=object).size > 0 and not self.allow_empty:
                            rbboxes.pop(i)
                            gt_poly.pop(i)
                            np.delete(is_crowd, i)
                            np.delete(gt_class, i)
                            np.delete(gt_bbox, i)
                        else:
                            gt_poly[i] = rbox['segmentation_rbbox']
                        has_segmentation = True

                    if 'track_id' in rbox:
                        gt_track_id[i][0] = rbox['track_id']
                        has_track_id = True

                if has_segmentation and not any(
                        gt_poly) and not self.allow_empty:
                    continue

                gt_rec = {
                    'is_crowd': is_crowd,
                    'gt_class': gt_class,
                    # 'gt_pose': gt_pose,
                    'gt_bbox': gt_bbox,
                    'gt_rad': gt_rad,
                    'gt_keypoint': gt_keypoint,
                    # 'gt_poly': gt_poly,
                }
                if has_track_id:
                    gt_rec.update({'gt_track_id': gt_track_id})

                for k, v in gt_rec.items():
                    if k in self.data_fields:
                        coco_rec[k] = v

                # TODO: remove load_semantic
                if self.load_semantic and 'semantic' in self.data_fields:
                    seg_path = os.path.join(self.dataset_dir, 'stuffthingmaps',
                                            'train2017', im_fname[:-3] + 'png')
                    coco_rec.update({'semantic': seg_path})

            logger.debug('Load file: {}, im_id: {}, h: {}, w: {}.'.format(
                im_path, img_id, im_h, im_w))
            if is_empty:
                empty_records.append(coco_rec)
            else:
                records.append(coco_rec)
            ct += 1
            if self.sample_num > 0 and ct >= self.sample_num:
                break
        assert ct > 0, 'not found any coco record in %s' % (anno_path)
        logger.info('Load [{} samples valid, {} samples invalid] in file {}.'.
                    format(ct, len(img_ids) - ct, anno_path))
        if self.allow_empty and len(empty_records) > 0:
            empty_records = self._sample_empty(empty_records, len(records))
            records += empty_records
        self.roidbs = records