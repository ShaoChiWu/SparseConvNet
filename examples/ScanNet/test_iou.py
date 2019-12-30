# Copyright 2016-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import torch, numpy as np


CLASS_LABELS = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'desk', 'curtain', 'refrigerator', 'shower curtain', 'toilet', 'sink', 'bathtub', 'otherfurniture']


def confusion_matrix(pred_ids, gt_ids):
    assert pred_ids.shape == gt_ids.shape, (pred_ids.shape, gt_ids.shape)
    idxs= gt_ids>=0
    return np.bincount(pred_ids[idxs]*20+gt_ids[idxs],minlength=400).reshape((20,20)).astype(np.ulonglong)

def get_iou(label_id, confusion):
    # true positives
    tp = np.longlong(confusion[label_id, label_id])
    # false negatives
    fn = np.longlong(confusion[label_id, :].sum()) - tp
    # false positives
    not_ignored = [l for l in range(20) if not l == label_id]
    fp = np.longlong(confusion[not_ignored, label_id].sum())

    denom = (tp + fp + fn)
    if denom == 0:
        return False
    return (float(tp) / denom, tp, denom)

def evaluate(pred_ids,gt_ids):
    print('evaluating', gt_ids.size, 'points...')
    confusion=confusion_matrix(pred_ids,gt_ids)
    class_ious = {}
    for i in range(20):
        label_name = CLASS_LABELS[i]
        label_id = i
        class_iou = get_iou(label_id, confusion)
        if class_iou is not False:
            class_ious[label_name] = get_iou(label_id, confusion)

    sum_iou = 0
    for label_name in class_ious:
        sum_iou+=class_ious[label_name][0]
    mean_iou = sum_iou/len(class_ious)

    print('classes          IoU')
    print('----------------------------')
    for i in range(20):
        label_name = CLASS_LABELS[i]
        if label_name in class_ious:
            print('{0:<14s}: {1:>5.3f}   ({2:>6d}/{3:<6d})'.format(label_name, class_ious[label_name][0], class_ious[label_name][1], class_ious[label_name][2]))
        else:
            print('{0:<14s}: {1}'.format(label_name, 'missing'))
    print('mean IOU', mean_iou) 