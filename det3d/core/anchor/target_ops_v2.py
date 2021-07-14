import logging

import numba
import numpy as np
import numpy.random as npr
import torch

logger = logging.getLogger(__name__)


def create_target_np(
    all_anchors,      # (70400, 7)
    gt_boxes,         # (M, 7)
    similarity_fn,
    box_encoding_fn,
    prune_anchor_fn=None,   # None
    gt_classes=None,        # (M,)
    matched_threshold=0.6,
    unmatched_threshold=0.45,
    bbox_inside_weight=None,
    positive_fraction=None, # None
    rpn_batch_size=300,     # 512
    norm_by_num_examples=False,
    box_code_size=7,
):
    """

    """
    num_anchors = all_anchors.shape[0]   # 70400
    anchors = all_anchors                # [70400, 7]
    num_inside = num_anchors             # 70400

    logger.debug("total_anchors: {}".format(num_anchors))
    logger.debug("anchors.shape: {}".format(anchors.shape))

    if gt_classes is None:
        gt_classes = np.ones([gt_boxes.shape[0]], dtype=np.int32)

    # label=1 is positive, 0 is negative, -1 is don't care (ignore)
    labels = np.empty((num_anchors,), dtype=np.int32)
    gt_ids = np.empty((num_anchors,), dtype=np.int32)
    labels.fill(-1)           # (70400,)
    gt_ids.fill(-1)           # (70400,)

    if len(gt_boxes) > 0:
        # anchors: [N, 7], gt_boxes: [M, 7]
        anchor_by_gt_overlap = similarity_fn(anchors, gt_boxes)      # (N, M), nearest bev iou

        anchor_to_gt_argmax = anchor_by_gt_overlap.argmax(axis=1)    # (N,), get the index of the gt_box with the largest iou for each anchor, (iou=0 -> index:0)
        anchor_to_gt_max = anchor_by_gt_overlap[np.arange(num_anchors), anchor_to_gt_argmax]  # (N,), get the largest iou with one gt_box for each anchor
        gt_to_anchor_argmax = anchor_by_gt_overlap.argmax(axis=0)    # (M,), get the index of the anchor with the largest iou for each gt_box.
        gt_to_anchor_max = anchor_by_gt_overlap[gt_to_anchor_argmax, np.arange(anchor_by_gt_overlap.shape[1])]  # (M,), get the largest iou with one anchor for each gt_box

        # must remove gt which doesn't match any anchor. but it should be impossible in voxelization scene.
        empty_gt_mask = gt_to_anchor_max == 0
        gt_to_anchor_max[empty_gt_mask] = -1

        # todo: Important
        # (M+m, ) Find all anchors that have the same max iou with the same gt_box, for each of gt_boxes.
        # which means sometimes one gt box may have the same iou (also maximum among all anchors) with multiple anchors.
        pos_inds_force = np.where(anchor_by_gt_overlap == gt_to_anchor_max)[0]  # (M+m,), index of positive anchors

        # fg label: for each gt, use anchors of highest overlap with it as positive targets.
        # usually one anchor can have only one gt_box with large iou. While one gt_box can have multiple anchors with large iou.
        # While I still feel there exist a little possibility that one gt box has max iou with one anchor, and this anchor has larger iou
        # with another gt_box, but this gt_box has max iou with another anchor, so the orginal gt box may not have corresponind target.
        gt_inds_force = anchor_to_gt_argmax[pos_inds_force]   # (M+m,) indices of targeted gt_boxes for the M+m positive anchors
        labels[pos_inds_force] = gt_classes[gt_inds_force]    # these anchors are labeled with `1` as positive
        gt_ids[pos_inds_force] = gt_inds_force                # these anchors are saved with corresponding gt_box indices as targets

        # fg label: above threshold IOU. Notice we use anchor_to_gt_max
        pos_inds = anchor_to_gt_max >= matched_threshold
        gt_inds = anchor_to_gt_argmax[pos_inds]
        labels[pos_inds] = gt_classes[gt_inds]
        gt_ids[pos_inds] = gt_inds

        # bg label: below threshold IOU. Notice we use anchor_to_gt_max
        bg_inds = np.where(anchor_to_gt_max < unmatched_threshold)[0]
    else:
        bg_inds = np.arange(num_anchors) # all set as background

    fg_inds = np.where(labels > 0)[0]           # indices of positive anchors
    fg_max_overlap = None
    if len(gt_boxes) > 0:
        fg_max_overlap = anchor_to_gt_max[fg_inds]  # array of max iou of one positive anchor with all gt boxes
    gt_pos_inds = gt_ids[fg_inds]

    if len(gt_boxes) == 0:
        labels[:] = 0
    else:
        labels[bg_inds] = 0

        ###################### TODO: my modification #######################
        # aim to reduce number of negative samples.
        # zero_iou_anchor_mask = anchor_by_gt_overlap[bg_inds].any(-1)
        # bg_ignore_mask = np.random.choice(np.arange(bg_inds.shape[0])[zero_iou_anchor_mask], size=int(zero_iou_anchor_mask.sum()*0.1), replace=False)
        # labels[bg_inds[bg_ignore_mask]] = -1
        ###################### TODO: end ####################################
        # todo: Important
        # some gt box may have iou with anchors less than threshold and be taken as bg_inds, so needed to label re-assignment.
        # notice fg_inds and gt_ids keep unchanged for positive targets with assignment of negative labels.
        labels[pos_inds_force] = gt_classes[gt_inds_force]

    bbox_targets = np.zeros((num_anchors, box_code_size), dtype=all_anchors.dtype)   # (70400, 7)
    if len(gt_boxes) > 0:
        # see box_np_ops.second_box_encode
        bbox_targets[fg_inds, :] = box_encoding_fn(gt_boxes[anchor_to_gt_argmax[fg_inds], :], anchors[fg_inds, :])  # targets: (num_pos_anchor, 7)
        #bbox_targets[fg_inds, :] = box_encoding_fn(gt_boxes[gt_pos_inds, :], anchors[fg_inds, :])

    bbox_outside_weights = np.zeros((num_inside,), dtype=all_anchors.dtype)

    if norm_by_num_examples:  # False
        num_examples = np.sum(labels >= 0)  # neg + pos
        num_examples = np.maximum(1.0, num_examples)
        bbox_outside_weights[labels > 0] = 1.0 / num_examples
    else:
        bbox_outside_weights[labels > 0] = 1.0

    ret = { "labels": labels,                             # [70400,]
            "bbox_targets": bbox_targets,                 # [num_pos_anchors, 7]
            "bbox_outside_weights": bbox_outside_weights, # [70400,]
            "assigned_anchors_overlap": fg_max_overlap,   # [num_pos_anchors,], max iou with each anchor among all gt boxes
            "positive_gt_id": gt_pos_inds,                # [num_pos_anchors,], index of targeted gt boxes for positive anchors
            "assigned_anchors_inds": fg_inds,}            # [num_pos_anchors, ], indices of positive anchors

    return ret
