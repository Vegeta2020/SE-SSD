import torch
import torch.nn as nn
import numpy as np
import numba

from ..registry import LOSSES
from .utils import weighted_loss
from det3d.core.bbox.box_torch_ops import center_to_corner_box2d, corner_to_standup_nd

'''
@numba.jit(nopython=True)
def iou_for_loss_jit(boxes, query_boxes, eps=1.0):
    """
        boxes: (K, 4) ndarray of float, (xmin, ymin, xmax, ymax).
        query_boxes: (K, 4) ndarray of float, (xmin, ymin, xmax, ymax).
    Returns
        overlaps: (K,) ndarray of overlap between boxes and query_boxes
    """
    K = boxes.shape[0]
    overlaps = np.zeros((K,), dtype=boxes.dtype)
    iou_bev = np.zeros((K,), dtype=boxes.dtype)
    for k in range(K):
        box_area = (query_boxes[k, 2] - query_boxes[k, 0] + eps) * (query_boxes[k, 3] - query_boxes[k, 1] + eps)
        iw = (min(boxes[k, 2], query_boxes[k, 2]) - max(boxes[k, 0], query_boxes[k, 0]) + eps)
        if iw > 0:
            ih = (min(boxes[k, 3], query_boxes[k, 3]) - max(boxes[k, 1], query_boxes[k, 1]) + eps)
            if ih > 0:
                ua = ((boxes[k, 2] - boxes[k, 0] + eps) * (boxes[k, 3] - boxes[k, 1] + eps) + box_area - iw * ih)
                overlaps[k] = iw * ih
                iou_bev[k] = overlaps[k] / ua
    return overlaps, iou_bev
'''


def bbox_overlaps(bboxes1, bboxes2, offset, eps):
    """Calculate overlap between two set of bboxes.
    Args:
        bboxes1 (Tensor): shape (m, 4), (xmin, ymin, xmax, ymax)
        bboxes2 (Tensor): shape (n, 4)
    Returns:
        ious(Tensor): shape (m, n).
    """
    left_top = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])      # [m, n, 2]
    right_bottom = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [m, n, 2]

    wh = (right_bottom - left_top + offset).clamp(min=0)            # [m, n, 2]
    overlap = wh[:, :, 0] * wh[:, :, 1]                             # [m, n]
    area1 = (bboxes1[:, 2] - bboxes1[:, 0] + offset) * (bboxes1[:, 3] - bboxes1[:, 1] + offset)   # [m,]
    area2 = (bboxes2[:, 2] - bboxes2[:, 0] + offset) * (bboxes2[:, 3] - bboxes2[:, 1] + offset)   # [n,]
    iou_bev = (overlap / (area1[:, None] + area2 - overlap)).clamp(min=eps, max=1-eps)   # [m, n]
    return iou_bev, overlap

def iou3d_eval(pred, target, iou_type="iou3d", offset=1e-6, eps=1e-6):
    """
        IoU: this is not strictly iou, it use standup boxes to cal iou.
            Computing the IoU between a set of predicted bboxes and target bboxes.
            The loss is calculated as negative log of IoU.
        Args:
            pred (Tensor): Predicted bboxes of format (x, y, z, w, l, h, ry) in velo coord, shape (m, 7).
            target (Tensor): Corresponding gt bboxes, shape (n, 7).
            eps (float): Eps to avoid log(0).
        Return:
            Tensor: Loss tensor. shape (m, n).
    """
    pred_corner_2d = center_to_corner_box2d(pred[:, 0:2], pred[:, 3:5], pred[:, -1])          # rotated corners: (m, 4, 2)
    pred_corner_sd = corner_to_standup_nd(pred_corner_2d)                                     # (m, 4), (xmin, ymin, xmax, ymax).
    target_corner_2d = center_to_corner_box2d(target[:, 0:2], target[:, 3:5], target[:, -1])  # (n, 4, 2)
    target_corner_sd = corner_to_standup_nd(target_corner_2d)                                 # (n, 4)
    iou_bev, overlap_bev = bbox_overlaps(pred_corner_sd, target_corner_sd, offset, eps)       # (m, n)

    pred_corner_2d = center_to_corner_box2d(pred[:, 1:3], pred[:, 4:6])                  # rotated corners: (m, 4, 2)
    pred_corner_sd = corner_to_standup_nd(pred_corner_2d)                                # (m, 4), (ymin, zmin, ymax, zmax).
    target_corner_2d = center_to_corner_box2d(target[:, 1:3], target[:, 4:6])            # (n, 4, 2)
    target_corner_sd = corner_to_standup_nd(target_corner_2d)                            # (n, 4)
    iou_face, overlap_face = bbox_overlaps(pred_corner_sd, target_corner_sd, offset, eps)  # (m, n)

    iou3d = None
    if iou_type == "iou3d":
        pred_height_min = (pred[:, 2] - pred[:, 5]).view(-1, 1)        # z - h, (m, 1)
        pred_height_max = pred[:, 2].view(-1, 1)                       # z
        target_height_min = (target[:, 2] - target[:, 5]).view(1, -1)  # (1, n)
        target_height_max = target[:, 2].view(1, -1)

        max_of_min = torch.max(pred_height_min, target_height_min)  # (m, 1)
        min_of_max = torch.min(pred_height_max, target_height_max)  # (1, n)
        overlap_h = torch.clamp(min_of_max - max_of_min, min=0)     # (m, n)
        overlap_3d = overlap_bev * overlap_h                        # (m, n)

        pred_vol = (pred[:, 3] * pred[:, 4] * pred[:, 5]).view(-1, 1)  # (m, 1)
        target_vol = (target[:, 3] * target[:, 4] * target[:, 5]).view(1, -1)  # (1, n)  -> broadcast (m, n)
        iou3d = (overlap_3d / torch.clamp(pred_vol + target_vol - overlap_3d, min=eps)).clamp(min=eps, max=1.0)

    return iou_bev, iou_face, iou3d

@LOSSES.register_module
class IoU3DLoss(nn.Module):
    def __init__(self, iou_type='iou3d', offset=1e-6, eps=1e-6, loss_weight=1.0):
        super(IoU3DLoss, self).__init__()
        self.iou_type = iou_type
        self.eps = eps
        self.offset = offset
        self.loss_weight = loss_weight


    def forward(self, pred, target, weights=None, **kwargs):
        """
            pred: [m, 7], (x,y,z,w,l,h,ry) in velo coord.
            target: [m, 7], (x,y,z,w,l,h,ry) in velo coord.
            iou_type: "iou3d" or "iou_bev"
            Boxes in pred and target should be matched one by one for calculation of iou loss.
        """
        pred = pred.float()
        target = target.float()

        valid_mask = (pred[:, 3] > 0) & (pred[:, 4] > 0) & (pred[:, 5] > 0)
        pred = pred[valid_mask]
        target = target[valid_mask]

        num_pos_pred = pred.shape[0]
        iou_loss = torch.tensor(0.0, dtype=torch.float32).cuda()

        if num_pos_pred > 0:
            diag = torch.arange(num_pos_pred)
            iou_bev, iou_face, iou3d = iou3d_eval(pred, target, self.iou_type, self.offset, self.eps)

            if self.iou_type == 'iou3d':
                log_iou3d = -iou3d.log()
                iou_loss = log_iou3d[diag, diag].sum() / num_pos_pred
            else:
                log_iou_face = - iou_face.log()
                log_iou_bev = - iou_bev.log()
                iou_loss = (log_iou_bev[diag, diag].sum() + log_iou_face[diag, diag].sum()) / num_pos_pred

        return iou_loss * self.loss_weight

 # if __name__ == "__main__":
 #    box_a = torch.from_numpy(np.array([0, 0, 0, 2, 2, 2, 0], dtype=np.float32).reshape(-1, 7))
 #    box_b = torch.from_numpy(np.array([0, 0, 0, 2, 2, 2, 0], dtype=np.float32).reshape(-1, 7))
 #    box_a = np.array([21.4, 1., 56.6, 1.52563191, 1.6285674, 3.8831164, 0.], dtype=np.float32).reshape(-1, 7)
 #    box_b = np.array([20.86000061, 2.50999999, 56.68999863, 1.67999995, 1.38999999, 4.26000023, 3.04999995], dtype=np.float32).reshape(-1, 7)
 #
 #    box_a = torch.from_numpy(box_a)
 #    box_b = torch.from_numpy(box_b)
 #
 #    iou3dloss = IoU3DLoss()
 #    print(iou3dloss(box_a, box_b))
 #
 #    import ipdb; ipdb.set_trace()
