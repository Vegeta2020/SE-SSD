import torch
import iou3d_cuda
import sys
import det3d.core.iou3d.utils as utils


def boxes_iou_bev_cpu(boxes_a, boxes_b, box_mode='wlh', metric='rotate_iou', rect=False):
    '''
    The box_np_ops.riou_cc is about 1.7x faster than the rotate boxes_iou_bev_cpu.
    The box_np_ops.iou_jit is about 20x faster than the nearest boxes_iou_bev_cpu.
    Input:
        boxes_a: (N, 7), [x, y, z, h, w, l, ry], torch tensor with type float32
        boxes_b: (M, 7)
        rect: True (rect coord), False (velo coord)
    Output:
        iou_bev: (N, M)
    '''
    if metric == 'rotate_iou':
        boxes_a_bev = utils.boxes3d_to_bev_torch(boxes_a, box_mode, rect)
        boxes_b_bev = utils.boxes3d_to_bev_torch(boxes_b, box_mode, rect)
    elif metric == 'nearest_iou':
        boxes_a_bev = utils.rbbox2d_to_near_bbox_torch(boxes_a, box_mode, rect)
        boxes_b_bev = utils.rbbox2d_to_near_bbox_torch(boxes_b, box_mode, rect)
    else:
        raise NotImplementedError
    iou_bev = torch.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    iou3d_cuda.boxes_iou_bev_cpu(boxes_a_bev.contiguous(), boxes_b_bev.contiguous(), iou_bev)

    return iou_bev


def boxes_iou_bev_gpu(boxes_a, boxes_b, box_mode='wlh', metric='rotate_iou', rect=False):
    """
    This rotate boxes_iou_bev_gpu is about 9.25x faster than box_np_ops.riou_cc.
    The box_np_ops.iou_jit is about 4.1x faster than the nearest boxes_iou_bev_gpu.
    :param boxes_a: (M, 7), [x, y, z, h, w, l, ry], torch tensor with type float32
    :param boxes_b: (N, 7), [x, y, z, h, w, l, ry], torch tensor with type float32
    :return:
        ans_iou: (M, N)
    """
    if metric == 'rotate_iou':
        boxes_a_bev = utils.boxes3d_to_bev_torch(boxes_a, box_mode, rect)
        boxes_b_bev = utils.boxes3d_to_bev_torch(boxes_b, box_mode, rect)
    elif metric == 'nearest_iou':
        boxes_a_bev = utils.rbbox2d_to_near_bbox_torch(boxes_a, box_mode, rect).cuda()
        boxes_b_bev = utils.rbbox2d_to_near_bbox_torch(boxes_b, box_mode, rect).cuda()
    else:
        raise NotImplementedError

    ans_iou = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()
    iou3d_cuda.boxes_iou_bev_gpu(boxes_a_bev.contiguous(), boxes_b_bev.contiguous(), ans_iou)
    return ans_iou

def boxes_iou3d_cpu_test(boxes_a, boxes_b, box_mode='wlh', rect=False):
    """
    # todo: need to be test with boxes_iou3d_cpu()
    Input (torch):
        boxes_a: (N, 7) [x, y, z, h, w, l, ry], torch tensor with type float32
        boxes_b: (M, 7) [x, y, z, h, w, l, ry], torch tensor with type float32
        rect: True/False means boxes in camera/velodyne coord system.
    Output:
        iou_3d: (N, M)
    """
    boxes_a_bev_3d = utils.boxes3d_to_bev_3d_torch(boxes_a, box_mode, rect)
    boxes_b_bev_3d = utils.boxes3d_to_bev_3d_torch(boxes_b, box_mode, rect)

    # bev overlap
    iou3d = torch.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    iou3d_cuda.boxes_iou3d_cpu(boxes_a_bev_3d.contiguous(), boxes_b_bev_3d.contiguous(), iou3d)
    return iou3d

def boxes_iou3d_cpu(boxes_a, boxes_b, box_mode='wlh', rect=False, need_bev=False):
    """
    Input (torch):
        boxes_a: (N, 7) [x, y, z, h, w, l, ry], torch tensor with type float32
        boxes_b: (M, 7) [x, y, z, h, w, l, ry], torch tensor with type float32
        rect: True/False means boxes in camera/velodyne coord system.
    Output:
        iou_3d: (N, M)
    """
    w_index, l_index, h_index = box_mode.index('w') + 3, box_mode.index('l') + 3, box_mode.index('h') + 3
    boxes_a_bev = utils.boxes3d_to_bev_torch(boxes_a, box_mode, rect)
    boxes_b_bev = utils.boxes3d_to_bev_torch(boxes_b, box_mode, rect)

    overlaps_bev = torch.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    iou3d_cuda.boxes_overlap_bev_cpu(boxes_a_bev.contiguous(), boxes_b_bev.contiguous(), overlaps_bev)

    # bev iou
    area_a = (boxes_a[:, w_index] * boxes_a[:, l_index]).view(-1, 1)  # (N, 1)
    area_b = (boxes_b[:, w_index] * boxes_b[:, l_index]).view(1, -1)  # (1, M)  -> broadcast (N, M)
    iou_bev = overlaps_bev / torch.clamp(area_a + area_b - overlaps_bev, min=1e-7)

    # height overlap
    if rect:
        boxes_a_height_min = (boxes_a[:, 1] - boxes_a[:, h_index]).view(-1, 1)  # y - h
        boxes_a_height_max = boxes_a[:, 1].view(-1, 1)  # y
        boxes_b_height_min = (boxes_b[:, 1] - boxes_b[:, h_index]).view(1, -1)
        boxes_b_height_max = boxes_b[:, 1].view(1, -1)
    else:
        boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, h_index]).view(-1, 1)  # z - h, (N, 1)
        boxes_a_height_max = boxes_a[:, 2].view(-1, 1)  # z
        boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, h_index]).view(1, -1)  # (1, M)
        boxes_b_height_max = boxes_b[:, 2].view(1, -1)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)  # (N, 1)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)  # (1, M)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)  # (N, M)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h  # broadcast: (N, M)

    vol_a = (boxes_a[:, h_index] * boxes_a[:, w_index] * boxes_a[:, l_index]).view(-1, 1)  # (N, 1)
    vol_b = (boxes_b[:, h_index] * boxes_b[:, w_index] * boxes_b[:, l_index]).view(1, -1)  # (1, M)  -> broadcast (N, M)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-7)

    if need_bev:
        return iou3d, iou_bev

    return iou3d



def boxes_iou3d_gpu_test(boxes_a, boxes_b, box_mode='wlh', rect=False):
    """
    # todo: need to be test with boxes_iou3d_gpu()
    Input (torch):
        IMPORTANT: (x, y, z) is real center
        boxes_a: (N, 7) [x, y, z, h, w, l, ry], torch tensor with type float32
        boxes_b: (M, 7) [x, y, z, h, w, l, ry], torch tensor with type float32
        rect: True/False means boxes in camera/velodyne coord system.
    Output:
        iou_3d: (N, M)
    """
    boxes_a_bev_3d = utils.boxes3d_to_bev_3d_torch(boxes_a, box_mode, rect)
    boxes_b_bev_3d = utils.boxes3d_to_bev_3d_torch(boxes_b, box_mode, rect)

    # bev overlap
    iou3d = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    iou3d_cuda.boxes_iou3d_gpu(boxes_a_bev_3d.contiguous(), boxes_b_bev_3d.contiguous(), iou3d)
    return iou3d

def boxes_iou3d_gpu(boxes_a, boxes_b, box_mode='wlh', rect=False, need_bev=False):
    """
    Input (torch):
        boxes_a: (N, 7) [x, y, z, h, w, l, ry], torch tensor with type float32
        boxes_b: (M, 7) [x, y, z, h, w, l, ry], torch tensor with type float32
        rect: True/False means boxes in camera/velodyne coord system.
    Output:
        iou_3d: (N, M)
    """
    w_index, l_index, h_index = box_mode.index('w') + 3, box_mode.index('l') + 3, box_mode.index('h') + 3
    boxes_a_bev = utils.boxes3d_to_bev_torch(boxes_a, box_mode, rect)
    boxes_b_bev = utils.boxes3d_to_bev_torch(boxes_b, box_mode, rect)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    iou3d_cuda.boxes_overlap_bev_gpu(boxes_a_bev.contiguous(), boxes_b_bev.contiguous(), overlaps_bev)

    # bev iou
    area_a = (boxes_a[:, w_index] * boxes_a[:, l_index]).view(-1, 1)  # (N, 1)
    area_b = (boxes_b[:, w_index] * boxes_b[:, l_index]).view(1, -1)  # (1, M)  -> broadcast (N, M)
    iou_bev = overlaps_bev / torch.clamp(area_a + area_b - overlaps_bev, min=1e-7)

    # height overlap
    if rect:
        boxes_a_height_min = (boxes_a[:, 1] - boxes_a[:, h_index]).view(-1, 1)  # y - h
        boxes_a_height_max = boxes_a[:, 1].view(-1, 1)                    # y
        boxes_b_height_min = (boxes_b[:, 1] - boxes_b[:, h_index]).view(1, -1)
        boxes_b_height_max = boxes_b[:, 1].view(1, -1)
    else:
        # todo: notice if (x, y, z) is the real center
        half_h_a = boxes_a[:, h_index] / 2.0
        half_h_b = boxes_b[:, h_index] / 2.0
        boxes_a_height_min = (boxes_a[:, 2] - half_h_a).view(-1, 1)  # z - h/2, (N, 1)
        boxes_a_height_max = (boxes_a[:, 2] + half_h_a).view(-1, 1)  # z + h/2, (N, 1)
        boxes_b_height_min = (boxes_b[:, 2] - half_h_b).view(1, -1)
        boxes_b_height_max = (boxes_b[:, 2] + half_h_b).view(1, -1)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)   # (N, M)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)   # (N, M)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)         # (N, M)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h    # broadcast: (N, M)

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)   # (N, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)   # (1, M)  -> broadcast (N, M)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-7)

    if need_bev:
        return iou3d, iou_bev
    return iou3d


def boxes_aligned_iou3d_gpu(boxes_a, boxes_b, box_mode='wlh', rect=False, need_bev=False):
    """
    Input (torch):
        boxes_a: (N, 7) [x, y, z, w, l, h, ry], torch tensor with type float32.
        boxes_b: (N, 7) [x, y, z, w, l, h, ry], torch tensor with type float32.
        rect: True/False means boxes in camera/velodyne coord system.
        Notice: (x, y, z) are real center.
    Output:
        iou_3d: (N)
    """
    assert boxes_a.shape[0] == boxes_b.shape[0]
    w_index, l_index, h_index = box_mode.index('w') + 3, box_mode.index('l') + 3, box_mode.index('h') + 3
    boxes_a_bev = utils.boxes3d_to_bev_torch(boxes_a, box_mode, rect)
    boxes_b_bev = utils.boxes3d_to_bev_torch(boxes_b, box_mode, rect)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], 1))).zero_()  # (N, 1)
    iou3d_cuda.boxes_aligned_overlap_bev_gpu(boxes_a_bev.contiguous(), boxes_b_bev.contiguous(), overlaps_bev)

    # bev iou
    area_a = (boxes_a[:, w_index] * boxes_a[:, l_index]).view(-1, 1)  # (N, 1)
    area_b = (boxes_b[:, w_index] * boxes_b[:, l_index]).view(-1, 1)  # (N, 1)
    iou_bev = overlaps_bev / torch.clamp(area_a + area_b - overlaps_bev, min=1e-7)  # [N, 1]

    # height overlap
    if rect:
        raise NotImplementedError
        # boxes_a_height_min = (boxes_a[:, 1] - boxes_a[:, h_index]).view(-1, 1)  # y - h
        # boxes_a_height_max = boxes_a[:, 1].view(-1, 1)                    # y
        # boxes_b_height_min = (boxes_b[:, 1] - boxes_b[:, h_index]).view(1, -1)
        # boxes_b_height_max = boxes_b[:, 1].view(1, -1)
    else:
        # todo: notice if (x, y, z) is the real center
        half_h_a = boxes_a[:, h_index] / 2.0
        half_h_b = boxes_b[:, h_index] / 2.0
        boxes_a_height_min = (boxes_a[:, 2] - half_h_a).view(-1, 1)  # z - h/2, (N, 1)
        boxes_a_height_max = (boxes_a[:, 2] + half_h_a).view(-1, 1)  # z + h/2, (N, 1)
        boxes_b_height_min = (boxes_b[:, 2] - half_h_b).view(-1, 1)
        boxes_b_height_max = (boxes_b[:, 2] + half_h_b).view(-1, 1)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)   # (N, 1)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)   # (N, 1)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)         # (N, 1)

    # 3d iou
    overlaps_3d = overlaps_bev * overlaps_h

    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)   # (N, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(-1, 1)   # (N, 1)

    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-7)

    if need_bev:
        return iou3d, iou_bev

    return iou3d

def nms_gpu(boxes, scores, thresh, box_mode='wlh'):
    """
    filter overlapped boxes based on bev iou
    :param boxes(torch): (N, 7), [x, y, z, h, w, l, ry], torch tensor with type float32
    :param scores: (N)
    :param thresh: in the range [0, 1] for iou thresh
    :return:
    """
    # areas = (x2 - x1) * (y2 - y1)
    boxes = utils.boxes3d_to_bev_torch(boxes, box_mode, rect=True)

    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_cuda.nms_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous()

def nms_3d_gpu(boxes, scores, thresh, box_mode='wlh'):
    """
    filter overlapped boxes based on 3d iou
    :param boxes: (N, 7), [x, y, z, h, w, l, ry], torch tensor with type float32
    :param scores: (N)
    :param thresh: in the range [0, 1] for iou thresh
    :return:
    """
    boxes = utils.boxes3d_to_bev_3d_torch(boxes, box_mode, rect=False)

    order = scores.sort(0, descending=True)[1]
    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_cuda.nms_3d_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous()


def nms_normal_gpu(boxes, scores, thresh):
    """
    Overlap calculated differently from nms_gpu(): boxes without rotated
    :param boxes: (N, 5) [x1, y1, x2, y2, ry]
    :param scores: (N)
    :param thresh:
    :return:
    """
    # areas = (x2 - x1) * (y2 - y1)
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_cuda.nms_normal_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous()


# if __name__ == '__main__':
#     import numpy as np
#     import os
#     os.environ['CUDA_VISIBLE_DEVICES'] = '7'
#     box_a = torch.Tensor([1, 1, 1, 2, 2, 2, np.pi / 4]).view(-1, 7).float().cuda().repeat(2, 1)
#     box_b = torch.Tensor([0, 0, 0, 1, 1, 1, 0]).view(-1, 7).float().cuda().repeat(2, 1)
#
#     boxes_a = torch.rand(20, 7) * torch.tensor([10, 10, 3, 4, 4, 4, np.pi], dtype=torch.float32) \
#               + torch.tensor([0, -10, -1, 0, 0, 0, -np.pi], dtype=torch.float32)
#     boxes_b = torch.rand(20, 7) * torch.tensor([10, 10, 3, 4, 4, 4, np.pi], dtype=torch.float32) \
#               + torch.tensor([0, -10, -1, 0, 0, 0, -np.pi], dtype=torch.float32)
#
#     iou3d_0, iou_bev_0 = boxes_iou3d_gpu(boxes_a.cuda(), boxes_b.cuda(), rect=False, need_bev=True)
#     print(iou_bev_0, iou3d_0)
#
#     import ipdb; ipdb.set_trace()
#
#     iou3d = boxes_aligned_iou3d_gpu(boxes_a.cuda(), boxes_b.cuda(), rect=False)
#
#     print(iou3d)
