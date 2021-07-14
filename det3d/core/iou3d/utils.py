import numpy as np
import torch


def limit_period(val, offset=0.5, period=2 * np.pi):
    return val - np.floor(val / period + offset) * period

def limit_period_torch(val, offset=0.5, period=2 * np.pi):
    return val - torch.floor(val / period + offset) * period

def center_to_minmax_2d(centers, dims):
    return np.concatenate([centers - dims / 2, centers + dims / 2], axis=-1)

def center_to_minmax_2d_torch(centers, dims):
    return torch.cat([centers - dims / 2, centers + dims / 2], dim=-1)


def rbbox2d_to_near_bbox(in_boxes, box_mode='wlh', rect=False):
    """
       convert rotated bbox to nearest 'standing' or 'lying' bbox.
        Args:
            inboxes: [N, 5(x, y, w, l, ry)] or [N, 7(x,y,z,w,l,h,ry)]
        Returns:
            outboxes: [N, 4(xmin, ymin, xmax, ymax)] bboxes
    """
    if in_boxes.shape[-1] == 7:
        w_index, l_index = box_mode.index('w') + 3, box_mode.index('l') + 3
        if rect:
            in_boxes = in_boxes[:, [0, 2, w_index, l_index, -1]]
        else:
            in_boxes = in_boxes[:, [0, 1, w_index, l_index, -1]]

    elif in_boxes.shape[-1] == 5:
        w_index, l_index = box_mode.index('w') + 2, box_mode.index('l') + 2
        in_boxes = in_boxes[:, [0, 1, w_index, l_index, -1]]

    rots = in_boxes[..., -1]
    rots_0_pi_div_2 = np.abs(limit_period(rots, 0.5, np.pi))   # limit ry in range np.abs([-np.pi/2., np.pi/2.])
    # this line aims to rotate the box to a vertial or horizonal direction with abs(angle) less than 45'.
    cond = (rots_0_pi_div_2 > np.pi / 4)[..., np.newaxis]
    in_boxes_center = np.where(cond, in_boxes[:, [0, 1, 3, 2]], in_boxes[:, :4])  # if True, change w and l; otherwise keep the same;
    out_boxes = np.zeros([in_boxes.shape[0], 5], dtype=in_boxes.dtype)
    out_boxes[:, :4] = center_to_minmax_2d(in_boxes_center[:, :2], in_boxes_center[:, 2:])
    return out_boxes

def rbbox2d_to_near_bbox_torch(in_boxes, box_mode='wlh', rect=False):
    """
       convert rotated bbox to nearest 'standing' or 'lying' bbox.
        Args:
            inboxes: [N, 5(x, y, w, l, ry)] or [N, 7(x,y,z,w,l,h,ry)]
        Returns:
            outboxes: [N, 4(xmin, ymin, xmax, ymax)] bboxes
    """
    if in_boxes.shape[-1] == 7:
        w_index, l_index = box_mode.index('w') + 3, box_mode.index('l') + 3
        if rect:
            in_boxes = in_boxes[:, [0, 2, w_index, l_index, -1]]
        else:
            in_boxes = in_boxes[:, [0, 1, w_index, l_index, -1]]

    elif in_boxes.shape[-1] == 5:
        w_index, l_index = box_mode.index('w') + 2, box_mode.index('l') + 2
        in_boxes = in_boxes[:, [0, 1, w_index, l_index, -1]]

    rots = in_boxes[..., -1]
    rots_0_pi_div_2 = torch.abs(limit_period_torch(rots, 0.5, np.pi))   # limit ry in range np.abs([-np.pi/2., np.pi/2.])
    # this line aims to rotate the box to a vertial or horizonal direction with abs(angle) less than 45'.
    cond = (rots_0_pi_div_2 > np.pi / 4).unsqueeze(-1)
    in_boxes_center = torch.where(cond, in_boxes[:, [0, 1, 3, 2]], in_boxes[:, :4])  # if True, change w and l; otherwise keep the same;
    out_boxes = torch.zeros([in_boxes.shape[0], 5], dtype=in_boxes.dtype)
    out_boxes[:, :4] = center_to_minmax_2d_torch(in_boxes_center[:, :2], in_boxes_center[:, 2:])
    return out_boxes

def boxes3d_to_bev_torch(boxes3d, box_mode='wlh',rect=False):
    """
    Input(torch):
        boxes3d: (N, 7) [x, y, z, h, w, l, ry]
        rect: True/False means boxes in camera/velodyne coord system.
    Output:
        boxes_bev: (N, 5) [x1, y1, x2, y2, ry/rz], left-bottom: (x1, y1), right-top: (x2, y2), ry/rz: clockwise rotation angle
    """
    boxes_bev = boxes3d.new(torch.Size((boxes3d.shape[0], 5)))
    if boxes3d.shape[-1] == 5:
        w_index, l_index = box_mode.index('w') + 2, box_mode.index('l') + 2
    elif boxes3d.shape[-1] == 7:
        w_index, l_index = box_mode.index('w') + 3, box_mode.index('l') + 3
    else:
        raise NotImplementedError

    half_w, half_l = boxes3d[:, w_index] / 2., boxes3d[:, l_index] / 2.
    if rect:
        cu, cv = boxes3d[:, 0], boxes3d[:, 2]   # cam coord: x, z
        boxes_bev[:, 0], boxes_bev[:, 1] = cu - half_l, cv - half_w  # left-bottom in cam coord
        boxes_bev[:, 2], boxes_bev[:, 3] = cu + half_l, cv + half_w  # right-top in cam coord
    else:
        cu, cv = boxes3d[:, 0], boxes3d[:, 1]   # velo coord: x, y
        boxes_bev[:, 0], boxes_bev[:, 1] = cu - half_w, cv - half_l  # left-bottom in velo coord
        boxes_bev[:, 2], boxes_bev[:, 3] = cu + half_w, cv + half_l  # right-top in cam coord
    # rz in velo should have the same effect as ry of cam coord in 2D. Points in box will clockwisely rotate with rz/ry angle.
    boxes_bev[:, 4] = boxes3d[:, -1]
    return boxes_bev


def boxes3d_to_bev_3d_torch(boxes3d, box_mode='wlh', rect=False):
    """
    Input:
        boxes3d: (N, 7) [x, y, z, h, w, l, ry]
        rect: True/False means boxes in camera/velodyne coord system.
    Output:
        boxes_bev: (N, 7) [x1, y1, z1, x2, y2, z2, ry/rz], neither velo nor cam coord
        left-bottom:(x1, y1, z1), right-top: (x2, y2, z2), ry/rz: clockwise rotation angle
    """
    boxes_bev = boxes3d.new(torch.Size((boxes3d.shape[0], 7)))
    w_index, l_index, h_index = box_mode.index('w') + 3, box_mode.index('l') + 3, box_mode.index('h') + 3
    half_w, half_l, height = boxes3d[:, w_index] / 2., boxes3d[:, l_index] / 2., boxes3d[:, h_index]
    if rect:
        cu, cv, cw = boxes3d[:, 0], boxes3d[:, 2], boxes3d[:, 1]   # cam coord: x, z, y
        boxes_bev[:, 0], boxes_bev[:, 1], boxes_bev[:, 2] = cu - half_l, cv - half_w, cw - height   # left-bottom in cam coord
        boxes_bev[:, 3], boxes_bev[:, 4], boxes_bev[:, 5] = cu + half_l, cv + half_w, cw            # right-top in cam coord
    else:
        cu, cv, cw = boxes3d[:, 0], boxes3d[:, 1], boxes3d[:, 2]   # velo coord: x, y, z
        boxes_bev[:, 0], boxes_bev[:, 1], boxes_bev[:, 2] = cu - half_w, cv - half_l, cw - height / 2.           # left-bottom in velo coord
        boxes_bev[:, 3], boxes_bev[:, 4], boxes_bev[:, 5] = cu + half_w, cv + half_l, cw + height / 2.  # right-top in cam coord
    # rz in velo should have the same effect as ry of cam coord in 2D. Points in box will clockwisely rotate with rz/ry angle.
    boxes_bev[:, 6] = boxes3d[:, 6]
    return boxes_bev

'''
# for debug
box_a = np.array([[ 5.8137197 ,  0.19487036,  1.68447933,  3.42000344, -1.48419428]])
print(rbbox2d_to_near_bbox(box_a))
print(torch.from_numpy(box_a).dtype)
print(rbbox2d_to_near_bbox_torch(torch.from_numpy(box_a)))
'''
