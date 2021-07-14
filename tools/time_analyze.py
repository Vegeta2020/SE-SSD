n = 500
import time
t1 = time.time()
for i in range(n):
    coll_mat = prep.box_collision_test(total_bv, total_bv)   # todo: too slow here
t2 = time.time()
print((t2-t1)/n)

import det3d.core.iou3d.iou3d_utils as iou3d
import torch
boxes_torch = torch.from_numpy(boxes).float()
t1 = time.time()
for i in range(n):
    x2 = iou3d.boxes_iou_bev_cpu(boxes_torch, boxes_torch)
t2 = time.time()
print((t2 - t1) / n)

t1 = time.time()
for i in range(n):
    x3 = iou3d.boxes_iou_bev_gpu(boxes_torch.cuda(), boxes_torch.cuda())
t2 = time.time()
print((t2 - t1) / n)