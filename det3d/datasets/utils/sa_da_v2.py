
import numpy as np
import numba
import torch

from det3d.core.bbox import box_np_ops
from det3d.core.bbox.geometry import points_in_convex_polygon_3d_jit

from scipy.spatial import cKDTree
from ifp import ifp_sample
import traceback


def one_hot(x, num_class=None):
    if not num_class:
        num_class = np.max(x) + 1
    ohx = np.zeros((len(x), num_class))
    ohx[range(len(x)), x] = 1
    return ohx

def get_pyramids(gt_boxes):
    '''
        corners (in velo coord): [N, 8, 3]
            6 -------- 5
           /|         /|
          2 -------- 1 .
          | |        | |
          . 7 -------- 4
          |/         |/
          3 -------- 0

    gt_boxes: [N, 7].
    return [N, 6, 15=5*3] pyramids.
    surface order:
             3
     ----------------
     |              |            ^ x
    4|     5(6)     |2           |
     |              |      y     |
     ----------------      <------
             1
    '''
    pyramid_orders = np.array([
        [0, 1, 5, 4],
        [4, 5, 6, 7],
        [7, 6, 2, 3],
        [3, 2, 1, 0],
        [1, 2, 6, 5],
        [0, 4, 7, 3]
    ])
    boxes_corners = box_np_ops.center_to_corner_box3d(gt_boxes[:, 0:3], gt_boxes[:, 3:6], gt_boxes[:, 6], \
                                                      origin=[0.5, 0.5, 0.5], axis=2).reshape(-1, 24)
    pyramid_list = []
    for order in pyramid_orders:
        # frustum polygon: 5 corners, 5 surfaces
        pyramid = np.concatenate((gt_boxes[:, 0:3],
                                  boxes_corners[:, 3*order[0] : 3*order[0]+3],
                                  boxes_corners[:, 3*order[1] : 3*order[1]+3],
                                  boxes_corners[:, 3*order[2] : 3*order[2]+3],
                                  boxes_corners[:, 3*order[3] : 3*order[3]+3]), axis=1)
        pyramid_list.append(pyramid[:, None, :])
    pyramids = np.concatenate(pyramid_list, axis=1) # [N, 6, 15], 15=5*3
    return pyramids


def points_in_pyramids_mask(points, pyramids):
    '''
    pyramids: [N', 15]
    surfaces: [N', 5, 3, 3]
               N': num of pyramids, '5: num of surfaces per pyramid, 3: num of corners per surface, 3: dim of each corner
    '''
    indices = [1, 2, 0, 2, 3, 0, 3, 4, 0, 4, 1, 0, 4, 3, 2]
    surfaces = np.concatenate([pyramids[:, 3*k : 3*k+3] for k in indices], axis=1).reshape(-1, 5, 3, 3)
    point_masks = points_in_convex_polygon_3d_jit(points[:, :3], surfaces)
    return point_masks

def pyramid_augment_v0(gt_boxes, points,
                       enable_sa_dropout=0.1,
                       enable_sa_sparsity=[0.05, 50],
                       enable_sa_swap=[0.05, 50],
                       ):

    try:
        pyramids = get_pyramids(gt_boxes)
        # dropout
        if enable_sa_dropout is not None and gt_boxes.shape[0] > 0:
            drop_prob = enable_sa_dropout
            drop_pyramid_indices = np.random.randint(0, 6, (pyramids.shape[0]))
            drop_pyramid_one_hot = one_hot(drop_pyramid_indices, num_class=6)
            drop_box_mask = np.random.uniform(0, 1, (pyramids.shape[0])) <= drop_prob

            drop_pyramid_mask = (np.tile(drop_box_mask, [6, 1]).transpose(1, 0) * drop_pyramid_one_hot) > 0
            drop_pyramids = pyramids[drop_pyramid_mask]
            point_masks = points_in_pyramids_mask(points, drop_pyramids)
            points = points[np.logical_not(point_masks.any(-1))]
            pyramids = pyramids[np.logical_not(drop_box_mask)]

        # sparsify
        if enable_sa_sparsity is not None and pyramids.shape[0] > 0:
            sparsity_prob, sparsity_num = enable_sa_sparsity
            sparsify_pyramid_indices = np.random.randint(0, 6, (pyramids.shape[0]))
            sparsify_pyramid_one_hot = one_hot(sparsify_pyramid_indices, num_class=6)
            sparsify_box_mask = np.random.uniform(0, 1, (pyramids.shape[0])) <= sparsity_prob
            sparsify_pyramid_mask = (np.tile(sparsify_box_mask, [6, 1]).transpose(1, 0) * sparsify_pyramid_one_hot) > 0

            point_masks = points_in_pyramids_mask(points, pyramids.reshape(-1, 15))
            pyramid_points_num = point_masks.sum(0)
            valid_pyramid_mask = pyramid_points_num > sparsity_num
            sparsify_pyramid_mask = sparsify_pyramid_mask & valid_pyramid_mask.reshape(-1, 6)
            sparsify_pyramids = pyramids[sparsify_pyramid_mask]

            if sparsify_pyramids.shape[0] > 0:
                point_masks = points_in_pyramids_mask(points, sparsify_pyramids)
                remain_points = points[np.logical_not(point_masks.any(-1))]
                to_sparsify_points = [points[point_masks[:, i]] for i in range(point_masks.shape[1])]

                sparsified_points = []
                for sample in to_sparsify_points:
                    dists, indices = cKDTree(sample[:, 0:3]).query(sample[:, 0:3], sample.shape[0])
                    sampled_indices = ifp_sample(dists, indices, sparsity_num)
                    sparsified_points.append(sample[sampled_indices])
                sparsified_points = np.concatenate(sparsified_points, axis=0)
                points = np.concatenate([remain_points, sparsified_points], axis=0)
            pyramids = pyramids[np.logical_not(sparsify_box_mask)]

        # swap partition
        if enable_sa_swap is not None:
            swap_prob, num_thres = enable_sa_swap
            swap_pyramid_mask = np.random.uniform(0, 1, (pyramids.shape[0])) <= swap_prob

            if swap_pyramid_mask.sum() > 0:
                point_masks = points_in_pyramids_mask(points, pyramids.reshape(-1, 15))
                point_nums = point_masks.sum(0).reshape(pyramids.shape[0], -1)  # [N, 6]
                non_zero_pyramids_mask = point_nums > num_thres  # ingore dropout pyramids or highly occluded pyramids
                selected_pyramids = non_zero_pyramids_mask * swap_pyramid_mask[:, None]  # selected boxes and all their valid pyramids

                if selected_pyramids.sum() > 0:
                    # get to_swap pyramids
                    index_i, index_j = np.nonzero(selected_pyramids)
                    selected_pyramid_indices = [np.random.choice(index_j[index_i == i]) \
                                                    if e and (index_i == i).any() else 0 for i, e in
                                                enumerate(swap_pyramid_mask)]
                    selected_pyramids_mask = selected_pyramids * one_hot(selected_pyramid_indices, num_class=6) == 1
                    to_swap_pyramids = pyramids[selected_pyramids_mask]

                    # get swapped pyramids
                    index_i, index_j = np.nonzero(selected_pyramids_mask)
                    non_zero_pyramids_mask[selected_pyramids_mask] = False
                    swapped_index_i = np.array([np.random.choice(np.where(non_zero_pyramids_mask[:, j])[0]) if \
                                                    np.where(non_zero_pyramids_mask[:, j])[0].shape[0] > 0 else
                                                index_i[i] for i, j in enumerate(index_j.tolist())])
                    swapped_indicies = np.concatenate([swapped_index_i[:, None], index_j[:, None]], axis=1)
                    swapped_pyramids = pyramids[swapped_indicies[:, 0].astype(np.int32), swapped_indicies[:, 1].astype(np.int32)]

                    # concat to_swap&swapped pyramids
                    swap_pyramids = np.concatenate([to_swap_pyramids, swapped_pyramids], axis=0)
                    swap_point_masks = points_in_pyramids_mask(points, swap_pyramids)
                    remain_points = points[np.logical_not(swap_point_masks.any(-1))]

                    # swap pyramids
                    points_res = []
                    num_swapped_pyramids = swapped_pyramids.shape[0]
                    for i in range(num_swapped_pyramids):
                        to_swap_pyramid = to_swap_pyramids[i]
                        swapped_pyramid = swapped_pyramids[i]

                        to_swap_points = points[swap_point_masks[:, i]]
                        swapped_points = points[swap_point_masks[:, i + num_swapped_pyramids]]
                        # for intensity transform
                        to_swap_points_intensity_ratio = (to_swap_points[:, -1:] - to_swap_points[:, -1:].min()) / \
                                                         np.clip((to_swap_points[:, -1:].max() - to_swap_points[:, -1:].min()), 1e-6, 1)
                        swapped_points_intensity_ratio = (swapped_points[:, -1:] - swapped_points[:, -1:].min()) / \
                                                         np.clip((swapped_points[:, -1:].max() - swapped_points[:, -1:].min()), 1e-6, 1)

                        to_swap_points_ratio = get_points_ratio(to_swap_points, to_swap_pyramid)
                        swapped_points_ratio = get_points_ratio(swapped_points, swapped_pyramid)
                        new_to_swap_points = recover_points_by_ratio(swapped_points_ratio, to_swap_pyramid)
                        new_swapped_points = recover_points_by_ratio(to_swap_points_ratio, swapped_pyramid)
                        # for intensity transform
                        new_to_swap_points_intensity = recover_points_intensity_by_ratio(
                            swapped_points_intensity_ratio, to_swap_points[:, -1:].max(),
                            to_swap_points[:, -1:].min())
                        new_swapped_points_intensity = recover_points_intensity_by_ratio(
                            to_swap_points_intensity_ratio, swapped_points[:, -1:].max(),
                            swapped_points[:, -1:].min())

                        # new_to_swap_points = np.concatenate([new_to_swap_points, swapped_points[:, -1:]], axis=1)
                        # new_swapped_points = np.concatenate([new_swapped_points, to_swap_points[:, -1:]], axis=1)

                        new_to_swap_points = np.concatenate([new_to_swap_points, new_to_swap_points_intensity], axis=1)
                        new_swapped_points = np.concatenate([new_swapped_points, new_swapped_points_intensity], axis=1)

                        points_res.append(new_to_swap_points)
                        points_res.append(new_swapped_points)

                    points_res = np.concatenate(points_res, axis=0)
                    points = np.concatenate([remain_points, points_res], axis=0)

        return points.astype(np.float32)

    except Exception as e:
        traceback.print_exc()
        print(e)
        import ipdb; ipdb.set_trace()


def get_points_ratio(points, pyramid):
    surface_center = (pyramid[3:6] + pyramid[6:9] + pyramid[9:12] + pyramid[12:]) / 4.0
    vector_0, vector_1, vector_2 = pyramid[6:9] - pyramid[3:6],  pyramid[12:] - pyramid[3:6], pyramid[0:3] - surface_center
    alphas = ((points[:, 0:3] - pyramid[3:6]) * vector_0).sum(-1) / np.power(vector_0, 2).sum()
    betas = ((points[:, 0:3] - pyramid[3:6]) * vector_1).sum(-1) / np.power(vector_1, 2).sum()
    gammas = ((points[:, 0:3] - surface_center) * vector_2).sum(-1) / np.power(vector_2, 2).sum()
    return [alphas, betas, gammas]

def recover_points_by_ratio(points_ratio, pyramid):
    alphas, betas, gammas = points_ratio
    surface_center = (pyramid[3:6] + pyramid[6:9] + pyramid[9:12] + pyramid[12:]) / 4.0
    vector_0, vector_1, vector_2 = pyramid[6:9] - pyramid[3:6], pyramid[12:] - pyramid[3:6], pyramid[0:3] - surface_center
    points = (alphas[:, None] * vector_0 + betas[:, None] * vector_1) + pyramid[3:6] + gammas[:, None] * vector_2
    return points

def recover_points_intensity_by_ratio(points_intensity_ratio, max_intensity, min_intensity):
    return points_intensity_ratio * (max_intensity - min_intensity) + min_intensity



