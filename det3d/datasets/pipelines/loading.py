import os.path as osp
import warnings
import numpy as np
from functools import reduce

import pycocotools.mask as maskUtils

from pathlib import Path
from copy import deepcopy
from det3d import torchie
from det3d.core.bbox import box_np_ops
from det3d.datasets.kitti import kitti_common as kitti

from ..registry import PIPELINES

'''
def read_file(path, tries=2, num_point_feature=4):
    points = None
    try_cnt = 0
    while points is None and try_cnt < tries:
        try_cnt += 1
        try:
            points = np.fromfile(path, dtype=np.float32)
            s = points.shape[0]
            if s % 5 != 0:
                points = points[: s - (s % 5)]
            points = points.reshape(-1, 5)[:, :num_point_feature]
        except Exception:
            points = None

    return points


def remove_close(points, radius: float) -> None:
    """
    Removes point too close within a certain radius from origin.
    :param radius: Radius below which points are removed.
    """
    x_filt = np.abs(points[0, :]) < radius
    y_filt = np.abs(points[1, :]) < radius
    not_close = np.logical_not(np.logical_and(x_filt, y_filt))
    points = points[:, not_close]
    return points


def read_sweep(sweep):
    min_distance = 1.0
    # points_sweep = np.fromfile(str(sweep["lidar_path"]),
    #                            dtype=np.float32).reshape([-1,
    #                                                       5])[:, :4].T
    points_sweep = read_file(str(sweep["lidar_path"])).T

    nbr_points = points_sweep.shape[1]
    if sweep["transform_matrix"] is not None:
        points_sweep[:3, :] = sweep["transform_matrix"].dot(
            np.vstack((points_sweep[:3, :], np.ones(nbr_points)))
        )[:3, :]
    # points_sweep[3, :] /= 255
    points_sweep = remove_close(points_sweep, min_distance)
    curr_times = sweep["time_lag"] * np.ones((1, points_sweep.shape[1]))

    return points_sweep.T, curr_times.T
'''

@PIPELINES.register_module
class LoadPointCloudFromFile(object):
    '''Loading points (x, y, z, r in velo coord) from pre-reduced .bin file'''
    def __init__(self, dataset="KittiDataset", **kwargs):
        self.type = dataset
        self.random_select = kwargs.get("random_select", False)
        self.npoints = kwargs.get("npoints", 16834)

    def __call__(self, res, info):
        # set datatype "KittiDataset"
        res["type"] = self.type

        if self.type == "KittiDataset":
            # get reduced points .bin file path
            pc_info = info["point_cloud"]
            velo_path = Path(pc_info["velodyne_path"])
            if not velo_path.is_absolute():
                velo_path = (Path(res["metadata"]["image_prefix"]) / pc_info["velodyne_path"])

            velo_reduced_path = (velo_path.parent.parent/ (velo_path.parent.stem + "_reduced") / velo_path.name)
            if velo_reduced_path.exists():
                velo_path = velo_reduced_path

            # load points: loaded points are in the image range
            points = np.fromfile(str(velo_path), dtype=np.float32, count=-1).reshape([-1, res["metadata"]["num_point_features"]])
            res["lidar"]["points"] = points

        else:
            raise NotImplementedError

        return res, info


@PIPELINES.register_module
class LoadPointCloudAnnotations(object):
    '''Loading gt_boxes and calib configs, remove dontcare objects;
       transform gt_boxes (xyz: cam -> velo; l,h,w -> w, l, h) and moved to the real center'''
    def __init__(self, with_bbox=True, **kwargs):
        self.enable_difficulty_level = kwargs.get("enable_difficulty_level", False)
        pass

    def __call__(self, res, info):

        if res["type"] in ["NuScenesDataset", "LyftDataset"] and "gt_boxes" in info:

            res["lidar"]["annotations"] = {
                "boxes": info["gt_boxes"].astype(np.float32),
                "names": info["gt_names"],
                "tokens": info["gt_boxes_token"],
                "velocities": info["gt_boxes_velocity"].astype(np.float32),
            }
        # True
        elif res["type"] == "KittiDataset":
            # only select useful calib matrix
            calib = info["calib"]
            calib_dict = {
                "rect": calib["R0_rect"],
                "Trv2c": calib["Tr_velo_to_cam"],
                "P2": calib["P2"],
                "frustum": box_np_ops.get_valid_frustum(calib["R0_rect"], calib["Tr_velo_to_cam"], calib["P2"], info["image"]["image_shape"]),
            }
            res["calib"] = calib_dict

            if "annos" in info:

                annos = info["annos"]
                # annos = kitti.remove_dontcare_v2(annos)  # todo: try my remove_dontcare_v2 later
                annos = kitti.remove_dontcare(annos)
                locs = annos["location"]   # x, y, z (cam)
                dims = annos["dimensions"] # l, h, w
                rots = annos["rotation_y"] # ry in label file, count from neg y-axis (0 degree), clockwise is postive,
                                           # anti-clockwise is negative.
                gt_names = annos["name"]
                gt_boxes = np.concatenate([locs, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
                calib = info["calib"]

                # x,y,z(cam), l, h, w, ry -> x,y,z(velo), w, l, h, ry
                gt_boxes = box_np_ops.box_camera_to_lidar(gt_boxes, calib["R0_rect"], calib["Tr_velo_to_cam"])
                # real center of gt_boxes_velo: z_value + h/2
                box_np_ops.change_box3d_center_(gt_boxes, [0.5, 0.5, 0], [0.5, 0.5, 0.5])

                res["lidar"]["annotations"] = {"boxes": gt_boxes, "names": gt_names,}    # without difficulty here
                if self.enable_difficulty_level:
                    res["lidar"]["annotations"].update({"difficulty": annos["difficulty"]})

                res["cam"]["annotations"] = {"boxes": annos["bbox"],"names": gt_names,}  # image 2d bbox
        else:
            return NotImplementedError

        return res, info
