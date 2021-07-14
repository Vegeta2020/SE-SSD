from det3d import torchie
import numpy as np
import torch

from ..registry import PIPELINES


class DataBundle(object):
    def __init__(self, data):
        self.data = data


@PIPELINES.register_module
class Reformat(object):
    def __init__(self, **kwargs):
        pass

    def __call__(self, res, info):
        meta = res["metadata"]
        points = res["lidar"]["points"]
        voxels = res["lidar"]["voxels"]
        anchors = res["lidar"]["targets"]["anchors"]

        data_bundle = dict(
            metadata=meta,                     # image_prefix/shape/id/num_points_features;
            points=points,                     # dim:5, padding with batch_id like coors;
            voxels=voxels["voxels"],           # [num_voxels, max_num_points(T/5), point_dim(4)]
            shape=voxels["shape"],             # [[1408, 1600,   40], [1408, 1600,   40]]
            num_points=voxels["num_points"],   # record num_points in each voxel;
            num_voxels=voxels["num_voxels"],   # num_voxels in each sample;
            coordinates=voxels["coordinates"], # coor and batch_id of each voxel;
            anchors=anchors,                   # anchors, only one group
        )

        if "points_raw" in res["lidar"].keys():
            data_bundle["points_raw"] = res["lidar"]["points_raw"]
            data_bundle["voxels_raw"] = res["lidar"]["voxels_raw"]["voxels"]
            data_bundle["shape_raw"] = res["lidar"]["voxels_raw"]["shape"]
            data_bundle["num_points_raw"] = res["lidar"]["voxels_raw"]["num_points"]
            data_bundle["num_voxels_raw"] = res["lidar"]["voxels_raw"]["num_voxels"]
            data_bundle["coordinates_raw"] = res["lidar"]["voxels_raw"]["coordinates"]
            data_bundle["anchors_raw"] = res["lidar"]["targets_raw"]["anchors"]

        if "anchors_mask" in res["lidar"]["targets"].keys():
            anchors_mask = res["lidar"]["targets"]["anchors_mask"]
            data_bundle.update(dict(anchors_mask=anchors_mask))

        if res["mode"] == "val":
            data_bundle.update(dict(metadata=meta,))

        calib = res.get("calib", None)
        data_bundle.update(dict(calib=calib)) if calib is not None else None

        if res["mode"] != "test":
            annos = res["lidar"]["annotations"]
            data_bundle.update(annos=annos,)
            if "annotations_raw" in res["lidar"].keys():
                annos_raw = res["lidar"]["annotations_raw"]
                data_bundle.update(annos_raw=annos_raw,)

        if res["mode"] == "train" and res['labeled']:
            ground_plane = res["lidar"].get("ground_plane", None)
            labels = res["lidar"]["targets"].get("labels", None)
            reg_targets = res["lidar"]["targets"].get("reg_targets", None)
            reg_weights = res["lidar"]["targets"].get("reg_weights", None)
            positive_gt_id = dict(positive_gt_id=res["lidar"]["targets"].get("positive_gt_id", None))

            data_bundle.update(dict(ground_plane=ground_plane)) if ground_plane is not None else None
            data_bundle.update(dict(labels=labels)) if labels is not None else None
            data_bundle.update(dict(reg_targets=reg_targets)) if reg_targets is not None else None
            data_bundle.update(dict(reg_weights=reg_weights)) if reg_weights is not None else None
            data_bundle.update(dict(positive_gt_id=positive_gt_id)) if positive_gt_id is not None else None

            # import ipdb; ipdb.set_trace()

            if "targets_raw" in res["lidar"].keys():
                data_bundle.update(dict(labels_raw=res["lidar"]["targets_raw"]["labels"]))
                data_bundle.update(dict(reg_targets_raw=res["lidar"]["targets_raw"]["reg_targets"]))
                data_bundle.update(dict(reg_weights_raw=res["lidar"]["targets_raw"]["reg_weights"]))
                data_bundle.update(dict(positive_gt_id_raw={"positive_gt_id": res["lidar"]["targets_raw"]["positive_gt_id"]}))
                data_bundle.update(dict(transformation=res["lidar"]["transformation"]))

        if res["mode"] == "train" and not res['labeled']:
            data_bundle.update(dict(transformation=res["lidar"]["transformation"]))

        return data_bundle, info


@PIPELINES.register_module
class PointCloudCollect(object):
    def __init__(
        self,
        keys,
        meta_keys=(
            "filename",
            "ori_shape",
            "img_shape",
            "pad_shape",
            "scale_factor",
            "flip",
            "img_norm_cfg",
        ),
    ):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, info):

        results = info["res"]

        data = {}
        img_meta = {}

        for key in self.meta_keys:
            img_meta[key] = results[key]
        data["img_meta"] = DC(img_meta, cpu_only=True)

        for key in self.keys:
            data[key] = results[key]
        return data

    def __repr__(self):
        return self.__class__.__name__ + "(keys={}, meta_keys={})".format(
            self.keys, self.meta_keys
        )
