import numpy as np
import pickle
import os

import os.path as osp

import warnings

from copy import deepcopy

from det3d.core.bbox import box_np_ops
from det3d.datasets.custom import PointCloudDataset
from det3d.datasets.registry import DATASETS


from det3d.datasets.kitti.kitti_common import *
from det3d.datasets.kitti.eval import get_official_eval_result, get_coco_eval_result, get_official_eval_result_v2
#from det3d.datasets.kitti.eval_2 import get_official_eval_result as get_official_eval_result_v2

from mmdet.datasets.transforms import (ImageTransform, BboxTransform)
from mmdet.datasets.utils import to_tensor, random_scale
from mmdet.datasets.kitti_utils import read_label, read_lidar, load_proposals, \
    project_rect_to_velo, Calibration, get_lidar_in_image_fov, \
    project_rect_to_image, project_rect_to_right
from mmdet.core.bbox3d.geometry import rbbox2d_to_near_bbox, filter_gt_box_outside_range, \
    sparse_sum_for_anchors_mask, fused_get_anchors_area, limit_period, center_to_corner_box3d, points_in_rbbox
import os
from mmdet.core.point_cloud.voxel_generator import VoxelGenerator
from mmdet.ops.points_op import points_op_cpu
from mmcv.parallel import DataContainer as DC
import mmcv

from mmdet.core.point_cloud import voxel_generator
from mmdet.core.point_cloud import point_augmentor
from mmdet.core.bbox3d import bbox3d_target
from mmdet.core.anchor import anchor3d_generator
from mmcv.runner import obj_from_dict

#@DATASETS.register_module
class KittiDataset(PointCloudDataset):
    NumPointFeatures = 4
    def __init__(self, root_path, info_path, cfg=None, pipeline=None, class_names=None, test_mode=False, **kwargs):
        super(KittiDataset, self).__init__(root_path, info_path, pipeline, test_mode=test_mode)
        assert self._info_path is not None
        if not hasattr(self, "_kitti_infos"):
            with open(self._info_path, "rb") as f:
                infos = pickle.load(f)
            self._kitti_infos = infos
        self._num_point_features = __class__.NumPointFeatures
        # print("remain number of infos:", len(self._kitti_infos))
        self._class_names = class_names
        self.plane_dir = root_path + "/training/planes"   # todo: check whether need it on val or test datasets

        ######## addition for sassd ##########
        self.anchor_area_threshold = 1
        self.sa_class_names = ['Car', 'Van']
        self.sample_ids = []
        for info in self._kitti_infos:
            self.sample_ids.append(int(info['image']['image_idx']))


        mode = info_path.split('.')[0].split('_')[-1]
        self.test_mode = mode != 'train'
        if mode in ['train', 'val', 'trainval']:
            subdir = '/training'
        elif mode == 'test':
            subdir = '/testing'
        else:
            raise NotImplementedError

        self.with_label = not self.test_mode
        self.img_prefix = root_path + subdir + "/image_2"
        self.label_prefix = root_path + subdir + "/label_2"
        self.calib_prefix = root_path + subdir + "/calib"
        self.lidar_prefix = "/mnt/proj50/zhengwu/KITTI-SA/object" + subdir + "/velodyne_reduced"
        img_norm_cfg = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
        self.img_transform = ImageTransform(size_divisor=32, **img_norm_cfg)

        anchor_generator_cfg = dict(
            type='AnchorGeneratorStride',
            sizes=[1.6, 3.9, 1.56],
            anchor_strides=[0.4, 0.4, 1.0],
            anchor_offsets=[0.2, -39.8, -1.78],
            rotations=[0, 1.57],
        )
        generator_cfg = dict(
            type='VoxelGenerator',
            voxel_size=[0.05, 0.05, 0.1],
            point_cloud_range=[0, -40., -3., 70.4, 40., 1.],
            max_num_points=5,
            max_voxels=20000
        )
        augmentor_cfg = dict(
            type='PointAugmentor',
            root_path="/mnt/proj50/zhengwu/KITTI/object/training",
            info_path="/mnt/proj50/zhengwu/KITTI-SA/object/kitti_dbinfos_train.pkl",   # x, y, z, l, h, w, ry.  bottom_center_z
            sample_classes=['Car'],
            min_num_points=5,
            sample_max_num=15,
            removed_difficulties=[-1],
            global_rot_range=[-0.78539816, 0.78539816],
            gt_rot_range=[-0.78539816, 0.78539816],
            center_noise_std=[1., 1., .5],
            scale_range=[0.95, 1.05]
        )



        self.anchor_generator = obj_from_dict(anchor_generator_cfg, anchor3d_generator)
        self.generator = obj_from_dict(generator_cfg, voxel_generator)
        self.augmentor = obj_from_dict(augmentor_cfg, point_augmentor)

        feature_map_size = self.generator.grid_size[:2] // 8
        feature_map_size = [*feature_map_size, 1][::-1]
        anchors = self.anchor_generator(feature_map_size)
        self.anchors = anchors.reshape([-1, 7])
        self.anchors_bv = rbbox2d_to_near_bbox(self.anchors[:, [0, 1, 3, 4, 6]])   # x, y, w, l, ry


    def __len__(self):
        if not hasattr(self, "_kitti_infos"):
            with open(self._info_path, "rb") as f:
                self._kitti_infos = pickle.load(f)

        return len(self._kitti_infos)

    def get_road_plane(self, idx):
        plane_file = os.path.join(self.plane_dir, "%06d.txt" % idx)
        with open(plane_file, "r") as f:
            lines = f.readlines()
        lines = [float(i) for i in lines[3].split()]
        plane = np.asarray(lines)

        # Ensure normal is always facing up, this is in the rectified camera coordinate
        if plane[1] > 0:
            plane = -plane

        norm = np.linalg.norm(plane[0:3])
        plane = plane / norm
        return plane

    @property
    def num_point_features(self):
        return self._num_point_features

    @property
    def ground_truth_annotations(self):
        if "annos" not in self._kitti_infos[0]:
            return None

        gt_annos = [info["annos"] for info in self._kitti_infos]

        return gt_annos

    def convert_detection_to_kitti_annos(self, detection, partial=False):
        class_names = self._class_names
        det_image_idxes = [k for k in detection.keys()]
        gt_image_idxes = [str(info["image"]["image_idx"]) for info in self._kitti_infos]
        image_idxes = [gt_image_idxes, det_image_idxes]
        # print(f"det_image_idxes: {det_image_idxes[:10]}")
        # print(f"gt_image_idxes: {gt_image_idxes[:10]}")
        annos = []
        # for i in range(len(detection)):
        for det_idx in image_idxes[int(partial==True)]:
            det = detection[det_idx]
            info = self._kitti_infos[gt_image_idxes.index(det_idx)]
            # info = self._kitti_infos[i]
            calib = info["calib"]
            rect = calib["R0_rect"]
            Trv2c = calib["Tr_velo_to_cam"]
            P2 = calib["P2"]
            final_box_preds = det["box3d_lidar"].detach().cpu().numpy()
            label_preds = det["label_preds"].detach().cpu().numpy()
            scores = det["scores"].detach().cpu().numpy()

            anno = get_start_result_anno()
            num_example = 0

            if final_box_preds.shape[0] != 0:
                final_box_preds[:, -1] = box_np_ops.limit_period(final_box_preds[:, -1], offset=0.5, period=np.pi * 2,)
                final_box_preds[:, 2] -= final_box_preds[:, 5] / 2   # center_z -> bottom_z

                # aim: x, y, z, w, l, h, r -> -y, -z, x, h, w, l, r
                # (x, y, z, w, l, h r) in lidar -> (x', y', z', l, h, w, r) in camera
                box3d_camera = box_np_ops.box_lidar_to_camera(final_box_preds, rect, Trv2c)
                camera_box_origin = [0.5, 1.0, 0.5]
                box_corners = box_np_ops.center_to_corner_box3d(box3d_camera[:, :3], box3d_camera[:, 3:6], box3d_camera[:, 6], camera_box_origin, axis=1,)
                box_corners_in_image = box_np_ops.project_to_image(box_corners, P2)
                # box_corners_in_image: [N, 8, 2]
                minxy = np.min(box_corners_in_image, axis=1)
                maxxy = np.max(box_corners_in_image, axis=1)
                bbox = np.concatenate([minxy, maxxy], axis=1)

                for j in range(box3d_camera.shape[0]):
                    image_shape = info["image"]["image_shape"]
                    if bbox[j, 0] > image_shape[1] or bbox[j, 1] > image_shape[0]:
                        continue
                    if bbox[j, 2] < 0 or bbox[j, 3] < 0:
                        continue
                    bbox[j, 2:] = np.minimum(bbox[j, 2:], image_shape[::-1])
                    bbox[j, :2] = np.maximum(bbox[j, :2], [0, 0])
                    anno["bbox"].append(bbox[j])

                    anno["alpha"].append(-np.arctan2(-final_box_preds[j, 1], final_box_preds[j, 0]) + box3d_camera[j, 6])
                    # anno["dimensions"].append(box3d_camera[j, [4, 5, 3]])
                    anno["dimensions"].append(box3d_camera[j, 3:6])
                    anno["location"].append(box3d_camera[j, :3])
                    anno["rotation_y"].append(box3d_camera[j, 6])
                    anno["name"].append(class_names[int(label_preds[j])])
                    anno["truncated"].append(0.0)
                    anno["occluded"].append(0)
                    anno["score"].append(scores[j])

                    num_example += 1

            if num_example != 0:
                anno = {n: np.stack(v) for n, v in anno.items()}
                annos.append(anno)
            else:
                annos.append(empty_result_anno())
            num_example = annos[-1]["name"].shape[0]
            annos[-1]["metadata"] = det["metadata"]
        return annos

    def evaluation(self, detections, output_dir=None, get_results=True):
        """
        When you want to eval your own dataset, you MUST set correct
        the z axis and box z center.
        """
        gt_annos = self.ground_truth_annotations
        dt_annos = self.convert_detection_to_kitti_annos(detections)

        # firstly convert standard detection to kitti-format dt annos
        z_axis = 1  # KITTI camera format use y as regular "z" axis.
        z_center = 1.0  # KITTI camera box's center is [0.5, 1, 0.5]
        # for regular raw lidar data, z_axis = 2, z_center = 0.5.

        results = None
        if get_results:
            result_official_dict = get_official_eval_result(gt_annos, dt_annos, self._class_names, z_axis=z_axis, z_center=z_center)
            result_coco_dict = get_coco_eval_result(gt_annos, dt_annos, self._class_names, z_axis=z_axis, z_center=z_center)

            result_official_dict_2 = get_official_eval_result_v2(gt_annos, dt_annos, self._class_names, z_axis=z_axis, z_center=z_center)

            results = {"results": {"official_AP_11": result_official_dict["result"],},
                       "results_2": {"official_AP_40": result_official_dict_2["result"],},
                       "detail": {"eval.kitti": {
                                       "official": result_official_dict["detail"],
                                       "coco": result_coco_dict["detail"],}},}

        return results, dt_annos

    def __getitem__(self, idx):
        return self.get_sensor_data(idx, with_gp=False)

    def get_sensor_data(self, idx, with_image=False, with_gp=False, by_index=False):

        sample_id = self.sample_ids[idx]

        # load image
        img = mmcv.imread(osp.join(self.img_prefix, '%06d.png' % sample_id))

        img, img_shape, pad_shape, scale_factor = self.img_transform(img, 1, False)

        objects = read_label(osp.join(self.label_prefix, '%06d.txt' % sample_id))
        calib = Calibration(osp.join(self.calib_prefix, '%06d.txt' % sample_id))

        gt_bboxes = [object.box3d for object in objects if object.type not in ["DontCare"]]  # label: bottom_z
        gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
        gt_types = [object.type for object in objects if object.type not in ["DontCare"]]

        # transfer from cam to lidar coordinates
        if len(gt_bboxes) != 0:
            gt_bboxes[:, :3] = project_rect_to_velo(gt_bboxes[:, :3], calib)

        img_meta = dict(
            img_shape=img_shape,
            sample_idx=sample_id,
            calib=calib
        )

        data = dict(
            img=to_tensor(img),
            img_meta=DC(img_meta, cpu_only=True)
        )

        if self.anchors is not None:
            data['anchors'] = DC(to_tensor(self.anchors.astype(np.float32)))

        points = read_lidar(osp.join(self.lidar_prefix, '%06d.bin' % sample_id))

        import ipdb; ipdb.set_trace()

        if self.augmentor is not None and self.test_mode is False:
            sampled_gt_boxes, sampled_gt_types, sampled_points = self.augmentor.sample_all(gt_bboxes, gt_types)
            assert sampled_points.dtype == np.float32
            gt_bboxes = np.concatenate([gt_bboxes, sampled_gt_boxes])
            gt_types = gt_types + sampled_gt_types
            assert len(gt_types) == len(gt_bboxes)

            # to avoid overlapping point (option)
            masks = points_in_rbbox(points, sampled_gt_boxes)
            points = points[np.logical_not(masks.any(-1))]

            # paste sampled points to the scene
            points = np.concatenate([sampled_points, points], axis=0)

            # select the interest classes
            selected = [i for i in range(len(gt_types)) if gt_types[i] in self.sa_class_names]
            gt_bboxes = gt_bboxes[selected, :]
            gt_types = [gt_types[i] for i in range(len(gt_types)) if gt_types[i] in self.sa_class_names]

            # force van to have same label as car
            gt_types = ['Car' if n == 'Van' else n for n in gt_types]
            gt_labels = np.array([self.sa_class_names.index(n) + 1 for n in gt_types], dtype=np.int64)

            self.augmentor.noise_per_object_(gt_bboxes, points, num_try=100)
            gt_bboxes, points = self.augmentor.random_flip(gt_bboxes, points)
            gt_bboxes, points = self.augmentor.global_rotation(gt_bboxes, points)
            gt_bboxes, points = self.augmentor.global_scaling(gt_bboxes, points)


        if isinstance(self.generator, VoxelGenerator):
            # voxels, coordinates, num_points = self.generator.generate(points)
            voxel_size = self.generator.voxel_size
            pc_range = self.generator.point_cloud_range
            grid_size = self.generator.grid_size

            keep = points_op_cpu.points_bound_kernel(points, pc_range[:3], pc_range[3:])
            voxels = points[keep, :]
            coordinates = ((voxels[:, [2, 1, 0]] - np.array(pc_range[[2, 1, 0]], dtype=np.float32)) / np.array(
                voxel_size[::-1], dtype=np.float32)).astype(np.int32)
            num_points = np.ones(len(keep)).astype(np.int32)

            data['voxels'] = DC(to_tensor(voxels.astype(np.float32)))
            data['coordinates'] = DC(to_tensor(coordinates))
            data['num_points'] = DC(to_tensor(num_points))



            if self.anchor_area_threshold >= 0 and self.anchors is not None:
                dense_voxel_map = sparse_sum_for_anchors_mask(
                    coordinates, tuple(grid_size[::-1][1:]))  # shape:  (1600, 1408)
                dense_voxel_map = dense_voxel_map.cumsum(0)
                dense_voxel_map = dense_voxel_map.cumsum(1)
                # dense_voxel_map : (1600, 1408)
                # self.anchors_bv: [70400, 4]
                # voxel_size: [0.05, 0.05, 0.1 ]
                # pc_range: [ 0. , -40. ,  -3. ,  70.4,  40. ,   1. ]
                # grid_size: [1408, 1600,   40]
                anchors_area = fused_get_anchors_area(
                    dense_voxel_map, self.anchors_bv, voxel_size, pc_range, grid_size)
                anchors_mask = anchors_area > self.anchor_area_threshold
                data['anchors_mask'] = DC(to_tensor(anchors_mask.astype(np.uint8)))

            # filter gt_bbox out of range
            bv_range = self.generator.point_cloud_range[[0, 1, 3, 4]]
            mask = filter_gt_box_outside_range(gt_bboxes, bv_range)
            gt_bboxes = gt_bboxes[mask]
            gt_labels = gt_labels[mask]

        else:
            NotImplementedError

        # skip the image if there is no valid gt bbox
        if len(gt_bboxes) == 0:
            return None

        # limit rad to [-pi, pi]
        gt_bboxes[:, 6] = limit_period(
            gt_bboxes[:, 6], offset=0.5, period=2 * np.pi)

        if self.with_label:
            data['gt_labels'] = DC(to_tensor(gt_labels))
            data['gt_bboxes'] = DC(to_tensor(gt_bboxes))


        #import ipdb;
        #ipdb.set_trace()

        res = {
            "voxels": data['voxels'].data.numpy(),
            "coordinates": data['coordinates'].data.numpy(),
            "num_points": data['num_points'].data.numpy(),
            "anno": {
                "gt_boxes": data['gt_bboxes'].data.numpy(),
                "gt_classes": data['gt_labels'].data.numpy(),
            },
            "anchors": data['anchors'].data.numpy(),
            "anchors_mask": data['anchors_mask'].data.numpy(),
            "metadata": {
                'img_shape': np.array(data['img_meta'].data['img_shape']),
                'sample_idx': data['img_meta'].data['sample_idx'],
            }
        }



        return res


# todo: for debug
if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    data_path = "/mnt/proj50/zhengwu/KITTI/object"
    info_path = "/mnt/proj50/zhengwu/KITTI/object/kitti_infos_train.pkl"

    from det3d.torchie import Config
    cfg = Config.fromfile("../../../examples/second/configs/config.py")
    pipeline = cfg.train_pipeline
    kitti = KittiDataset(data_path, info_path, pipeline=pipeline)
    data = kitti.get_sensor_data(99, by_index=True)
    # for i in range(3000):
    #     print(i)
    #     data = kitti.get_sensor_data(i, by_index=False)
    import ipdb; ipdb.set_trace()

