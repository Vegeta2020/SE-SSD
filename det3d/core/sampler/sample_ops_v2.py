import copy
import pathlib
import pickle
import time
from functools import partial, reduce

import numpy as np
import torch
from det3d.core.bbox import box_np_ops
from det3d.core.sampler import preprocess as prep
from det3d.utils.check import shape_mergeable


class DataBaseSamplerV2:
    def __init__(
        self,
        db_infos,                        # object/dbinfos_train.pkl
        groups,                          # [dict(Car=15,),],
        db_prepor=None,                  # filter_by_min_num_points, filter_by_difficulty
        rate=1.0,                        # rate=1.0
        global_rot_range=None,           # [0, 0]
        logger=None,                     # logging.getLogger("build_dbsampler")
        gt_random_drop=-1.0,
        gt_aug_with_context=-1.0,
        gt_aug_similar_type=False,
    ):
        # load all gt database here.
        for k, v in db_infos.items():
            logger.info(f"load {len(v)} {k} database infos")

        # preprocess: filter_by_min_num_points/difficulty.
        if db_prepor is not None:
            db_infos = db_prepor(db_infos)
            logger.info("After filter database:")
            for k, v in db_infos.items():
                logger.info(f"load {len(v)} {k} database infos")

        self.db_infos = db_infos
        self._rate = rate
        self._groups = groups
        self._group_db_infos = {}
        self._group_name_to_names = []
        self._sample_classes = []
        self._sample_max_nums = []
        self.gt_point_random_drop = gt_random_drop
        self.gt_aug_with_context = gt_aug_with_context

        # get group_name: Car and group_max_num: 15
        self._group_db_infos = self.db_infos  # just use db_infos
        for group_info in groups:
            self._sample_classes += list(group_info.keys())           # ['Car']
            self._sample_max_nums += list(group_info.values())        # [15]

        # get sampler dict for each class like Car, Cyclist, Pedestrian...
        # this sampler can ensure batch samples selected randomly.
        self._sampler_dict = {}
        for k, v in self._group_db_infos.items():
            self._sampler_dict[k] = prep.BatchSampler(v, k)

        if gt_aug_similar_type:
            self._sampler_dict["Car"] = prep.BatchSampler(self._group_db_infos["Car"] + self._group_db_infos["Van"], "Car")

    def sample_all(
        self,
        root_path,
        gt_boxes,
        gt_names,
        num_point_features,
        random_crop=False,
        gt_group_ids=None,
        calib=None,
        targeted_class_names=None,
        with_road_plane_cam=None,
    ):
        '''
            This func aims to sample some gt-boxes and corresponding points to perform gt augmentation;
            notice that the non-targeted gt boxes (like pedestrian) have been considered into collision test;
            notice that the points in corresponding gt-box are read from pre-saved gt database;
        '''
        # record the num of gt-aug samples with a dict and a list
        sampled_num_dict = {}
        sample_num_per_class = []
        for class_name, max_sample_num in zip(self._sample_classes, self._sample_max_nums):  # actual only once for ['Car': 15]
            sampled_num = int(max_sample_num - np.sum([n == class_name for n in gt_names]))
            #sampled_num = int(max_sample_num - np.sum([name in targeted_class_names for name in gt_names]))
            sampled_num = np.round(self._rate * sampled_num).astype(np.int64)
            sampled_num_dict[class_name] = sampled_num
            sample_num_per_class.append(sampled_num)

        sampled = []
        sampled_gt_boxes = []
        all_gt_boxes = gt_boxes

        # gt-augmentation: sample gt boxes and add them to current gt_boxes.
        # todo: we may sample box one by one to ensure num of gt-boxes is fulfilled.
        for class_name, sampled_num in zip(self._sample_classes, sample_num_per_class):
            # if sampled_num > 0:
            #     sampled_objects = self.sample_class_v2(class_name, sampled_num, all_gt_boxes)
            #     sampled += sampled_objects

            #     if len(sampled_objects) > 0:
            #         if len(sampled_objects) == 1:
            #             sampled_boxes = sampled_objects[0]["box3d_lidar"][np.newaxis, ...]
            #         else:
            #             sampled_boxes = np.stack([s["box3d_lidar"] for s in sampled_objects], axis=0)

            #         sampled_gt_boxes += [sampled_boxes]
            #         all_gt_boxes = np.concatenate([all_gt_boxes, sampled_boxes], axis=0)

            # ensure final num_boxes fulfill the num requirement after collision test.
            times = 0
            while sampled_num > 0 and times < 2:
                sampled_objects = self.sample_class_v2(class_name, sampled_num, all_gt_boxes)
                sampled += sampled_objects
            
                if len(sampled_objects) > 0:
                    if len(sampled_objects) == 1:
                        sampled_boxes = sampled_objects[0]["box3d_lidar"][np.newaxis, ...]
                    else:
                        sampled_boxes = np.stack([s["box3d_lidar"] for s in sampled_objects], axis=0)
            
                    sampled_gt_boxes += [sampled_boxes]
                    all_gt_boxes = np.concatenate([all_gt_boxes, sampled_boxes], axis=0)
                sampled_num -= len(sampled_objects)
                times += 1

        if len(sampled) > 0:
            ''' get points in sampled gt_boxes '''
            sampled_gt_boxes = np.concatenate(sampled_gt_boxes, axis=0)
            num_sampled = len(sampled)
            s_points_list = []
            # get points in sampled gt-boxes from pre-generated gt database.
            for info in sampled:
                try:
                    s_points = np.fromfile(str(pathlib.Path(root_path) / info["path"]), dtype=np.float32).reshape(-1, num_point_features)
                    # gt_points are saved with relative distance; so need to recover by adding box center.
                    s_points[:, :3] += info["box3d_lidar"][:3]

                    if with_road_plane_cam is not None:
                        a, b, c, d = with_road_plane_cam
                        box3d_cam_center = info['box3d_cam'][0:3]  # x,y,z with cam coord. bottom center.
                        cur_height_cam = (-d - a * box3d_cam_center[0] - c * box3d_cam_center[2]) / b
                        move_height_cam = info['box3d_cam'][1] - cur_height_cam  # cal y dist, > 0: move up, < 0: move down.

                        s_points[:, 2] += move_height_cam
                        index = sampled.index(info)
                        sampled_gt_boxes[index, 2] += move_height_cam



                    # random drop points in gt_boxes to make model more robust.
                    if self.gt_point_random_drop > 0:
                        num_point = s_points.shape[0]
                        if num_point > 10:
                            drop_num = int(np.random.uniform(0, self.gt_point_random_drop) * num_point)
                            choice = np.random.choice(np.arange(num_point), num_point - drop_num, replace=False)
                            s_points = s_points[choice]
                    s_points_list.append(s_points)


                except Exception:
                    print(info["path"])
                    continue

            '''todo: do something about random crop'''
            if random_crop:  # False
                s_points_list_new = []
                assert calib is not None
                rect = calib["rect"]
                Trv2c = calib["Trv2c"]
                P2 = calib["P2"]
                gt_bboxes = box_np_ops.box3d_to_bbox(sampled_gt_boxes, rect, Trv2c, P2)
                crop_frustums = prep.random_crop_frustum(gt_bboxes, rect, Trv2c, P2)
                for i in range(crop_frustums.shape[0]):
                    s_points = s_points_list[i]
                    mask = prep.mask_points_in_corners(s_points, crop_frustums[i : i + 1]).reshape(-1)
                    num_remove = np.sum(mask)
                    if num_remove > 0 and (s_points.shape[0] - num_remove) > 15:
                        s_points = s_points[np.logical_not(mask)]
                    s_points_list_new.append(s_points)
                s_points_list = s_points_list_new

            ret = {
                "gt_names": np.array([s["name"] for s in sampled]),
                "difficulty": np.array([s["difficulty"] for s in sampled]),
                "gt_boxes": sampled_gt_boxes,
                "points": np.concatenate(s_points_list, axis=0),
                "gt_masks": np.ones((num_sampled,), dtype=np.bool_),
                "group_ids": np.arange(gt_boxes.shape[0], gt_boxes.shape[0] + len(sampled)),
            }
        else:
            ret = None
        return ret

    def sample_class_v2(self, name, num, gt_boxes):
        '''
            This func aims to select fixed number of gt boxes from gt database with collision (bev iou) test performed.
        '''

        # sample num gt_boxes from gt_database
        sampled = self._sampler_dict[name].sample(num)
        sampled = copy.deepcopy(sampled)

        num_sampled = len(sampled)
        num_gt = gt_boxes.shape[0]

        # get all boxes: gt_boxes + sp_boxes
        sp_boxes = np.stack([i["box3d_lidar"] for i in sampled], axis=0)  # todo: need modification here
        boxes = np.concatenate([gt_boxes, sp_boxes], axis=0).copy()

        offset = [0.0, 0.0]
        if self.gt_aug_with_context > 0.0:
            offset = [self.gt_aug_with_context, self.gt_aug_with_context]

        # get all boxes_bev: gt_boxes_bev + sampled_boxes_bev
        sp_boxes_new = boxes[num_gt:]
        gt_boxes_bv = box_np_ops.center_to_corner_box2d(gt_boxes[:, 0:2], gt_boxes[:, 3:5], gt_boxes[:, -1])
        sp_boxes_bv = box_np_ops.center_to_corner_box2d(sp_boxes_new[:, 0:2], sp_boxes_new[:, 3:5] + offset, sp_boxes_new[:, -1])
        total_bv = np.concatenate([gt_boxes_bv, sp_boxes_bv], axis=0)

        # collision test on bev (stricter than 3d)
        coll_mat = prep.box_collision_test(total_bv, total_bv)  # todo: too slow here
        diag = np.arange(total_bv.shape[0])
        coll_mat[diag, diag] = False

        # get valid samples
        valid_samples = []
        for i in range(num_gt, num_gt + num_sampled):  # todo: the overall box num may not meet the requirement
            if coll_mat[i].any():
                # i-th sampled box is not considered into auged gt-boxes.
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                # i-th sampled box is considered into auged gt-boxes.
                valid_samples.append(sampled[i - num_gt])

        return valid_samples


    def sample_class_v3(self, name, num, gt_boxes):
        '''
            This func aims to selected fixed number of gt boxes from gt database with collision test performed.
        '''

        # sample num gt_boxes from gt_database
        sampled_objects = copy.deepcopy(self._sampler_dict[name].sample(num))
        sampled_boxes = np.stack([i["box3d_lidar"] for i in sampled_objects], axis=0)  # todo: need modification here
        all_boxes = np.concatenate([gt_boxes, sampled_boxes], axis=0).copy()
        all_boxes_torch = torch.from_numpy(all_boxes).float()

        # collision test: prep.box_collision_test is still a little faster than boxes_iou_bev_cpu.
        iou_bev = iou3d.boxes_iou_bev_cpu(all_boxes_torch, all_boxes_torch).numpy()
        coll_mat = iou_bev > 0.
        diag = np.arange(all_boxes.shape[0])
        coll_mat[diag, diag] = False

        # get valid samples
        valid_samples = []
        num_gt = gt_boxes.shape[0]
        num_sampled = len(sampled_objects)
        for i in range(num_gt, num_gt + num_sampled):  # todo: without multiple try times, sometimes, the overall box num may not meet the requirement
            if coll_mat[i].any():
                # i-th sampled box is not considered into auged gt-boxes.
                coll_mat[i] = False
                coll_mat[:, i] = False
            else:
                # i-th sampled box is considered into auged gt-boxes.
                valid_samples.append(sampled_objects[i - num_gt])

        return valid_samples

