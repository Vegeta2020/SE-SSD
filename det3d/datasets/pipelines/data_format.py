


'''
###################################### After LoadPointCloudFromFile##########################################
Loading points (x, y, z, r in velo coord) from pre-reduced .bin file
        res = {
                 "type": "KittiDataset",
                 "lidar": {
                            "type": "lidar",
                            "points": [N, 4],
                            "ground_plane": -gp[-1] if with_gp else None,
                            "annotations": None,
                            "names": None,
                            "targets": None,
                           },
                "metadata": {
                            "image_prefix": self._root_path,
                            "num_point_features": KittiDataset.NumPointFeatures,
                            "image_idx": info["image"]["image_idx"],
                            "image_shape": info["image"]["image_shape"],
                            "token": str(info["image"]["image_idx"]),
                            },
                "calib": None,
                "cam": {
                         "annotations": None,
                        },
                "mode": "val" if self.test_mode else "train",
              }


########################################### After LoadPointCloudAnnotations#########################################
        Loading gt_boxes and calib configs, remove dontcare objects; transform gt_boxes
        (xyz: cam -> velo; l,h,w -> w, l, h) and moved to the real center

            res = {
                 "type": "KittiDataset";
                 "lidar": {
                                "type": "lidar",
                                "points": [N, 4],
                                "ground_plane": -gp[-1] if with_gp else None,
                                "annotations": {    # dc obj removed.
                                                    "boxes": gt_boxes, # lidar: [x, y, z, w, l, h, ry], xyz (velo), h (real center).
                                                    "names": gt_names,
                                                    "difficulty": difficulty, # todo: we may add difficulty here
                                                    "cam_boxes": gt_boxes_cam, # todo: my addition for ground plane cal.
                                                }
                                "names": None,
                                "targets": None,
                           },
                "metadata": {
                                "image_prefix": self._root_path,
                                "num_point_features": KittiDataset.NumPointFeatures,
                                "image_idx": info["image"]["image_idx"],
                                "image_shape": info["image"]["image_shape"],
                                "token": str(info["image"]["image_idx"]),
                            },
                "calib": {
                            "rect": calib["R0_rect"],
                            "Trv2c": calib["Tr_velo_to_cam"],
                            "P2": calib["P2"],
                         }
                "cam": {
                         "annotations": {
                                           "boxes": annos["bbox"],
                                           "names": gt_names,
                                        }
                        },
                "mode": "val" if self.test_mode else "train",
              }



######################################### After Preprocess##########################################
    res = {
                     "type": "KittiDataset",
                     "lidar": {
                                    "type": "lidar",
                                    "points": [N, 4],    # shuffled
                                    "ground_plane": -gp[-1] if with_gp else None,
                                    "annotations": {    # untargeted objs removed first time.
                                                        "gt_classes": gt_classes,   # index + 1 in self.class_names.
                                                        "gt_boxes": gt_boxes,       # with gt/per-object/global aug.
                                                        "gt_names": gt_names,
                                                    }
                                    "names": None,
                                    "targets": None,
                               },
                    "metadata": {
                                    "image_prefix": self._root_path,
                                    "num_point_features": KittiDataset.NumPointFeatures,
                                    "image_idx": info["image"]["image_idx"],
                                    "image_shape": info["image"]["image_shape"],
                                    "token": str(info["image"]["image_idx"]),
                                },
                    "calib": {
                                "rect": calib["R0_rect"],
                                "Trv2c": calib["Tr_velo_to_cam"],
                                "P2": calib["P2"],
                             }
                    "cam": {
                             "annotations": {
                                               "boxes": annos["bbox"],
                                               "names": gt_names,
                                            }
                            },
                    "mode": "val" if self.test_mode else "train",
                  }


########################################### After Voxelization ######################################
    res = {
                     "type": "KittiDataset",
                     "lidar": {
                                    "type": "lidar",
                                    "points": [N, 4],    # shuffled
                                    "ground_plane": -gp[-1] if with_gp else None,
                                    "annotations": {    # gt_boxes out of valid range removed.
                                                        "gt_classes": gt_classes,
                                                        "gt_boxes": gt_boxes,
                                                        "gt_names": gt_names,
                                                    }
                                    "Voxels": {
                                                 voxels=voxels,
                                                 coordinates=coordinates,
                                                 num_points=num_points_per_voxel,
                                                 num_voxels=num_voxels,
                                                 shape=grid_size,
                                               }
                                    "names": None,
                                    "targets": None,

                               },
                    "metadata": {
                                    "image_prefix": self._root_path,
                                    "num_point_features": KittiDataset.NumPointFeatures,
                                    "image_idx": info["image"]["image_idx"],
                                    "image_shape": info["image"]["image_shape"],
                                    "token": str(info["image"]["image_idx"]),
                                },
                    "calib": {
                                "rect": calib["R0_rect"],
                                "Trv2c": calib["Tr_velo_to_cam"],
                                "P2": calib["P2"],
                             }
                    "cam": {
                             "annotations": {
                                               "boxes": annos["bbox"],
                                               "names": gt_names,
                                            }
                            },
                    "mode": "val" if self.test_mode else "train",
                  }

########################################### After AssignTarget ######################################
    res = {
                     "type": "KittiDataset",
                     "lidar": {
                                    "type": "lidar",
                                    "points": [N, 4],    # shuffled
                                    "ground_plane": -gp[-1] if with_gp else None,
                                    "annotations": {    # untargeted class of gt objs removed the second time.
                                                        "gt_classes": gt_classes,
                                                        "gt_boxes": gt_boxes, # ry range limited in [-pi, pi].
                                                        "gt_names": gt_names,
                                                    }
                                    "Voxels": {
                                                 voxels=voxels,
                                                 coordinates=coordinates,
                                                 num_points=num_points_per_voxel,
                                                 num_voxels=num_voxels,
                                                 shape=grid_size,
                                               }
                                    "targets": {
                                                 "labels": [targets_dict["labels"]],                      # [num_pos_anchors,]
                                                 "reg_targets": [targets_dict["bbox_targets"]],           # [num_pos_anchors, 7]
                                                 "reg_weights": [targets_dict["bbox_outside_weights"]],   # positive samples set as 1.
                                                }
                                    "names": None,
                                    "targets": None,

                               },
                    "metadata": {
                                    "image_prefix": self._root_path,
                                    "num_point_features": KittiDataset.NumPointFeatures,
                                    "image_idx": info["image"]["image_idx"],
                                    "image_shape": info["image"]["image_shape"],
                                    "token": str(info["image"]["image_idx"]),
                                },
                    "calib": {
                                "rect": calib["R0_rect"],
                                "Trv2c": calib["Tr_velo_to_cam"],
                                "P2": calib["P2"],
                             }
                    "cam": {
                             "annotations": {
                                               "boxes": annos["bbox"],
                                               "names": gt_names,
                                            }
                            },
                    "mode": "val" if self.test_mode else "train",
                  }


########################################### After Reformat ######################################
    data_bundle = {
                    "metadata": {
                                    "image_prefix": self._root_path,
                                    "num_point_features": KittiDataset.NumPointFeatures,
                                    "image_idx": info["image"]["image_idx"],
                                    "image_shape": info["image"]["image_shape"],
                                    "token": str(info["image"]["image_idx"]),
                                },
                    "points": [N, 4],
                    "voxels": voxels,              # [num_voxels, max_num_points(T/5), point_dim(4)].
                    "shape":  [1408, 1600,   40],
                    "num_points" : ,               # record num_points in each voxel;
                    "num_voxels" : ,               # num_voxels in each sample;
                    "coordinates" : ,              # coor and batch_id of each voxel;
                    "anchors" : anchors,           # anchors, only one group

                    "calib": {
                                "rect": calib["R0_rect"],
                                "Trv2c": calib["Tr_velo_to_cam"],
                                "P2": calib["P2"],
                             }

                    "annos": {
                                "gt_classes": gt_classes,
                                "gt_boxes": gt_boxes, # ry range limited in [-pi, pi].
                                "gt_names": gt_names,
                             }

                    "ground_plane": -gp[-1],  # one scalar
                    "labels": [targets_dict["labels"]],                       # [num_pos_anchors,]
                    "reg_targets": [targets_dict["bbox_targets"]],           # [num_pos_anchors, 7]
                    "reg_weights": [targets_dict["bbox_outside_weights"]],   # positive samples set as 1.
                  }

########################################### After collate_kitti ######################################
    batch_samples = {  # or called example in code.
                    'metadata': [
                                   {
                                      'image_prefix': '/mnt/proj50/zhengwu/KITTI/object',
                                      'num_point_features': 4,
                                      'image_idx': 2771, # int
                                      'image_shape': array([375, 1242], dtype=int32),
                                      'token': '2771',
                                   }, ...
                                ],
                    'points': tensor([159443, 5], dtype=torch.float32),  # Batch points. First dim is sample_id like coordinate.
                    'voxels': tensor([137042, 5, 4], dtype=torch.float32),
                    'shape':  ((1408, 1600, 40),...), # [batch_size, 3].
                    'num_points': (...),  # (137042,), num_points_per_voxel, total 137042 voxels in all 8 samples.
                    'num_voxels': (...),  # (batch_size,), num_voxels_per_sample,
                    'coordinates':(...),  # [137042, 4].
                    'anchors': tensor(([8, 70400, 7]), ..), # (batch_size, 3).
                    'calib':{
                              'rect': tensor(...),  # (batch_size, 4, 4).
                              'Trv2c': tensor(...), # (batch_size, 4, 4).
                              'P2': tensor(...),    # (batch_size, 4, 4).
                            }
                    'annos': [   # length: batch_size. num_gt is different in samples. np.stack is useful.
                                {
                                  "gt_boxes": [np.array([num_gt_in_sampler, 7], dtype=np.float64)],
                                  "gt_names": [np.array([num_gt_in_sampler,], dtype=<U8)],
                                  "gt_classes": [np.array([num_gt_in_sampler,], dtype=int32)],
                                }, ...
                             ]
                    'ground_plane': np.array([batch_size,], dtype=np.float64),
                    'labels':  [np.array([batch_size, 70400]),],
                    'reg_targets': [tensor([8, 70400, 7], dtype=torch.float32)],
                    'reg_weights': [tensor([8, 70400], dtype=torch.float32)],    # positive samples as 1, others as 0.
                  }

'''

