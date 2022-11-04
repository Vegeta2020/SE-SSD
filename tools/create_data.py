import copy
from pathlib import Path
import pickle

import fire

from det3d.datasets.kitti import kitti_common as kitti_ds
from det3d.datasets.utils.create_gt_database import create_groundtruth_database
from det3d.torchie import Config

cfg = Config.fromfile("../examples/second/configs/config.py")

def kitti_data_prep(root_path):
    # compress info of image(path), velodyne(path), label (all info but dc removed),
    # calib (all) into pkl file; DontCare has been removed
    # root_path: "/mnt/proj50/zhengwu/KITTI/object"
    #
    # kitti_infos_train/trainval/val/test for each sample in train.txt/trainval.txt/val.txt/test.txt
    # info{
    #      "point_cloud": {
    #                       "num_featuers": 4,
    #                       "velodyne_path": "/training/velodyne/000000.bin",
    #                      }
    #      "image": {
    #                 "image_idx": index,  (eg. element in ImageSets/trainval.txt)
    #                 "image_path": "/training/image_2/000000.png",
    #                 "image_shape":  [H(370, y), W(1224, x)], (eg. in image coord)
    #                }
    #      "calib": {
    #                  "P0": P0,
    #                  "P1": P1,
    #                  "P2": P2,
    #                  "P3": P3,
    #                  "R0_rect":  rect_4x4,
    #                  "Tr_velo_to_cam":  Tr_velo_to_cam,
    #                  "Tr_imu_to_velo":  Tr_imu_to_velo,
    #                }
    #        "annos": {
    #                   "name": ,
    #                   "truncated": ,
    #                   "occluded": ,
    #                   "alpha": ,
    #                   "bbox": ,
    #                   "dimensions": lhw, # including dc obj.
    #                   "location": xyz, (eg. in cam coord)
    #                   "rotation_y": ry,
    #                   "index": array([0,1,2..num_non_dc_obj, -1, -1, -1]), # there are num_dc_obj of -1.
    #                   "group_ids": array([0,1,2.., num_dc&non_dc_obj]),
    #                   "difficulty": [d0, d1, ...num_all_obj],    # di in [0 (easy), 1 (moderate), 2 (hard), -1].
    #                   "num_points_in_gt": [n0, n1, n2, ..., -1, -1, -1],  # dc gt are counted as -1.
    #                  }
    #        }
    #kitti_ds.create_kitti_info_file(root_path)

    # all points outside of image_range are removed and kept as reduced point_cloud .bin file.
    kitti_ds.create_reduced_point_cloud(root_path)


    # dbinfos_train.pkl
    # all_db_infos['Car'] = [db_info_0, db_info_1, ...], grouped by class_names.
    # db_info = {
    #             "name": names[i],            # "Car"
    #             "path": "object/training/velodyne_reduced/xxxxx.bin",  # saved with relative distance to the box center.
    #             "image_idx": image_idx,
    #             "gt_idx": i,                 # index in a single scene.
    #             "box3d_lidar": gt_boxes[i],  # dc removed.
    #             "num_points_in_gt": gt_points.shape[0],
    #             "difficulty": difficulty[i]  # todo: not accurate, all are set as 0.
    #            }

    # save each gt box points separately and all gt info in a
    create_groundtruth_database("KITTI", root_path, Path(root_path) / "kitti_infos_train.pkl", gt_aug_with_context=cfg.my_paras.gt_aug_with_context)

'''
def nuscenes_data_prep(root_path, version, nsweeps=10):
    nu_ds.create_nuscenes_infos(root_path, version=version, nsweeps=nsweeps)
    create_groundtruth_database(
        "NUSC",
        root_path,
        Path(root_path) / "infos_train_{:02d}sweeps_withvelo.pkl".format(nsweeps),
        nsweeps=nsweeps,
    )
    # nu_ds.get_sample_ground_plane(root_path, version=version)


def lyft_data_prep(root_path, version="trainval"):
    lyft_ds.create_lyft_infos(root_path, version=version)
    create_groundtruth_database(
        "LYFT", root_path, Path(root_path) / "lyft_info_train.pkl"
    )
'''

if __name__ == "__main__":
    #fire.Fire()
    kitti_data_prep("/data/zhengwu/KITTI/object")
