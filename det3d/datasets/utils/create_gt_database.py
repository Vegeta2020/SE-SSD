import pickle
from pathlib import Path

import numpy as np

from det3d.core import box_np_ops
from det3d.datasets.dataset_factory import get_dataset
from det3d.torchie import Config

from joblib import Parallel, delayed
from tqdm import tqdm

dataset_name_map = {
    "KITTI": "KittiDataset",
    "NUSC": "NuScenesDataset",
    "LYFT": "LyftDataset",
}


def create_groundtruth_database(
    dataset_class_name,
    data_path,
    info_path=None,
    used_classes=None,
    db_path=None,
    dbinfo_path=None,
    relative_path=True,
    add_rgb=False,
    lidar_only=False,
    bev_only=False,
    coors_range=None,
    gt_aug_with_context=-1.0,
    **kwargs,
):
    gt_aug_with_context = gt_aug_with_context
    pipeline = [
        {"type": "LoadPointCloudFromFile", "dataset": dataset_name_map[dataset_class_name],},
        {"type": "LoadPointCloudAnnotations", "with_bbox": True, "enable_difficulty_level": True},
    ]
    # get KittiDataset loaded points and annos.
    dataset = get_dataset(dataset_class_name)(info_path=info_path, root_path=data_path, test_mode=True, pipeline=pipeline)

    # prepare dbinfo_path and db_path.
    root_path = Path(data_path)
    if db_path is None:
        db_path = root_path / "gt_database"
    if dbinfo_path is None:
        dbinfo_path = root_path / "dbinfos_train.pkl"
    db_path.mkdir(parents=True, exist_ok=True)

    all_db_infos = {}
    group_counter = 0

    for index in tqdm(range(len(dataset))):
        image_idx = index
        sensor_data = dataset.get_sensor_data(index)   # see in loading.py after pipelines.
        if "image_idx" in sensor_data["metadata"]:     # True, image_idx = file_name (000001)
            image_idx = sensor_data["metadata"]["image_idx"]

        points = sensor_data["lidar"]["points"]
        annos = sensor_data["lidar"]["annotations"]
        gt_boxes = annos["boxes"]   # gt_boxes of all classes, dc has been removed in pipeline LoadPointCloudAnnotations.
        names = annos["names"]      # gt_names.
        group_dict = {}
        group_ids = np.full([gt_boxes.shape[0]], -1, dtype=np.int64)

        if "group_ids" in annos:    # False
            group_ids = annos["group_ids"]
        else:
            group_ids = np.arange(gt_boxes.shape[0], dtype=np.int64)

        difficulty = np.zeros(gt_boxes.shape[0], dtype=np.int32)
        if "difficulty" in annos:  # False
            difficulty = annos["difficulty"]

        num_obj = gt_boxes.shape[0]

        # todo: maybe we need add some contexual points here.
        offset = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]   # [x, y, z, w, l, h, ry]
        if gt_aug_with_context > 0.0:
            offset = [0.0, 0.0, 0.0, gt_aug_with_context, gt_aug_with_context, 0.0, 0.0]
            db_path = root_path / "gt_enlarged_database"
            dbinfo_path = root_path / "dbinfos_enlarged_train.pkl"
            db_path.mkdir(parents=True, exist_ok=True)

        point_indices_for_num = box_np_ops.points_in_rbbox(points, gt_boxes)
        point_indices = box_np_ops.points_in_rbbox(points, gt_boxes + offset)
        for i in range(num_obj):   # in a single scene.
            filename = f"{image_idx}_{names[i]}_{i}.bin"
            filepath = db_path / filename
            gt_points = points[point_indices[:, i]]
            num_points_in_gt = point_indices_for_num[:, i].sum()
            gt_points[:, :3] -= gt_boxes[i, :3]  # only record relative distance
            with open(filepath, "w") as f:       # db: gt points in each gt_box are saved
                gt_points[:, :4].tofile(f)

            if (used_classes is None) or names[i] in used_classes:
                if relative_path:
                    db_dump_path = str(db_path.stem + "/" + filename)
                else:
                    db_dump_path = str(filepath)

                db_info = {
                    "name": names[i],
                    "path": db_dump_path,
                    "image_idx": image_idx,
                    "gt_idx": i,
                    "box3d_lidar": gt_boxes[i],
                    "num_points_in_gt": num_points_in_gt,
                    "difficulty": difficulty[i]  # todo: not accurate, all are set as 0.
                }
                local_group_id = group_ids[i]
                if local_group_id not in group_dict:
                    group_dict[local_group_id] = group_counter
                    group_counter += 1
                db_info["group_id"] = group_dict[local_group_id]  # count from 0 to total_num_of_specific_class[like 13442 for car]
                if "score" in annos:  # False
                    db_info["score"] = annos["score"][i]
                if names[i] in all_db_infos:                 # all_db_infos are grouped by class_names (like car, pedestrian, cyclist, cycle)
                    all_db_infos[names[i]].append(db_info)   # all db infos include info of all db
                else:
                    all_db_infos[names[i]] = [db_info]



    print("dataset length: ", len(dataset))
    for k, v in all_db_infos.items():
        print(f"load {len(v)} {k} database infos")

    with open(dbinfo_path, "wb") as f:
       pickle.dump(all_db_infos, f)
