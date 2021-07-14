import itertools
import logging
from pathlib import Path

from det3d.builder import build_box_coder
from det3d.utils.config_tool import get_downsample_factor

data_root_prefix = "/mnt/proj50/zhengwu"

# norm_cfg = dict(type='SyncBN', eps=1e-3, momentum=0.01)
norm_cfg = None

tasks = [dict(num_class=1, class_names=["Car"],),]

class_names = list(itertools.chain(*[t["class_names"] for t in tasks]))

# training and testing settings
box_coder = dict(type="ground_box3d_coder", n_dim=7, linear_dim=False, encode_angle_vector=False,)

# exp_sesssd_release_v0_0: based on v1_0, remove sada
# exp_sesssd_release_v1_0: default settings of sesssd


TAG = 'exp_se_ssd_v1_8'
# torch.set_printoptions(precision=4, sci_mode=False)
my_paras = dict(
    batch_size=4,
    data_mode="train",        # "train" or "trainval": the set to train the model;
    enable_ssl=True,         # Ensure "False" in CIA-SSD training
    eval_training_set=False,  # True: eval on "data_mode" set; False: eval on validation set.[Ensure "False" in training; Switch in Testing]

    # unused
    enable_difficulty_level=False,
    remove_difficulty_points=False,  # act with neccessary condition: enable_difficulty_level=True.
    gt_random_drop=-1,
    data_aug_random_drop=-1,
    far_points_first=False,
    data_aug_with_context=-1,        # enlarged size for w and l in data aug.
    gt_aug_with_context=-1,
    gt_aug_similar_type=False,
    min_points_in_gt=-1,
    loss_iou=None,
)

# model settings
model = dict(
    type="VoxelNet",
    pretrained=None,
    reader=dict(type="VoxelFeatureExtractorV3", num_input_features=4, norm_cfg=norm_cfg,),
    backbone=dict(type="SpMiddleFHD", num_input_features=4, ds_factor=8, norm_cfg=norm_cfg,),
    neck=dict(
        type="SSFA",
        layer_nums=[5,],
        ds_layer_strides=[1,],
        ds_num_filters=[128,],
        us_layer_strides=[1,],
        us_num_filters=[128,],
        num_input_features=128,
        norm_cfg=norm_cfg,
        logger=logging.getLogger("RPN"),
    ),
    bbox_head=dict(
        type="MultiGroupHead",
        mode="3d",
        in_channels=sum([128,]),
        norm_cfg=norm_cfg,
        tasks=tasks,
        weights=[1,],
        box_coder=build_box_coder(box_coder),
        encode_background_as_zeros=True,
        loss_norm=dict(type="NormByNumPositives", pos_cls_weight=1.0, neg_cls_weight=1.0,),
        loss_cls=dict(type="SigmoidFocalLoss", alpha=0.25, gamma=2.0, loss_weight=1.0,),
        use_sigmoid_score=True,
        loss_bbox=dict(type="WeightedSmoothL1Loss", sigma=3.0, code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], codewise=True, loss_weight=2.0, ),
        encode_rad_error_by_sin=True,
        loss_aux=dict(type="WeightedSoftmaxClassificationLoss", name="direction_classifier", loss_weight=0.2,),
        direction_offset=0.0,
        #loss_iou=my_paras['loss_iou'],
    ),
)

target_assigner = dict(
    type="iou",
    anchor_generators=[
        dict(
            type="anchor_generator_range",
            sizes=[1.6, 3.9, 1.56],  # w, l, h
            anchor_ranges=[0, -40.0, -1.0, 70.4, 40.0, -1.0],
            rotations=[0, 1.57],
            matched_threshold=0.6,
            unmatched_threshold=0.45,
            class_name="Car",
        ),
    ],
    sample_positive_fraction=-1,
    sample_size=512,
    region_similarity_calculator=dict(type="nearest_iou_similarity",),
    pos_area_threshold=-1,
    tasks=tasks,
)

assigner = dict(
    box_coder=box_coder,
    target_assigner=target_assigner,
    out_size_factor=8,
    debug=False,
    enable_similar_type=True,
)


train_cfg = dict(assigner=assigner)

test_cfg = dict(
    nms=dict(
        use_rotate_nms=True,
        use_multi_class_nms=False,
        nms_pre_max_size=1000,
        nms_post_max_size=100,
        nms_iou_threshold=0.01,
    ),
    score_threshold=0.3,
    post_center_limit_range=[0, -40.0, -5.0, 70.4, 40.0, 5.0],
    max_per_img=100,
)

# dataset settings
dataset_type = "KittiDataset"

db_sampler = dict(
    type="GT-AUG",
    enable=True,
    db_info_path=data_root_prefix + "/KITTI/object/dbinfos_" + my_paras['data_mode'] +".pkl",
    sample_groups=[dict(Car=15,),],
    db_prep_steps=[
        dict(filter_by_min_num_points=dict(Car=5,)),
        dict(filter_by_difficulty=[-1],),    # todo: need to check carefully
    ],
    global_random_rotation_range_per_object=[0, 0],
    rate=1.0,
    gt_random_drop=my_paras['gt_random_drop'],
    gt_aug_with_context=my_paras['gt_aug_with_context'],
    gt_aug_similar_type=my_paras['gt_aug_similar_type'],
)
train_preprocessor = dict(
    mode="train",
    shuffle_points=True,
    gt_loc_noise=[1.0, 1.0, 0.5],
    gt_rot_noise=[-0.785, 0.785],
    global_rot_noise=[-0.785, 0.785],
    global_scale_noise=[0.95, 1.05],
    global_rot_per_obj_range=[0, 0],
    global_trans_noise=[0.0, 0.0, 0.0],
    remove_points_after_sample=True,
    gt_drop_percentage=0.0,
    gt_drop_max_keep_points=15,
    remove_environment=False,
    remove_unknown_examples=my_paras.get("remove_difficulty_points", False),
    db_sampler=db_sampler,
    class_names=class_names,   # 'Car'
    symmetry_intensity=False,
    enable_similar_type=True,
    min_points_in_gt=my_paras["min_points_in_gt"],
    data_aug_with_context=my_paras["data_aug_with_context"],
    data_aug_random_drop=my_paras["data_aug_random_drop"],
)

val_preprocessor = dict(
    mode="val",
    shuffle_points=False,
    remove_environment=False,
    remove_unknown_examples=False,
)

voxel_generator = dict(
    range=[0, -40.0, -3.0, 70.4, 40.0, 1.0],
    voxel_size=[0.05, 0.05, 0.1],
    max_points_in_voxel=5,
    max_voxel_num=20000,
    far_points_first=my_paras['far_points_first'],
)

train_pipeline = [
    dict(type="LoadPointCloudFromFile"),
    dict(type="LoadPointCloudAnnotations", with_bbox=True, enable_difficulty_level=my_paras.get("enable_difficulty_level", False)),
    dict(type="Preprocess", cfg=train_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignTarget", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
    # dict(type='PointCloudCollect', keys=['points', 'voxels', 'annotations', 'calib']),
]
test_pipeline = [
    dict(type="LoadPointCloudFromFile"),
    dict(type="LoadPointCloudAnnotations", with_bbox=True),
    dict(type="Preprocess", cfg=val_preprocessor),
    dict(type="Voxelization", cfg=voxel_generator),
    dict(type="AssignTarget", cfg=train_cfg["assigner"]),
    dict(type="Reformat"),
]
training_pipeline = test_pipeline if my_paras['eval_training_set'] else train_pipeline


data_root = data_root_prefix + "/KITTI/object"
train_anno = data_root_prefix + "/KITTI/object/kitti_infos_" + my_paras['data_mode'] + ".pkl"
val_anno = data_root_prefix + "/KITTI/object/kitti_infos_val.pkl"
test_anno = data_root_prefix + "/KITTI/object/kitti_infos_test.pkl"
trainval_anno = data_root_prefix + "/KITTI/object/kitti_infos_trainval.pkl"

data = dict(
    samples_per_gpu=my_paras['batch_size'],  # batch_size: 4
    workers_per_gpu=2,  # default: 2
    train=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=train_anno,
        class_names=class_names,
        pipeline=training_pipeline,
    ),
    val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
    trainval=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=trainval_anno,
        class_names=class_names,
        pipeline=test_pipeline,
    ),
    train_unlabel_val=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=val_anno,
        class_names=class_names,
        pipeline=train_pipeline,
        labeled=False,
    ),
    train_unlabel_test=dict(
        type=dataset_type,
        root_path=data_root,
        info_path=test_anno,
        class_names=class_names,
        pipeline=train_pipeline,
        labeled=False,
    ),
)

# for cia optimizer
optimizer = dict(type="adam", amsgrad=0.0, wd=0.01, fixed_wd=True, moving_average=False,)
optimizer_config = dict(grad_clip=dict(max_norm=35, norm_type=2))
lr_config = dict(type="one_cycle", lr_max=0.003, moms=[0.95, 0.85], div_factor=10.0, pct_start=0.4,)  # learning policy in training hooks



checkpoint_config = dict(interval=1)
log_config = dict(interval=10,hooks=[dict(type="TextLoggerHook"),],) # dict(type='TensorboardLoggerHook')

# runtime settings
total_epochs = 60
device_ids = range(8)
dist_params = dict(backend="nccl", init_method="env://")
log_level = "INFO"
work_dir = "/mnt/proj50/zhengwu/saved_model/KITTI/proj52/megvii/second/" + TAG
# load_from: "path of pre-trained checkpoint to initialize both teacher & student, e.g., CIA-SSD pre-trained model"
# load_from = "/xxx/xxx/xxx/epoch_60.pth"
load_from = "/mnt/proj50/zhengwu/saved_model/KITTI/proj52/megvii/second/pre_trained_model_2/epoch_60.pth"
resume_from = None
workflow = [("train", 60), ("val", 1)] if my_paras['enable_ssl'] else [("train", 60), ("val", 1)]
save_file = False if TAG == "debug" or TAG == "exp_debug" or Path(work_dir, "Det3D").is_dir() else True


