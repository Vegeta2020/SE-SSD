#!/bin/bash
TASK_DESC='second'


if [ ! $TASK_DESC ] 
then
    echo "TASK_DESC must be specified."
    echo "Usage: train.sh task_description"
    exit $E_ASSERT_FAILED
fi

# Voxelnet
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --master_port=2020 --nproc_per_node=8 train.py
CUDA_VISIBLE_DEVICES=4,5 nohup  python -m torch.distributed.launch --nproc_per_node=2 --master_port=2040 train.py  & -> proj52: dist_newest_default_3
CUDA_VISIBLE_DEVICES=1,2,3 nohup python -m torch.distributed.launch --nproc_per_node=3 --master_port=2090 train.py  & -> proj52: dist_newest_default_4

CUDA_VISIBLE_DEVICES=6,7 nohup python -m torch.distributed.launch --nproc_per_node=2 --master_port=3000 train.py & --resume latest.pth &
CUDA_VISIBLE_DEVICES=7 nohup python train.py --resume latest.pth &
CUDA_VISIBLE_DEVICES=1,2,3 python -m torch.distributed.launch --nproc_per_node=3 --master_port=2020 train.py  -> proj52: dist_newest_default_3
#python -m torch.distributed.launch --nproc_per_node=8 ./tools/train.py examples/second/configs/nusc_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn.py --work_dir=$NUSC_CBGS_WORK_DIR
# python -m torch.distributed.launch --nproc_per_node=8 ./tools/train.py examples/second/configs/lyft_all_vfev3_spmiddleresnetfhd_rpn2_mghead_syncbn.py --work_dir=$LYFT_CBGS_WORK_DIR

# PointPillars
# python -m torch.distributed.launch --nproc_per_node=8 ./tools/train.py ./examples/point_pillars/configs/original_pp_mghead_syncbn_kitti.py --work_dir=$PP_WORK_DIR
export PYTHONPATH=/research/dept7/wuzheng/software/spconv/spconv/:$PYTHONPATH