## SE-SSD: Self-Ensembling Single-Stage Object Detector From Point Cloud (CVPR 2021) [[Paper]](https://arxiv.org/abs/2104.09804)

An accurate and fast single-stage 3D object detection framework on KITTI dataset.

**Authors**: [Wu Zheng](https://github.com/Vegeta2020), Weiliang Tang, [Li Jiang](https://github.com/llijiang), Chi-Wing Fu.


## TensorRT Version
A faster [TensorRT version](https://github.com/jingyue202205/SE-SSD-AI-TRT) of SE-SSD is going to be available thanks to [@jingyue202205](https://github.com/jingyue202205).


## AP on KITTI Dataset


Val Split (11 recall points):
```
AP_11: car AP(Average Precision)@0.70, 0.70, 0.70:
bbox AP:98.72, 90.10, 89.57
bev  AP:90.61, 88.76, 88.18
3d   AP:90.21, 86.25, 79.22
aos  AP:98.67, 89.86, 89.16


AP_40: car AP(Average Precision)@0.70, 0.70, 0.70:
bbox AP:99.57, 95.58, 93.16
bev  AP:96.70, 92.15, 89.74
3d   AP:93.75, 86.18, 83.50
aos  AP:99.52, 95.28, 92.69

```

Test Split: [Submission link](http://www.cvlibs.net/datasets/kitti/eval_object_detail.php?&result=14e5c4daac79d3aef85a842f79538defb1b37ad1)

You may download the trained student model [here](https://drive.google.com/file/d/1M2nP_bGpOy0Eo90xWFoTIUkjhdw30Pjs/view?usp=sharing), which is trained on the train split (3712 samples).

## Pipeline

![pipeline](https://github.com/Vegeta2020/SE-SSD/blob/master/pictures/pipeline.png)
The framework of our Self-Ensembling Single-Stage object Detector (SE-SSD) with a teacher SSD and a student SSD. To start, we feed the input point cloud to the teacher to produce relatively precise bounding boxes and confidence, and take these predictions (after
global transformations) as soft targets to supervise the student with our consistency loss. On the top branch, we apply the same global transformations to the input, then perform our shape-aware data augmentation to generate augmented samples
as inputs to the student. Further, we formulate the Orientation-aware Distance-IoU loss to supervise the student with hard targets, and update the teacher parameters based on the student parameters with the exponential moving average (EMA) strategy. In this
way, the framework can boost the precisions of the detector significantly without incurring extra computation during the inference.

## Installation

```bash
$ git clone https://github.com/Vegeta2020/SE-SSD.git
$ cd ./SE-SSD/det3d/core/iou3d
$ python setup.py install
$ cd ./SE-SSD
$ python setup.py build develop

$ git clone https://github.com/jackd/ifp-sample.git
$ pip install -e ifp-sample
```
Please follow Det3D for installation of other [related packages](https://github.com/poodarchu/Det3D/blob/master/INSTALLATION.md) and [data preparation](https://github.com/poodarchu/Det3D/blob/master/GETTING_STARTED.md).

## Train and Eval

Configure the model in
```bash
$ /SE-SSD/examples/second/configs/config.py
```

Please use our code to generate ground truth data:
```bash
$ python ./SE-SSD/tools/create_data.py
```

Train the SE-SSD:
```bash
$ cd ./SE-SSD/tools
$ python train.py  # Single GPU
$ python -m torch.distributed.launch --nproc_per_node=4 train.py   # Multiple GPU
```

Evaluate the SE-SSD:
```bash
$ cd ./SE-SSD/tools
$ python test.py
```

## Citation
If you find this work useful in your research, please star our repository and consider citing:
```
@inproceedings{zheng2021se,
  title={SE-SSD: Self-Ensembling Single-Stage Object Detector From Point Cloud},
  author={Wu Zheng, Weiliang Tang, Li Jiang, Chi-Wing Fu},
  booktitle={CVPR},
  pages={14494--14503},
  year={2021}
}
```


## Acknowledgement
Thanks for previous works [Det3D](https://github.com/poodarchu/det3d) and [CIA-SSD](https://github.com/Vegeta2020/CIA-SSD), as our codebase are mainly based on them. 

Special thanks for [Dingfu Zhou](https://github.com/dingfuzhou) for his shared code.

Thanks for the reviewers's valuable comments on this paper.

## Contact
If you have any question or suggestion about this repo, please feel free to contact me (zheng-w10@foxmail.com)
