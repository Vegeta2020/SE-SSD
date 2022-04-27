import logging
from collections import defaultdict
from enum import Enum
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np
import torch
from det3d.core.bbox import box_torch_ops
from det3d.models.builder import build_loss
from det3d.models.losses import metrics
from det3d.torchie.cnn import constant_init, kaiming_init
from det3d.torchie.trainer import load_checkpoint
from torch import nn
from torch.nn import functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from det3d.core.iou3d import iou3d_utils
from det3d.core.bbox import box_np_ops, box_torch_ops

from .. import builder
from ..losses import accuracy
from ..registry import HEADS
from det3d.core.sampler import preprocess as prep
import time


def one_hot_f(tensor, depth, dim=-1, on_value=1.0, dtype=torch.float32):
    '''
        Tensor: [4, 70400], elem is in range(depth), like 0 or 1 for depth=2, which means index of label 1 in last \
                dimension of tensor_onehot;
        tensor_onehot.scatter_(dim, index_matrix, value): dim mean target dim of tensor_onehot to pad value, index_matrix
                has the shape of tensor_onehot.shape[:-1], denoting index of label 1 in the target dim.
    '''
    tensor_onehot = torch.zeros(*list(tensor.shape), depth, dtype=dtype, device=tensor.device)  # [4, 70400, 2]
    tensor_onehot.scatter_(dim, tensor.unsqueeze(dim).long(), on_value)  # [4, 70400, 2]
    return tensor_onehot


def add_sin_difference(boxes1, boxes2):
    rad_pred_encoding = torch.sin(boxes1[..., -1:]) * torch.cos(boxes2[..., -1:])  # ry -> sin(pred_ry)*cos(gt_ry)
    rad_gt_encoding = torch.cos(boxes1[..., -1:]) * torch.sin(boxes2[..., -1:])  # ry -> cos(pred_ry)*sin(gt_ry)
    res_boxes1 = torch.cat([boxes1[..., :-1], rad_pred_encoding], dim=-1)
    res_boxes2 = torch.cat([boxes2[..., :-1], rad_gt_encoding], dim=-1)
    return res_boxes1, res_boxes2


def _get_pos_neg_loss(cls_loss, labels):
    # cls_loss: [N, num_anchors, num_class], it has been averaged on each sample
    # labels: [N, num_anchors]
    batch_size = cls_loss.shape[0]
    if cls_loss.shape[-1] == 1 or len(cls_loss.shape) == 2:
        cls_pos_loss = (labels > 0).type_as(cls_loss) * cls_loss.view(batch_size, -1)
        cls_neg_loss = (labels == 0).type_as(cls_loss) * cls_loss.view(batch_size, -1)
        cls_pos_loss = cls_pos_loss.sum() / batch_size  # averaged on batch
        cls_neg_loss = cls_neg_loss.sum() / batch_size
    else:
        cls_pos_loss = cls_loss[..., 1:].sum() / batch_size
        cls_neg_loss = cls_loss[..., 0].sum() / batch_size
    return cls_pos_loss, cls_neg_loss


def get_direction_target(anchors, reg_targets, one_hot=True, dir_offset=0.0):
    '''
        Paras:
            anchors: [batch_size, w*h*num_anchor_per_pos, anchor_dim], [4, 70400, 7];
            reg_targets: same shape as anchors, [4, 70400, 7] here;
        return:
            dir_clas_targets: [batch_size, w*h*num_anchor_per_pos, 2]
    '''
    batch_size = reg_targets.shape[0]
    anchors = anchors.view(batch_size, -1, anchors.shape[-1])
    rot_gt = reg_targets[..., -1] + anchors[..., -1]  # original ry, [4, 70400]
    dir_cls_targets = ((rot_gt - dir_offset) > 0).long()  # [4, 70400], elem: 0 or 1, todo: physical scene
    if one_hot:
        dir_cls_targets = one_hot_f(dir_cls_targets, 2, dtype=anchors.dtype)
    return dir_cls_targets


def smooth_l1_loss(pred, gt, sigma):
    def _smooth_l1_loss(pred, gt, sigma):
        sigma2 = sigma ** 2
        cond_point = 1 / sigma2
        x = pred - gt
        abs_x = torch.abs(x)

        in_mask = abs_x < cond_point
        out_mask = 1 - in_mask

        in_value = 0.5 * (sigma * x) ** 2
        out_value = abs_x - 0.5 / sigma2

        value = in_value * in_mask.type_as(in_value) + out_value * out_mask.type_as(
            out_value
        )
        return value

    value = _smooth_l1_loss(pred, gt, sigma)
    loss = value.mean(dim=1).sum()
    return loss


def smooth_l1_loss_detectron2(input, target, beta: float, reduction: str = "none"):
    """
    Smooth L1 loss defined in the Fast R-CNN paper as:
                  | 0.5 * x ** 2 / beta   if abs(x) < beta
    smoothl1(x) = |
                  | abs(x) - 0.5 * beta   otherwise,
    where x = input - target.
    Smooth L1 loss is related to Huber loss, which is defined as:
                | 0.5 * x ** 2                  if abs(x) < beta
     huber(x) = |
                | beta * (abs(x) - 0.5 * beta)  otherwise
    Smooth L1 loss is equal to huber(x) / beta. This leads to the following
    differences:
     - As beta -> 0, Smooth L1 loss converges to L1 loss, while Huber loss
       converges to a constant 0 loss.
     - As beta -> +inf, Smooth L1 converges to a constant 0 loss, while Huber loss
       converges to L2 loss.
     - For Smooth L1 loss, as beta varies, the L1 segment of the loss has a constant
       slope of 1. For Huber loss, the slope of the L1 segment is beta.
    Smooth L1 loss can be seen as exactly L1 loss, but with the abs(x) < beta
    portion replaced with a quadratic function such that at abs(x) = beta, its
    slope is 1. The quadratic segment smooths the L1 loss near x = 0.
    Args:
        input (Tensor): input tensor of any shape
        target (Tensor): target value tensor with the same shape as input
        beta (float): L1 to L2 change point.
            For beta values < 1e-5, L1 loss is computed.
        reduction: 'none' | 'mean' | 'sum'
                 'none': No reduction will be applied to the output.
                 'mean': The output will be averaged.
                 'sum': The output will be summed.
    Returns:
        The loss with the reduction option applied.
    Note:
        PyTorch's builtin "Smooth L1 loss" implementation does not actually
        implement Smooth L1 loss, nor does it implement Huber loss. It implements
        the special case of both in which they are equal (beta=1).
        See: https://pytorch.org/docs/stable/nn.html#torch.nn.SmoothL1Loss.
     """
    if beta < 1e-5:
        # if beta == 0, then torch.where will result in nan gradients when
        # the chain rule is applied due to pytorch implementation details
        # (the False branch "0.5 * n ** 2 / 0" has an incoming gradient of
        # zeros, rather than "no gradient"). To avoid this issue, we define
        # small values of beta to be exactly l1 loss.
        loss = torch.abs(input - target)
    else:
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


def create_loss(
        loc_loss_ftor,
        cls_loss_ftor,
        box_preds,
        cls_preds,
        cls_targets,
        cls_weights,
        reg_targets,
        reg_weights,
        num_class,
        encode_background_as_zeros=True,
        encode_rad_error_by_sin=True,
        bev_only=False,
        box_code_size=9,
):
    batch_size = int(box_preds.shape[0])
    box_preds = box_preds.view(batch_size, -1, box_code_size)  # [8, 200, 176, 14] -> [8, 70400, 7]
    cls_preds = cls_preds.view(batch_size, -1, num_class)  # [8, 70400] -> [8, 70400, 1]

    if encode_rad_error_by_sin:  # True
        # sin(a - b) = sinacosb-cosasinb, a: pred, b: gt; box_preds: ry_a -> sinacosb; reg_targets: ry_b -> cosasinb
        box_preds, reg_targets = add_sin_difference(box_preds, reg_targets)

    loc_losses = loc_loss_ftor(box_preds, reg_targets, weights=reg_weights)  # [N, M]
    cls_losses = cls_loss_ftor(cls_preds, cls_targets, weights=cls_weights)  # [N, M]
    return loc_losses, cls_losses


class LossNormType(Enum):
    NormByNumPositives = "norm_by_num_positives"
    NormByNumExamples = "norm_by_num_examples"
    NormByNumPosNeg = "norm_by_num_pos_neg"
    DontNorm = "dont_norm"


@HEADS.register_module
class Head(nn.Module):
    def __init__(self, num_input, num_pred, num_cls, use_dir=False, num_dir=0, header=True, name="",
                 focal_loss_init=False, **kwargs, ):
        super(Head, self).__init__(**kwargs)
        self.use_dir = use_dir

        self.conv_box = nn.Conv2d(num_input, num_pred, 1)  # 128 -> 14
        self.conv_cls = nn.Conv2d(num_input, num_cls, 1)  # 128 -> 2
        self.conv_iou = nn.Conv2d(num_input, 2, 1)  # 128 -> 2
        #self.conv_iou = nn.Conv2d(in_channels=128, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False,)

        self.trans_conv = None
        # self.trans_conv = nn.Sequential(
        #     nn.Conv2d(num_input, num_input, 1),
        #     nn.BatchNorm2d(num_input, eps=1e-3, momentum=0.01),
        #     nn.ReLU(),
        # )

        if self.use_dir:
            self.conv_dir = nn.Conv2d(num_input, num_dir, 1)  # 128 -> 4

    def forward(self, x):
        ret_list = []  # x.shape=[8, 128, 200, 176]
        if self.trans_conv:
            x = self.trans_conv(x)
        box_preds = self.conv_box(x).permute(0, 2, 3, 1).contiguous()  # box_preds.shape=[8, 200, 176, 14]
        cls_preds = self.conv_cls(x).permute(0, 2, 3, 1).contiguous()  # cls_preds.shape=[8, 200, 176, 2]
        ret_dict = {"box_preds": box_preds, "cls_preds": cls_preds}
        if self.use_dir:
            dir_preds = self.conv_dir(x).permute(0, 2, 3, 1).contiguous()  # dir_preds.shape=[8, 200, 176, 4]
            ret_dict["dir_cls_preds"] = dir_preds

        ret_dict["iou_preds"] = self.conv_iou(x).permute(0, 2, 3, 1).contiguous()

        return ret_dict


@HEADS.register_module
class RegHead(nn.Module):
    def __init__(self, mode="z", in_channels=sum([128, ]), norm_cfg=None, tasks=None, name="rpn", logger=None,
                 **kwargs, ):
        super(RegHead, self).__init__()

        num_classes = [len(t["class_names"]) for t in tasks]
        self.class_names = [t["class_names"] for t in tasks]

        if norm_cfg is None:
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self._norm_cfg = norm_cfg

        num_preds = []
        for idx, num_class in enumerate(num_classes):
            num_preds.append(2)  # h, z, gp

        self.tasks = nn.ModuleList()
        num_input = in_channels
        for task_id, num_pred in enumerate(num_preds):
            conv_box = nn.Conv2d(num_input, num_pred, 1)
            self.tasks.append(conv_box)

        self.crop_cfg = kwargs.get("crop_cfg", None)
        self.z_mode = kwargs.get("z_type", "top")
        self.iou_loss = kwargs.get("iou_loss", False)

    def forward(self, x):

        ret_dicts = []
        for task in self.tasks:
            out = task(x)
            out = F.max_pool2d(out, kernel_size=out.size()[2:])
            ret_dicts.append(out.permute(0, 2, 3, 1).contiguous())

        return ret_dicts

    def loss(self, example, preds, **kwargs):
        batch_size_dev = example["targets"].shape[0]
        rets = []
        for task_id, task_pred in enumerate(preds):

            zg = example["targets"][:, 2:3]
            hg = example["targets"][:, 3:4]
            gg = example["targets"][:, 4:5]
            gp = example["ground_plane"].view(-1, 1)

            zt = task_pred[:, :, :, 0:1].view(-1, 1)
            ht = task_pred[:, :, :, 1:2].view(-1, 1)

            height_loss = smooth_l1_loss(ht, hg, 3.0) / batch_size_dev

            height_a = self.crop_cfg.anchor.height
            z_center_a = self.crop_cfg.anchor.center

            if self.z_mode == "top":
                z_top_a = z_center_a + height_a / 2
                gt = z_top_a + zt - (height_a + ht) - gp
                z_loss = smooth_l1_loss(zt, zg, 3.0) / batch_size_dev
                # IoU Loss
                yg_top, yg_down = zg + z_top_a, zg + z_top_a - (hg + height_a)
                yp_top, yp_down = zt + z_top_a, zt + z_top_a - (ht + height_a)

            elif self.z_mode == "center":
                gt = z_center_a + zt - (height_a + ht) / 2.0 - gp
                z_loss = smooth_l1_loss(zt, zg, 3.0) / batch_size_dev
                # IoU Loss
                yg_top, yg_down = (
                    zg + z_center_a + (hg + height_a) / 2.0,
                    zg + z_center_a - (hg + height_a) / 2.0,
                )
                yp_top, yp_down = (
                    zt + z_center_a + (ht + height_a) / 2.0,
                    zt + z_center_a - (ht + height_a) / 2.0,
                )

            gp_loss = smooth_l1_loss(gt, gg, 3.0) / batch_size_dev

            h_intersect = torch.min(yp_top, yg_top) - torch.max(yp_down, yg_down)
            iou = h_intersect / (hg + height_a + ht + height_a - h_intersect)
            iou[iou < 0] = 0.0
            iou[iou > 1] = 1.0
            iou_loss = (1 - iou).sum() / batch_size_dev

            ret = dict(
                loss=z_loss + height_loss + gp_loss,
                z_loss=z_loss,
                height_loss=height_loss,
                gp_loss=gp_loss,
            )

            if self.iou_loss:
                ret["loss"] += iou_loss
                ret.update(dict(iou_loss=iou_loss, ))
            rets.append(ret)
        """convert batch-key to key-batch
        """
        rets_merged = defaultdict(list)
        for ret in rets:
            for k, v in ret.items():
                rets_merged[k].append(v)

        return rets_merged

    def predict(self, example, preds, stage_one_outputs):
        rets = []
        for task_id, task_pred in enumerate(preds):
            if self.z_mode == "top":
                height_a = self.crop_cfg.anchor.height
                z_top_a = self.crop_cfg.anchor.center + height_a / 2
                # z top
                task_pred[:, :, :, 1] += height_a
                task_pred[:, :, :, 0] += z_top_a - task_pred[:, :, :, 1] / 2.0
            elif self.z_mode == "center":
                height_a = self.crop_cfg.anchor.height
                z_center_a = self.crop_cfg.anchor.center
                # regress h and z
                task_pred[:, :, :, 0] += z_center_a
                task_pred[:, :, :, 1] += height_a

            rets.append(task_pred.view(-1, 2))

        # Merge branches results
        num_tasks = len(rets)
        ret_list = []
        # len(rets) == task num
        # len(rets[0]) == batch_size
        num_preds = len(rets)
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            ret = torch.cat(rets)
            ret_list.append(ret)

        cnt = 0
        for idx, sample in enumerate(stage_one_outputs):
            for box in sample["box3d_lidar"]:
                box[2] = ret_list[0][cnt, 0]
                box[5] = ret_list[0][cnt, 1]

                cnt += 1

        return stage_one_outputs


@HEADS.register_module
class MultiGroupHead(nn.Module):
    def __init__(self,
                 mode="3d",
                 in_channels=[128, ],
                 norm_cfg=None,
                 tasks=[],  # [dict(num_class=1, class_names=["Car"],),].
                 weights=[],  # [1].
                 num_classes=[1, ],  #
                 box_coder=None,  # func: box_torch_ops.second_box_encode/second_box_decode
                 with_cls=True,
                 with_reg=True,
                 reg_class_agnostic=False,
                 encode_background_as_zeros=True,  # actually discard this category
                 loss_norm=dict(type="NormByNumPositives", pos_cls_weight=1.0, neg_cls_weight=1.0, ),
                 loss_cls=dict(type="SigmoidFocalLoss", alpha=0.25, gamma=2.0, loss_weight=1.0, ),
                 use_sigmoid_score=True,
                 loss_bbox=dict(type="WeightedSmoothL1Loss", sigma=3.0,
                                code_weights=[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0], codewise=True, loss_weight=2.0, ),
                 encode_rad_error_by_sin=True,
                 loss_aux=dict(type="WeightedSoftmaxClassificationLoss", name="direction_classifier",
                               loss_weight=0.2, ),
                 direction_offset=0.0,
                 name="rpn",
                 logger=None,
                 ):
        super(MultiGroupHead, self).__init__()

        assert with_cls or with_reg

        num_classes = [len(t["class_names"]) for t in tasks]  # [1]
        self.class_names = [t["class_names"] for t in tasks]  # [['Car']]
        self.num_anchor_per_locs = [2 * n for n in num_classes]  # [2]

        self.box_coder = box_coder
        box_code_sizes = [box_coder.n_dim] * len(num_classes)  # [7]*1

        self.with_cls = with_cls  # True
        self.with_reg = with_reg  # True
        self.in_channels = in_channels  # 128
        self.num_classes = num_classes  # 1
        self.reg_class_agnostic = reg_class_agnostic  # False
        self.encode_rad_error_by_sin = encode_rad_error_by_sin  # True
        self.encode_background_as_zeros = encode_background_as_zeros  # True
        self.use_sigmoid_score = use_sigmoid_score  # True
        self.box_n_dim = self.box_coder.n_dim  # 7

        self.loss_cls = build_loss(loss_cls)
        self.loss_reg = build_loss(loss_bbox)

        from det3d.models.losses.odious import odiou_3D
        self.odiou_3d_loss = odiou_3D()
        self.loss_iou_pred = build_loss(dict(type="WeightedSmoothL1Loss", sigma=3.0, code_weights=None, codewise=True, loss_weight=1.0, ))

        if loss_aux is not None:  # True
            self.loss_aux = build_loss(loss_aux)

        self.loss_norm = loss_norm

        if not logger:
            logger = logging.getLogger("MultiGroupHead")
        self.logger = logger

        self.dcn = None
        self.zero_init_residual = False

        self.use_direction_classifier = loss_aux is not None  # WeightedSoftmaxClassificationLoss
        if loss_aux:
            self.direction_offset = direction_offset  # 0

        self.bev_only = True if mode == "bev" else False  # mode='3d' -> False

        # get output_size by calculating num_cls&num_pred(loc)&num_dir
        num_clss = []
        num_preds = []
        num_dirs = []
        for num_c, num_a, box_cs in zip(num_classes, self.num_anchor_per_locs, box_code_sizes):  # 1, 2, 7
            if self.encode_background_as_zeros:  # actually discard this category
                num_cls = num_a * num_c  # 2
            else:
                num_cls = num_a * (num_c + 1)
            num_clss.append(num_cls)  # [2]

            if self.bev_only:
                num_pred = num_a * (box_cs - 2)
            else:
                num_pred = num_a * box_cs  # 14
            num_preds.append(num_pred)  # [14]

            if self.use_direction_classifier:
                num_dir = num_a * 2  # 4, 2 * softmax(2,)
                num_dirs.append(num_dir)  # [4]
            else:
                num_dir = None

        logger.info(f"num_classes: {num_classes}, num_preds: {num_preds}, num_dirs: {num_dirs}")

        # it seems can add multiple head here.
        self.tasks = nn.ModuleList()
        for task_id, (num_pred, num_cls) in enumerate(zip(num_preds, num_clss)):
            self.tasks.append(Head(in_channels, num_pred, num_cls, use_dir=self.use_direction_classifier, \
                                   num_dir=num_dirs[task_id] if self.use_direction_classifier else None,
                                   header=False, ))

        logger.info("Finish MultiGroupHead Initialization")
        post_center_range = [0, -40.0, -5.0, 70.4, 40.0, 5.0]
        self.post_center_range = torch.tensor(post_center_range, dtype=torch.float).cuda()
        self.thresh = torch.tensor([0.3], dtype=torch.float).cuda()
        self.top_labels = torch.zeros([70400], dtype=torch.long, ).cuda()  # [70400]
        self.loss_size_consistency = nn.MSELoss(reduction='mean')
        self.loss_iou_consistency = build_loss(dict(type="WeightedSmoothL1Loss", sigma=3.0, code_weights=None, codewise=True, loss_weight=1.0, ))
        self.loss_score_consistency = build_loss(dict(type="WeightedSmoothL1Loss", sigma=3.0, code_weights=None, codewise=True, loss_weight=1.0, ))
        self.loss_dir_consistency = nn.MSELoss(reduction='mean')

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = logging.getLogger()
            load_checkpoint(self, pretrained, strict=False, logger=logger)
        elif pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv2d):
                    kaiming_init(m)
                elif isinstance(m, (_BatchNorm, nn.GroupNorm)):
                    constant_init(m, 1)

            if self.dcn is not None:
                for m in self.modules():
                    if isinstance(m, Bottleneck) and hasattr(m, "conv2_offset"):
                        constant_init(m.conv2_offset, 0)

            if self.zero_init_residual:
                for m in self.modules():
                    if isinstance(m, Bottleneck):
                        constant_init(m.norm3, 0)
                    elif isinstance(m, BasicBlock):
                        constant_init(m.norm2, 0)
        else:
            raise TypeError("pretrained must be a str or None")

    def forward(self, x):
        ret_dicts = []
        for task in self.tasks:
            # import det3d.visualization.kitti_data_vis.vis_features as vis_features
            ret_dicts.append(task(x))
        return ret_dicts

    def prepare_loss_weights(self, labels, loss_norm=None, dtype=torch.float32, ):
        '''
            get weight of each anchor in each sample; all weights in each sample sum as 1.
        '''
        loss_norm_type = getattr(LossNormType, loss_norm["type"])  # norm_by_num_positives
        pos_cls_weight = loss_norm["pos_cls_weight"]  # 1.0
        neg_cls_weight = loss_norm["neg_cls_weight"]  # 1.0

        cared = labels >= 0  # [N, num_anchors]
        positives = labels > 0
        negatives = labels == 0
        negative_cls_weights = negatives.type(dtype) * neg_cls_weight
        positive_cls_weights = positives.type(dtype) * pos_cls_weight
        cls_weights = negative_cls_weights + positive_cls_weights
        reg_weights = positives.type(dtype)

        if loss_norm_type == LossNormType.NormByNumExamples:
            num_examples = cared.type(dtype).sum(1, keepdim=True)
            num_examples = torch.clamp(num_examples, min=1.0)
            cls_weights /= num_examples
            bbox_normalizer = positives.sum(1, keepdim=True).type(dtype)
            reg_weights /= torch.clamp(bbox_normalizer, min=1.0)

        elif loss_norm_type == LossNormType.NormByNumPositives:  # True
            pos_normalizer = positives.sum(1, keepdim=True).type(dtype)  # [batch_size, 1]
            reg_weights /= torch.clamp(pos_normalizer, min=1.0)  # [N, num_anchors], average in each sample;
            cls_weights /= torch.clamp(pos_normalizer, min=1.0)  # todo: interesting, how about the negatives samples

        elif loss_norm_type == LossNormType.NormByNumPosNeg:
            pos_neg = torch.stack([positives, negatives], dim=-1).type(dtype)
            normalizer = pos_neg.sum(1, keepdim=True)  # [N, 1, 2]
            cls_normalizer = (pos_neg * normalizer).sum(-1)  # [N, M]
            cls_normalizer = torch.clamp(cls_normalizer, min=1.0)
            # cls_normalizer will be pos_or_neg_weight/num_pos_or_neg
            normalizer = torch.clamp(normalizer, min=1.0)
            reg_weights /= normalizer[:, 0:1, 0]
            cls_weights /= cls_normalizer

        elif loss_norm_type == LossNormType.DontNorm:  # support ghm loss
            pos_normalizer = positives.sum(1, keepdim=True).type(dtype)
            reg_weights /= torch.clamp(pos_normalizer, min=1.0)

        else:
            raise ValueError(f"unknown loss norm type. available: {list(LossNormType)}")

        return cls_weights, reg_weights, cared


    def nn_distance(self, box1, box2, iou_thres=0.7, return_loss='10'):
        """
            box1: (N,C) torch tensor;   box2: (M,C) torch tensor
        """

        ans_iou = iou3d_utils.boxes_iou_bev_gpu(box1, box2)
        iou1, idx1 = torch.max(ans_iou, dim=1)
        iou2, idx2 = torch.max(ans_iou, dim=0)
        mask1, mask2 = iou1 > iou_thres, iou2 > iou_thres
        ans_iou = ans_iou[mask1]
        ans_iou = ans_iou[:, mask2]

        if ans_iou.shape[0] == 0 or ans_iou.shape[1] == 0:  # for unlabeled data (some scenes wo cars)
            return [None] * 5

        iou1, idx1 = torch.max(ans_iou, dim=1)
        iou2, idx2 = torch.max(ans_iou, dim=0)
        val_box1, val_box2 = box1[mask1], box2[mask2]
        aligned_box1, aligned_box2 = val_box1[idx2], val_box2[idx1]

        # iou3d, iou_bev = iou3d_utils.boxes_aligned_iou3d_gpu(val_box2, aligned_box1, need_bev=True)
        encoded_box_preds, encoded_reg_targets = add_sin_difference(val_box1, aligned_box2)
        loss1 = self.loss_reg(encoded_box_preds, encoded_reg_targets).sum(-1) / 7.
        encoded_box_preds, encoded_reg_targets = add_sin_difference(val_box2, aligned_box1)
        loss2 = self.loss_reg(encoded_box_preds, encoded_reg_targets).sum(-1) / 7.
        if return_loss == '10':
            box_consistency_loss = loss1.sum() / loss1.shape[0]
        elif return_loss == '01':
            box_consistency_loss = loss2.sum() / loss2.shape[0]
        elif return_loss == '11':
            box_consistency_loss = (loss1.sum() + loss2.sum()) / (loss1.shape[0] + loss2.shape[0])
        else:
            raise NotImplementedError

        return box_consistency_loss, idx1, idx2, mask1, mask2

    def per_box_loc_trans(self, boxes, gt_boxes, trans):
        gt_indices = iou3d_utils.boxes_iou3d_gpu(boxes.cuda(), torch.from_numpy(gt_boxes).float().cuda()).max(-1)[1]
        trans_loc = torch.from_numpy(trans['translation_loc']).float().cuda()[gt_indices]
        rot_loc = torch.from_numpy(trans['rotation_loc']).float().cuda()[gt_indices]
        boxes[:, :3] += trans_loc
        boxes[:, 6] += rot_loc
        return boxes


    def consistency_loss(self, preds_stu, preds_tea, example):
        '''
            each prediction of student matched with one prediction of teacher
        '''
        batch_size = preds_stu[0]['box_preds'].shape[0]
        # trans, trans_res = example['transformation'][unsupervision_mask], {}
        # for key in trans[0].keys():
        #     val_list = [trans[i][key] for i in range(batch_size)]
        #     trans_res.update({key: np.stack(val_list)})
        batch_trans = example['transformation']
        batch_gt_dict_raw = example['annos_raw']
        batch_box_preds_stu = preds_stu[0]["box_preds"].view(batch_size, -1, 7)
        batch_cls_preds_stu = preds_stu[0]["cls_preds"].view(batch_size, -1, 1)
        batch_dir_preds_stu = preds_stu[0]["dir_cls_preds"].view(batch_size, -1, 2)
        batch_iou_preds_stu = preds_stu[0]["iou_preds"].view(batch_size, -1, 1)
        batch_box_preds_tea = preds_tea[0]["box_preds"].view(batch_size, -1, 7)
        batch_cls_preds_tea = preds_tea[0]["cls_preds"].view(batch_size, -1, 1)
        batch_dir_preds_tea = preds_tea[0]["dir_cls_preds"].view(batch_size, -1, 2)
        batch_iou_preds_tea = preds_tea[0]["iou_preds"].view(batch_size, -1, 1)

        batch_box_loss = torch.tensor([0.], dtype=torch.float32).cuda()
        batch_cls_loss = torch.tensor([0.], dtype=torch.float32).cuda()
        batch_iou_loss = torch.tensor([0.], dtype=torch.float32).cuda()
        batch_dir_loss = torch.tensor([0.], dtype=torch.float32).cuda()

        batch_id = 0
        for box_preds_stu_offset, cls_preds_stu, dir_preds_stu, iou_preds_stu, \
            box_preds_tea_offset, cls_preds_tea, dir_preds_tea, iou_preds_tea, trans in \
                zip(batch_box_preds_stu, batch_cls_preds_stu, batch_dir_preds_stu, batch_iou_preds_stu,
                    batch_box_preds_tea, batch_cls_preds_tea, batch_dir_preds_tea, batch_iou_preds_tea, batch_trans):
            batch_id += 1
            box_preds_stu = self.box_coder.decode_torch(box_preds_stu_offset, example["anchors"][0][0])
            box_preds_tea = self.box_coder.decode_torch(box_preds_tea_offset, example["anchors"][0][0])

            # filter predicted boxes
            top_scores_keep_stu = torch.sigmoid(cls_preds_stu).squeeze(-1) >= 0.3  # [70400]
            top_scores_keep_tea = torch.sigmoid(cls_preds_tea).squeeze(-1) >= 0.3  # [70400]
            pos_anchors = example['anchors'][0][0][top_scores_keep_tea]
            mask_stu = (box_preds_stu[:, :3] >= self.post_center_range[:3]).all(1)
            mask_stu &= (box_preds_stu[:, :3] <= self.post_center_range[3:]).all(1)
            mask_stu &= top_scores_keep_stu
            mask_tea = (box_preds_tea[:, :3] >= self.post_center_range[:3]).all(1)
            mask_tea &= (box_preds_tea[:, :3] <= self.post_center_range[3:]).all(1)
            mask_tea &= top_scores_keep_tea
            top_box_preds_stu, top_cls_preds_stu, top_dir_preds_stu, top_iou_preds_stu, \
            top_box_preds_tea, top_cls_preds_tea, top_dir_preds_tea, top_iou_preds_tea \
            = box_preds_stu[mask_stu], cls_preds_stu[mask_stu], dir_preds_stu[mask_stu], iou_preds_stu[mask_stu], \
              box_preds_tea[mask_tea], cls_preds_tea[mask_tea], dir_preds_tea[mask_tea], iou_preds_tea[mask_tea]

            if mask_stu.sum() > 0 and mask_tea.sum() > 0:
                # transform boxes predicted by teacher with local & global augmentation
                # top_box_preds_tea = self.per_box_loc_trans(top_box_preds_tea, gt_dict_raw['gt_boxes'][0], trans)
                top_box_preds_tea[:, 1] = - top_box_preds_tea[:, 1] if trans["flipped"] else top_box_preds_tea[:, 1]
                top_box_preds_tea[:, -1] = - top_box_preds_tea[:, -1] + np.pi if trans["flipped"] else top_box_preds_tea[:, -1]
                top_box_preds_tea[:, :3] = box_torch_ops.rotation_points_single_angle(top_box_preds_tea[:, :3], trans["noise_rotation"], axis=2)
                top_box_preds_tea[:, -1] += trans["noise_rotation"]
                top_box_preds_tea[:, :-1] *= trans["noise_scale"]
                # top_box_preds_tea[:, :3] += torch.from_numpy(trans['noise_trans']).float().cuda()

                # center consistency loss
                box_consistency_loss, idx1, idx2, mask1, mask2 = self.nn_distance(top_box_preds_stu, top_box_preds_tea)
                if box_consistency_loss is None:
                    continue
                batch_box_loss += box_consistency_loss

                # cls_score consistency loss
                aligned_cls_preds_stu, aligned_cls_preds_tea = top_cls_preds_stu[mask1][idx2], top_cls_preds_tea[mask2][idx1]
                scores_stu, scores_tea = torch.sigmoid(top_cls_preds_stu[mask1]), torch.sigmoid(aligned_cls_preds_tea)
                score_consistency_loss = self.loss_score_consistency(scores_stu, scores_tea).mean()
                batch_cls_loss += score_consistency_loss

                # iou consistency loss
                aligned_iou_preds_tea = (top_iou_preds_tea[mask2][idx1] + 1) * 0.5
                top_iou_preds_stu = (top_iou_preds_stu[mask1] + 1) * 0.5
                iou_consistency_loss = self.loss_iou_consistency(top_iou_preds_stu, aligned_iou_preds_tea).mean()
                batch_iou_loss += iou_consistency_loss

                # dir consistency loss
                aligned_dir_preds_tea = top_dir_preds_tea[mask2][idx1]
                aligned_dir_preds_tea = F.softmax(aligned_dir_preds_tea, dim=-1)
                top_dir_preds_stu = F.softmax(top_dir_preds_stu[mask1], dim=-1)
                dir_consistency_loss = self.loss_dir_consistency(top_dir_preds_stu, aligned_dir_preds_tea)
                batch_dir_loss += dir_consistency_loss

        consistency_loss = (1.0 * batch_box_loss + 1.0 * batch_cls_loss + 1.0 * batch_iou_loss) / batch_size
        return consistency_loss


    def loss(self, example, preds_dicts, preds_ema, **kwargs):
        supervision_mask = example["ssl_labeled"] == 1 if "ssl_labeled" in example.keys() else torch.ones(len(example['metadata'])) == 1
        consistency_loss = self.consistency_loss(preds_dicts, preds_ema, example)
        loss_ema = self.get_model_ema_loss(example, preds_ema)

        batch_anchors = example["anchors"][0][supervision_mask]
        batch_size_device = batch_anchors.shape[0]

        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            # get predictions.
            box_preds = preds_dict["box_preds"][supervision_mask]
            cls_preds = preds_dict["cls_preds"][supervision_mask]

            # get targets and weights.
            labels = example["labels"][task_id]  # cls_labels: [batch_size, 70400], elem in [-1, 0, 1].
            reg_targets = example["reg_targets"][task_id]  # reg_labels: [batch_size, 70400, 7].
            cls_weights, reg_weights, cared = self.prepare_loss_weights(labels, loss_norm=self.loss_norm, dtype=torch.float32, )  # all: [batch_size, 70400]
            cls_targets = labels * cared.type_as(labels)  # filter -1 in labels.
            cls_targets = cls_targets.unsqueeze(-1)  # [batch_size, 70400, 1].

            # get localization and classification loss.
            batch_size = int(box_preds.shape[0])
            box_preds = box_preds.view(batch_size, -1, self.box_n_dim)  # [batch_size, 200, 176, 14] -> [batch_size, 70400, 7].
            cls_preds = cls_preds.view(batch_size, -1, self.num_classes[task_id])  # [batch_size, 70400] -> [batch_size, 70400, 1].

            if self.encode_rad_error_by_sin:  # True
                # todo: Notice, box_preds.ry has changed to box_preds.sinacosb.
                # sin(a - b) = sinacosb-cosasinb, a: pred, b: gt; box_preds: ry_a -> sinacosb; reg_targets: ry_b -> cosasinb.
                encoded_box_preds, encoded_reg_targets = add_sin_difference(box_preds, reg_targets)

            loc_loss = self.loss_reg(encoded_box_preds, encoded_reg_targets, weights=reg_weights)  # [N, 70400, 7], WeightedSmoothL1Loss, averaged in sample.
            cls_loss = self.loss_cls(cls_preds, cls_targets, weights=cls_weights)  # [N, 70400, 1], SigmoidFocalLoss, averaged in sample.
            loc_loss_reduced = self.loss_reg._loss_weight * loc_loss.sum() / batch_size_device  # 2.0, averaged on batch_size
            cls_loss_reduced = self.loss_cls._loss_weight * cls_loss.sum() / batch_size_device  # 1.0, average on batch_size

            cls_pos_loss, cls_neg_loss = _get_pos_neg_loss(cls_loss, labels)  # for analysis, average on batch
            cls_pos_loss /= self.loss_norm["pos_cls_weight"]
            cls_neg_loss /= self.loss_norm["neg_cls_weight"]

            if self.use_direction_classifier:  # True
                dir_targets = get_direction_target(example["anchors"][task_id][supervision_mask], reg_targets, dir_offset=self.direction_offset, )  # [8, 70400, 2]
                dir_logits = preds_dict["dir_cls_preds"][supervision_mask].view(batch_size_device, -1, 2)  # [8, 70400, 2], WeightedSoftmaxClassificationLoss.
                weights = (labels > 0).type_as(dir_logits)  # [8, 70400], only for positive anchors.
                weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)  # [8, 70400], averaged in sample.
                dir_loss = self.loss_aux(dir_logits, dir_targets, weights=weights)  # [8, 70400].
                dir_loss = self.loss_aux._loss_weight * dir_loss.sum() / batch_size_device  # averaged in batch.


            # for analysis.
            loc_loss_elem = [loc_loss[:, :, i].sum() / batch_size_device for i in range(loc_loss.shape[-1])]

            # for iou prediction
            iou_preds = preds_dict["iou_preds"][supervision_mask]
            pos_pred_mask = reg_weights > 0
            iou_pos_preds = iou_preds.view(batch_size, -1, 1)[pos_pred_mask]
            qboxes = self.box_coder.decode_torch(box_preds[pos_pred_mask], example["anchors"][0][supervision_mask][pos_pred_mask])
            gboxes = self.box_coder.decode_torch(reg_targets[pos_pred_mask], example["anchors"][0][supervision_mask][pos_pred_mask])
            iou_weights = reg_weights[pos_pred_mask]
            iou_pos_targets = iou3d_utils.boxes_aligned_iou3d_gpu(qboxes, gboxes).detach()
            iou_pos_targets = 2 * iou_pos_targets - 1
            iou_pred_loss = self.loss_iou_pred(iou_pos_preds, iou_pos_targets, iou_weights)
            iou_pred_loss = iou_pred_loss.sum() / batch_size

            # for iou3d loss
            pos_pred_mask = reg_weights > 0
            if pos_pred_mask.sum() > 0:
                qboxes = self.box_coder.decode_torch(preds_dict["box_preds"][supervision_mask].view(batch_size, -1, 7)[pos_pred_mask], \
                                                     example["anchors"][0][supervision_mask][pos_pred_mask])
                gboxes = self.box_coder.decode_torch(example["reg_targets"][0][pos_pred_mask], example["anchors"][0][supervision_mask][pos_pred_mask])
                weights = reg_weights[pos_pred_mask]
                ious_loss = self.odiou_3d_loss(gboxes, qboxes, weights, batch_size)

            # loc_loss_reduced / ious_loss
            loss = cls_loss_reduced + ious_loss + dir_loss + iou_pred_loss

            ret = {
                "loss": loss,
                "cls_loss_reduced": cls_loss_reduced.detach().cpu().mean(),
                "loc_loss_reduced": loc_loss_reduced.detach().cpu().mean(),
                "dir_loss_reduced": dir_loss.detach().cpu() if self.use_direction_classifier else None,
                "iou_pred_loss": iou_pred_loss.detach().cpu(),
                "consistency_loss": consistency_loss,
                "loc_loss_elem": [elem.detach().cpu() for elem in loc_loss_elem],
                "cls_pos_loss": cls_pos_loss.detach().cpu(),
                "cls_neg_loss": cls_neg_loss.detach().cpu(),
                "ious_loss": ious_loss.detach().cpu(),
                "num_pos": (labels > 0)[0].sum(),
                "num_neg": (labels == 0)[0].sum(),
            }

            rets.append(ret)

        """convert batch-key to key-batch"""
        rets_merged = defaultdict(list)
        for ret in rets:
            for k, v in ret.items():
                rets_merged[k].append(v)

        for k, v in loss_ema.items():
            rets_merged[k].append(v[0])

        return rets_merged

    def get_model_ema_loss(self, example, preds_dicts):
        supervision_mask = example["ssl_labeled"] == 1 if "ssl_labeled" in example.keys() else torch.ones(len(example['metadata'])) == 1
        batch_anchors = example["anchors"][0][supervision_mask]
        batch_size_device = batch_anchors.shape[0]

        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            box_preds = preds_dict["box_preds"][supervision_mask]
            cls_preds = preds_dict["cls_preds"][supervision_mask]

            # get targets and weights.
            labels = example["labels_raw"][task_id]
            reg_targets = example["reg_targets_raw"][task_id]
            cls_weights, reg_weights, cared = self.prepare_loss_weights(labels, loss_norm=self.loss_norm, dtype=torch.float32, )
            cls_targets = labels * cared.type_as(labels)
            cls_targets = cls_targets.unsqueeze(-1)

            # get localization and classification loss.
            batch_size = int(box_preds.shape[0])
            box_preds = box_preds.view(batch_size, -1, self.box_n_dim)
            cls_preds = cls_preds.view(batch_size, -1, self.num_classes[task_id])

            if self.encode_rad_error_by_sin:
                encoded_box_preds, encoded_reg_targets = add_sin_difference(box_preds, reg_targets)

            loc_loss = self.loss_reg(encoded_box_preds, encoded_reg_targets, weights=reg_weights)
            cls_loss = self.loss_cls(cls_preds, cls_targets, weights=cls_weights)
            loc_loss_reduced = self.loss_reg._loss_weight * loc_loss.sum() / batch_size_device
            cls_loss_reduced = self.loss_cls._loss_weight * cls_loss.sum() / batch_size_device

            cls_pos_loss, cls_neg_loss = _get_pos_neg_loss(cls_loss, labels)
            cls_pos_loss /= self.loss_norm["pos_cls_weight"]
            cls_neg_loss /= self.loss_norm["neg_cls_weight"]

            if self.use_direction_classifier:  # True
                dir_targets = get_direction_target(example["anchors_raw"][task_id][supervision_mask], reg_targets, dir_offset=self.direction_offset, )
                dir_logits = preds_dict["dir_cls_preds"][supervision_mask].view(batch_size_device, -1, 2)
                weights = (labels > 0).type_as(dir_logits)
                weights /= torch.clamp(weights.sum(-1, keepdim=True), min=1.0)
                dir_loss = self.loss_aux(dir_logits, dir_targets, weights=weights)
                dir_loss = self.loss_aux._loss_weight * dir_loss.sum() / batch_size_device

            # for analysis.
            loc_loss_elem = [loc_loss[:, :, i].sum() / batch_size_device for i in range(loc_loss.shape[-1])]

            # for iou prediction
            iou_preds = preds_dict["iou_preds"][supervision_mask]
            pos_pred_mask = reg_weights > 0
            iou_pos_preds = iou_preds.view(batch_size, -1, 1)[pos_pred_mask]
            qboxes = self.box_coder.decode_torch(box_preds[pos_pred_mask], example["anchors_raw"][0][supervision_mask][pos_pred_mask])
            gboxes = self.box_coder.decode_torch(reg_targets[pos_pred_mask], example["anchors_raw"][0][supervision_mask][pos_pred_mask])
            iou_weights = reg_weights[pos_pred_mask]
            iou_pos_targets = iou3d_utils.boxes_aligned_iou3d_gpu(qboxes, gboxes).detach()
            iou_pos_targets = 2 * iou_pos_targets - 1
            iou_pred_loss = self.loss_iou_pred(iou_pos_preds, iou_pos_targets, iou_weights)
            iou_pred_loss = iou_pred_loss.sum() / batch_size

            loss = cls_loss_reduced + dir_loss + iou_pred_loss


            ret = {
                "loss_ema": loss.detach().cpu(),
                "cls_loss_reduced_ema": cls_loss_reduced.detach().cpu().mean(),
                "loc_loss_reduced_ema": loc_loss_reduced.detach().cpu().mean(),
                "dir_loss_reduced_ema": dir_loss.detach().cpu() if self.use_direction_classifier else None,
                "iou_pred_loss_ema": iou_pred_loss.detach().cpu(),
                "loc_loss_elem_ema": [elem.detach().cpu() for elem in loc_loss_elem],
                "cls_pos_loss_ema": cls_pos_loss.detach().cpu(),
                "cls_neg_loss_ema": cls_neg_loss.detach().cpu(),
                "num_pos_ema": (labels > 0)[0].sum(),
                "num_neg_ema": (labels == 0)[0].sum(),
            }

            rets.append(ret)

        """convert batch-key to key-batch"""
        rets_merged = defaultdict(list)
        for ret in rets:
            for k, v in ret.items():
                rets_merged[k].append(v)
        return rets_merged


    def predict(self, example, preds_dicts, test_cfg, **kwargs):
        batch_valid_frustum = example['calib']['frustum']  # [batch_size, 1, 6, 4, 3]
        batch_anchors = example["anchors"]

        rets = []
        for task_id, preds_dict in enumerate(preds_dicts):
            meta_list = example["metadata"]  # length: 8
            num_class_with_bg = self.num_classes[task_id]  # 1
            batch_size = batch_anchors[task_id].shape[0]
            batch_task_anchors = example["anchors"][task_id].view(batch_size, -1, self.box_n_dim)  # [8, 70400, 7]
            batch_anchors_mask = [None] * batch_size
            batch_cls_preds = preds_dict["cls_preds"].view(batch_size, -1, num_class_with_bg)  # [8, 70400, 1]
            batch_box_preds = preds_dict["box_preds"].view(batch_size, -1, self.box_n_dim)  # [batch_size, 70400, 7]
            batch_iou_preds = preds_dict["iou_preds"].view(batch_size, -1, 1)

            # pred_box(abs_pos):[8, 70400, 7] <- pred_offset: [8, 70400, 7], anchors: [8, 70400, 7]
            batch_reg_preds = self.box_coder.decode_torch(batch_box_preds[:, :, : self.box_coder.code_size], batch_task_anchors)
            batch_dir_preds = preds_dict["dir_cls_preds"].view(batch_size, -1, 2)  # [8, 200, 176, 4]) -> [8, 70400, 2]

            rets.append(self.get_task_detections(task_id,
                                                 num_class_with_bg,
                                                 test_cfg,
                                                 batch_cls_preds,
                                                 batch_reg_preds,
                                                 batch_dir_preds,
                                                 batch_iou_preds,
                                                 batch_anchors,
                                                 batch_anchors_mask,
                                                 meta_list,
                                                 batch_valid_frustum))
        num_samples = len(rets[0])

        ret_list = []
        for i in range(num_samples):
            ret = {}
            for k in rets[0][i].keys():
                if k in ["box3d_lidar", "scores"]:
                    ret[k] = torch.cat([ret[i][k] for ret in rets])

                elif k in ["label_preds"]:
                    flag = 0
                    for j, num_class in enumerate(self.num_classes):
                        rets[j][i][k] += flag
                        flag += num_class
                    ret[k] = torch.cat([ret[i][k] for ret in rets])

                elif k == "metadata":
                    ret[k] = rets[0][i][k]

            ret_list.append(ret)
        return ret_list

    def get_task_detections(self, task_id, num_class_with_bg, test_cfg, batch_cls_preds, batch_reg_preds,
                            batch_dir_preds=None,
                            batch_iou_preds=None, batch_anchors=None, batch_anchors_mask=None, meta_list=None,
                            batch_valid_frustum=None):
        predictions_dicts = []
        post_center_range = self.post_center_range
        anchors = batch_anchors[0][0]

        for box_preds, cls_preds, dir_preds, iou_preds, a_mask, meta, valid_frustum in zip(batch_reg_preds,  batch_cls_preds, batch_dir_preds,
                                                                                          batch_iou_preds, batch_anchors_mask, meta_list, batch_valid_frustum):
            # get dir labels
            if self.use_direction_classifier:
                dir_labels = torch.max(dir_preds, dim=-1)[1]  # [70400]

            # get scores from cls_preds
            total_scores = torch.sigmoid(cls_preds)  # [70400, 1]
            top_scores = total_scores.squeeze(-1)  # [70400]
            top_labels = self.top_labels

            # SCORE_THRESHOLD: REMOVE those boxes lower than 0.3.
            if test_cfg.score_threshold > 0.0:
                thresh = self.thresh
                top_scores_keep = top_scores >= thresh
                top_scores = top_scores.masked_select(top_scores_keep)

                # Still add IoU rectification in CIA-SSD to SE-SSD due to its minor positive effect.
                iou_preds = (iou_preds.squeeze() + 1) * 0.5
                top_scores *= torch.pow(iou_preds.masked_select(top_scores_keep), 4)

            # NMS: obtain remained box_preds & dir_labels & cls_labels after score threshold.
            if top_scores.shape[0] != 0:
                if test_cfg.score_threshold > 0.0:
                    box_preds = box_preds[top_scores_keep]
                    if self.use_direction_classifier:
                        dir_labels = dir_labels[top_scores_keep]
                    top_labels = top_labels[top_scores_keep]

                boxes_for_nms = box_preds[:, [0, 1, 3, 4, -1]]

                # REMOVE overlap boxes by bev rotate-nms.
                nms_type = "rotate_nms"
                if nms_type == "rotate_nms":      # DEFAULT NMS
                    nms_func = box_torch_ops.rotate_nms
                    selected = nms_func(boxes_for_nms,
                                        top_scores,
                                        pre_max_size=test_cfg.nms.nms_pre_max_size,
                                        post_max_size=test_cfg.nms.nms_post_max_size,
                                        iou_threshold=test_cfg.nms.nms_iou_threshold, )
                    box_preds = box_preds[selected]
                    if self.use_direction_classifier:
                        dir_labels = dir_labels[selected]
                    label_preds = top_labels[selected]
                    scores = top_scores[selected]

                # Still add DI-NMS in CIA-SSD to SE-SSD due to its minor positive effect.
                elif nms_type == 'rotate_weighted_nms':  # DI-NMS
                    nms_func = box_torch_ops.rotate_weighted_nms
                    box_preds, dir_labels, label_preds, scores, selected = nms_func(box_preds,
                                                                          boxes_for_nms,
                                                                          dir_labels,
                                                                          top_labels,
                                                                          top_scores,
                                                                          iou_preds[top_scores_keep],
                                                                          anchors[top_scores_keep],
                                                                          pre_max_size=test_cfg.nms.nms_pre_max_size,
                                                                          post_max_size=test_cfg.nms.nms_post_max_size,
                                                                          iou_threshold=test_cfg.nms.nms_iou_threshold,
                                                                          enable_centerness=True,
                                                                          centerness_pow=2,
                                                                          nms_cnt_thresh=2.6,  # 2.6
                                                                          nms_sigma_dist_interval=(0, 20, 40, 60),
                                                                          nms_sigma_square=(0.0009, 0.009, 0.1, 1),
                                                                          suppressed_thresh=0.3,
                                                                          )
                else:
                    raise NotImplementedError
            else:
                box_preds = torch.zeros([0, 7], dtype=float)

            if box_preds.shape[0] > 0:
                from det3d.core.bbox.geometry import points_in_convex_polygon_3d_jit
                indices = points_in_convex_polygon_3d_jit(box_preds[:, :3].cpu().numpy(), valid_frustum.cpu().numpy())
                box_preds = box_preds[indices.reshape([-1])]
                dir_labels = dir_labels[indices.reshape([-1])]
                label_preds = label_preds[indices.reshape([-1])]
                scores = scores[indices.reshape([-1])]

            # POST-PROCESSING of predictions.
            if box_preds.shape[0] != 0:
                # move pred boxes direction by pi, eg. pred_ry < 0 while pred_dir_label > 0.
                if self.use_direction_classifier:
                    opp_labels = ((box_preds[..., -1] - self.direction_offset) > 0) ^ (dir_labels.byte() == 1)
                    box_preds[..., -1] += torch.where(opp_labels, torch.tensor(np.pi).type_as(box_preds), torch.tensor(0.0).type_as(box_preds), )  # useful for dir accuracy, but has no impact on localization

                # remove pred boxes out of POST_VALID_RANGE
                mask = (box_preds[:, :3] >= post_center_range[:3]).all(1)
                mask &= (box_preds[:, :3] <= post_center_range[3:]).all(1)
                predictions_dict = {"box3d_lidar": box_preds[mask],
                                    "scores": scores[mask],
                                    "label_preds": label_preds[mask],
                                    "metadata": meta, }
            else:
                dtype = batch_reg_preds.dtype
                device = batch_reg_preds.device
                predictions_dict = {
                    "box3d_lidar": torch.zeros([0, self.box_n_dim], dtype=dtype, device=device),
                    "scores": torch.zeros([0], dtype=dtype, device=device),
                    "label_preds": torch.zeros([0], dtype=top_labels.dtype, device=device),
                    "metadata": meta,
                }
            predictions_dicts.append(predictions_dict)

        return predictions_dicts
