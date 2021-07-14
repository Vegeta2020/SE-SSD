# import time
# import numpy as np
# import math
# import matplotlib.pyplot as plt

# import torch

# from torch import nn
# from torch.nn import functional as F
# from torchvision.models import resnet
# from torch.nn.modules.batchnorm import _BatchNorm

# from det3d.torchie.cnn import constant_init, kaiming_init, xavier_init
# from det3d.torchie.trainer import load_checkpoint
# from det3d.models.utils import Empty, GroupNorm, Sequential
# from det3d.models.utils import change_default_args

# from .. import builder
# from ..registry import NECKS
# from ..utils import build_norm_layer


# # v1.0: feature-fusion rpn
# @NECKS.register_module
# class FF_RPN(nn.Module):
#     def __init__(self, layer_nums, ds_layer_strides, ds_num_filters, us_layer_strides, us_num_filters,
#                  num_input_features, norm_cfg=None, name="rpn", logger=None, **kwargs):
#         super(FF_RPN, self).__init__()
#         self._layer_strides = ds_layer_strides  # [1,]
#         self._num_filters = ds_num_filters  # [128,]
#         self._layer_nums = layer_nums  # [5,]
#         self._upsample_strides = us_layer_strides  # [1,]
#         self._num_upsample_filters = us_num_filters  # [128,]
#         self._num_input_features = num_input_features  # 128
#         self.reduction_rate = 4

#         if norm_cfg is None:  # True
#             norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
#         self._norm_cfg = norm_cfg

#         self.bottom_up_block_0 = Sequential(
#             nn.ZeroPad2d(1),
#             nn.Conv2d(128, 128, 3, stride=1, bias=False),
#             build_norm_layer(self._norm_cfg, 128)[1],
#             nn.ReLU(),

#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
#             build_norm_layer(self._norm_cfg, 128, )[1],
#             nn.ReLU(),

#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
#             build_norm_layer(self._norm_cfg, 128, )[1],
#             nn.ReLU(),
#         )

#         self.bottom_up_block_1 = Sequential(
#             # [200, 176] -> [100, 88]
#             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False, ),
#             build_norm_layer(self._norm_cfg, 256, )[1],
#             nn.ReLU(),

#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False, ),
#             build_norm_layer(self._norm_cfg, 256, )[1],
#             nn.ReLU(),

#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False, ),
#             build_norm_layer(self._norm_cfg, 256, )[1],
#             nn.ReLU(),

#         )

#         self.trans_0 = Sequential(
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False, ),
#             build_norm_layer(self._norm_cfg, 128, )[1],
#             nn.ReLU(),
#         )

#         self.trans_1 = Sequential(
#             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False, ),
#             build_norm_layer(self._norm_cfg, 256, )[1],
#             nn.ReLU(),
#         )

#         self.deonv_block_0 = Sequential(
#             nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False, ),
#             build_norm_layer(self._norm_cfg, 128, )[1],
#             nn.ReLU(),
#         )

#         self.deonv_block_1 = Sequential(
#             nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False, ),
#             build_norm_layer(self._norm_cfg, 128, )[1],
#             nn.ReLU(),
#         )

#         self.conv_0 = Sequential(
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
#             build_norm_layer(self._norm_cfg, 128, )[1],
#             nn.ReLU(),
#         )

#         self.conv_1 = Sequential(
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
#             build_norm_layer(self._norm_cfg, 128, )[1],
#             nn.ReLU(),
#         )

#         # v1.0.4&7
#         # self.trans_2 = Sequential(
#         #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False, ),
#         #     build_norm_layer(self._norm_cfg, 128, )[1],
#         #     nn.ReLU(),
#         # )
#         # self.deonv_block_2 = Sequential(
#         #     nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, output_padding=0, bias=False, ),
#         #     build_norm_layer(self._norm_cfg, 128, )[1],
#         #     nn.ReLU(),
#         # )
#         # self.conv_2 = Sequential(
#         #     nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
#         #     build_norm_layer(self._norm_cfg, 128, )[1],
#         #     nn.ReLU(),
#         # )


#         # v1.0.6
#         # self.ch_avg_att_0 = Sequential(
#         # 	nn.AdaptiveAvgPool2d(1),
#         # 	nn.Flatten(),
#         #     nn.Linear(128, int(128/self.reduction_rate), bias=True),
#         #     nn.ReLU(),
#         #     nn.Linear(int(128/self.reduction_rate), 128, bias=True),
#         #     nn.Sigmoid(),
#         # )
#         # self.ch_avg_att_1 = Sequential(
#         # 	nn.AdaptiveAvgPool2d(1),
#         # 	nn.Flatten(),
#         #     nn.Linear(128, int(128/self.reduction_rate), bias=True),
#         #     nn.ReLU(),
#         #     nn.Linear(int(128/self.reduction_rate), 128, bias=True),
#         #     nn.Sigmoid(),
#         # )

#         logger.info("Finish RPN Initialization")

#     # default init_weights for conv(msra) and norm in ConvModule
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 xavier_init(m, distribution="uniform")

#     def forward(self, x):
#         # FF_RPN: v1.0
#         # v1.0.1: remove self.conv_1 based on v1.0. x_output_1 = x_middle_1.
#         # v1.0.2: remove x_output_1 in x_output. based on v1.0.
#         # v1.0.3: add input x as low-level features in fpn.

#         # v1.0.4: based on v1.0.3, remove x_output_1&2.
#         # v1.0.5: add ch att to after lateral conv1x1. 
#         # v1.0.6: based on v1.0, add ch att on x_output_0/1.
#         # v1.0.7: based on 1.0.4, add ch att on x_output.
#         # v1.0.8: based on 1.0.0, add ch att on x_output.

#         B, C, H, W = x.shape

#         # v1.0
#         x_0 = self.bottom_up_block_0(x)
#         x_1 = self.bottom_up_block_1(x_0)
#         x_trans_0 = self.trans_0(x_0)
#         x_trans_1 = self.trans_1(x_1)
#         x_middle_0 = self.deonv_block_0(x_trans_1) + x_trans_0
#         x_middle_1 = self.deonv_block_1(x_trans_1)
#         x_output_0 = self.conv_0(x_middle_0)
#         x_output_1 = self.conv_1(x_middle_1)
#         x_output = x_output_0 + x_output_1


#         # v1.0.4
#         # x_0 = self.bottom_up_block_0(x)
#         # x_1 = self.bottom_up_block_1(x_0)
#         # x_trans_1 = self.trans_1(x_1)
#         # x_middle_0 = self.deonv_block_0(x_trans_1) + self.trans_0(x_0)
#         # x_middle_2 = self.deonv_block_2(x_middle_0) + self.trans_2(x)
#         # x_output_2 = self.conv_2(x_middle_2)
#         # x_output = x_output_2

#         # v1.0.5
#         # x_0 = self.bottom_up_block_0(x)
#         # x_1 = self.bottom_up_block_1(x_0)
#         # x_trans_0 = self.trans_0(x_0)
#         # x_trans_1 = self.trans_1(x_1)

#         # ch_avg_0 = F.adaptive_avg_pool2d(x_trans_0, 1).squeeze()
#         # ch_weight_0 = self.ch_att_0(ch_avg_0).reshape(B, C, 1, 1)
#         # ch_avg_1 = F.adaptive_avg_pool2d(x_trans_1, 1).squeeze()
#         # ch_weight_1 = self.ch_att_1(ch_avg_1).reshape(B, 2*C, 1, 1)

#         # x_trans_0 = x_trans_0 * ch_weight_0
#         # x_trans_1 = x_trans_1 * ch_weight_1
        
#         # x_middle_0 = self.deonv_block_0(x_trans_1) + x_trans_0
#         # x_middle_1 = self.deonv_block_1(x_trans_1)
#         # x_output_0 = self.conv_0(x_middle_0)
#         # x_output_1 = self.conv_1(x_middle_1)
#         # x_output = x_output_0 + x_output_1

#         # v1.0.6
#         # x_0 = self.bottom_up_block_0(x)
#         # x_1 = self.bottom_up_block_1(x_0)
#         # x_trans_0 = self.trans_0(x_0)
#         # x_trans_1 = self.trans_1(x_1)
#         # x_middle_0 = self.deonv_block_0(x_trans_1) + x_trans_0
#         # x_middle_1 = self.deonv_block_1(x_trans_1)
#         # x_output_0 = self.conv_0(x_middle_0)
#         # x_output_1 = self.conv_1(x_middle_1)
#         # x_output_0 = x_output_0 * self.ch_avg_att_0(x_output_0).view(B, C, 1, 1)
#         # x_output_1 = x_output_1 * self.ch_avg_att_1(x_output_1).view(B, C, 1, 1)
#         # x_output = x_output_0 + x_output_1 

#         # v1.0.7
#         # B, C, H, W = x.shape
#         # x_0 = self.bottom_up_block_0(x)
#         # x_1 = self.bottom_up_block_1(x_0)
#         # x_trans_1 = self.trans_1(x_1)
#         # x_middle_0 = self.deonv_block_0(x_trans_1) + self.trans_0(x_0)
#         # x_middle_2 = self.deonv_block_2(x_middle_0) + self.trans_2(x)
#         # x_output_2 = self.conv_2(x_middle_2)
#         # ch_avg = F.adaptive_avg_pool2d(x_output_2, 1).squeeze()
#         # ch_weight = self.ch_att(ch_avg).reshape(B, C, 1, 1)
#         # x_output = x_output_2 * ch_weight

#         # v1.0.8
#         # x_0 = self.bottom_up_block_0(x)
#         # x_1 = self.bottom_up_block_1(x_0)
#         # x_trans_0 = self.trans_0(x_0)
#         # x_trans_1 = self.trans_1(x_1)
#         # x_middle_0 = self.deonv_block_0(x_trans_1) + x_trans_0
#         # x_middle_1 = self.deonv_block_1(x_trans_1)
#         # x_output_0 = self.conv_0(x_middle_0)
#         # x_output_1 = self.conv_1(x_middle_1)
#         # x_output = x_output_0 + x_output_1
#         # ch_avg = F.adaptive_avg_pool2d(x_output, 1).squeeze()
#         # ch_weight = self.ch_att(ch_avg).reshape(B, C, 1, 1)
#         # x_output = x_output * ch_weight


#         return x_output


# # v1.0: feature-fusion rpn
# # @NECKS.register_module
# # class AFF_RPN(nn.Module):
# #     def __init__(self, layer_nums, ds_layer_strides, ds_num_filters, us_layer_strides, us_num_filters,
# #                  num_input_features, norm_cfg=None, name="rpn", logger=None, **kwargs):
# #         super(AFF_RPN, self).__init__()
# #         self._layer_strides = ds_layer_strides  # [1,]
# #         self._num_filters = ds_num_filters  # [128,]
# #         self._layer_nums = layer_nums  # [5,]
# #         self._upsample_strides = us_layer_strides  # [1,]
# #         self._num_upsample_filters = us_num_filters  # [128,]
# #         self._num_input_features = num_input_features  # 128

# #         self.reduction_rate = 4

# #         if norm_cfg is None:  # True
# #             norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
# #         self._norm_cfg = norm_cfg

# #         self.bottom_up_block_0_0 = Sequential(
# #             nn.ZeroPad2d(1),
# #             nn.Conv2d(128, 128, 3, stride=1, bias=False),
# #             build_norm_layer(self._norm_cfg, 128)[1],
# #             nn.ReLU(),
# #         )

# #         self.bottom_up_block_0_1 = Sequential(
# #             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
# #             build_norm_layer(self._norm_cfg, 128, )[1],
# #             nn.ReLU(),
# #         )

# #         self.bottom_up_block_0_2 = Sequential(
# #             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
# #             build_norm_layer(self._norm_cfg, 128, )[1],
# #             nn.ReLU(),
# #         )

# #         self.bottom_up_block_1_0 = Sequential(
# #             # [200, 176] -> [100, 88]
# #             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False, ),
# #             build_norm_layer(self._norm_cfg, 256, )[1],
# #             nn.ReLU(),
# #         )

# #         self.bottom_up_block_1_1 = Sequential(
# #             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False, ),
# #             build_norm_layer(self._norm_cfg, 256, )[1],
# #             nn.ReLU(),
# #         )
# #         self.bottom_up_block_1_2 = Sequential(
# #             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False, ),
# #             build_norm_layer(self._norm_cfg, 256, )[1],
# #             nn.ReLU(),

# #         )

# #         self.trans_0 = Sequential(
# #             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False, ),
# #             build_norm_layer(self._norm_cfg, 128, )[1],
# #             nn.ReLU(),
# #         )

# #         self.trans_1 = Sequential(
# #             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False, ),
# #             build_norm_layer(self._norm_cfg, 256, )[1],
# #             nn.ReLU(),
# #         )

# #         self.deonv_block_0 = Sequential(
# #             nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False, ),
# #             build_norm_layer(self._norm_cfg, 128, )[1],
# #             nn.ReLU(),
# #         )

# #         self.deonv_block_1 = Sequential(
# #             nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False, ),
# #             build_norm_layer(self._norm_cfg, 128, )[1],
# #             nn.ReLU(),
# #         )

# #         self.conv_0 = Sequential(
# #             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
# #             build_norm_layer(self._norm_cfg, 128, )[1],
# #             nn.ReLU(),
# #         )

# #         self.conv_1 = Sequential(
# #             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
# #             build_norm_layer(self._norm_cfg, 128, )[1],
# #             nn.ReLU(),
# #         )


# #         # v1.0.1&3
# #         self.ch_avg_att_0 = Sequential(
# #         	nn.AdaptiveAvgPool2d(1),
# #         	nn.Flatten(),
# #             nn.Linear(128, int(128/self.reduction_rate), bias=True),
# #             nn.ReLU(),
# #             nn.Linear(int(128/self.reduction_rate), 128, bias=True),
# #             nn.Sigmoid(),
# #         )
# #         self.ch_avg_att_1 = Sequential(
# #         	nn.AdaptiveAvgPool2d(1),
# #         	nn.Flatten(),
# #             nn.Linear(256, int(256/self.reduction_rate), bias=True),
# #             nn.ReLU(),
# #             nn.Linear(int(256/self.reduction_rate), 256, bias=True),
# #             nn.Sigmoid(),
# #         )

# #         # V1.0.2&3
# #         self.ch_avg_att_0_0 = Sequential(
# #         	nn.AdaptiveAvgPool2d(1),
# #         	nn.Flatten(),
# #             nn.Linear(128, int(128/self.reduction_rate), bias=True),
# #             nn.ReLU(),
# #             nn.Linear(int(128/self.reduction_rate), 128, bias=True),
# #             nn.Sigmoid(),
# #         )
# #         self.ch_avg_att_0_1 = Sequential(
# #         	nn.AdaptiveAvgPool2d(1),
# #         	nn.Flatten(),
# #             nn.Linear(128, int(128/self.reduction_rate), bias=True),
# #             nn.ReLU(),
# #             nn.Linear(int(128/self.reduction_rate), 128, bias=True),
# #             nn.Sigmoid(),
# #         )
# #         self.ch_avg_att_1_0 = Sequential(
# #         	nn.AdaptiveAvgPool2d(1),
# #         	nn.Flatten(),
# #             nn.Linear(256, int(256/self.reduction_rate), bias=True),
# #             nn.ReLU(),
# #             nn.Linear(int(256/self.reduction_rate), 256, bias=True),
# #             nn.Sigmoid(),
# #         )
# #         self.ch_avg_att_1_1 = Sequential(
# #         	nn.AdaptiveAvgPool2d(1),
# #         	nn.Flatten(),
# #             nn.Linear(256, int(256/self.reduction_rate), bias=True),
# #             nn.ReLU(),
# #             nn.Linear(int(256/self.reduction_rate), 256, bias=True),
# #             nn.Sigmoid(),
# #         )



# #         logger.info("Finish RPN Initialization")

# #     # default init_weights for conv(msra) and norm in ConvModule
# #     def init_weights(self):
# #         for m in self.modules():
# #             if isinstance(m, nn.Conv2d):
# #                 xavier_init(m, distribution="uniform")

# #     def forward(self, x):
# #         # AFF_RPN: v1.0
# #         # v1.0.1: ch att after block 0&1
# #         # v1.0.2: ch att in block 0&1
# #         # v1.0.3: ch att in and after block 0&1

# #         # try nn.LeakyReLU() later
        

# #         B, C, H, W = x.shape

# #         # v1.0.1
# #         # x = self.bottom_up_block_0_0(x)
# #         # x = self.bottom_up_block_0_1(x)
# #         # x_0 = self.bottom_up_block_0_2(x)
# #         # ch_weight_0 = self.ch_avg_att_0(x_0).view(B, C, 1, 1)
# #         # x_0 = x_0 * ch_weight_0

# #         # x = self.bottom_up_block_1_0(x_0)
# #         # x = self.bottom_up_block_1_1(x)
# #         # x_1 = self.bottom_up_block_1_2(x)
# #         # ch_weight_1 = self.ch_avg_att_1(x_1).view(B, 2*C, 1, 1)
# #         # x_1 = x_1 * ch_weight_1

# #         # x_trans_0 = self.trans_0(x_0)
# #         # x_trans_1 = self.trans_1(x_1)

# #         # x_middle_0 = self.deonv_block_0(x_trans_1) + x_trans_0
# #         # x_middle_1 = self.deonv_block_1(x_trans_1)

# #         # x_output_0 = self.conv_0(x_middle_0)
# #         # x_output_1 = self.conv_1(x_middle_1)
# #         # x_output = x_output_0 + x_output_1



# #         # v1.0.2
# #         x = self.bottom_up_block_0_0(x)
# #         x = x * self.ch_avg_att_0_0(x).view(B, C, 1, 1)
# #         x = self.bottom_up_block_0_1(x)
# #         x = x * self.ch_avg_att_0_1(x).view(B, C, 1, 1)
# #         x_0 = self.bottom_up_block_0_2(x)
# #         x_0 = x_0 * self.ch_avg_att_0(x_0).view(B, C, 1, 1)

# #         x = self.bottom_up_block_1_0(x_0)
# #         x = x * self.ch_avg_att_1_0(x).view(B, 2*C, 1, 1)
# #         x = self.bottom_up_block_1_1(x)
# #         x = x * self.ch_avg_att_1_1(x).view(B, 2*C, 1, 1)
# #         x_1 = self.bottom_up_block_1_2(x)
# #         x_1 = x_1 * self.ch_avg_att_1(x_1).view(B, 2*C, 1, 1)

# #         x_trans_0 = self.trans_0(x_0)
# #         x_trans_1 = self.trans_1(x_1)

# #         x_middle_0 = self.deonv_block_0(x_trans_1) + x_trans_0
# #         x_middle_1 = self.deonv_block_1(x_trans_1)

# #         x_output_0 = self.conv_0(x_middle_0)
# #         x_output_1 = self.conv_1(x_middle_1)
# #         x_output = x_output_0 + x_output_1


       
# #         # v1.0.3
# #         # x = self.bottom_up_block_0_0(x)
# #         # x = x * self.ch_avg_att_0_0(x).view(B, C, 1, 1)
# #         # x = self.bottom_up_block_0_1(x)
# #         # x = x * self.ch_avg_att_0_1(x).view(B, C, 1, 1)
# #         # x_0 = self.bottom_up_block_0_2(x)
# #         # x_0 = x_0 * self.ch_avg_att_0(x_0).view(B, C, 1, 1)

# #         # x = self.bottom_up_block_1_0(x_0)
# #         # x = x * self.ch_avg_att_1_0(x).view(B, 2*C, 1, 1)
# #         # x = self.bottom_up_block_1_1(x)
# #         # x = x * self.ch_avg_att_1_1(x).view(B, 2*C, 1, 1)
# #         # x_1 = self.bottom_up_block_1_2(x)
# #         # x_1 = x_1 * self.ch_avg_att_1(x_1).view(B, 2*C, 1, 1)

# #         # x_trans_0 = self.trans_0(x_0)
# #         # x_trans_1 = self.trans_1(x_1)

# #         # x_middle_0 = self.deonv_block_0(x_trans_1) + x_trans_0
# #         # x_middle_1 = self.deonv_block_1(x_trans_1)

# #         # x_output_0 = self.conv_0(x_middle_0)
# #         # x_output_1 = self.conv_1(x_middle_1)
# #         # x_output = x_output_0 + x_output_1


# #         return x_output

