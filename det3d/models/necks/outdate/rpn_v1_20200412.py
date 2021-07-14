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

# # exp_rpn_v2.0_0/1/2:   # - / 79.08 / 78.77
# # exp_rpn_v2.2_0/1/2:   # 79.14/ dead / 78.92
# # exp_rpn_v1.0_0/1:     # 79.0x

# # # v2.1 add conv after deconv
# # # v2.2 add conv before deconv

# # exp_rpn_v1.6_0: 4 5 6 7
# # exp_rpn_v1.7_0/1: 0&1, 2&3

# # exp_rpn_v1.0.2_0: 0&1, remove fpn, directly concat 2-level features with same weights.        79.02
# # exp_rpn_v1.0.3_0: 2&3, remove fpn, directly concat 3-level features with different weights.   79.07

# # exp_ds_rpn_v1.0_0: # 78.71
# # exp_ds_rpn_v1.1_0: # 78.28

# # exp_ds_rpn_v1.2_0/1：79.01 / 78.95
# # exp_rpn_v1.3.3_3_0/1: 4, 5
# # exp_rpn_v1.3.3_4_0/1: 6, 7 


# @NECKS.register_module
# class RPN_v1_5(nn.Module):
#     def __init__(self, layer_nums, ds_layer_strides, ds_num_filters, us_layer_strides, us_num_filters, num_input_features, norm_cfg=None, name="rpn", logger=None, **kwargs):
#         super(RPN_v1_5, self).__init__()
#         self._layer_strides = ds_layer_strides         # [1,]
#         self._num_filters = ds_num_filters             # [128,]
#         self._layer_nums = layer_nums                  # [5,]
#         self._upsample_strides = us_layer_strides      # [1,]
#         self._num_upsample_filters = us_num_filters    # [128,]
#         self._num_input_features = num_input_features  # 128

#         if norm_cfg is None: # True
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

#         self.deconv_block_0 = Sequential(
#             nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False, ),
#             build_norm_layer(self._norm_cfg, 128, )[1],
#             nn.ReLU(),
#         )

#         self.deconv_block_1 = Sequential(
#             nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False, ),
#             build_norm_layer(self._norm_cfg, 128, )[1],
#             nn.ReLU(),
#         )

#         self.trans_0 = Sequential(
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False, ),
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

#         self.fc_0 = Sequential(
#             nn.Linear(128, 64, bias=True),
#             nn.ReLU(),
#             nn.Linear(64, 128, bias=True),
#         )

#         self.fc_1 = Sequential(
#             nn.Linear(256, 128, bias=True),
#             nn.ReLU(),
#             nn.Linear(128, 256, bias=True),
#         )


#         # only for v1.7:
#         self.sp_att_conv_0 = Sequential(
#         	nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 9), stride=1, padding=(0, 4), bias=False, ),
#         	build_norm_layer(self._norm_cfg, 64, )[1],
#             nn.ReLU(),

#             nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(9, 1), stride=1, padding=(4, 0), bias=False, ),
#         	build_norm_layer(self._norm_cfg, 1, )[1],
#             nn.ReLU(),

#         )

#         self.sp_att_conv_1 = Sequential(
#         	nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(9, 1), stride=1, padding=(4, 0), bias=False, ),
#         	build_norm_layer(self._norm_cfg, 64, )[1],
#             nn.ReLU(),

#             nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 9), stride=1, padding=(0, 4), bias=False, ),
#         	build_norm_layer(self._norm_cfg, 1, )[1],
#             nn.ReLU(),
#         )



#         logger.info("Finish RPN Initialization")

#     # default init_weights for conv(msra) and norm in ConvModule
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 xavier_init(m, distribution="uniform")

#     def forward(self, x):
#         # v1.5: with sp and ch attention;
#         # v1.6: with only ch attention;
#         # v1.7: perform sp attention with global convolution network.
#         x_0 = self.bottom_up_block_0(x)    # low-level feature, [200, 176, 128]
#         x_1 = self.bottom_up_block_1(x_0)  # high-level feature, [100, 88, 256]

#         B, C, H, W = x_0.shape

#         # only for v1.5
#         # avg_output_0 = F.adaptive_max_pool2d(x_0, 1).squeeze()   # B, C
#         # sp_att_mask = (self.fc_0(avg_output_0).reshape(B, C, 1, 1) * x_0).sum(dim=1)
#         # sp_weight = F.sigmoid(sp_att_mask).reshape(B, 1, H, W)
#         # x_mid_0 = x_0 * sp_weight

#         # only for v1.6
#         # x_mid_0 = self.trans_0(x_0)

#         # only for v1.7
#         sp_weight = F.sigmoid(self.sp_att_conv_0(x_0) + self.sp_att_conv_1(x_0)).reshape(B, 1, H, W)
#         x_mid_0 = sp_weight * x_0


#         avg_output_1 = F.adaptive_avg_pool2d(x_1, 1).squeeze()  # B, C
#         ch_weight = F.sigmoid(self.fc_1(avg_output_1)).reshape(B, C*2, 1, 1)
#         x_mid_1 = x_1 * ch_weight

#         x_output_0 = self.conv_0(self.deconv_block_0(x_mid_1) + x_mid_0)
#         x_output_1 = self.conv_1(self.deconv_block_1(x_mid_1))
#         x_output = 1.0 * x_output_0 + 1.0 * x_output_1

#         return x_output


# # v1.0.2&3: remove fpn, directly concat multi-scale features.
# @NECKS.register_module
# class RPN_v1(nn.Module):
#     def __init__(self, layer_nums, ds_layer_strides, ds_num_filters, us_layer_strides, us_num_filters,
#                  num_input_features, norm_cfg=None, name="rpn", logger=None, **kwargs):
#         super(RPN_v1, self).__init__()
#         self._layer_strides = ds_layer_strides  # [1,]
#         self._num_filters = ds_num_filters  # [128,]
#         self._layer_nums = layer_nums  # [5,]
#         self._upsample_strides = us_layer_strides  # [1,]
#         self._num_upsample_filters = us_num_filters  # [128,]
#         self._num_input_features = num_input_features  # 128

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
#             nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1,
#                                bias=False, ),
#             build_norm_layer(self._norm_cfg, 128, )[1],
#             nn.ReLU(),
#         )

#         self.deonv_block_1 = Sequential(
#             nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1,
#                                bias=False, ),
#             build_norm_layer(self._norm_cfg, 128, )[1],
#             nn.ReLU(),
#         )

#         self.conv_0 = Sequential(
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
#             build_norm_layer(self._norm_cfg, 128, )[1],
#             nn.ReLU(),
#         )

#         self.conv_1 = Sequential(
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False, ),
#             build_norm_layer(self._norm_cfg, 128, )[1],
#             nn.ReLU(),
#         )

#         logger.info("Finish RPN Initialization")

#     # default init_weights for conv(msra) and norm in ConvModule
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 xavier_init(m, distribution="uniform")

#     def forward(self, x):
#         x_0 = self.bottom_up_block_0(x)
#         x_1 = self.bottom_up_block_1(x_0)

#         # only for v1.0.2
#         # x_middle_0 = self.deonv_block_0(x_1) + self.trans_0(x_0)

#         # only for v1.0.3
#         # v1.0.3_0: [1, 0.4, 0.2]
#         # v1.0.3_1: [1, 1, 1], change kernel of self.conv_1 from 3x3 to 1x1
#         x_middle_0 = self.deonv_block_0(x_1) + self.trans_0(x_0) + self.conv_1(x)

#         x_output = self.conv_0(x_middle_0)

#         return x_output


# # v1.0
# @NECKS.register_module
# class DS_RPN(nn.Module):
#     def __init__(self, layer_nums, ds_layer_strides, ds_num_filters, us_layer_strides, us_num_filters,
#                  num_input_features, norm_cfg=None, name="rpn", logger=None, **kwargs):
#         super(DS_RPN, self).__init__()
#         self._layer_strides = ds_layer_strides  # [1,]
#         self._num_filters = ds_num_filters  # [128,]
#         self._layer_nums = layer_nums  # [5,]
#         self._upsample_strides = us_layer_strides  # [1,]
#         self._num_upsample_filters = us_num_filters  # [128,]
#         self._num_input_features = num_input_features  # 128

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

#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
#             build_norm_layer(self._norm_cfg, 128, )[1],
#             nn.ReLU(),

#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
#             build_norm_layer(self._norm_cfg, 128, )[1],
#             nn.ReLU(),

#         )

#         self.sp_att_0 = Sequential(
#         	nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 9), stride=1, padding=(0, 4), bias=False, ),
#         	build_norm_layer(self._norm_cfg, 64, )[1],
#             nn.ReLU(),

#             nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(9, 1), stride=1, padding=(4, 0), bias=False, ),
#         	build_norm_layer(self._norm_cfg, 1, )[1],
#             nn.ReLU(),

#         )

#         self.sp_att_1 = Sequential(
#         	nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(9, 1), stride=1, padding=(4, 0), bias=False, ),
#         	build_norm_layer(self._norm_cfg, 64, )[1],
#             nn.ReLU(),

#             nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 9), stride=1, padding=(0, 4), bias=False, ),
#         	build_norm_layer(self._norm_cfg, 1, )[1],
#             nn.ReLU(),
#         )

#         self.ch_att_0 = Sequential(
#             nn.Linear(128, 128, bias=True),
#             nn.ReLU(),
#             nn.Linear(128, 128, bias=True),
#         )

#         self.fuse_conv_0 = Sequential(
#         	nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
#             build_norm_layer(self._norm_cfg, 128, )[1],
#             nn.ReLU(),
#         )


#         # only for v1.1
#         # self.sp_att_v11_0 = Sequential(
#         # 	nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3, bias=False, ),
#         # 	build_norm_layer(self._norm_cfg, 1, )[1],
#         #     nn.ReLU(),
#         # )

#         # self.ch_att_v11_0 = Sequential(
#         #     nn.Linear(256, 128, bias=True),
#         #     nn.ReLU(),
#         #     nn.Linear(128, 128, bias=True),
#         # )

#         logger.info("Finish RPN Initialization")

#     # default init_weights for conv(msra) and norm in ConvModule
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 xavier_init(m, distribution="uniform")

#     def forward(self, x):
#     	# DS_RPN: double_stream rpn
#     	# v1.0: sp weight with global convolution network; ch_weight only consider gap values with mlp.
#     	# v1.1: sp weight with gap&gmp&conv7,7; ch_weight consider gap&gmp&mlp.
#     	# v1.2: based on v1.0, block2 add two conv.

#     	# only for v1.0
#         x_0 = self.bottom_up_block_0(x)
#         x_1 = self.bottom_up_block_1(x)
#         B, C, H, W = x_0.shape
#         sp_weight = F.sigmoid(self.sp_att_0(x_0) +  self.sp_att_1(x_0)) 
#         x_output_0 = x_0 * sp_weight

#         avg_middle_1 = F.adaptive_avg_pool2d(x_1, 1).squeeze()  # B, C
#         ch_weight = F.sigmoid(self.ch_att_0(avg_middle_1)).reshape(B, C, 1, 1)
#         x_output_1 = x_1 * ch_weight

#         x_output = self.fuse_conv_0(1.0 * x_output_0 + 1.0 * x_output_1)

#         # only for v1.1
#         # x_0 = self.bottom_up_block_0(x)
#         # x_1 = self.bottom_up_block_1(x)

#         # B, C, H, W = x_0.shape

#         # sp_avg = F.adaptive_avg_pool1d(x_0.reshape(B, C, H*W).permute(0, 2, 1), 1).permute(0, 2, 1).reshape(B, 1, H, W)
#         # sp_max = F.adaptive_max_pool1d(x_0.reshape(B, C, H*W).permute(0, 2, 1), 1).permute(0, 2, 1).reshape(B, 1, H, W)
#         # sp_cat = torch.cat([sp_avg, sp_max], dim=1)
#         # sp_weight = F.sigmoid(self.sp_att_v11_0(sp_cat))
#         # x_output_0 = x_0 * sp_weight

#         # ch_avg = F.adaptive_avg_pool2d(x_1, 1).squeeze()  # B, C
#         # ch_max = F.adaptive_max_pool2d(x_1, 1).squeeze()
#         # ch_cat = torch.cat([ch_avg, ch_max], dim=1)
#         # ch_weight = F.sigmoid(self.ch_att_v11_0(ch_cat)).reshape(B, C, 1, 1)
#         # x_output_1 = x_1 * ch_weight

#         # x_output = self.fuse_conv_0(1.0 * x_output_0 + 1.0 * x_output_1)

#         return x_output

# # v1.0.1: only add feature channles based on v1.0.
# @NECKS.register_module
# class FS_RPN(nn.Module):
#     def __init__(self, layer_nums, ds_layer_strides, ds_num_filters, us_layer_strides, us_num_filters,
#                  num_input_features, norm_cfg=None, name="rpn", logger=None, **kwargs):
#         super(FS_RPN, self).__init__()
#         self._layer_strides = ds_layer_strides  # [1,]
#         self._num_filters = ds_num_filters  # [128,]
#         self._layer_nums = layer_nums  # [5,]
#         self._upsample_strides = us_layer_strides  # [1,]
#         self._num_upsample_filters = us_num_filters  # [128,]
#         self._num_input_features = num_input_features  # 128

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

#         self.fc_0 = Sequential(
#             nn.Linear(128, 64, bias=True),
#             nn.ReLU(),
#             nn.Linear(64, 128, bias=True),
#         )

#         self.fc_1 = Sequential(
#             nn.Linear(128, 64, bias=True),
#             nn.ReLU(),
#             nn.Linear(64, 128, bias=True),
#         )

#         # only for v1.0.2
#         self.trans_2 = Sequential(
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False, ),
#             build_norm_layer(self._norm_cfg, 128, )[1],
#             nn.ReLU(),
#         )
#         self.conv_2 = Sequential(
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
#             build_norm_layer(self._norm_cfg, 128, )[1],
#             nn.ReLU(),
#         )
#         self.conv_3 = Sequential(
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
#             build_norm_layer(self._norm_cfg, 128, )[1],
#             nn.ReLU(),
#         )

#         # only for v1.3.3
       

#         self.ch_att_0 = Sequential(
#             nn.Linear(128*2, 64, bias=True),
#             nn.ReLU(),
#             nn.Linear(64, 128, bias=True),
#         )

#         self.ch_att_1 = Sequential(
#             nn.Linear(256*2, 96, bias=True),
#             nn.ReLU(),
#             nn.Linear(96, 256, bias=True),
#         )

#         self.fuse_conv_0 = Sequential(
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False, ),
#             build_norm_layer(self._norm_cfg, 128, )[1],
#             nn.ReLU(),
#         )

#         self.att_conv_0 =  Sequential(
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False, ),
#             build_norm_layer(self._norm_cfg, 128, )[1],
#             nn.ReLU(),
#         )
#         self.att_conv_1 = Sequential(
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False, ),
#             build_norm_layer(self._norm_cfg, 128, )[1],
#             nn.ReLU(),
#         )
#         self.att_conv_2 = Sequential(
#             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False, ),
#             build_norm_layer(self._norm_cfg, 128, )[1],
#             nn.ReLU(),
#         )

#         self.sp_att_0 = Sequential(
#         	nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(1, 9), stride=1, padding=(0, 4), bias=False, ),
#         	build_norm_layer(self._norm_cfg, 64, )[1],
#             nn.ReLU(),

#             nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(9, 1), stride=1, padding=(4, 0), bias=False, ),
#         	build_norm_layer(self._norm_cfg, 1, )[1],
#             nn.ReLU(),

#         )

#         self.sp_att_1 = Sequential(
#         	nn.Conv2d(in_channels=128, out_channels=64, kernel_size=(9, 1), stride=1, padding=(4, 0), bias=False, ),
#         	build_norm_layer(self._norm_cfg, 64, )[1],
#             nn.ReLU(),

#             nn.Conv2d(in_channels=64, out_channels=1, kernel_size=(1, 9), stride=1, padding=(0, 4), bias=False, ),
#         	build_norm_layer(self._norm_cfg, 1, )[1],
#             nn.ReLU(),
#         )

#         logger.info("Finish RPN Initialization")

#     # default init_weights for conv(msra) and norm in ConvModule
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 xavier_init(m, distribution="uniform")

#     def forward(self, x):
#         # v1.3.1: based on v1.0.1, combine low&high-level features with weights;
#         # v1.3.2: remove fpn, directly add low&high-level features.
#         # v1.3.3: based on 1.3.2, but add attention on low&mid&high level features.
#         #           v1.3.3_0: low with sp att, mid with sp&ch att, high with ch att;
#         #           v1.3.3_1: low with sp att, mid with sp att, high with ch att;
#         #           v1.3.3_2: low with sp att, mid without att, high with ch att;
#         #           v1.3.3_3: low with sp att, mid with ch att, high with ch att, changed to global conv.
#         #           v1.3.3_4: low with sp att, mid without att, high with ch att, changed to global conv.

#         # v1.0.2: add input x as low-level feature; then fpn has low&mid&high level features;

#         x_0 = self.bottom_up_block_0(x)
#         x_1 = self.bottom_up_block_1(x_0)

#         # only for v1.0.1
#         # x_trans_0 = self.trans_0(x_0)
#         # x_trans_1 = self.trans_1(x_1)
#         # x_middle_0 = self.deonv_block_0(x_trans_1) + x_trans_0
#         # x_middle_1 = self.deonv_block_1(x_trans_1)
#         # x_output_0 = self.conv_0(x_middle_0)
#         # x_output_1 = self.conv_1(x_middle_1)

#         # only for v1.0.2
#         # x_middle_2 = self.trans_2(x) + self.conv_2(x_middle_0)
#         # x_output_2 = self.conv_3(x_middle_2)
#         # x_output = x_output_0 + x_output_1 + x_output_2


#         # only for v1.3.2
#         # x_output_0 = self.conv_0(x_0)
#         # x_output_1 = self.deonv_block_0(x_1)
#         # B, C, H, W = x_output_0.shape

#         # only for v1.3.1&2
#         # avg_output_0 = F.adaptive_avg_pool2d(x_output_0, 1).squeeze()  # B, C
#         # avg_output_1 = F.adaptive_avg_pool2d(x_output_1, 1).squeeze()  # B, C
#         # channel_weight_output_0 = F.sigmoid(self.fc_0(avg_output_0)).reshape(B, C, 1, 1)
#         # channel_weight_output_1 = F.sigmoid(self.fc_1(avg_output_1)).reshape(B, C, 1, 1)
#         # x_output = 1.0 * x_output_0 * channel_weight_output_0 + 1.0 * x_output_1 * channel_weight_output_1
#         # x_output = self.conv_1(x_output)


#         # only for v1.3.3
#         # low-level feature with sp&ch attention
#         B, C, H, W = x.shape
#         sp_weight = F.sigmoid(self.sp_att_0(x) + self.sp_att_1(x))  # (B, 1, H, W)
#         x = x * sp_weight

#         # mid-level feature with ch attention
#         # ch_avg = F.adaptive_avg_pool2d(x_0, 1).squeeze()  # B, C
#         # ch_max = F.adaptive_max_pool2d(x_0, 1).squeeze()
#         # ch_cat = torch.cat([ch_avg, ch_max], dim=1)
#         # ch_weight = F.sigmoid(self.ch_att_0(ch_cat).reshape(B, C, 1, 1))  # (B, C, 1, 1)
#         # x_0 = x_0 * ch_weight

#         # high-level
#         ch_avg = F.adaptive_avg_pool2d(x_1, 1).squeeze()  # B, C
#         ch_max = F.adaptive_max_pool2d(x_1, 1).squeeze()
#         ch_cat = torch.cat([ch_avg, ch_max], dim=1)
#         ch_weight = F.sigmoid(self.ch_att_1(ch_cat)).reshape(B, 2*C, 1, 1)  # (B, C, 1, 1)
#         x_1 = self.deonv_block_0(x_1 * ch_weight)
#         x_output = self.fuse_conv_0(x + x_0 + x_1)

#         return x_output



# # # v1.0.1: only increase feature channles based on v1.0.
# # @NECKS.register_module
# # class RPN_v1(nn.Module):
# #     def __init__(self, layer_nums, ds_layer_strides, ds_num_filters, us_layer_strides, us_num_filters,
# #                  num_input_features, norm_cfg=None, name="rpn", logger=None, **kwargs):
# #         super(RPN_v1_6, self).__init__()
# #         self._layer_strides = ds_layer_strides  # [1,]
# #         self._num_filters = ds_num_filters  # [128,]
# #         self._layer_nums = layer_nums  # [5,]
# #         self._upsample_strides = us_layer_strides  # [1,]
# #         self._num_upsample_filters = us_num_filters  # [128,]
# #         self._num_input_features = num_input_features  # 128

# #         if norm_cfg is None:  # True
# #             norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
# #         self._norm_cfg = norm_cfg

# #         self.bottom_up_block_0 = Sequential(
# #             nn.ZeroPad2d(1),
# #             nn.Conv2d(128, 128, 3, stride=1, bias=False),
# #             build_norm_layer(self._norm_cfg, 128)[1],
# #             nn.ReLU(),

# #             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
# #             build_norm_layer(self._norm_cfg, 128, )[1],
# #             nn.ReLU(),

# #             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
# #             build_norm_layer(self._norm_cfg, 128, )[1],
# #             nn.ReLU(),
# #         )

# #         self.bottom_up_block_1 = Sequential(
# #             # [200, 176] -> [100, 88]
# #             nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False, ),
# #             build_norm_layer(self._norm_cfg, 256, )[1],
# #             nn.ReLU(),

# #             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False, ),
# #             build_norm_layer(self._norm_cfg, 256, )[1],
# #             nn.ReLU(),

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
# #             nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1,
# #                                bias=False, ),
# #             build_norm_layer(self._norm_cfg, 128, )[1],
# #             nn.ReLU(),
# #         )

# #         self.deonv_block_1 = Sequential(
# #             nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1,
# #                                bias=False, ),
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

# #         logger.info("Finish RPN Initialization")

# #     # default init_weights for conv(msra) and norm in ConvModule
# #     def init_weights(self):
# #         for m in self.modules():
# #             if isinstance(m, nn.Conv2d):
# #                 xavier_init(m, distribution="uniform")

# #     def forward(self, x):
# #         x_0 = self.bottom_up_block_0(x)
# #         x_1 = self.bottom_up_block_1(x_0)

# #         x_trans_0 = self.trans_0(x_0)
# #         x_trans_1 = self.trans_1(x_1)

# #         x_middle_0 = self.deonv_block_0(x_trans_1) + x_trans_0
# #         x_middle_1 = self.deonv_block_1(x_trans_1)

# #         x_output_0 = self.conv_0(x_middle_0)
# #         x_output_1 = self.conv_1(x_middle_1)

# #         x_output = x_output_0 + x_output_1

# #         return x_output


# # @NECKS.register_module
# # class RPN_v2(nn.Module):
# #     def __init__(self, layer_nums, ds_layer_strides, ds_num_filters, us_layer_strides, us_num_filters, num_input_features, norm_cfg=None, name="rpn", logger=None, **kwargs):
# #         super(RPN_v2, self).__init__()
# #         self._layer_strides = ds_layer_strides         # [1,]
# #         self._num_filters = ds_num_filters             # [128,]
# #         self._layer_nums = layer_nums                  # [5,]
# #         self._upsample_strides = us_layer_strides      # [1,]
# #         self._num_upsample_filters = us_num_filters    # [128,]
# #         self._num_input_features = num_input_features  # 128

# #         if norm_cfg is None: # True
# #             norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
# #         self._norm_cfg = norm_cfg


# #         self.bottom_up_block_0 = Sequential(
# #             nn.ZeroPad2d(1),
# #             nn.Conv2d(128, 128, 3, stride=1, bias=False),
# #             build_norm_layer(self._norm_cfg, 128)[1],
# #             nn.ReLU(),

# #             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
# #             build_norm_layer(self._norm_cfg, 128, )[1],
# #             nn.ReLU(),
# #         )

# #         self.bottom_up_block_1 = Sequential(
# #             # [200, 176] -> [100, 88]
# #             nn.Conv2d(in_channels=128, out_channels=192, kernel_size=3, stride=2, padding=1, bias=False, ),
# #             build_norm_layer(self._norm_cfg, 192, )[1],
# #             nn.ReLU(),

# #             nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1, bias=False, ),
# #             build_norm_layer(self._norm_cfg, 192, )[1],
# #             nn.ReLU(),
# #         )

# #         self.bottom_up_block_2 = Sequential(
# #             # [100, 88] -> [50, 44]
# #             nn.Conv2d(in_channels=192, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False, ),
# #             build_norm_layer(self._norm_cfg, 256, )[1],
# #             nn.ReLU(),

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
# #             nn.Conv2d(in_channels=192, out_channels=192, kernel_size=1, stride=1, padding=0, bias=False, ),
# #             build_norm_layer(self._norm_cfg, 192, )[1],
# #             nn.ReLU(),
# #         )

# #         self.trans_2 = Sequential(
# #             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False, ),
# #             build_norm_layer(self._norm_cfg, 256, )[1],
# #             nn.ReLU(),
# #         )

# #         self.deonv_block_0_0 = Sequential(
# #             nn.ConvTranspose2d(in_channels=256, out_channels=192, kernel_size=2, stride=2, padding=0, output_padding=0, bias=False,),
# #             build_norm_layer(self._norm_cfg, 192, )[1],
# #             nn.ReLU(),
# #         )

# #         self.deonv_block_0_1 = Sequential(
# #             nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=4, stride=4, padding=0, output_padding=0, bias=False, ),
# #             build_norm_layer(self._norm_cfg, 128, )[1],
# #             nn.ReLU(),
# #         )

# #         self.deonv_block_1_0 = Sequential(
# #             nn.ConvTranspose2d(in_channels=192, out_channels=128, kernel_size=2, stride=2, padding=0, output_padding=0, bias=False, ),
# #             build_norm_layer(self._norm_cfg, 128, )[1],
# #             nn.ReLU(),
# #         )

# #         self.deonv_block_1_1 = Sequential(
# #             nn.ConvTranspose2d(in_channels=192, out_channels=128, kernel_size=2, stride=2, padding=0, output_padding=0, bias=False, ),
# #             build_norm_layer(self._norm_cfg, 128, )[1],
# #             nn.ReLU(),
# #         )

# #         self.deonv_block_2_0 = Sequential(
# #             nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, output_padding=0, bias=False, ),
# #             build_norm_layer(self._norm_cfg, 128, )[1],
# #             nn.ReLU(),
# #         )

# #         self.deonv_block_2_1 = Sequential(
# #             nn.ConvTranspose2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, output_padding=0, bias=False, ),
# #             build_norm_layer(self._norm_cfg, 128, )[1],
# #             nn.ReLU(),
# #         )

# #         self.conv_0 = Sequential(
# #             nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False, ),
# #             build_norm_layer(self._norm_cfg, 256, )[1],
# #             nn.ReLU(),
# #         )

# #         self.conv_1 = Sequential(
# #             nn.Conv2d(in_channels=192, out_channels=192, kernel_size=3, stride=1, padding=1, bias=False, ),
# #             build_norm_layer(self._norm_cfg, 192, )[1],
# #             nn.ReLU(),
# #         )

# #         self.conv_2 = Sequential(
# #             nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
# #             build_norm_layer(self._norm_cfg, 128, )[1],
# #             nn.ReLU(),
# #         )


# #         logger.info("Finish RPN Initialization")

# #     # default init_weights for conv(msra) and norm in ConvModule
# #     def init_weights(self):
# #         for m in self.modules():
# #             if isinstance(m, nn.Conv2d):
# #                 xavier_init(m, distribution="uniform")

# #     def forward(self, x):
# #         x_0 = self.bottom_up_block_0(x)
# #         x_1 = self.bottom_up_block_1(x_0)
# #         x_2 = self.bottom_up_block_2(x_1)

# #         x_trans_0 = self.trans_0(x_0)
# #         x_trans_1 = self.trans_1(x_1)
# #         x_trans_2 = self.trans_2(x_2)

# #         x_middle_0 = self.deonv_block_0_0(x_trans_2)
# #         x_middle_1 = x_middle_0 + x_trans_1
# #         x_middle_2 = x_trans_0 + self.deonv_block_1_0(x_middle_1)

# #         # v2.1 add conv after deconv
# #         # v2.2 add conv before deconv
# #         # x_output_0 = self.deonv_block_0_1(self.conv_0(x_trans_2))
# #         # x_output_1 = self.deonv_block_1_1(self.conv_1(x_middle_1))
# #         # x_output_2 = self.deonv_block_2_1(self.conv_2(x_middle_2))

# #         x_output_0 = self.deonv_block_0_1(x_trans_2)
# #         x_output_1 = self.deonv_block_1_1(x_middle_1)
# #         x_output_2 = self.deonv_block_2_1(x_middle_2)

# #         x_output = x_output_0 + x_output_1 + x_output_2

# #         return x_output



# @NECKS.register_module
# class RPN(nn.Module):
#     def __init__(self, layer_nums, ds_layer_strides, ds_num_filters, us_layer_strides, us_num_filters, num_input_features, norm_cfg=None, name="rpn", logger=None, **kwargs):

#         super(RPN, self).__init__()
#         self._layer_strides = ds_layer_strides       # [1,]
#         self._num_filters = ds_num_filters           # [128,]
#         self._layer_nums = layer_nums                # [5,]
#         self._upsample_strides = us_layer_strides    # [1,]
#         self._num_upsample_filters = us_num_filters  # [128,]
#         self._num_input_features = num_input_features # 128

#         if norm_cfg is None: # True
#             norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
#         self._norm_cfg = norm_cfg

#         assert len(self._layer_strides) == len(self._layer_nums)
#         assert len(self._num_filters) == len(self._layer_nums)
#         assert len(self._num_upsample_filters) == len(self._upsample_strides)

#         self._upsample_start_idx = len(self._layer_nums) - len(self._upsample_strides)   # 0

#         must_equal_list = []
#         for i in range(len(self._upsample_strides)):
#             must_equal_list.append(self._upsample_strides[i] / np.prod(self._layer_strides[: i + self._upsample_start_idx + 1]))  # [1]

#         for val in must_equal_list:
#             assert val == must_equal_list[0]

#         in_filters = [self._num_input_features, *self._num_filters[:-1]]     # [128]
#         blocks = []
#         deblocks = []

#         for i, layer_num in enumerate(self._layer_nums):
#             block, num_out_filters = self._make_layer(in_filters[i], self._num_filters[i], layer_num, stride=self._layer_strides[i],)  # 128, 128, 5, 1
#             blocks.append(block)
#             if i - self._upsample_start_idx >= 0:
#                 deblock = Sequential(
#                     nn.ConvTranspose2d(
#                         num_out_filters,                                             # 128
#                         self._num_upsample_filters[i - self._upsample_start_idx],    # 128
#                         self._upsample_strides[i - self._upsample_start_idx],        # 1
#                         stride=self._upsample_strides[i - self._upsample_start_idx], # 1
#                         bias=False,
#                     ),
#                     # todo: attention here, add extra batch_norm implementation
#                     build_norm_layer(self._norm_cfg, self._num_upsample_filters[i - self._upsample_start_idx],)[1],  # 128
#                     nn.ReLU(),
#                 )
#                 deblocks.append(deblock)
#         self.blocks = nn.ModuleList(blocks)
#         '''
#         6 层二维卷积
#         self.blocks:
#             (0): ZeroPad2d(padding=(1, 1, 1, 1), value=0.0)
#             (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), bias=False)
#             (2): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#             (3): ReLU()
#             (4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (5): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#             (6): ReLU()
#             (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (8): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#             (9): ReLU()
#             (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (11): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#             (12): ReLU()
#             (13): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (14): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#             (15): ReLU()
#             (16): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#             (17): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#             (18): ReLU()

#         '''
#         self.deblocks = nn.ModuleList(deblocks)
#         '''
#         1层反卷积
#         self.deblocks:
#             (0): Sequential(
#                 (0): ConvTranspose2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#                 (1): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#                 (2): ReLU()
#             )
#         '''
#         logger.info("Finish RPN Initialization")

#     @property
#     def downsample_factor(self):
#         factor = np.prod(self._layer_strides)    # 1
#         if len(self._upsample_strides) > 0:      # True
#             factor /= self._upsample_strides[-1] # 1.0
#         return factor  # 1.0

#     def _make_layer(self, inplanes, planes, num_blocks, stride=1):
#         # 128, 128, 5, 1
#         block = Sequential(
#             nn.ZeroPad2d(1),   # to keep size of input and output feature map the same=128
#             nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
#             build_norm_layer(self._norm_cfg, planes)[1],
#             # nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01),
#             nn.ReLU(),
#         )

#         for j in range(num_blocks):
#             block.add(nn.Conv2d(planes, planes, 3, padding=1, bias=False))
#             block.add(build_norm_layer(self._norm_cfg, planes)[1],) # nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
#             block.add(nn.ReLU())

#         return block, planes

#     # default init_weights for conv(msra) and norm in ConvModule
#     def init_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 xavier_init(m, distribution="uniform")

#     def forward(self, x):
#         ups = []
#         #import ipdb; ipdb.set_trace()
#         for i in range(len(self.blocks)):
#             x = self.blocks[i](x)       # [4, 128, 200, 176]
#             if i - self._upsample_start_idx >= 0:  # True
#                 ups.append(self.deblocks[i - self._upsample_start_idx](x))  # torch.Size([4, 128, 200, 176])
#         if len(ups) > 0:   # True
#             x = torch.cat(ups, dim=1)        # [4, 128, 200, 176]
#         return x







# '''
#        6 层二维卷积
#        self.blocks:
#            (0): ZeroPad2d(padding=(1, 1, 1, 1), value=0.0)
#            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), bias=False)
#            (2): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#            (3): ReLU()
#            (4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#            (5): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#            (6): ReLU()
#            (7): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#            (8): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#            (9): ReLU()
#            (10): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#            (11): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#            (12): ReLU()
#            (13): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#            (14): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#            (15): ReLU()
#            (16): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
#            (17): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#            (18): ReLU()
           
#         1层反卷积
#         self.deblocks:
#             (0): Sequential(
#                 (0): ConvTranspose2d(128, 128, kernel_size=(1, 1), stride=(1, 1), bias=False)
#                 (1): BatchNorm2d(128, eps=0.001, momentum=0.01, affine=True, track_running_stats=True)
#                 (2): ReLU()
#             )

# '''


# @NECKS.register_module
# class PointModule(nn.Module):
#     def __init__(
#         self,
#         num_input_features,
#         layers=[1024, 128,],
#         norm_cfg=None,
#         name="rpn",
#         logger=None,
#         **kwargs
#     ):
#         super(PointModule, self).__init__()

#         if norm_cfg is None:
#             norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
#         self._norm_cfg = norm_cfg

#         blocks = [
#             nn.Conv2d(num_input_features, layers[0], 1, bias=False),
#             build_norm_layer(self._norm_cfg, layers[0])[1],
#             nn.ReLU(),
#             nn.Conv2d(layers[0], layers[1], 1, bias=False),
#             build_norm_layer(self._norm_cfg, layers[1])[1],
#             nn.ReLU(),
#         ]
#         self.pn = nn.ModuleList(blocks)
#         self.out = nn.MaxPool1d(3, stride=1, padding=1)

#     def forward(self, x):
#         x = x.flatten(1, -1)
#         x = x.view(x.shape[0], x.shape[1], 1, 1)

#         for l in self.pn:
#             x = l(x)

#         x = x.view(x.shape[0], 1, x.shape[1])
#         x = self.out(x).view(x.shape[0], x.shape[2], 1, 1)

#         return x
