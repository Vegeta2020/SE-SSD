import time
import numpy as np
import math
import matplotlib.pyplot as plt

import torch

from torch import nn
from torch.nn import functional as F
from torchvision.models import resnet
from torch.nn.modules.batchnorm import _BatchNorm

from det3d.torchie.cnn import constant_init, kaiming_init, xavier_init
from det3d.torchie.trainer import load_checkpoint
from det3d.models.utils import Empty, GroupNorm, Sequential
from det3d.models.utils import change_default_args

from .. import builder
from ..registry import NECKS
from ..utils import build_norm_layer


@NECKS.register_module
class RPN(nn.Module):
    def __init__(self, layer_nums, ds_layer_strides, ds_num_filters, us_layer_strides, us_num_filters, num_input_features, norm_cfg=None, name="rpn", logger=None, **kwargs):

        super(RPN, self).__init__()
        self._layer_strides = ds_layer_strides       # [1,]
        self._num_filters = ds_num_filters           # [128,]
        self._layer_nums = layer_nums                # [5,]
        self._upsample_strides = us_layer_strides    # [1,]
        self._num_upsample_filters = us_num_filters  # [128,]
        self._num_input_features = num_input_features # 128

        if norm_cfg is None: # True
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self._norm_cfg = norm_cfg

        assert len(self._layer_strides) == len(self._layer_nums)
        assert len(self._num_filters) == len(self._layer_nums)
        assert len(self._num_upsample_filters) == len(self._upsample_strides)

        self._upsample_start_idx = len(self._layer_nums) - len(self._upsample_strides)   # 0

        must_equal_list = []
        for i in range(len(self._upsample_strides)):
            must_equal_list.append(self._upsample_strides[i] / np.prod(self._layer_strides[: i + self._upsample_start_idx + 1]))  # [1]

        for val in must_equal_list:
            assert val == must_equal_list[0]

        in_filters = [self._num_input_features, *self._num_filters[:-1]]     # [128]
        blocks = []
        deblocks = []

        for i, layer_num in enumerate(self._layer_nums):
            block, num_out_filters = self._make_layer(in_filters[i], self._num_filters[i], layer_num, stride=self._layer_strides[i],)  # 128, 128, 5, 1
            blocks.append(block)
            if i - self._upsample_start_idx >= 0:
                deblock = Sequential(
                    nn.ConvTranspose2d(
                        num_out_filters,                                             # 128
                        self._num_upsample_filters[i - self._upsample_start_idx],    # 128
                        self._upsample_strides[i - self._upsample_start_idx],        # 1
                        stride=self._upsample_strides[i - self._upsample_start_idx], # 1
                        bias=False,
                    ),
                    # todo: attention here, add extra batch_norm implementation
                    build_norm_layer(self._norm_cfg, self._num_upsample_filters[i - self._upsample_start_idx],)[1],  # 128
                    nn.ReLU(),
                )
                deblocks.append(deblock)
        self.blocks = nn.ModuleList(blocks)
        self.deblocks = nn.ModuleList(deblocks)
        logger.info("Finish RPN Initialization")

    @property
    def downsample_factor(self):
        factor = np.prod(self._layer_strides)    # 1
        if len(self._upsample_strides) > 0:      # True
            factor /= self._upsample_strides[-1] # 1.0
        return factor  # 1.0

    def _make_layer(self, inplanes, planes, num_blocks, stride=1):
        # 128, 128, 5, 1
        block = Sequential(
            nn.ZeroPad2d(1),   # to keep size of input and output feature map the same=128
            nn.Conv2d(inplanes, planes, 3, stride=stride, bias=False),
            build_norm_layer(self._norm_cfg, planes)[1],
            # nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01),
            nn.ReLU(),
        )

        for j in range(num_blocks):
            block.add(nn.Conv2d(planes, planes, 3, padding=1, bias=False))
            block.add(build_norm_layer(self._norm_cfg, planes)[1],) # nn.BatchNorm2d(planes, eps=1e-3, momentum=0.01)
            block.add(nn.ReLU())

        return block, planes

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, x):
        ups = []
        #import ipdb; ipdb.set_trace()
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)       # [4, 128, 200, 176]
            if i - self._upsample_start_idx >= 0:  # True
                ups.append(self.deblocks[i - self._upsample_start_idx](x))  # torch.Size([4, 128, 200, 176])
        if len(ups) > 0:   # True
            x = torch.cat(ups, dim=1)        # [4, 128, 200, 176]
        return x

# SSFA: Spatial-semantic feature aggregation module
@NECKS.register_module
class SSFA(nn.Module):
    def __init__(self, layer_nums, ds_layer_strides, ds_num_filters, us_layer_strides, us_num_filters,
                 num_input_features, norm_cfg=None, name="rpn", logger=None, **kwargs):
        super(SSFA, self).__init__()
        self._layer_strides = ds_layer_strides  # [1,]
        self._num_filters = ds_num_filters      # [128,]
        self._layer_nums = layer_nums           # [5,]
        self._upsample_strides = us_layer_strides      # [1,]
        self._num_upsample_filters = us_num_filters    # [128,]
        self._num_input_features = num_input_features  # 128

        if norm_cfg is None:  # True
            norm_cfg = dict(type="BN", eps=1e-3, momentum=0.01)
        self._norm_cfg = norm_cfg

        self.bottom_up_block_0 = Sequential(
            nn.ZeroPad2d(1),
            nn.Conv2d(128, 128, 3, stride=1, bias=False),
            build_norm_layer(self._norm_cfg, 128)[1],
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
            build_norm_layer(self._norm_cfg, 128, )[1],
            nn.ReLU(),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
            build_norm_layer(self._norm_cfg, 128, )[1],
            nn.ReLU(),
        )

        self.bottom_up_block_1 = Sequential(
            # [200, 176] -> [100, 88]
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1, bias=False, ),
            build_norm_layer(self._norm_cfg, 256, )[1],
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False, ),
            build_norm_layer(self._norm_cfg, 256, )[1],
            nn.ReLU(),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1, bias=False, ),
            build_norm_layer(self._norm_cfg, 256, )[1],
            nn.ReLU(),

        )

        self.trans_0 = Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=1, stride=1, padding=0, bias=False, ),
            build_norm_layer(self._norm_cfg, 128, )[1],
            nn.ReLU(),
        )

        self.trans_1 = Sequential(
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1, stride=1, padding=0, bias=False, ),
            build_norm_layer(self._norm_cfg, 256, )[1],
            nn.ReLU(),
        )

        self.deconv_block_0 = Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False, ),
            build_norm_layer(self._norm_cfg, 128, )[1],
            nn.ReLU(),
        )

        self.deconv_block_1 = Sequential(
            nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False, ),
            build_norm_layer(self._norm_cfg, 128, )[1],
            nn.ReLU(),
        )

        self.conv_0 = Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
            build_norm_layer(self._norm_cfg, 128, )[1],
            nn.ReLU(),
        )

        self.w_0 = Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False, ),
            build_norm_layer(self._norm_cfg, 1, )[1],
        )

        self.conv_1 = Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1, bias=False, ),
            build_norm_layer(self._norm_cfg, 128, )[1],
            nn.ReLU(),
        )

        self.w_1 = Sequential(
            nn.Conv2d(in_channels=128, out_channels=1, kernel_size=1, stride=1, padding=0, bias=False, ),
            build_norm_layer(self._norm_cfg, 1, )[1],
        )

        logger.info("Finish RPN Initialization")

    # default init_weights for conv(msra) and norm in ConvModule
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                xavier_init(m, distribution="uniform")

    def forward(self, x):
        x_0 = self.bottom_up_block_0(x)
        x_1 = self.bottom_up_block_1(x_0)
        x_trans_0 = self.trans_0(x_0)
        x_trans_1 = self.trans_1(x_1)
        x_middle_0 = self.deconv_block_0(x_trans_1) + x_trans_0
        x_middle_1 = self.deconv_block_1(x_trans_1)
        x_output_0 = self.conv_0(x_middle_0)
        x_output_1 = self.conv_1(x_middle_1)

        x_weight_0 = self.w_0(x_output_0)
        x_weight_1 = self.w_1(x_output_1)
        x_weight = torch.softmax(torch.cat([x_weight_0, x_weight_1], dim=1), dim=1)
        x_output = x_output_0 * x_weight[:, 0:1, :, :] + x_output_1 * x_weight[:, 1:, :, :]

        return x_output



