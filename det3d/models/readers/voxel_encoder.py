import time

import numpy as np
import torch
from det3d.models.utils import Empty, change_default_args, get_paddings_indicator
from torch import nn
from torch.nn import functional as F

from .. import builder
from ..registry import READERS


@READERS.register_module
class VFELayer(nn.Module):
    def __init__(self, in_channels, out_channels, use_norm=True, name="vfe"):
        super(VFELayer, self).__init__()
        self.name = name
        self.units = int(out_channels / 2)
        if use_norm:
            BatchNorm1d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        self.linear = Linear(in_channels, self.units)
        self.norm = BatchNorm1d(self.units)

    def forward(self, inputs):
        # [K, T, 7] tensordot [7, units] = [K, T, units]
        voxel_count = inputs.shape[1]
        x = self.linear(inputs)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        pointwise = F.relu(x)
        # [K, T, units]

        aggregated = torch.max(pointwise, dim=1, keepdim=True)[0]
        # [K, 1, units]
        repeated = aggregated.repeat(1, voxel_count, 1)

        concatenated = torch.cat([pointwise, repeated], dim=2)
        # [K, T, 2 * units]
        return concatenated


@READERS.register_module
class VoxelFeatureExtractor(nn.Module):
    def __init__(
        self,
        num_input_features=4,
        use_norm=True,
        num_filters=[32, 128],
        with_distance=False,
        voxel_size=(0.2, 0.2, 4),
        name="VoxelFeatureExtractor",
    ):
        super(VoxelFeatureExtractor, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm1d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        assert len(num_filters) == 2
        num_input_features += 3  # add mean features
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance
        self.vfe1 = VFELayer(num_input_features, num_filters[0], use_norm)
        self.vfe2 = VFELayer(num_filters[0], num_filters[1], use_norm)
        self.linear = Linear(num_filters[1], num_filters[1])
        # var_torch_init(self.linear.weight)
        # var_torch_init(self.linear.bias)
        self.norm = BatchNorm1d(num_filters[1])

    def forward(self, features, num_voxels, coors):
        # features: [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: [concated_num_points]
        # t = time.time()
        # torch.cuda.synchronize()

        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(
            features
        ).view(-1, 1, 1)
        features_relative = features[:, :, :3] - points_mean
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features = torch.cat([features, features_relative, points_dist], dim=-1)
        else:
            features = torch.cat([features, features_relative], dim=-1)
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        # mask = features.max(dim=2, keepdim=True)[0] != 0

        # torch.cuda.synchronize()
        # print("vfe prep forward time", time.time() - t)
        x = self.vfe1(features)
        x *= mask
        x = self.vfe2(x)
        x *= mask
        x = self.linear(x)
        x = self.norm(x.permute(0, 2, 1).contiguous()).permute(0, 2, 1).contiguous()
        x = F.relu(x)
        x *= mask
        # x: [concated_num_points, num_voxel_size, 128]
        voxelwise = torch.max(x, dim=1)[0]
        return voxelwise


@READERS.register_module
class VoxelFeatureExtractorV2(nn.Module):
    def __init__(
        self,
        num_input_features=4,
        use_norm=True,
        num_filters=[32, 128],
        with_distance=False,
        voxel_size=(0.2, 0.2, 4),
        name="VoxelFeatureExtractor",
    ):
        super(VoxelFeatureExtractorV2, self).__init__()
        self.name = name
        if use_norm:
            BatchNorm1d = change_default_args(eps=1e-3, momentum=0.01)(nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)
        assert len(num_filters) > 0
        num_input_features += 3
        if with_distance:
            num_input_features += 1
        self._with_distance = with_distance

        num_filters = [num_input_features] + num_filters
        filters_pairs = [
            [num_filters[i], num_filters[i + 1]] for i in range(len(num_filters) - 1)
        ]
        self.vfe_layers = nn.ModuleList(
            [VFELayer(i, o, use_norm) for i, o in filters_pairs]
        )
        self.linear = Linear(num_filters[-1], num_filters[-1])
        # var_torch_init(self.linear.weight)
        # var_torch_init(self.linear.bias)
        self.norm = BatchNorm1d(num_filters[-1])

    def forward(self, features, num_voxels, coors):
        # features: [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: [concated_num_points]
        points_mean = features[:, :, :3].sum(dim=1, keepdim=True) / num_voxels.type_as(
            features
        ).view(-1, 1, 1)
        features_relative = features[:, :, :3] - points_mean
        if self._with_distance:
            points_dist = torch.norm(features[:, :, :3], 2, 2, keepdim=True)
            features = torch.cat([features, features_relative, points_dist], dim=-1)
        else:
            features = torch.cat([features, features_relative], dim=-1)
        voxel_count = features.shape[1]
        mask = get_paddings_indicator(num_voxels, voxel_count, axis=0)
        mask = torch.unsqueeze(mask, -1).type_as(features)
        for vfe in self.vfe_layers:
            features = vfe(features)
            features *= mask
        features = self.linear(features)
        features = (
            self.norm(features.permute(0, 2, 1).contiguous())
            .permute(0, 2, 1)
            .contiguous()
        )
        features = F.relu(features)
        features *= mask
        # x: [concated_num_points, num_voxel_size, 128]
        voxelwise = torch.max(features, dim=1)[0]
        return voxelwise


@READERS.register_module
class VFEV3_ablation(nn.Module):
    def __init__(self, num_input_features=4, norm_cfg=None, name="VFEV3_ablation"):
        super(VFEV3_ablation, self).__init__()
        self.name = name
        self.num_input_features = num_input_features

    def forward(self, features, num_voxels, coors=None):
        points_mean = features[:, :, [0, 1, 3]].sum(
            dim=1, keepdim=False
        ) / num_voxels.type_as(features).view(-1, 1)
        points_mean = torch.cat(
            [points_mean, 1.0 / num_voxels.to(torch.float32).view(-1, 1)], dim=1
        )

        return points_mean.contiguous()


@READERS.register_module
class VoxelFeatureExtractorV3(nn.Module):
    '''Interesting! Take the mean value of points in voxel as its feature'''
    def __init__(self, num_input_features=4, norm_cfg=None, name="VoxelFeatureExtractorV3"):
        super(VoxelFeatureExtractorV3, self).__init__()
        self.name = name
        self.num_input_features = num_input_features

    def forward(self, voxels, num_points_per_voxel, coors=None):
        # features: [batch_size * num_voxels, num_points_in_voxel, num_input_features],
        # num_points_per_voxel:  [batch_size * num_voxels, num_points].
        # todo: maybe we should add some info about the voxel
        points_mean = voxels[:, :, : self.num_input_features].sum(dim=1, keepdim=False) / num_points_per_voxel.type_as(voxels).view(-1, 1)
        return points_mean.contiguous()


@READERS.register_module
class VoxelFeatureExtractorV3_sassd(nn.Module):
    '''Interesting! Take the mean value of points in voxel as its feature'''
    def __init__(self, num_input_features=4, norm_cfg=None, name="VoxelFeatureExtractorV3"):
        super(VoxelFeatureExtractorV3_sassd, self).__init__()
        self.name = name
        self.num_input_features = num_input_features

    def forward(self, voxels, num_points_per_voxel, coors=None):
        return voxels

# @READERS.register_module
# class VoxelFeatureExtractorV4(nn.Module):
#     '''Interesting! Take the mean value of points in voxel as its feature'''
#     def __init__(self, num_input_features=4, norm_cfg=None, name="VoxelFeatureExtractorV3"):
#         super(VoxelFeatureExtractorV4, self).__init__()
#         self.name = name
#         self.num_input_features = num_input_features
#
#     def forward(self, voxels, num_points_per_voxel, coors=None):
#         # features: [batch_size * num_voxels, num_points_in_voxel, num_input_features],
#         # num_points_per_voxel:  [batch_size * num_voxels, num_points].
#         # todo: maybe we should add some info about the voxel
#         points_mean = voxels[:, :, : self.num_input_features].sum(dim=1, keepdim=False) / num_points_per_voxel.type_as(voxels).view(-1, 1)
#         zero_mask = (voxels[:, :, : self.num_input_features] == 0).all(2)
#         non_zero_mask = torch.logical_not(zero_mask)
#         points_delta = torch.zeros_like(voxels[:, :, : self.num_input_features])
#         points_mean_repeat = points_mean[:, None, :].repeat(1, 5, 1)
#         points_delta[non_zero_mask] = voxels[:, :, : self.num_input_features][non_zero_mask] - points_mean_repeat[non_zero_mask]
#         points_delta_max = points_delta.max(axis=1)[0][:, 0:3]
#         points_delta_min = points_delta.min(axis=1)[0][:, 0:3]
#         points_input = torch.cat([points_mean, points_delta_max, points_delta_min], dim=1)
#
#         return points_input.contiguous()


# @READERS.register_module
# class VoxelFeatureExtractorV5(nn.Module):
#     '''Interesting! Take the mean value of points in voxel as its feature'''
#     def __init__(self, num_input_features=4, norm_cfg=None, name="VoxelFeatureExtractorV5"):
#         super(VoxelFeatureExtractorV5, self).__init__()
#         self.name = name
#         self.num_input_features = num_input_features
#         self.conv1 = torch.nn.Conv1d(4, 16, 1)
#         self.conv2 = torch.nn.Conv1d(16, 4, 1)
#         self.bn1 = nn.BatchNorm1d(16)
#         self.bn2 = nn.BatchNorm1d(4)
#
#
#     def forward(self, voxels, num_points_per_voxel, coors=None):
#         points_mean = voxels[:, :, : self.num_input_features].sum(dim=1, keepdim=False) / num_points_per_voxel.type_as(voxels).view(-1, 1)
#         zero_mask = (voxels[:, :, : self.num_input_features] == 0).all(2)
#         non_zero_mask = torch.logical_not(zero_mask)
#         points_delta = torch.zeros_like(voxels[:, :, : self.num_input_features])
#         points_mean_repeat = points_mean[:, None, :].repeat(1, 5, 1)
#         points_delta[non_zero_mask] = voxels[:, :, : self.num_input_features][non_zero_mask] - points_mean_repeat[non_zero_mask]
#
#         points_delta_features = - 1e6 * torch.ones(voxels.shape[0], voxels.shape[1], 4).cuda().contiguous()
#         x = F.relu(self.bn1(self.conv1(points_delta[non_zero_mask][:, :, None])))
#         x = F.relu(self.bn2(self.conv2(x))).permute(0, 2, 1).squeeze()
#         points_delta_features[non_zero_mask] = x
#         points_delta_features_max = torch.max(points_delta_features, dim=1)[0]
#         cat_points = torch.cat([points_mean, points_delta_features_max], dim=1)
#
#         return cat_points.contiguous()


@READERS.register_module
class SimpleVoxel(nn.Module):
    """Simple voxel encoder. only keep r, z and reflection feature.
    """

    def __init__(self, num_input_features=4, norm_cfg=None, name="SimpleVoxel"):

        super(SimpleVoxel, self).__init__()

        self.num_input_features = num_input_features
        self.name = name

    def forward(self, features, num_voxels, coors=None):
        # features: [concated_num_points, num_voxel_size, 3(4)]
        # num_voxels: [concated_num_points]
        points_mean = features[:, :, : self.num_input_features].sum(
            dim=1, keepdim=False
        ) / num_voxels.type_as(features).view(-1, 1)
        feature = torch.norm(points_mean[:, :2], p=2, dim=1, keepdim=True)
        # z is important for z position regression, but x, y is not.
        res = torch.cat([feature, points_mean[:, 2 : self.num_input_features]], dim=1)
        return res
