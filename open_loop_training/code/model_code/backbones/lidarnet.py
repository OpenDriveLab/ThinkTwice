import copy
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from mmcv.runner import BaseModule
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import BACKBONES
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
from mmdet3d.models.builder import MIDDLE_ENCODERS
from mmdet3d.models.middle_encoders.sparse_encoder import SparseEncoder
from mmdet3d.ops.spconv import IS_SPCONV2_AVAILABLE
if IS_SPCONV2_AVAILABLE:
    from spconv.pytorch import SparseConvTensor
else:
    from mmcv.ops import SparseConvTensor


#To avoid strange FP16 inf norm bug
@MIDDLE_ENCODERS.register_module()
class SparseEncoder_fp32(SparseEncoder):
    
    @force_fp32()
    def forward(self, voxel_features, coors, batch_size):
        """Forward of SparseEncoder.

        Args:
            voxel_features (torch.Tensor): Voxel features in shape (N, C).
            coors (torch.Tensor): Coordinates in shape (N, 4),
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            dict: Backbone features.
        """
        coors = coors.int()
        input_sp_tensor = SparseConvTensor(voxel_features, coors,
                                           self.sparse_shape, batch_size)
        x = self.conv_input(input_sp_tensor)

        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(encode_features[-1])
        spatial_features = out.dense()

        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)

        return spatial_features


@BACKBONES.register_module()
class LidarNet(MVXTwoStageDetector):
    def __init__(self,
                 bev_h=None,
                 bev_w=None,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 pts_backbone=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 ):
        self.bev_h = bev_h
        self.bev_w = bev_w
        super(LidarNet,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             None, pts_backbone, None, pts_neck,
                             pts_bbox_head, None, None,
                             train_cfg, test_cfg, None)
        self.fp16_enabled = False
    
    @auto_fp16()
    def forward(self, pts):
        voxels, num_points, coors = self.voxelize(pts)
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors)
        batch_size = coors[-1, 0] + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size) 
        if self.with_pts_backbone:
            x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        return x


