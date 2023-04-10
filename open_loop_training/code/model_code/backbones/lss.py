## Copyright (c) Megvii Inc. All rights reserved.
## From BEVDepth
## Modified by Xiaosong Jia
import torch
import torch.nn.functional as F
from mmcv.runner import BaseModule
from mmdet.models import BACKBONES
from mmcv.cnn import build_conv_layer
from mmdet3d.models import build_neck
from mmdet.models import build_backbone
from mmdet.models.backbones.resnet import BasicBlock
from torch import nn
import numpy as np
from ops.voxel_pooling import voxel_pooling
from mmcv.runner import force_fp32, auto_fp16

__all__ = ['LSS']


class _ASPPModule(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, padding, dilation,
                 BatchNorm):
        super(_ASPPModule, self).__init__()
        self.atrous_conv = nn.Conv2d(inplanes,
                                     planes,
                                     kernel_size=kernel_size,
                                     stride=1,
                                     padding=padding,
                                     dilation=dilation,
                                     bias=False)
        self.bn = BatchNorm(planes)
        self.relu = nn.ReLU()
        self._init_weight()

    def forward(self, x):
        x = self.atrous_conv(x)
        x = self.bn(x)
        return self.relu(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class ASPP(nn.Module):
    def __init__(self, inplanes, mid_channels=256, BatchNorm=nn.BatchNorm2d):
        super(ASPP, self).__init__()
        dilations = [1, 6, 12, 18]

        self.aspp1 = _ASPPModule(inplanes,
                                 mid_channels,
                                 1,
                                 padding=0,
                                 dilation=dilations[0],
                                 BatchNorm=BatchNorm)
        self.aspp2 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[1],
                                 dilation=dilations[1],
                                 BatchNorm=BatchNorm)
        self.aspp3 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[2],
                                 dilation=dilations[2],
                                 BatchNorm=BatchNorm)
        self.aspp4 = _ASPPModule(inplanes,
                                 mid_channels,
                                 3,
                                 padding=dilations[3],
                                 dilation=dilations[3],
                                 BatchNorm=BatchNorm)

        self.global_avg_pool = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(inplanes, mid_channels, 1, stride=1, bias=False),
            BatchNorm(mid_channels),
            nn.ReLU(),
        )
        self.conv1 = nn.Conv2d(int(mid_channels * 5),
                               mid_channels,
                               1,
                               bias=False)
        self.bn1 = BatchNorm(mid_channels)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self._init_weight()

    def forward(self, x):
        x1 = self.aspp1(x)
        x2 = self.aspp2(x)
        x3 = self.aspp3(x)
        x4 = self.aspp4(x)
        x5 = self.global_avg_pool(x)
        x5 = F.interpolate(x5,
                           size=x4.size()[2:],
                           mode='bilinear',
                           align_corners=True)
        x = torch.cat((x1, x2, x3, x4, x5), dim=1)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        return self.dropout(x)

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


class Mlp(nn.Module):
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.ReLU,
                 drop=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.conv_reduce = nn.Conv2d(channels, channels, 1, bias=True)
        self.act1 = act_layer()
        self.conv_expand = nn.Conv2d(channels, channels, 1, bias=True)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.conv_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.conv_expand(x_se)
        return x * self.gate(x_se)


class DepthNet(nn.Module):
    def __init__(self, in_channels, mid_channels, context_channels,
                 depth_channels):
        super(DepthNet, self).__init__()
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(in_channels,
                      mid_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
        )
        self.context_conv = nn.Conv2d(mid_channels,
                                      context_channels,
                                      kernel_size=1,
                                      stride=1,
                                      padding=0)
        self.bn = nn.BatchNorm1d(22)
        self.depth_mlp = Mlp(22, mid_channels, mid_channels)
        self.depth_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.context_mlp = Mlp(22, mid_channels, mid_channels)
        self.context_se = SELayer(mid_channels)  # NOTE: add camera-aware
        self.depth_conv = nn.Sequential(
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            BasicBlock(mid_channels, mid_channels),
            ASPP(mid_channels, mid_channels),
            build_conv_layer(cfg=dict(
                type='DCN',
                in_channels=mid_channels,
                out_channels=mid_channels,
                kernel_size=3,
                padding=1,
                groups=4,
                im2col_step=128,
            )),
            nn.Conv2d(mid_channels,
                      depth_channels,
                      kernel_size=1,
                      stride=1,
                      padding=0),
        )

    def forward(self, x, mats_dict):    # mats_dict   B x T x N
        intrins = mats_dict['intrin_mats'][:, -1:, ..., :3, :3]
        batch_size = intrins.shape[0]
        num_cams = intrins.shape[2]
        ida = mats_dict['ida_mats'][:,  -1:, ...]
        sensor2ego = mats_dict['sensor2ego_mats'][:,  -1:, ..., :3, :]
        mlp_input = torch.cat(
            [
                torch.stack(
                    [
                        intrins[:, 0:1, ..., 0, 0],
                        intrins[:, 0:1, ..., 1, 1],
                        intrins[:, 0:1, ..., 0, 2],
                        intrins[:, 0:1, ..., 1, 2],
                        ida[:, 0:1, ..., 0, 0],
                        ida[:, 0:1, ..., 0, 1],
                        ida[:, 0:1, ..., 0, 3],
                        ida[:, 0:1, ..., 1, 0],
                        ida[:, 0:1, ..., 1, 1],
                        ida[:, 0:1, ..., 1, 3],
                    ],
                    dim=-1,
                ),
                sensor2ego.view(batch_size, 1, num_cams, -1),
            ],
            -1,
        )
        mlp_input = self.bn(mlp_input.reshape(-1, mlp_input.shape[-1]))
        x = self.reduce_conv(x)
        context_se = self.context_mlp(mlp_input)[..., None, None]
        context = self.context_se(x, context_se)
        context = self.context_conv(context)
        depth_se = self.depth_mlp(mlp_input)[..., None, None]
        depth = self.depth_se(x, depth_se)
        depth = self.depth_conv(depth)
        return torch.cat([depth, context], dim=1)


class UnetLayer(nn.Module):
    def __init__(self, in_channels, middle_channels, out_channels):
        super(UnetLayer, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv_relu = nn.Sequential(
            nn.Conv2d(middle_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            )
        self.fp16_enabled = False
    
    @force_fp32()
    def forward(self, x1, x2):
        x1 = self.up(x1)
        x1 = torch.cat((x1, x2), dim=1)
        x1 = self.conv_relu(x1)
        return x1

class UNet(nn.Module):
    def __init__(self, n_class, fpn_in_channels):
        super().__init__()
        self.unet_layer4 = UnetLayer(fpn_in_channels[3], 256+fpn_in_channels[2], 256)
        self.unet_layer3 = UnetLayer(256, 256+fpn_in_channels[1], 256)
        self.unet_layer2 = UnetLayer(256, 128+fpn_in_channels[0], 128)
        self.unet_layer0 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(128, 64, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
            )
        self.conv_last = nn.Conv2d(64, n_class, 1)
        self.fp16_enabled = False
    
    def forward(self, input):
        e1, e2, e3, e4 = input # input resnet's feature directly
        d4 = self.unet_layer4(e4, e3) 
        d3 = self.unet_layer3(d4, e2) 
        d2 = self.unet_layer2(d3, e1) 
        d0 = self.unet_layer0(d2) 
        out = self.conv_last(d0) #  1/2 origin shape
        return out

from mmdet.models.necks.pafpn import PAFPN   
from mmdet.models import NECKS 
@NECKS.register_module()
class PAFPN_fp32(PAFPN):
    @force_fp32()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            # fix runtime error of "+=" inplace operation in PyTorch 1.10
            laterals[i - 1] = laterals[i - 1] + F.interpolate(
                laterals[i], size=prev_shape, mode='nearest')

        # build outputs
        # part 1: from original levels
        inter_outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]

        # part 2: add bottom-up path
        for i in range(0, used_backbone_levels - 1):
            inter_outs[i + 1] += self.downsample_convs[i](inter_outs[i])

        outs = []
        outs.append(inter_outs[0])
        outs.extend([
            self.pafpn_convs[i - 1](inter_outs[i])
            for i in range(1, used_backbone_levels)
        ])

        # part 3: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[used_backbone_levels](orig))
                elif self.add_extra_convs == 'on_lateral':
                    outs.append(self.fpn_convs[used_backbone_levels](
                        laterals[-1]))
                elif self.add_extra_convs == 'on_output':
                    outs.append(self.fpn_convs[used_backbone_levels](outs[-1]))
                else:
                    raise NotImplementedError
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)


@BACKBONES.register_module()
class LSS(BaseModule):
    def __init__(self, x_bound, y_bound, z_bound, d_bound, final_dim,
                 downsample_factor, output_channels, img_backbone_conf,
                 img_neck_conf, depth_net_conf, seg_net_conf=None, queue_len=1, fpn_in_channels=[64, 128, 256, 512]):
        """Modified from `https://github.com/nv-tlabs/lift-splat-shoot`.
        Args:
            x_bound (list): Boundaries for x.
            y_bound (list): Boundaries for y.
            z_bound (list): Boundaries for z.
            d_bound (list): Boundaries for d.
            final_dim (list): Dimension for input images.
            downsample_factor (int): Downsample factor between feature map
                and input image.
            output_channels (int): Number of channels for the output
                feature map.
            img_backbone_conf (dict): Config for image backbone.
            img_neck_conf (dict): Config for image neck.
            depth_net_conf (dict): Config for depth net.
        """

        super(LSS, self).__init__()
        self.downsample_factor = downsample_factor
        self.fp16_enabled = False
        self.d_bound = d_bound
        self.final_dim = final_dim
        self.output_channels = output_channels
        self.queue_len = queue_len
        if self.queue_len!=1:
            self.bev_multiframe_merge = nn.Conv2d(256*queue_len,
                                     256,
                                     kernel_size=3,
                                     stride=1,
                                     padding=1,
                                     bias=False)
        self.register_buffer(
            'voxel_size',
            torch.Tensor([row[2] for row in [x_bound, y_bound, z_bound]]))
        self.register_buffer(
            'voxel_coord',
            torch.Tensor([
                row[0] + row[2] / 2.0 for row in [x_bound, y_bound, z_bound]
            ]))
        self.register_buffer(
            'voxel_num',
            torch.LongTensor([(row[1] - row[0]) / row[2]
                              for row in [x_bound, y_bound, z_bound]]))
        self.register_buffer('frustum', self.create_frustum())
        self.depth_channels, _, _, _ = self.frustum.shape

        self.img_backbone = build_backbone(img_backbone_conf)
        self.img_neck = build_neck(img_neck_conf)
        self.neck_conv = nn.Conv2d(in_channels=img_neck_conf["out_channels"], out_channels=depth_net_conf["in_channels"], kernel_size=1)
        self.depth_net = self._configure_depth_net(depth_net_conf)


        self.seg_net = self._configure_seg_net(seg_net_conf, fpn_in_channels)
        
        self.seg_res_to_image_feature =nn.Sequential(
                nn.Conv2d(seg_net_conf['out_channels'], 64, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.Conv2d(64, 16, 1),
                nn.BatchNorm2d(16),
                nn.ReLU(),

                nn.Conv2d(16, 32, 3, padding=1, stride=2),
                nn.BatchNorm2d(32),
                nn.ReLU(),


                nn.Conv2d(32, 32, 1),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                
                nn.Conv2d(32, 64, 3, padding=1, stride=2),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.Conv2d(64, 64, 1),
                nn.BatchNorm2d(64),
                nn.ReLU(),

                nn.Conv2d(64, 128, 3, padding=1, stride=2),
                nn.BatchNorm2d(128),
                nn.ReLU(),
            )
        self.merge_seg_and_image = nn.Conv2d(256+128, 256, 3, padding=1)
        self.img_neck.init_weights()
        self.img_backbone.init_weights()

    def _configure_depth_net(self, depth_net_conf):
        return DepthNet(
            depth_net_conf['in_channels'],
            depth_net_conf['mid_channels'],
            self.output_channels,
            self.depth_channels,
        )

    def _configure_seg_net(self, seg_net_conf, fpn_in_channels):
        return UNet(seg_net_conf['out_channels'], fpn_in_channels)

    def create_frustum(self):
        """Generate frustum"""
        # make grid in image plane
        ogfH, ogfW = self.final_dim
        fH, fW = ogfH // self.downsample_factor, ogfW // self.downsample_factor
        d_coords = torch.arange(*self.d_bound,
                                dtype=torch.float).view(-1, 1,
                                                        1).expand(-1, fH, fW)
        D, _, _ = d_coords.shape
        x_coords = torch.linspace(0, ogfW - 1, fW, dtype=torch.float).view(
            1, 1, fW).expand(D, fH, fW)
        y_coords = torch.linspace(0, ogfH - 1, fH,
                                  dtype=torch.float).view(1, fH,
                                                          1).expand(D, fH, fW)
        paddings = torch.ones_like(d_coords)
        # D x H x W x 3
        frustum = torch.stack((x_coords, y_coords, d_coords, paddings), -1)
        return frustum

    @force_fp32()
    def get_geometry(self, sensor2ego_mat, intrin_mat, ida_mat, bda_mat):
        """Transfer points from camera coord to ego coord.

        Args:
            rots(Tensor): Rotation matrix from camera to ego.
            trans(Tensor): Translation matrix from camera to ego.
            intrins(Tensor): Intrinsic matrix.
            post_rots_ida(Tensor): Rotation matrix for ida.
            post_trans_ida(Tensor): Translation matrix for ida
            post_rot_bda(Tensor): Rotation matrix for bda.

        Returns:
            Tensors: points ego coord.
        """
        batch_size, num_cams, _, _ = sensor2ego_mat.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum
        ida_mat = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4)
        ## ida_mat = aug_img2img
        ## img2aug_img
        points = ida_mat.inverse().matmul(points.unsqueeze(-1))     # B,N,D,H,W,4,1
        # cam_to_ego
        points = torch.cat(
            (points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],    # x/y * z 
             points[:, :, :, :, :, 2:]), 5)
       
        combine = sensor2ego_mat.matmul(torch.inverse(intrin_mat))
        # cam2lidar @ img2cam   --> img2lidar
        points = combine.view(batch_size, num_cams, 1, 1, 1, 4,
                              4).matmul(points)
        if bda_mat is not None:
            bda_mat = bda_mat.unsqueeze(1).repeat(1, num_cams, 1, 1).view(
                batch_size, num_cams, 1, 1, 1, 4, 4)
            points = (bda_mat @ points).squeeze(-1)
        else:
            points = points.squeeze(-1)
        return points[..., :3]

    @auto_fp16()
    def get_cam_feats(self, imgs):
        """Get feature maps from images."""
        batch_size, num_sweeps, num_cams, num_channels, imH, imW = imgs.shape
        imgs = imgs.flatten().view(batch_size * num_sweeps * num_cams,
                                   num_channels, imH, imW)
        fpn_feats = self.img_backbone(imgs)
        fpn_feats = self.img_neck(fpn_feats)

        img_feats = self.neck_conv_fp32(fpn_feats[2])
        img_feats = img_feats.reshape(batch_size, num_sweeps, num_cams,
                                      img_feats.shape[1], img_feats.shape[2],
                                      img_feats.shape[3])
        return img_feats, fpn_feats

    @force_fp32()
    def neck_conv_fp32(self, input):
        return self.neck_conv(input)

    @force_fp32()
    def _forward_depth_net(self, feat, mats_dict):
        return self.depth_net(feat, mats_dict)
    
    @auto_fp16()
    def _forward_voxel_net(self, img_feat_with_depth):
        return img_feat_with_depth

    @auto_fp16()
    def _forward_single_sweep(self,
                              sweep_index,
                              sweep_imgs,
                              mats_dict,
                              is_return_depth=False):
        """Forward function for single sweep.

        Args:
            sweep_index (int): Index of sweeps.
            sweep_imgs (Tensor): Input images.
            mats_dict (dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
            is_return_depth (bool, optional): Whether to return depth.
                Default: False.

        Returns:
            Tensor: BEV feature map.
        """
        batch_size, num_sweeps, num_cams, num_channels, img_height, \
            img_width = sweep_imgs.shape

        img_feats, fpn_feats = self.get_cam_feats(sweep_imgs)
        source_features = img_feats[:, 0, ...]
        source_features = source_features.reshape(batch_size * num_cams,
                                    source_features.shape[2],
                                    source_features.shape[3],
                                    source_features.shape[4])
        depth_feature = self._forward_depth_net(
            source_features,
            mats_dict,
        )
        depth = depth_feature[:, :self.depth_channels]
        depth_softmax = depth.softmax(1)

        img_feature = depth_feature[:, self.depth_channels:(self.depth_channels + self.output_channels)]

        ## Map 2D Segmentation to BEV as well 
        seg_output = self.seg_net(fpn_feats)    # 1/2 origin shape
        seg_feature = self.seg_res_to_image_feature_forward(seg_output.detach())# 1/16 origin shape
        img_feature = torch.cat((img_feature, seg_feature), dim=1)
        img_feature = self.merge_seg_and_image(img_feature)   # keep same dims

        img_feat_with_depth = depth_softmax.unsqueeze(1) * img_feature.unsqueeze(2)
        outs = {}
        outs["fpn_feats"] = fpn_feats
        img_feat_with_depth = self._forward_voxel_net(img_feat_with_depth)

        img_feat_with_depth = img_feat_with_depth.reshape(
            batch_size,
            num_cams,
            img_feat_with_depth.shape[1],
            img_feat_with_depth.shape[2],
            img_feat_with_depth.shape[3],
            img_feat_with_depth.shape[4],
        )
        geom_xyz = self.get_geometry(
            mats_dict['sensor2ego_mats'][:, sweep_index, ...],
            mats_dict['intrin_mats'][:, sweep_index, ...],
            mats_dict['ida_mats'][:, sweep_index, ...],
            mats_dict.get('bda_mat', None),
        )

        img_feat_with_depth = img_feat_with_depth.permute(0, 1, 3, 4, 5, 2)

        feature_map = self.voxel_pooling_method(geom_xyz, img_feat_with_depth.contiguous(), self.voxel_num.to(img_feat_with_depth.device))
        
        outs["bev"] = feature_map.contiguous()
        if is_return_depth:
            outs['depth'] = depth
        outs['seg'] = seg_output
        return outs
    
   
    @force_fp32()
    def seg_res_to_image_feature_forward(self, input):
        return self.seg_res_to_image_feature(input)

    @force_fp32()
    def voxel_pooling_method(self, geom_xyz, img_feat_with_depth, voxel_num):
        geom_xyz = ((geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) /
                    self.voxel_size).int()
        return voxel_pooling(geom_xyz, img_feat_with_depth, voxel_num)

    @auto_fp16()
    def forward(self,
                img,
                img_metas,
                timestamps=None,
                is_return_depth=False):
        """Forward function.
        Args:
            sweep_imgs(Tensor): Input images with shape of (B, num_sweeps,
                num_cameras, 3, H, W).
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from
                    camera to ego with shape of (B, num_sweeps,
                    num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape
                    of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix
                    from key frame camera to sweep frame camera with
                    shape of (B, num_sweeps, num_cameras, 4, 4).
            timestamps(Tensor): Timestamp for all images with the shape of(B,
                num_sweeps, num_cameras).

        Return:
            Tensor: bev feature map.
        """
        if img.dim() == 5:
            img = img.unsqueeze(1)    

        batch_size, num_sweeps, num_cams, num_channels, img_height, \
            img_width = img.shape

        intrins, ida, sensor2ego = [], [], []
        for b in range(len(img_metas)):
            _intrins, _ida, _sensor2ego = [], [], []
            for sweep_idx in range(len(img_metas[0])):
                mats = img_metas[b][sweep_idx]
                intrinsic_mat = torch.zeros((num_cams,4,4))
                intrinsic_mat[:,:3,:3] = mats['cam_intrinsic']
                intrinsic_mat[:,3,3] = 1
                _intrins.append(intrinsic_mat)
                _ida.append(mats['ida_mats'])   # Nx4x4
                _sensor2ego.append(mats['currlidar2keycam'].permute(0,2,1))     # cam2lidar
            intrins.append(torch.stack(_intrins))
            sensor2ego.append(torch.stack(_sensor2ego))
            ida.append(torch.stack(_ida))
        intrins, ida, sensor2ego = torch.stack(intrins), torch.stack(ida), torch.stack(sensor2ego)
        
        old_img_metas = img_metas
        img_metas = {}
        img_metas['intrin_mats'] = intrins.to(img.device)
        img_metas['sensor2ego_mats'] = sensor2ego.to(img.device)
        img_metas['ida_mats'] = ida.to(img.device)

        key_frame_res = self._forward_single_sweep(
            -1,
            img[:, -1:, ...],  
            img_metas,
            is_return_depth=is_return_depth)


        outs = {}
        bev_feature_list = [key_frame_res['bev']]
        if self.seg_net is not None:
            outs['seg'] = key_frame_res['seg']
        if is_return_depth:
            outs['depth'] = key_frame_res['depth']
        outs["fpn_feats"] = key_frame_res["fpn_feats"]

        current_ida = ida[:, -1, :, :].clone() #B, 4, 4, 4
        current_lidar2img = torch.stack([_[-1]["lidar2img"] for _ in old_img_metas], dim=0)
        outs['lidar2img'] = current_lidar2img
        outs['ida_mat'] = current_ida

        # multi-frame 
        for sweep_index in range(1, num_sweeps):
            with torch.no_grad():
                sweep_frame_res = self._forward_single_sweep(
                    -sweep_index,
                    img[:, -(sweep_index+1):-sweep_index, ...],
                    img_metas,
                    is_return_depth=False)
                bev_feature_list.append(sweep_frame_res['bev'])

        assert len(bev_feature_list)==self.queue_len, 'LSS.queue_len must be set correctly in config!'
        bev_feature = torch.cat(bev_feature_list, 1)
        if self.queue_len>1:
            bev_feature = self.bev_multiframe_merge(bev_feature)
        outs['bev'] = bev_feature
        return outs
        