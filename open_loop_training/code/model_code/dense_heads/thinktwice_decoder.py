import copy
from collections import OrderedDict
from distutils.errors import LibError
from .multi_scale_deformable_attn_function import SpatialCrossAttention
from .utils import SpatialGRU, sigmoid_focal_loss, init_weights
from mmcv.cnn import build_conv_layer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta

from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmdet.models import HEADS
from mmcv.runner import BaseModule

import numpy as np
import mmcv
import cv2 as cv
from mmcv.runner import force_fp32, auto_fp16

def inv_softplus(x):
    return x + torch.log(-torch.expm1(-x))


class PredictionModule(nn.Module):
    def __init__(self, act):
        super().__init__()
        self.act = act
        self.spatial_gru = SpatialGRU(input_size=6, hidden_size=32, act=self.act)
        ## Resiudal Update
        self.ffn = nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=1),
                self.act(),
                nn.Conv2d(64, 32, kernel_size=3, padding=1),
                self.act(),
                nn.Conv2d(32, 32, kernel_size=1),
            )
    
    @force_fp32()
    def forward(self, current_BEV_feature, current_wp, current_ctrl, future_bev_feat_from_last_layer):
        future_bev_feat = self.spatial_gru(torch.cat([current_wp, current_ctrl], dim=2).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, current_BEV_feature.shape[-2], current_BEV_feature.shape[-1]), state=current_BEV_feature).view(-1, current_BEV_feature.shape[1], current_BEV_feature.shape[2], current_BEV_feature.shape[3])
        ## Residual Update
        if future_bev_feat_from_last_layer is not None:
            current_future_bev_feat = self.ffn(future_bev_feat) + future_bev_feat_from_last_layer.view(-1, current_BEV_feature.shape[1], current_BEV_feature.shape[2], current_BEV_feature.shape[3])
        return future_bev_feat
     



class LookModule(nn.Module):
    def __init__(self, act):
        super().__init__()
        self.act = act
        self.cam_look_module = SpatialCrossAttention()
        self.lidar_look_module_atten = nn.Sequential(
                    nn.Linear(6+128, 256),
                    self.act(),
                    nn.Linear(256, 512),
                    nn.Sigmoid(),
                )
        self.lidar_look_module_MLP = nn.Sequential(
                    nn.Linear(512, 128),
                    self.act(),
                    nn.Flatten(start_dim=2),
                    nn.Linear(9*128, 256),
                    self.act(),
                )

        self.look_feature_MLP = nn.Sequential(
                nn.Linear(512*4, 512),
                self.act(),
                nn.Linear(512, 128),
            )
        self.point_cloud_range = [-8.0, -19.2, -4.0, 30.4, 19.2, 4.0]
        self.fp16_enabled = False
    
    @force_fp32()    
    def obtain_lidar_look_features(self, wp, lidar_grid):
        bs, T, xy = wp.size()
        relative_wp_x = 1.0 - torch.clamp(((wp[..., 0] - self.point_cloud_range[0])/(self.point_cloud_range[3]-self.point_cloud_range[0])).unsqueeze(-1) + torch.Tensor([0.0, -0.1, 0.1]).unsqueeze(0).unsqueeze(0).to(wp.device), min=0.0, max=1.0)
        relative_wp_y = torch.clamp(((wp[..., 1] - self.point_cloud_range[1])/(self.point_cloud_range[4]-self.point_cloud_range[1])).unsqueeze(-1)  + torch.Tensor([0.0, -0.1, 0.1]).unsqueeze(0).unsqueeze(0).to(wp.device), min=0.0, max=1.0)
        relative_wp = torch.stack([relative_wp_x.unsqueeze(-1).repeat(1, 1, 1, 3), relative_wp_y.unsqueeze(-2).repeat(1, 1, 3, 1)], dim=-1).view(bs*T, -1, 1, 2) * 2 - 1
        sampled_feat = F.grid_sample(input=lidar_grid.view(bs*T, *lidar_grid.shape[2:]), grid=relative_wp, align_corners=False).view(bs, T, -1, 9).transpose(2, 3)
        return sampled_feat

    @force_fp32()
    def obtain_cam_ref_points_query(self, reference_points, transform_mat, img_shape, query, mlvl_feats):
        reference_points = torch.cat((reference_points, torch.ones_like(reference_points[..., :1])), -1) # reference_points (B, num_queries, 4)

        B, num_query = reference_points.size()[:2]
        num_cam = transform_mat[0].size(1)
        reference_points = reference_points.view(B, 1, num_query, 4).repeat(1, num_cam, 1, 1).unsqueeze(-1)
        lidar2img = transform_mat[0].view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)
        ida_mat = transform_mat[1].view(B, num_cam, 1, 4, 4).repeat(1, 1, num_query, 1, 1)
        reference_points_cam = torch.matmul(lidar2img, reference_points).squeeze(-1)
        eps = 1e-5
        new_reference_points_cam = reference_points_cam.clone()
        new_reference_points_cam[..., 0:2] = reference_points_cam[..., 0:2] / torch.maximum(
            reference_points_cam[..., 2:3], torch.ones_like(reference_points_cam[..., 2:3])*eps)
        reference_points_cam = torch.matmul(ida_mat, new_reference_points_cam.unsqueeze(-1)).squeeze(-1)
        mask = (reference_points_cam[..., 2:3] > eps)

        reference_points_cam = reference_points_cam[..., :2]

        reference_points_cam[..., 0] /= img_shape[1]
        reference_points_cam[..., 1] /= img_shape[0]#B x Cam x Num_query x 2
        mask = (mask & (reference_points_cam[..., 1:2] > 0.0)
                        & (reference_points_cam[..., 1:2] < 1.0)
                        & (reference_points_cam[..., 0:1] < 1.0)    
                        & (reference_points_cam[..., 0:1] > 0.0))
        mask = mask.view(B, num_cam, num_query)
        mask = torch.nan_to_num(mask)
        mask = mask.permute(1, 0, 2)
        sampled_feats = []
        
        N = 4
        reference_points_cam_lvl = reference_points_cam.view(B*N, num_query, 1, 2) * 2 -1.0
        for lvl, feat in enumerate(mlvl_feats):
            BN, C, H, W = feat.size()
            B = int(BN/N)
            #feat = feat.view(B*N, C, H, W)
            sampled_feat = F.grid_sample(feat, reference_points_cam_lvl, align_corners=False)
            sampled_feat = sampled_feat.view(B, N, C, num_query, 1).permute(0, 2, 3, 1, 4)
            sampled_feats.append(sampled_feat)
        sampled_feats = torch.stack(sampled_feats, -1)
        sampled_feats = sampled_feats.view(B, C, num_query, num_cam, len(mlvl_feats)).permute(0, 2, 3, 1, 4).reshape(B, num_query, num_cam, -1)
        
        indexes = []
        # From bev_mask of each image, to obtain the available queries' index
        for i, mask_per_img in enumerate(mask):
            tmp_index = []
            for j in range(B):
                index_query_per_img = mask_per_img[j].nonzero().squeeze(-1)
                tmp_index.append(index_query_per_img)
            indexes.append(tmp_index)
        max_len = max([len(each_sample) for each_cam in indexes for each_sample in each_cam])
        bs, num_query, query_emb = query.shape
        num_cams = 4
        queries_rebatch = query.new_zeros([bs * num_cams, max_len, query_emb+1024])
        reference_points_rebatch = reference_points_cam.new_zeros([bs * num_cams, max_len, 2]) 

        reference_points_cam = reference_points_cam.permute(1, 0, 2, 3)
        for i, reference_points_per_img in enumerate(reference_points_cam):
            for j in range(bs):
                index_query_per_img = indexes[i][j]
                if len(index_query_per_img) != 0:
                    queries_rebatch[j * num_cams + i, :len(index_query_per_img)] = torch.cat([query[j, index_query_per_img], sampled_feats[j, index_query_per_img, i]], dim=-1)
                    reference_points_rebatch[j * num_cams + i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]
        return reference_points_rebatch.view(bs, num_cams, max_len, 2), queries_rebatch.view(bs, num_cams, max_len, query_emb+1024), indexes
    

    @force_fp32()
    def forward(self, current_wp, current_ctrl_softplus, measurement_feat, flattened_feat, coor2img, img_size, mlvl_feats, fpn_feat_flatten, spatial_shapes, level_start_index, lidar_feat_with_high_resolution, temporal_embedding, static_embedding):
        # Look Module - Image
        ## Avoid empty points situation which could cause bugs in code 
        static_point = torch.Tensor([[5.0, 0.0], [0.0, -5.0], [0.0, 5.0], [-5.0, 0.0],]).unsqueeze(0).repeat(current_wp.shape[0], 1, 1).to(current_wp.device)
        look_wp = torch.cat([current_wp, static_point], dim=1)
        ### Add z to 2D BEV coordinate uniformly
        look_wp_3d = torch.cat([look_wp.unsqueeze(2).repeat(1, 1, 15, 1), torch.linspace(-4, 10, 15, dtype=float, device=look_wp.device).unsqueeze(0).unsqueeze(0).unsqueeze(-1).repeat(look_wp.shape[0], look_wp.shape[1], 1, 1)], dim=-1).view(look_wp.shape[0], -1, 3).to(look_wp.device, look_wp.dtype) # B x T x z_level x 3 -> B x T*z_level x 3
        input_ctrl = torch.cat([current_ctrl_softplus.unsqueeze(2).repeat(1, 1, 15, 1).view(current_ctrl_softplus.shape[0], -1, 4), torch.zeros(current_wp.shape[0], 4*15, 4).to(current_wp.device)], dim=1) ## No control for static points

        ## For deformable Attn
        img_query = torch.cat([
            input_ctrl, ## predicted control 
            look_wp_3d, ## predict traj
            torch.cat([temporal_embedding.unsqueeze(0).unsqueeze(-2).repeat(current_wp.shape[0], 1, 15, 1).view(current_wp.shape[0], -1, temporal_embedding.shape[-1]), ## Temporal Emeddding
            static_embedding.unsqueeze(0).unsqueeze(-2).repeat(current_wp.shape[0], 1, 15, 1).view(current_wp.shape[0], -1, temporal_embedding.shape[-1])], dim=1), ## Static Point Embedding
            measurement_feat.unsqueeze(1).repeat(1, look_wp_3d.shape[1], 1), 
            flattened_feat.unsqueeze(1).repeat(1, look_wp_3d.shape[1], 1)
            ], dim=-1) #B x T*z_level * 4 + 3 + 128 + 128 + 256
        
        ## Transform into the desired format for the deformable attention
        ref_points_cam_rebatch, query_rebatch, indexes_rebatch = self.obtain_cam_ref_points_query(look_wp_3d, coor2img, img_size, img_query, mlvl_feats)
        img_look_features = self.cam_look_module(query=query_rebatch, value=fpn_feat_flatten, reference_points=ref_points_cam_rebatch, spatial_shapes=spatial_shapes, level_start_index=level_start_index, indexes=indexes_rebatch)
        img_look_features = img_look_features.unsqueeze(1).repeat(1, temporal_embedding.shape[0], 1)

        ## Look Module - Lidar
        lidar_attn_weight = self.lidar_look_module_atten(torch.cat([current_wp, current_ctrl_softplus, temporal_embedding.unsqueeze(0).repeat(current_wp.shape[0], 1, 1)], dim=-1)) ## Different feature maps for different control signals and time-steps

        temporal_lidar_feat = lidar_attn_weight.unsqueeze(-1).unsqueeze(-1) * lidar_feat_with_high_resolution.unsqueeze(1).float()
        ## Retrieve Lidar features
        lidar_look_features = self.obtain_lidar_look_features(current_wp, temporal_lidar_feat)
        lidar_look_features = self.lidar_look_module_MLP(lidar_look_features)

        look_features = torch.cat([img_look_features, torch.zeros_like(lidar_look_features)], dim=-1)
        return look_features

class ThinkTwiceDecoderLayer(nn.Module):
    def __init__(self, act):
        super().__init__()
        self.act = act
        self.prediction_module = PredictionModule(self.act)
        self.look_module = LookModule(self.act)

        self.mlp = nn.Sequential(
                            nn.LayerNorm(256+128+512+128),
                            ## flattened prediction BEV feat 256  + look feature 512 + temporal embedding 128 + measurement
                            nn.Linear(256+128+512+128, 512),
                            self.act(),
                            nn.Dropout(0.0),
                            nn.Linear(512, 512),
                            self.act(),
                        )
        
        self.traj_offset_module = nn.Sequential(
                            nn.Linear(512+2, 256),
                            self.act(),
                            nn.Linear(256, 64),
                            self.act(),
                            nn.Linear(64, 2),
                        )
        self.ctrl_offset_module =  nn.Sequential(
                            nn.Linear(512+4, 256),
                            self.act(),
                            nn.Linear(256, 64),
                            self.act(),
                            nn.Linear(64, 4),
                        )
        
        self.BEV_feat_update_module = nn.Sequential(
                nn.Conv2d(512*4+32, 128, kernel_size=3, padding=1),
                self.act(),
                nn.Conv2d(128, 32, kernel_size=3, padding=1),
            )
        
        self.flattened_BEV_feat_update_module = nn.Sequential(
                nn.Linear(256+512*4, 512),
                self.act(),
                nn.Linear(512, 256),
            )
        self.fp16_enabled = False


    @force_fp32()
    def forward(self, BEV_feat, current_wp, current_ctrl, future_bev_feat_from_last_layer, parent_module, grid2feat, measurement_feat, flattened_BEV_feat, coor2img, img_size, mlvl_feats, fpn_feat_flatten, spatial_shapes, level_start_index, lidar_feat_with_high_resolution, temporal_embedding, static_embedding):
        current_ctrl_softplus = F.softplus(current_ctrl)

        ## Prediction Module
        future_bev_feat = self.prediction_module(current_BEV_feature=BEV_feat, current_wp=current_wp, current_ctrl=current_ctrl_softplus, future_bev_feat_from_last_layer=future_bev_feat_from_last_layer)

        flattened_future_BEV_feat, _ = grid2feat(future_bev_feat, parent_module)
        flattened_future_BEV_feat = flattened_future_BEV_feat.view(BEV_feat.shape[0], 4, 256) ## Pred_len = 4
        future_bev_feat = future_bev_feat.view(BEV_feat.shape[0], -1, BEV_feat.shape[1], BEV_feat.shape[2], BEV_feat.shape[3])

        ## Look Module
        look_features = self.look_module(
            current_wp=current_wp, current_ctrl_softplus=current_ctrl_softplus, measurement_feat=measurement_feat, flattened_feat=flattened_BEV_feat, coor2img=coor2img, img_size=img_size, mlvl_feats=mlvl_feats, fpn_feat_flatten=fpn_feat_flatten, spatial_shapes=spatial_shapes, level_start_index=level_start_index, lidar_feat_with_high_resolution=lidar_feat_with_high_resolution, temporal_embedding=temporal_embedding, static_embedding=static_embedding
        )

        all_future_feat = self.mlp(torch.cat([flattened_future_BEV_feat, look_features, temporal_embedding.unsqueeze(0).repeat(flattened_future_BEV_feat.shape[0], 1, 1), measurement_feat.unsqueeze(1).repeat(1, flattened_future_BEV_feat.shape[1], 1)], dim=-1))

        traj_offset =self.traj_offset_module(torch.cat([current_wp, all_future_feat], dim=-1))
        ctrl_offset = self.ctrl_offset_module(torch.cat([current_ctrl, all_future_feat], dim=-1))

        ## Residual update of feature maps similar to DETR
        updated_BEV_feat = self.BEV_feat_update_module(torch.cat([BEV_feat, all_future_feat.view(BEV_feat.shape[0], -1).unsqueeze(-1).unsqueeze(-1).repeat(1, 1, BEV_feat.shape[2], BEV_feat.shape[3])], dim=1)) + BEV_feat

        updated_flattend_BEV_feat = self.flattened_BEV_feat_update_module(torch.cat([flattened_BEV_feat, all_future_feat.view(flattened_BEV_feat.shape[0], -1)], dim=-1)) + flattened_BEV_feat
        return traj_offset, ctrl_offset, future_bev_feat, updated_BEV_feat, updated_flattend_BEV_feat

@HEADS.register_module()
class ThinkTwiceDecoder(BaseModule):
    def __init__(self,
                 *args,
                 config=None,
                 bev_h=None,
                 bev_w=None,
                 BEV_feat_dim=256, ## BEVfFeature dimension
                 flattened_BEV_feat_dim=256, ## Flattened 1D feature dimension
                 **kwargs):

        super(ThinkTwiceDecoder, self).__init__(
            *args, **kwargs)
        
        self.config = config
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.BEV_feat_dim = BEV_feat_dim
        self.flattened_BEV_feat_dim = flattened_BEV_feat_dim
        self.act = nn.ReLU

        #For distil
        self.distil_index = [2, 3, 4, 5] ## All layers
        self.distil_kl_loss_weight_dict = {2:0.25, 3:1.0/3.0, 4:1.0/4.0, 5:1/11.0}
        self.wp_loss_weight = 15.0
        self.action_loss_weight = 15.0
        
        self.build_layers()
        self.fp16_enabled = False

    def build_layers(self):
        self.build_coarse_output_layer()
        self.build_refine_layers()
        self.apply(init_weights)

    ## Similar to TCP, give 1D feature map, predict 1D traj and control
    def build_coarse_output_layer(self):
        self.join_traj = nn.Sequential(
                            nn.Linear(128+self.flattened_BEV_feat_dim, 512),
                            self.act(),
                            nn.Linear(512, 512),
                            self.act(),
                            nn.Linear(512, 256),
                            self.act(),
                        )
        self.output_traj = nn.Sequential(
                nn.Linear(256, 512),
                self.act(),
                nn.Linear(512, 2*self.config.pred_len),
            )
        self.join_ctrl = nn.Sequential(
                            nn.Linear(128+self.flattened_BEV_feat_dim, 512),
                            self.act(),
                            nn.Linear(512, 512),
                            self.act(),
                            nn.Linear(512, 256),
                            self.act(),
                        )
        self.speed_branch = nn.Sequential(
                            nn.Linear(self.flattened_BEV_feat_dim, 256),
                            self.act(),
                            nn.Linear(256, 256),
                            self.act(),
                            nn.Linear(256, 1),
                        )

        self.value_branch_traj = nn.Sequential(
                    nn.Linear(256, 256),
                    self.act(),
                    nn.Linear(256, 256),
                    self.act(),
                    nn.Linear(256, 1),
                )
        self.value_branch_ctrl = nn.Sequential(
                    nn.Linear(256, 256),
                    self.act(),
                    nn.Linear(256, 256),
                    self.act(),
                    nn.Linear(256, 1),
                )
        # shared branches_neurons
        self.dim_out = 2

        self.policy_head = nn.Sequential(
                nn.Linear(256, 512),
                self.act(),
                nn.Linear(512, 512),
                self.act(),
            )
        self.dist_mu = nn.Sequential(
            nn.Linear(512, 512), 
            self.act(),
            nn.Linear(512, self.dim_out * self.config.pred_len),
            ) ## No softplus here!
        self.dist_sigma = nn.Sequential(
            nn.Linear(512, 512), 
            self.act(),
            nn.Linear(512, self.dim_out * self.config.pred_len),
            ) ## No softplus here!
        

    def build_refine_layers(self):
        ## Transform all FPN feature to the same dimension to conduct multi-scale deformable attention
        self.fpn_linear0 = nn.Conv2d(in_channels=self.config["FPN_out_channels"][0], out_channels=256, kernel_size=1)
        self.fpn_linear1 = nn.Conv2d(in_channels=self.config["FPN_out_channels"][1], out_channels=256, kernel_size=1)
        self.fpn_linear2 = nn.Conv2d(in_channels=self.config["FPN_out_channels"][2], out_channels=256, kernel_size=1)
        self.fpn_linear3 = nn.Conv2d(in_channels=self.config["FPN_out_channels"][3], out_channels=256, kernel_size=1)
        self.temporal_embedding = nn.parameter.Parameter(torch.zeros(self.config.pred_len, 128)) ## Embedding for different predicted time-step
        self.cams_embeds = nn.parameter.Parameter(torch.zeros(4, 256))
        self.static_embedding = nn.parameter.Parameter(torch.zeros(4, 128))
        self.level_embeds = nn.parameter.Parameter(torch.zeros(4, 256)) ## For multi-scale attention
        nn.init.trunc_normal_(self.temporal_embedding, mean=0, std=0.02)
        nn.init.trunc_normal_(self.cams_embeds, mean=0, std=0.02)
        nn.init.trunc_normal_(self.static_embedding, mean=0, std=0.02)
        nn.init.trunc_normal_(self.temporal_embedding, mean=0, std=0.02)

        self.decoder_layers = nn.ModuleList([ThinkTwiceDecoderLayer(self.act) for _ in range(self.config["refine_num"])])
    
    @force_fp32()
    def transform_fpn_feats(self, mlvl_feats):
        spatial_shapes = []
        feat_flatten = []
        """  flatten feats of each level """
        for lvl, feat in enumerate(mlvl_feats):
            bs_num_cam, c, h, w = feat.shape
            num_cam = 4
            bs = int(bs_num_cam/num_cam)
            feat = feat.view(bs, num_cam, c, h, w)
            spatial_shape = (h, w)
            feat = feat.flatten(3).permute(1, 0, 3, 2)  
            feat = feat + self.cams_embeds[:, None, None, :].to(feat.dtype)
            feat = feat + self.level_embeds[None, None, lvl:lvl + 1, :].to(feat.dtype)
            spatial_shapes.append(spatial_shape)
            feat_flatten.append(feat)
        """ concat flattened features """
        feat_flatten = torch.cat(feat_flatten, 2)  # (cam, bs, sum(h*w), 256)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=mlvl_feats[0].device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        fpn_feat_flatten = feat_flatten.permute(0, 2, 1, 3)  # (num_cam, sum(H*W), bs, embed_dims)
        return spatial_shapes, level_start_index, fpn_feat_flatten

    ## Convert 32x21x21 BEV feature to 256 dim 1D feature by the shared flatten module
    @auto_fp16()
    def grid2feat(self, grid_feat, parent_module):
        mid_feature = [None, None] #Currently no feature for 8x94x94 16x45x45
        mid_feature.append(grid_feat)
        feats64_10_10 = parent_module.MLP10(parent_module.inplace_act(parent_module.conv21_10(grid_feat)))
        mid_feature.append(feats64_10_10)
        feats128_4_4 = parent_module.MLP4(parent_module.inplace_act(parent_module.conv10_4(feats64_10_10)))
        mid_feature.append(feats128_4_4)
        feats256_2_2 = parent_module.MLP2(parent_module.inplace_act(parent_module.conv4_2((feats128_4_4))))
        mid_feature.append(feats256_2_2)
        flattened_BEV_feat = parent_module.output_fc(feats256_2_2.flatten(start_dim=1))
        return flattened_BEV_feat, mid_feature
    

    @force_fp32()
    def forward(self, flattend_BEV_feat, BEV_feat, measurement_feat, target_point, parent_module, teacher_forcing_data=None, look_feature_metadata=None):
        outs = {}
        outs['bev_feature'] = BEV_feat
        ## Coarase prediction similar to TCP
        ### Predict speed by sensor feature to alleviate copy cat problem
        outs['pred_speed'] = self.speed_branch(flattend_BEV_feat)
        ### Coarse Traj Branch
        j_traj = self.join_traj(torch.cat([flattend_BEV_feat, measurement_feat], 1))
        outs['pred_value_traj'] = self.value_branch_traj(j_traj)
        outs['pred_features_traj'] = j_traj
        pred_wp  = self.output_traj(j_traj).view(-1, self.config.pred_len, 2)
        pred_wp_lis = [pred_wp]
        ### Coarse Ctrl Branch
        j_ctrl = self.join_ctrl(torch.cat([flattend_BEV_feat, measurement_feat], dim=-1))
        outs['pred_value_ctrl'] = self.value_branch_ctrl(j_ctrl)
        outs['pred_features_ctrl'] = j_ctrl
        policy = self.policy_head(j_ctrl)
        predicted_mu = self.dist_mu(policy).view(-1, self.config.pred_len, self.dim_out)
        predicted_sigma = self.dist_sigma(policy).view(-1, self.config.pred_len, self.dim_out)
        pred_ctrl_lis = [torch.cat([predicted_mu, predicted_sigma], dim=-1)]

        ## Look Module
        ### Prepare features for Look Module - Image
        lidar2img = look_feature_metadata[0].to(flattend_BEV_feat.device)
        ida_mat = look_feature_metadata[1].to(flattend_BEV_feat.device)
        fpn_feats = look_feature_metadata[2]
        mlvl_feats = []
        mlvl_feats.append(self.fpn_linear0(fpn_feats[0]))
        mlvl_feats.append(self.fpn_linear1(fpn_feats[1]))
        mlvl_feats.append(self.fpn_linear2(fpn_feats[2]))
        mlvl_feats.append(self.fpn_linear3(fpn_feats[3]))
        spatial_shapes, level_start_index, fpn_feat_flatten = self.transform_fpn_feats(mlvl_feats)
        ### Prepare features for Look Module - Lidar
        lidar_feat_with_high_resolution = look_feature_metadata[3]


        future_bev_feat = None
        current_BEV_feat = BEV_feat.clone()
        current_flattened_BEV_feat = flattend_BEV_feat.clone()
        stored_BEV_feat = []
        stored_flattened_BEV_feat = []
        stored_future_BEV_feat = []
        for refine_layer_index in range(self.config["refine_num"]):
            current_wp = pred_wp_lis[-1].detach()
            current_ctrl = pred_ctrl_lis[-1].detach()

            traj_offset, ctrl_offset, updated_future_bev_feat, updated_BEV_feat, updated_flattend_BEV_feat = self.decoder_layers[refine_layer_index](BEV_feat=current_BEV_feat, current_wp=current_wp, current_ctrl=current_ctrl, future_bev_feat_from_last_layer=future_bev_feat, parent_module=parent_module, grid2feat=self.grid2feat, measurement_feat=measurement_feat, flattened_BEV_feat=current_flattened_BEV_feat, coor2img=[lidar2img, ida_mat], img_size=self.config["img_size"], mlvl_feats=mlvl_feats, fpn_feat_flatten=fpn_feat_flatten, spatial_shapes=spatial_shapes, level_start_index=level_start_index, lidar_feat_with_high_resolution=lidar_feat_with_high_resolution, temporal_embedding=self.temporal_embedding, static_embedding=self.static_embedding)

            pred_wp_lis.append(traj_offset.float()+current_wp.float())
            pred_ctrl_lis.append(ctrl_offset.float()+current_ctrl.float())

            stored_BEV_feat.append(updated_BEV_feat)
            current_BEV_feat = updated_BEV_feat

            stored_flattened_BEV_feat.append(updated_flattend_BEV_feat)
            current_flattened_BEV_feat = updated_flattend_BEV_feat

            stored_future_BEV_feat.append(updated_future_bev_feat)
            future_bev_feat = updated_future_bev_feat
        
        outs["refine_flattned_BEV_feature"] = torch.stack(stored_flattened_BEV_feat, dim=1)
        outs["refine_BEV_feature"] = torch.stack(stored_BEV_feat, dim=1)
        outs["refine_future_BEV_feature"] = torch.stack(stored_future_BEV_feat, dim=1).view(current_wp.shape[0], current_wp.shape[1], self.config["refine_num"], 32, 21, 21).transpose(1, 2)

        pred_wp_lis = torch.stack(pred_wp_lis, dim=1) #BxrefinexTx2
        pred_ctrl_lis = torch.clamp(F.softplus(torch.stack(pred_ctrl_lis, dim=1).float()), min=1e-3) #BxrefinexTx4
        outs['pred_wp'] = pred_wp_lis
        outs['mu_branches'] = pred_ctrl_lis[:, :, 0, :2]
        outs['sigma_branches'] = pred_ctrl_lis[:, :, 0, 2:]
        outs['future_mu'] = pred_ctrl_lis[:, :, 1:, :2]
        outs['future_sigma'] = pred_ctrl_lis[:, :, 1:, 2:]

        ## Apply teacher forcing
        if teacher_forcing_data is not None:
            teacher_pred_traj_offset_lis = []
            teacher_pred_ctrl_offset_lis = []
            ##GT input should obtain GT future feature
            current_wp = teacher_forcing_data["waypoints"].float()

            current_ctrl_softplus = torch.cat([torch.cat([teacher_forcing_data["action_mu"], teacher_forcing_data["action_sigma"]], dim=-1).unsqueeze(1), torch.cat([torch.stack(teacher_forcing_data["future_action_mu"][:-1], dim=1), torch.stack(teacher_forcing_data["future_action_sigma"][:-1], dim=1)], dim=-1)], dim=1).float()

            current_ctrl = inv_softplus(current_ctrl_softplus)

            future_bev_feat = None
            current_BEV_feat = BEV_feat.clone()
            current_flattened_BEV_feat = flattend_BEV_feat.clone()
            stored_BEV_feat = []
            stored_flattened_BEV_feat = []
            stored_future_BEV_feat = []
            for refine_layer_index in range(self.config["refine_num"]):
                traj_offset, ctrl_offset, updated_future_bev_feat, updated_BEV_feat, updated_flattend_BEV_feat = self.decoder_layers[refine_layer_index](BEV_feat=current_BEV_feat, current_wp=current_wp, current_ctrl=current_ctrl, future_bev_feat_from_last_layer=future_bev_feat, parent_module=parent_module, grid2feat=self.grid2feat, measurement_feat=measurement_feat, flattened_BEV_feat=current_flattened_BEV_feat, coor2img=[lidar2img, ida_mat], img_size=self.config["img_size"], mlvl_feats=mlvl_feats, fpn_feat_flatten=fpn_feat_flatten, spatial_shapes=spatial_shapes, level_start_index=level_start_index, lidar_feat_with_high_resolution=lidar_feat_with_high_resolution, temporal_embedding=self.temporal_embedding, static_embedding=self.static_embedding)

                teacher_pred_traj_offset_lis.append(traj_offset)
                teacher_pred_ctrl_offset_lis.append(ctrl_offset)

                stored_BEV_feat.append(updated_BEV_feat)
                current_BEV_feat = updated_BEV_feat

                stored_flattened_BEV_feat.append(updated_flattend_BEV_feat)
                current_flattened_BEV_feat = updated_flattend_BEV_feat

                stored_future_BEV_feat.append(updated_future_bev_feat)
                future_bev_feat = updated_future_bev_feat
                
            teacher_pred_traj_offset_lis = torch.stack(teacher_pred_traj_offset_lis, dim=1)
            teacher_pred_ctrl_offset_lis = torch.stack(teacher_pred_ctrl_offset_lis, dim=1)
            teacher_future_BEV_lis = torch.stack(stored_future_BEV_feat, dim=1)

            outs["teacher_pred_wp_offset"] = teacher_pred_traj_offset_lis
            outs["teacher_pred_ctrl_offset_lis"] = teacher_pred_ctrl_offset_lis
            outs["teacher_future_BEV_feature"] = teacher_future_BEV_lis

            outs["teacher_refine_flattned_BEV_feature"] = torch.stack(stored_flattened_BEV_feat, dim=1)
            outs["teacher_refine_BEV_feature"] = torch.stack(stored_BEV_feat, dim=1)
        return outs

    @force_fp32()
    def loss(self,
             batch,
             pred,
             mid_BEV_feature, indices_with_gradient=None,
            ):
        loss_dict = dict()

        gt_speed = batch['speed'].to(dtype=torch.float32).view(-1,1) / 12.
        gt_value = batch['value'].view(-1,1)
        gt_flattened_BEV_feature = batch['feature']
        gt_waypoints = batch['waypoints']

        ## Open Loop Statistics
        with torch.no_grad():
            pred_action = self._get_action_beta(pred['mu_branches'][:, -1, :], pred['sigma_branches'][:, -1, :])
            gt_action = self._get_action_beta(batch['action_mu'], batch['action_sigma'])
            l1_action = F.l1_loss(pred_action, gt_action, reduction="none").detach().mean(dim=0)
            loss_dict["current_throttle_brake_offset"] = l1_action[0]
            loss_dict['current_steer_offset'] = l1_action[1]
            wp_offset = F.l1_loss(pred['pred_wp'][:, -1, :, :], gt_waypoints[:, :, :],reduction="none").detach().mean(dim=0)
            mean_wp_offset = wp_offset.mean(dim=0) 
            loss_dict["longitudinal_offset"] = mean_wp_offset[0] 
            loss_dict["lateral_offset"] = mean_wp_offset[1]
            loss_dict["longitudinal_offset"] = mean_wp_offset[0] 
            loss_dict["lateral_offset"] = mean_wp_offset[1]

        ## Current Action Loss
        gt_ctrl_distribution = Beta(batch['action_mu'].unsqueeze(1), batch['action_sigma'].unsqueeze(1))
        pred_ctrl_disttribution = Beta(pred['mu_branches'], pred['sigma_branches'])
        kl_div = torch.distributions.kl_divergence(gt_ctrl_distribution, pred_ctrl_disttribution)
        loss_dict['action_loss'] = kl_div.mean() * self.action_loss_weight


        loss_dict['speed_loss'] = F.smooth_l1_loss(pred['pred_speed'], gt_speed, reduction="mean")
        loss_dict['value_loss'] = (F.smooth_l1_loss(pred['pred_value_traj'], gt_value, reduction="mean") + F.smooth_l1_loss(pred['pred_value_ctrl'], gt_value, reduction="none")) * self.config.value_weight

        loss_dict['flattened_feature_loss'] = (F.smooth_l1_loss(pred['pred_features_traj'], gt_flattened_BEV_feature, reduction="mean") + F.smooth_l1_loss(pred['pred_features_ctrl'], gt_flattened_BEV_feature, reduction="mean")) * self.config.features_weight

        ## Future Action Loss
        gt_future_action_mu = torch.stack(batch['future_action_mu'][:-1], axis=1).unsqueeze(1)
        gt_future_action_sigma = torch.stack(batch['future_action_sigma'][:-1], axis=1).unsqueeze(1)
        gt_future_ctrl_distribution = Beta(gt_future_action_mu, gt_future_action_sigma)
        pred_future_ctrl_distribution = Beta(pred['future_mu'], pred['future_sigma'])
        future_kl_div = torch.distributions.kl_divergence(gt_future_ctrl_distribution, pred_future_ctrl_distribution)
        loss_dict['future_action_loss'] = future_kl_div.mean() * self.action_loss_weight * 0.25

        ## Traj Loss
        wp_loss = F.smooth_l1_loss(pred['pred_wp'], gt_waypoints.unsqueeze(1).repeat(1, pred['pred_wp'].shape[1], 1, 1), reduction="mean")
        loss_dict['wp_loss'] = wp_loss * self.wp_loss_weight
        
        ## Feature Disillation Loss
        ## BEV Feature Loss of Encoder
        for feature_map_index in self.distil_index:
            gt_BEV_feature =  batch["grid_feature"][feature_map_index]
            pred_BEV_feature = mid_BEV_feature[feature_map_index]
            loss_dict['BEV_feature_loss'+str(feature_map_index)] = torch.clamp(F.smooth_l1_loss(pred_BEV_feature, gt_BEV_feature, reduction="none"), min=-5.0, max=5.0).mean() * self.distil_kl_loss_weight_dict[feature_map_index]
        
        ## BEV Feature Loss of Look Module
        pred_BEV_feature = pred["refine_BEV_feature"]
        gt_BEV_feature =  batch["grid_feature"][2].unsqueeze(1).repeat(1, pred_BEV_feature.shape[1], 1, 1, 1) ## 21x21 BEV feature map
        loss_dict['refine_BEV_feature_loss2'] =  torch.clamp(F.smooth_l1_loss(pred_BEV_feature, gt_BEV_feature, reduction="none"), min=-5.0, max=5.0).mean() * self.distil_kl_loss_weight_dict[2]

        loss_dict['refine_flattened_feature_loss'] = torch.clamp(F.smooth_l1_loss(pred['refine_flattned_BEV_feature'], gt_flattened_BEV_feature.unsqueeze(1).repeat(1, pred['refine_flattned_BEV_feature'].shape[1], 1), reduction="none"), min=-5.0, max=5.0).mean() * self.config.features_weight * 0.1

        # Teacher Forcing Part
        ## All offset should be zero
        loss_dict['teacher_wp_loss'] = F.smooth_l1_loss(pred['teacher_pred_wp_offset'], torch.zeros_like(pred['teacher_pred_wp_offset']), reduction='mean')
        loss_dict['teacher_action_loss'] = F.smooth_l1_loss(pred['teacher_pred_ctrl_offset_lis'], torch.zeros_like(pred['teacher_pred_ctrl_offset_lis']), reduction='mean')

        ## Future Feature Supervision
        feature_map_index = 2 ## 21x21 feature map
        gt_future_BEV_feature = torch.stack([_[feature_map_index] for _ in batch["future_grid_feature"]], dim=1)
        pred_future_BEV_feature = pred["teacher_future_BEV_feature"]
        ## NumSamples, RefineNum, TemporalStep, Channel, Width, Height
        N,R,T,C,W,H = pred_future_BEV_feature.shape
        gt_future_BEV_feature = gt_future_BEV_feature.unsqueeze(1).repeat(1, R, 1, 1, 1, 1)
        loss_dict['teacher_future_BEV_feature_loss'+str(feature_map_index)] = torch.clamp(F.smooth_l1_loss(pred_future_BEV_feature, gt_future_BEV_feature, reduction="none"), min=-5.0, max=5.0).mean() * self.distil_kl_loss_weight_dict[2]

        ## BEV Feature Loss of Look Module
        pred_BEV_feature = pred["teacher_refine_BEV_feature"]
        gt_BEV_feature =  batch["grid_feature"][2].unsqueeze(1).repeat(1, pred_BEV_feature.shape[1], 1, 1, 1) ## 21x21 BEV feature map
        loss_dict['teacher_refine_BEV_feature_loss'+str(feature_map_index)] = torch.clamp(F.smooth_l1_loss(pred_BEV_feature, gt_BEV_feature, reduction="none"), min=-5.0, max=5.0).mean() * self.distil_kl_loss_weight_dict[2]

        loss_dict['teacher_refine_flattened_feature_loss'] = torch.clamp(F.smooth_l1_loss(pred['teacher_refine_flattned_BEV_feature'], gt_flattened_BEV_feature.unsqueeze(1).repeat(1, pred['teacher_refine_flattned_BEV_feature'].shape[1], 1), reduction="none"), min=-5.0, max=5.0).mean() * self.config.features_weight
        return loss_dict
    
    @force_fp32()
    def _get_action_beta(self, alpha, beta):
        x = torch.zeros_like(alpha)
        x[:, 1] += 0.5
        mask1 = (alpha > 1) & (beta > 1)
        x[mask1] = (alpha[mask1]-1)/(alpha[mask1]+beta[mask1]-2)
        mask2 = (alpha <= 1) & (beta > 1)
        x[mask2] = 0.0
        mask3 = (alpha > 1) & (beta <= 1)
        x[mask3] = 1.0
        # mean
        mask4 = (alpha <= 1) & (beta <= 1)
        x[mask4] = alpha[mask4]/torch.clamp((alpha[mask4]+beta[mask4]), min=1e-5)
        x = x * 2 - 1
        return x