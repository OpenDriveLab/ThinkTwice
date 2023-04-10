from collections import deque, OrderedDict
from distutils.command.config import config
import cv2
import numpy as np
import matplotlib
matplotlib.use('AGG')
import matplotlib.pyplot as plt

import torch 
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
from torchvision import transforms as T

from mmcv.runner import BaseModule
from mmdet.models import DETECTORS
import mmcls.models
from mmdet3d.models import builder
from mmcv.runner import force_fp32, auto_fp16
from .utils import PIDController, FocalLoss, set_dropout_zero, init_weights, SEBasicBlock


@DETECTORS.register_module()
class EncoderDecoder(BaseModule):
    def __init__(self,  
                img_encoder,
                decoder,
                lidar_encoder=None,
                num_cams = 4,
                use_depth=False,
                use_seg=False,
                downsample_factor=16,
                seg_downsample_factor=2,
                train_cfg=None,
                test_cfg=None,
                ):
        super().__init__()
        self.config = train_cfg
        self.fp16_enabled = False

        self.num_cams = num_cams
        self.use_depth = False if 'use_depth' not in train_cfg else train_cfg['use_depth']
        self.use_seg  = False if 'use_seg' not in train_cfg else train_cfg['use_seg']
        self.downsample_factor = downsample_factor
        self.seg_downsample_factor = seg_downsample_factor
        
        self.turn_controller = PIDController(K_P=self.config['turn_KP'], K_I=self.config['turn_KI'], K_D=self.config['turn_KD'], n=self.config['turn_n'])
        self.speed_controller = PIDController(K_P=self.config['speed_KP'], K_I=self.config['speed_KI'], K_D=self.config['speed_KD'], n=self.config['speed_n'])

        ## Encoder
        self.img_encoder = builder.build_backbone(img_encoder)
        self.depth_resize = None
        self.dbound = self.img_encoder.d_bound
        self.depth_channels = int((self.dbound[1] - self.dbound[0]) / self.dbound[2])
        self.seg_loss_func = FocalLoss()
        self.lidar_encoder = builder.build_backbone(lidar_encoder)
        self.act = nn.ReLU
        self.build_fusion_and_flatten_network_for_BEV()
        self.measurements_encoder = nn.Sequential(
                            nn.Linear(1+2+6, 128),
                            self.act(),
                            nn.Linear(128, 128),
                            self.act(),
                        )
        nn.init.xavier_normal_(self.measurements_encoder[0].weight)
        nn.init.constant_(self.measurements_encoder[0].bias, 0)
        nn.init.xavier_normal_(self.measurements_encoder[2].weight)
        nn.init.constant_(self.measurements_encoder[2].bias, 0)
        

        ## Decoder
        self.decoder= builder.build_head(decoder)
        self.max_epoch = 60
        if "total_epochs" in self.config:
            self.max_epoch = self.config["total_epochs"]
    
    def set_epoch(self, epoch):
        self.epoch = epoch
    
    @auto_fp16()
    def build_fusion_and_flatten_network_for_BEV(self):
        ### BEV Fusion Network 
        self.norm = nn.BatchNorm2d
        self.inplace_act = self.act()
        self.conv_cam = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            self.norm(256),
            self.act(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            self.norm(256),
        )
        self.conv_cam.apply(init_weights)

        self.conv_lidar = nn.Sequential(
            nn.Conv2d(256*2, 256, kernel_size=3, padding=1, bias=False, stride=2),
            self.norm(256),
            self.act(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False, stride=2),
            self.norm(256),
            self.act(),
        )
        self.conv_lidar.apply(init_weights)
        self.conv_fusion = nn.Sequential(
            nn.Conv2d(256*2, 256, kernel_size=3, padding=1, bias=False),
            self.norm(256),
            self.act(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1, bias=False),
            self.norm(256),
        )
        self.conv_fusion.apply(init_weights)

        ### Transform Raw BEV feature to 1D feature map
        ## The dimension of middle BEV feature map follows Roach to enable feature distillation
        self._256_to_32 = nn.Conv2d(256, 32, kernel_size=3, padding=1)
        self._256_to_32.apply(init_weights)
        self.MLP21 = SEBasicBlock(32, self.act) ## MLP for the 21x21 BEV feature map
        self.MLP10 = SEBasicBlock(64, self.act) ## MLP for the 10x10 BEV feature map
        self.MLP4 = SEBasicBlock(128, self.act) ## MLP for the 4x4 BEV feature map
        self.MLP2 = SEBasicBlock(256, self.act) ## MLP for the 2x2 BEV feature map
        self.MLP21.apply(init_weights)
        self.MLP10.apply(init_weights)
        self.MLP4.apply(init_weights)
        self.MLP2.apply(init_weights)
        self.conv21_10 = nn.Conv2d(32, 64, kernel_size=3, stride=2)
        self.conv10_4 = nn.Conv2d(64, 128, kernel_size=3, stride=2)
        self.conv4_2 = nn.Conv2d(128, 256, kernel_size=3, stride=1)
        self.conv21_10.apply(init_weights)
        self.conv10_4.apply(init_weights)
        self.conv4_2.apply(init_weights)
        
        self.output_fc = nn.Sequential(
            nn.Linear(1024, 512),
            self.act(),
            nn.BatchNorm1d(512),
            nn.Linear(512, 256),
            self.act()
        )
        self.output_fc.apply(init_weights)

    def train_step(self, data, optimizer):
        losses = self.forward_train(data)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=data['img'].shape[0])
        return outputs
    
    @force_fp32()
    def forward_train(self, batch):
        target_point = batch['target_point'].to(dtype=torch.float32)
        command = batch['target_command']
        speed = batch['speed'].to(dtype=torch.float32).view(-1,1) / 12.
        state = [speed, target_point, command,  batch['target_command_raw']]
        # extract all modal features
        points = batch['points'] if 'points' in batch else None
        cam_feat, lidar_feat, measurement_feat = self.extract_sensor_feat(
            img=batch['img'], 
            state=state, 
            img_metas=batch['img_metas'],
            points=points)
        
        flattend_BEV_feat, fusion_feats, mid_BEV_feature, mid_lidar_feat = self.get_fusion_feat(cam_feat, lidar_feat)
        
        teacher_forcing_input = {}
        for teacher_key in ["waypoints", "action_sigma", "action_mu", "future_action_sigma", "future_action_mu"]:
            teacher_forcing_input[teacher_key] = batch[teacher_key]

        ## Decoder
        pred = self.decoder(flattend_BEV_feat, fusion_feats, measurement_feat, target_point, self, teacher_forcing_input, [cam_feat["lidar2img"], cam_feat["ida_mat"], cam_feat["fpn_feats"], mid_lidar_feat])
        loss_dict = self.decoder.loss(batch, pred, mid_BEV_feature)

        ## seg supervision
        if self.use_seg:
            seg_gt = self.get_downsampled_gt_seg(batch['seg'])
            seg_pred = cam_feat['seg']
            seg_loss = self.seg_loss_func(seg_pred, seg_gt)
            loss_dict['seg_loss'] = seg_loss * 10
          
        ## depth supervision
        if self.use_depth:
            depth_gt = batch['depth']
            depth_gt = self.get_downsampled_gt_depth(depth_gt)
            depth_pred = cam_feat['depth'].permute(0, 2, 3, 1).contiguous().view(
                -1, self.depth_channels)
            fg_mask = torch.max(depth_gt, dim=1).values > 0.0
            depth_loss = F.binary_cross_entropy_with_logits(
                depth_pred[fg_mask],
                depth_gt[fg_mask],
                reduction='none',
            ).sum() / max(1.0, fg_mask.sum())
            loss_dict['depth_loss'] = depth_loss
        return loss_dict
    
    ## For close-loop evaluation
    def forward_inference(self, batch):
        self.epoch = 10000
        target_point = batch['target_point'].to(dtype=torch.float32)
        command = batch['target_command']
        speed = batch['speed'].to(dtype=torch.float32).view(-1,1) / 12.
        state = [speed, target_point, command,  batch['target_command_raw']]
        # extract all modal features
        points = batch['points'] if 'points' in batch else None
        cam_feat, lidar_feat, measurement_feat = self.extract_sensor_feat(
            img=batch['img'], 
            state=state, 
            img_metas=batch['img_metas'],
            points=points)
        flattend_BEV_feat, fusion_feats, mid_feature, mid_lidar_feat = self.get_fusion_feat(cam_feat, lidar_feat)
        teacher_forcing_data = None
        pred = self.decoder(flattend_BEV_feat, fusion_feats, measurement_feat, target_point, self, teacher_forcing_data, [cam_feat["lidar2img"], cam_feat["ida_mat"], cam_feat["fpn_feats"], mid_lidar_feat])
        return pred

    @auto_fp16()
    def get_fusion_feat(self, cam_feat, lidar_feat,):
        lidar_feat_with_high_resolution = lidar_feat[0].clone()
        #fusion
        cam_feats_reduce = self.inplace_act(self.conv_cam(cam_feat['bev']) + cam_feat['bev']) # Residual Connection
        if self.lidar_encoder is not None:
            pts_feats_reduce = self.conv_lidar(lidar_feat[0])
            bev_feats = self.inplace_act(self.conv_fusion(torch.cat([cam_feats_reduce, pts_feats_reduce], dim=1)) + cam_feats_reduce + pts_feats_reduce)  # Residual Connection  Bx256x21x21
        else:
            bev_feats = cam_feats_reduce # Bx256x21x21
        
        mid_feature = [None, None] #Currently no feature for 8x94x94 16x45x45
        feats32_21_21 = self.MLP21(self.inplace_act(self._256_to_32(bev_feats)))
        
        bev_feats = feats32_21_21
        mid_feature.append(feats32_21_21)
        feats64_10_10 = self.MLP10(self.inplace_act(self.conv21_10(feats32_21_21)))
        mid_feature.append(feats64_10_10)
        feats128_4_4 = self.MLP4(self.inplace_act(self.conv10_4(feats64_10_10)))
        mid_feature.append(feats128_4_4)
        feats256_2_2 = self.MLP2(self.inplace_act(self.conv4_2((feats128_4_4))))
        mid_feature.append(feats256_2_2)
        flattend_BEV_feat = self.output_fc(feats256_2_2.flatten(start_dim=1)) # Bx256
        return flattend_BEV_feat, bev_feats, mid_feature, lidar_feat_with_high_resolution

    @auto_fp16()
    def extract_sensor_feat(self, img, state, img_metas, points, is_train=True):
        # extract camera feature
        cam_feat = self.img_encoder(img=img, img_metas=img_metas, is_return_depth=self.use_depth and is_train)
        cam_feat['bev'] = torch.rot90(torch.flip(cam_feat['bev'],dims=[2]), 1, dims=[2,3]) ### Match with Roach BEV
        state = torch.cat(state[:3], dim=-1)
        measurement_feat = self.measurements_encoder(state)
        lidar_feat = self.lidar_encoder(points[:, -1, ...])
        for i in range(len(lidar_feat)):
            lidar_feat[i] = torch.rot90(torch.flip(lidar_feat[i],dims=[2]), 1, dims=[2,3]) ### Match with Roach BEV
        ## Visualization of BEV feature
        if False:
            self.plot(cam_feat['bev'], lidar_feat[0], img_metas[0])
        return cam_feat, lidar_feat, measurement_feat

    def plot(self, cam_feats, fusion_feats, img_metas):
        def get_plot(arr):
            arr2 = arr.cpu().detach().numpy()
            return np.max(arr2,axis=0)
        for i in [0]:
            plt.clf()
            plt.subplot(2,1,1)
            plt.imshow(get_plot(cam_feats[i]))#[::-1])
            plt.title('cam feature')
            plt.subplot(2,1,2)
            plt.imshow(get_plot(fusion_feats[i]))#[::-1])
            plt.title('lidar feature')
            plt.suptitle('x=[-8.0, 30.4], y=[-19.2, 19.2]')
            plt.savefig('cam_fusion_'+str(i)+'.png')

    @force_fp32()
    def process_action(self, pred, command, speed, target_point):
        action = self._get_action_beta(pred['mu_branches'][:, -1, :].view(1,2), pred['sigma_branches'][:, -1, :].view(1,2))
        acc, steer = action.cpu().numpy()[0].astype(np.float64)
        if acc >= 0.0:
            throttle = acc
            brake = 0.0
        else:
            throttle = 0.0
            brake = np.abs(acc)
        throttle = np.clip(throttle, 0, 1)
        steer = np.clip(steer, -1, 1)
        brake = np.clip(brake, 0, 1)
        metadata = {
            'speed': float(speed.cpu().numpy().astype(np.float64)),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
            'command': command,
            'target_point': target_point,
        }
        return steer, throttle, brake, metadata

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
    
    ## Copy from TCP
    ## Modified by Xiaosong Jia
    @force_fp32()
    def control_pid(self, waypoints, velocity, target, stuck_desired_speed=-1):
        ''' Predicts vehicle control with a PID controller.
        Args:
            waypoints (tensor): output of self.plan()
            velocity (tensor): speedometer input
        '''
        assert(waypoints.size(0)==1)
        waypoints = waypoints[0].data.cpu().numpy()
        saved_waypoints = waypoints.copy()
        saved_target = target.copy()
        waypoints = waypoints[:, ::-1]
        target = target[::-1]

        # iterate over vectors between predicted waypoints
        num_pairs = len(waypoints) - 1
        best_norm = 1e5
        desired_speed = 0
        aim = waypoints[0]
        for i in range(num_pairs):
            # magnitude of vectors, used for speed
            desired_speed += np.linalg.norm(
                    waypoints[i+1] - waypoints[i]) * 2.0 / num_pairs

            # norm of vector midpoints, used for steering
            norm = np.linalg.norm((waypoints[i+1] + waypoints[i]) / 2.0)
            if abs(self.config['aim_dist']-best_norm) > abs(self.config['aim_dist']-norm):
                aim = waypoints[i]
                best_norm = norm
        desired_speed = desired_speed.astype(np.float64)
        if stuck_desired_speed > 0:
            desired_speed = stuck_desired_speed
        aim_last = waypoints[-1] - waypoints[-2]

        angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
        angle_last = np.degrees(np.pi / 2 - np.arctan2(aim_last[1], aim_last[0])) / 90
        angle_target = np.degrees(np.pi / 2 - np.arctan2(target[1], target[0])) / 90

        # choice of point to aim for steering, removing outlier predictions
        # use target point if it has a smaller angle or if error is large
        # predicted point otherwise
        # (reduces noise in eg. straight roads, helps with sudden turn commands)
        use_target_to_aim = np.abs(angle_target) < np.abs(angle)
        #angle_thresh  
        use_target_to_aim = use_target_to_aim or (np.abs(angle_target-angle_last) > self.config['angle_thresh'] and target[1] < self.config['dist_thresh'])
        if use_target_to_aim:
            angle_final = angle_target
        else:
            angle_final = angle
        angle_final = angle_final.astype(np.float64)
        speed = velocity[0].data.cpu().numpy()
        if (speed < 0.01):
            angle_final = 0.0  # When we don't move we don't want the angle error to accumulate in the integral

        steer = self.turn_controller.step(angle_final)
        steer = np.clip(steer, -1.0, 1.0)

        brake = desired_speed < self.config['brake_speed'] or (speed / desired_speed) > self.config['brake_ratio']

        delta = np.clip(desired_speed - speed, 0.0, self.config['clip_delta'])
        throttle = self.speed_controller.step(delta)
        throttle = np.clip(throttle, 0.0, 1.0)#self.config['max_throttle'])
        throttle = throttle if not brake else 0.0

        metadata = {
            'speed': float(speed.astype(np.float64)),
            'steer': float(steer),
            'throttle': float(throttle),
            'brake': float(brake),
            'wp_4': tuple(saved_waypoints[3].astype(np.float64)),
            'wp_3': tuple(saved_waypoints[2].astype(np.float64)),
            'wp_2': tuple(saved_waypoints[1].astype(np.float64)),
            'wp_1': tuple(saved_waypoints[0].astype(np.float64)),
            'aim': tuple(aim.astype(np.float64)),
            'target': tuple(saved_target.astype(np.float64)),
            'desired_speed': float(desired_speed),
            'angle': float(angle.astype(np.float64)),
            'angle_last': float(angle_last.astype(np.float64)),
            'angle_target': float(angle_target.astype(np.float64)),
            'angle_final': float(angle_final),
            'delta': float(delta.astype(np.float64)),
        }
        return steer, throttle, brake, metadata

    @force_fp32()
    def forward(self, is_eval=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        losses = self.forward_train(kwargs)
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
        loss=loss, log_vars=log_vars, num_samples=kwargs['img'].shape[0])
        return outputs
    
    @force_fp32()
    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary infomation.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')
        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)
        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()
        return loss, log_vars
    
    @force_fp32()
    def get_downsampled_gt_depth(self, gt_depths, min_pooling=True):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        if min_pooling:
            gt_depths = gt_depths.view(
                B * N,
                H // self.downsample_factor,
                self.downsample_factor,
                W // self.downsample_factor,
                self.downsample_factor,
                1,
            )
            gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
            gt_depths = gt_depths.view(
                -1, self.downsample_factor * self.downsample_factor)
            gt_depths_tmp = torch.where(gt_depths == 0.0,
                                        1e5 * torch.ones_like(gt_depths),
                                        gt_depths)
            gt_depths = torch.min(gt_depths_tmp, dim=-1).values
            gt_depths = gt_depths.view(B * N, H // self.downsample_factor,
                                    W // self.downsample_factor)
        else:   # just use nearest sampling
            if self.depth_resize is None:
                self.depth_resize = T.Resize((H//self.downsample_factor,W//self.downsample_factor), interpolation=T.InterpolationMode.NEAREST)
            gt_depths = gt_depths.view(B*N, H, W)
            gt_depths = self.depth_resize(gt_depths)
        gt_depths = (gt_depths -
                     (self.dbound[0] - self.dbound[2])) / self.dbound[2]
        gt_depths = torch.where(
            (gt_depths < self.depth_channels + 1) & (gt_depths >= 0.0),
            gt_depths, torch.zeros_like(gt_depths))
        gt_depths = F.one_hot(gt_depths.long(),
                              num_classes=self.depth_channels + 1).view(
                                  -1, self.depth_channels + 1)[:, 1:]
        return gt_depths.float()

    @force_fp32()
    def get_downsampled_gt_seg(self, gt_seg):
        B, N, H, W = gt_seg.shape
        if self.depth_resize is None:
            self.depth_resize = T.Resize((H//self.seg_downsample_factor,W//self.seg_downsample_factor), interpolation=T.InterpolationMode.NEAREST)
        gt_seg = gt_seg.view(B*N, H, W)
        gt_seg = self.depth_resize(gt_seg)
        return gt_seg.long()


