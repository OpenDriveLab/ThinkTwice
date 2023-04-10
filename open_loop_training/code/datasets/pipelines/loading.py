# Copyright (c) OpenMMLab. All rights reserved.
import os
from types import new_class
import cv2
import time
import copy
import torch
import numpy as np

import os.path as osp
from PIL import Image
import matplotlib.pyplot as plt
from mmdet.datasets.builder import PIPELINES

from torchvision import transforms as T

import mmcv
from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC
from mmdet3d.datasets.pipelines.loading import LoadPointsFromFile
import io
from scipy.ndimage import label as sep_mask



@PIPELINES.register_module()
class LoadMultiImages(object):
    def __init__(self,
                 is_local=False,
                 camera_names = ['rgb_front', 'rgb_left', 'rgb_right', 'rgb_back'],
                 ceph_conf="~/petreloss.conf",
                ):
        self.is_local = is_local
        self.camera_names = camera_names
        if not self.is_local:
            from petrel_client.client import Client
            self.client = Client(ceph_conf)
        self.file_client_args=dict(backend='disk')
        self.file_client = mmcv.FileClient(**self.file_client_args)
    def load_img(self, filename):
        if self.is_local:
            return np.array(Image.open(filename))
        else:
            return np.frombuffer(memoryview(self.client.get(filename)), np.uint8).reshape(900, 1600, 3)
    def __call__(self, results):
        filenames = [os.path.join(results["scene_token"], name, str(results["frame_idx"]).zfill(4)+".png") for name in self.camera_names]
        results['img'] = [self.load_img(filename) for filename in filenames]
        results["img_filename"] = filenames
        return results
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(is_local={self.is_local}, '
        return repr_str



@PIPELINES.register_module()
class LoadPoints(LoadPointsFromFile):
    def __init__(self,
                 is_local,
                 ceph_conf='~/petreloss.conf',
                 **kwargs,
                ):
        super().__init__(**kwargs)
        self.is_local = is_local
        if not self.is_local:
            from petrel_client.client import Client
            self.client = Client(ceph_conf)
    def _load_points(self, pts_filename):
        if not self.is_local:
            points = np.load(io.BytesIO(self.client.get(pts_filename)), allow_pickle=True)
        else:
            points = np.load(pts_filename, allow_pickle=True)
        return points

@PIPELINES.register_module()
class LoadDepth(LoadMultiImages):
    def __init__(self,
                 is_local=False,     # carla's camera use unreal-coordination, which must be corrected
                 camera_names = ['rgb_front', 'rgb_left', 'rgb_right', 'rgb_back'],
                 ceph_conf="~/petreloss.conf",
                ):
        super().__init__(is_local, camera_names, ceph_conf)
    def __call__(self, results):
        filenames = [os.path.join(results["scene_token"], name.replace('rgb','depth'), str(results["frame_idx"]).zfill(4)+".png") for name in self.camera_names]
        depths = []
        for filename in filenames:
            rgb = self.load_img(filename).astype(np.float32)
            r,g,b = rgb[...,0], rgb[...,1], rgb[...,2]
            depth = (r + g*256 + b*256*256)/(256**3 - 1) * 1000
            depths.append(depth)
        results['depth'] = depths
        return results

## Segment Traffic Light
def red_green_yellow(rgb_image):
    hsv = cv2.cvtColor(rgb_image[:, None, :], cv2.COLOR_RGB2HSV)
    avg_saturation = int(hsv[:,:,1].mean()) # Sum the brightness values   
    sat_low = int(avg_saturation * 1.1)#1.3)
    val_low = 140
    # Green
    lower_green = np.array([70,sat_low,val_low])
    upper_green = np.array([100,255,255])
    sum_green = cv2.inRange(hsv, lower_green, upper_green).astype(np.bool8).sum()
    # Red
    lower_red = np.array([150,sat_low,val_low])
    upper_red = np.array([180,255,255])
    sum_red = cv2.inRange(hsv, lower_red, upper_red).astype(np.bool8).sum()
    if sum_red < 3 and sum_green < 3:
        return 0 #not sure or yellow
    if sum_red >= sum_green:
        return 1# Red
    return 2 # Green


@PIPELINES.register_module()
class LoadSeg(LoadMultiImages):
    def __init__(self,
                 is_local=False,     # carla's camera use unreal-coordination, which must be corrected
                 camera_names = ['rgb_front', 'rgb_left', 'rgb_right', 'rgb_back'],
                 seg_label_idxs=[0,1,2,3,4,5,6],
                 ceph_conf="~/petreloss.conf",
                ):
        super().__init__(is_local, camera_names, ceph_conf)
        self.seg_label_idxs = seg_label_idxs
    
    def load_img(self, filename):
        if self.is_local:
            return np.array(Image.open(filename))
        else:
            return np.frombuffer(memoryview(self.client.get(filename)), np.uint8).reshape(900, 1600,)
    def __call__(self, results):
        filenames = [os.path.join(results["scene_token"], name.replace('rgb','seg'), str(results["frame_idx"]).zfill(4)+".png") for name in self.camera_names]
        segs = []
        for f_index, filename in enumerate(filenames):
            src = self.load_img(filename).astype(np.float32)
            seg = np.zeros_like(src)
            for idx, label in enumerate(self.seg_label_idxs):
                ## Mannually Segment the traffic light into red, green, yellow
                if label == 18:
                    now_img = results['img'][f_index]
                    tl_part = src==label
                    #tl_pixel_index = np.where(tl_part)
                    tl_mask, num_tl = sep_mask(tl_part, structure=[[1,1,1],[1,1,1],[1,1,1]])
                    a_flattened = tl_mask.ravel()
                    sidx = np.argsort(a_flattened)
                    afs = a_flattened[sidx]
                    cut_idx = np.r_[0,np.flatnonzero(afs[1:] != afs[:-1])+1, a_flattened.size]
                    row, col = np.unravel_index(sidx, tl_mask.shape)
                    row_indices = [row[i:j] for i,j in zip(cut_idx[:-1],cut_idx[1:])][1:]
                    col_indices = [col[i:j] for i,j in zip(cut_idx[:-1],cut_idx[1:])][1:]
                    for tl_index in range(num_tl):
                        if len(row_indices[tl_index]) < 20:
                            continue
                        now_tl = now_img[row_indices[tl_index], col_indices[tl_index], :]
                        light_type=red_green_yellow(now_tl)
                        seg[row_indices[tl_index], col_indices[tl_index]] = idx + light_type
                else:
                    seg[np.where(src==label)] = idx
            segs.append(seg)
        results['seg'] = segs
        return results

