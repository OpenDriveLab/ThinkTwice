from email.mime import image
import os
import cv2
import copy
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from imgaug import augmenters as iaa
from torchvision import transforms as T
import torch
import mmcv
from mmdet.datasets.builder import PIPELINES
from mmcv.parallel import DataContainer as DC
from mmdet.datasets.pipelines import to_tensor

## Precompute the following matrice to speed
LIDAR2UNREAL = {
    "rgb_front": np.array([[1.0, 0.0, -0.0, -1.5], [0.0, 1.0, 0.0, 0.0], [-0.0, 0.0, 1.0, -2.5], [0.0, 0.0, 0.0, 1.0]]),
    "rgb_back":np.array([[-1.0, -0.0, 0.0, -1.6], [0.0, -1.0, 0.0, -0.0], [0.0, 0.0, 1.0, -2.5], [0.0, 0.0, 0.0, 1.0]]),
    "rgb_left":np.array([[0.0, -1.0, 0.0, -0.3], [1.0, 0.0, -0.0, -0.0], [0.0, 0.0, 1.0, -2.5], [0.0, 0.0, 0.0, 1.0]]),
    "rgb_right":np.array([[0.0, 1.0, -0.0, -0.3], [-1.0, -0.0, 0.0, 0.0], [0.0, 0.0, 1.0, -2.5], [0.0, 0.0, 0.0, 1.0]])
}     

unreal2cam = np.array([[0,1,0,0], [0,0,-1,0], [1,0,0,0], [0,0,0,1]])
LIDAR2CAM = {
    "rgb_front": np.array([ [0.0, 1.0, 0.0, 0.0] , [0.0, 0.0, -1.0, 2.5] , [1.0, 0.0, 0.0, -1.5] , [0.0, 0.0, 0.0, 1.0] ]),
    "rgb_back": np.array([ [0.0, -1.0, 0.0, 0.0] , [0.0, 0.0, -1.0, 2.5] , [-1.0, 0.0, 0.0, -1.6] , [0.0, 0.0, 0.0, 1.0] ]),
    "rgb_left": np.array([ [1.0, 0.0, 0.0, 0.0] , [0.0, 0.0, -1.0, 2.5] , [0.0, -1.0, 0.0, -0.3] , [0.0, 0.0, 0.0, 1.0] ]),
    "rgb_right": np.array([ [-1.0, 0.0, 0.0, 0.0] , [0.0, 0.0, -1.0, 2.5] , [0.0, 1.0, 0.0, -0.3] , [0.0, 0.0, 0.0, 1.0] ]),
}


UNDISTORT_LIDAR2IMG = {
    "rgb_front":np.array([[788.25758876, 304.14395142, 0.0, -1182.38638314], [449.78972161, 0.0, -221.49429321, -120.94884939000008], [1.0, 0.0, 0.0, -1.5], [0.0, 0.0, 0.0, 1.0]]),
    "rgb_left":np.array([[304.14395142, -788.25758876, 0.0, -236.47727662799997], [0.0, -449.78972161, -221.49429321, 418.79881654199994], [0.0, -1.0, 0.0, -0.3], [0.0, 0.0, 0.0, 1.0]]),
    "rgb_right":np.array([[-304.14395142, 788.25758876, 0.0, -236.47727662799997], [0.0, 449.78972161, -221.49429321, 418.79881654199994], [0.0, 1.0, 0.0, -0.3], [0.0, 0.0, 0.0, 1.0]]),
    "rgb_back":np.array([[-788.25758876, -304.14395142, 0.0, -1261.2121420160001], [-449.78972161, 0.0, -221.49429321, -165.9278215510001], [-1.0, 0.0, 0.0, -1.6], [0.0, 0.0, 0.0, 1.0]])
}

LIDAR2IMG = {
    "rgb_front":np.array([[800.0, 214.35935394, 0.0, -1200.0], [450.0, 0.0, -214.35935394, -139.10161515000004], [1.0, 0.0, 0.0, -1.5], [0.0, 0.0, 0.0, 1.0]]),
    "rgb_left":np.array([[214.35935394, -800.0, 0.0, -240.0], [0.0, -450.0, -214.35935394, 400.89838484999996], [0.0, -1.0, 0.0, -0.3], [0.0, 0.0, 0.0, 1.0]]),
    "rgb_right":np.array([[-214.35935394, 800.0, 0.0, -240.0], [0.0, 450.0, -214.35935394, 400.89838484999996], [0.0, 1.0, 0.0, -0.3], [0.0, 0.0, 0.0, 1.0]]),
    "rgb_back":np.array([[-800.0, -214.35935394, 0.0, -1280.0], [-450.0, 0.0, -214.35935394, -184.10161515000007], [-1.0, 0.0, 0.0, -1.6], [0.0, 0.0, 0.0, 1.0]])
}

mtx = np.array([[214.35935394, 0, 800,],[0, 214.35935394, 450],[0, 0, 1]])
dist = np.array([[ 0.00888296, -0.00130899,  0.00012061, -0.00338673,  0.00028834]])
newcameramtx = np.array([[304.14395142,   0,         788.25758876,],
[  0,        221.49429321, 449.78972161,],
[  0,           0,           1,        ],])
CAM_INTRINSIC = {
    "rgb_front": mtx,
    "rgb_back": mtx,
    "rgb_left": mtx,
    "rgb_right": mtx,
}
CAM_DIST= {
    "rgb_front": dist,
    "rgb_back": dist,
    "rgb_left": dist,
    "rgb_right": dist,
}

@PIPELINES.register_module(force=True)
class InitMultiImage(object):
    """Random scale the image
    Args:
        scales
    """
    def __init__(self, cfg):
        self.size = cfg["img_size"]
        self.camera_names = cfg['camera_names']
        self.resize_func = T.Resize(size=self.size)
        assert len(self.size)==2
        self.undistort = cfg["undistort"]
        self.unreal_coord = cfg["unreal_coord"]
        self.use_depth = cfg['use_depth']
        ## Undistort mat
        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (1600, 900), 5)
        mapx = (torch.from_numpy(mapx) - 800) / 800
        mapy= (torch.from_numpy(mapy) - 450) / 450
        self.num_cams = cfg["num_cams"]
        self.temporal_len = cfg["queue_length"]
        ## Precompute the mapping matrix to speed up
        self.map_grid = torch.stack([mapx, mapy], dim=-1).unsqueeze(0).repeat(self.temporal_len*self.num_cams, 1, 1, 1)
        if self.use_depth:
            self.map_grid_depth = torch.stack([mapx, mapy], dim=-1).unsqueeze(0).repeat(self.num_cams, 1, 1, 1)

        ##Initialize mats for all cameras
        self.cam_intrinsic = np.array([newcameramtx.copy() for _ in range(self.num_cams)] if self.undistort else [mtx.copy() for _ in range(self.num_cams)])
        self.lidar2cam = np.array([LIDAR2CAM[name] for name in self.camera_names])
        self.lidar2img = np.array([UNDISTORT_LIDAR2IMG[name] for name in self.camera_names] if self.undistort else [LIDAR2IMG[name] for name in self.camera_names])

    def __call__(self, queue):
        """Call function to pad images, masks, semantic segmentation maps.
        Args:
            results (dict): Result dict from loading pipeline.
        Returns:
            dict: Updated result dict.
        """
        #loading from queue
        imgs_list = np.stack([each['img'].data for each in queue], axis=0)
        temporal_len, num_cams, h, w, c = imgs_list.shape
        y_size, x_size = self.size
        y_scale, x_scale = y_size/h, x_size/w

        ## undistort
        imgs_list = torch.from_numpy(imgs_list).to(dtype=torch.get_default_dtype()).view(-1, h, w, c).permute(0, 3, 1, 2)
        if self.undistort:
            imgs_list = torch.nn.functional.grid_sample(imgs_list, self.map_grid, align_corners=False)
        imgs_list = self.resize_func(imgs_list).view(temporal_len, num_cams, c, y_size, x_size)

        ## depth label 
        if self.use_depth:
            depth = torch.from_numpy(np.stack(queue[-1]['depth'])).to(dtype=torch.get_default_dtype()).view(-1,1,h,w)
            if self.undistort:
                depth = torch.nn.functional.grid_sample(depth, self.map_grid_depth, align_corners=False)
            queue[-1]['depth'] = self.resize_func(depth)[:,0,...]
        ## camera mat
        scale_factor = np.eye(4)
        scale_factor[0, 0] *= x_scale
        scale_factor[1, 1] *= y_scale
        for i in range(len(queue)):
            queue[i]["img_metas"].data['cam_intrinsic'] = self.cam_intrinsic        
            queue[i]["img_metas"].data['lidar2cam'] = self.lidar2cam
            lidar2img = [scale_factor @ l2i for l2i in self.lidar2img]
            queue[i]["img_metas"].data['lidar2img'] = lidar2img
            queue[i]["img"] = imgs_list[i]
            queue[i]["img_metas"].data['img_shape'] = [imgs_list.shape[-2:]] * self.num_cams
            queue[i]["img_metas"].data['ori_shape'] = [imgs_list.shape[-2:]] * self.num_cams
            queue[i]["img_metas"].data['ida_mats'] = np.array([[[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]] * self.num_cams)
        return queue
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, '
        return repr_str



@PIPELINES.register_module()
class ImageTransformMulti(object):
    def __init__(self, aug, batch_size):
        self.transform = T.Compose([T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])])
        self.aug = aug
        self._batch_read_number = 0
        self._batch_size =  batch_size
    
    def __call__(self, queue):
        if self.aug:
            imgs_list = np.stack([each['img'].numpy().astype(np.uint8) for each in queue], axis=0).transpose((0, 1, 3, 4, 2))
            temporal_len, num_cams, H, W, C = imgs_list.shape
            #temporal length x num_camera x H x W x C
            now_augmentor = augmenter(self._batch_read_number/self._batch_size).to_deterministic()
            ##Same ImgAug for all history frames and camera of one single sample
            new_imgs_list = np.stack([now_augmentor.augment_image(imgs_list[i][j]) for i in range(temporal_len) for j in range(num_cams)], axis=0).reshape(-1, H, W, C)
            new_imgs_list = self.transform(torch.from_numpy(new_imgs_list).permute(0, 3, 1, 2).to(dtype=torch.get_default_dtype()).div(255)).view(temporal_len, num_cams, C, H, W)
            for temporal_index in range(imgs_list.shape[0]):
                queue[temporal_index]["img"] = to_tensor(new_imgs_list[temporal_index])
            self._batch_read_number += 1
        else:
            for temporal_index in range(len(queue)):
                queue[temporal_index]["img"] = self.transform(queue[temporal_index]["img"].to(dtype=torch.get_default_dtype()).div(255))
        return queue
    
    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(aug={self.aug}, '
        return repr_str

def augmenter(image_iteration):
    iteration = image_iteration
    frequency_factor = min(0.05 + float(iteration)/600000.0, 1.0)
    color_factor = min(float(iteration)/3000000.0, 1.0)
    dropout_factor = 0.198667 + (0.03856658 - 0.198667) / (1 + (iteration / 600000) ** 1.863486)

    blur_factor = min(0.5 + (0.5*iteration/300000.0), 1.0)

    add_factor = 10 + 10*iteration/300000.0

    multiply_factor_pos = 1 + (2.5*iteration/600000.0)
    multiply_factor_neg = 1 - (0.91 * iteration / 1500000.0)

    contrast_factor_pos = 1 + (0.5*iteration/1500000.0)
    contrast_factor_neg = 1 - (0.5 * iteration / 1500000.0)

    augmenter = iaa.Sequential([

        iaa.Sometimes(frequency_factor, iaa.GaussianBlur((0, blur_factor))),
        # blur images with a sigma between 0 and 1.5
        iaa.Sometimes(frequency_factor, iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, dropout_factor),
                                                                  per_channel=color_factor)),
        # add gaussian noise to images
        iaa.Sometimes(frequency_factor, iaa.CoarseDropout((0.0, dropout_factor), size_percent=(
            0.08, 0.2), per_channel=color_factor)),
        # randomly remove up to X% of the pixels
        iaa.Sometimes(frequency_factor, iaa.Dropout((0.0, dropout_factor), per_channel=color_factor)),
        # randomly remove up to X% of the pixels
        iaa.Sometimes(frequency_factor,
                      iaa.Add((-add_factor, add_factor), per_channel=color_factor)),
        # change brightness of images (by -X to Y of original value)
        iaa.Sometimes(frequency_factor,
                      iaa.Multiply((multiply_factor_neg, multiply_factor_pos), per_channel=color_factor)),
        # change brightness of images (X-Y% of original value)
        # iaa.Sometimes(frequency_factor, iaa.ContrastNormalization((contrast_factor_neg, contrast_factor_pos),
        #                                                                per_channel=color_factor)),
        iaa.Sometimes(frequency_factor, iaa.contrast.LinearContrast((contrast_factor_neg, contrast_factor_pos),
                                                                    per_channel=color_factor)),
        # improve or worsen the contrast
        iaa.Sometimes(frequency_factor, iaa.Grayscale((0.0, 1))),  # put grayscale

    ],
        random_order=True  # do all of the above in random order
    )

    return augmenter



### From BEVDepth: https://github.com/Megvii-BaseDetection/BEVDepth
@PIPELINES.register_module()
class IDAImageTransform(object):
    def __init__(self, cfg, ida_aug_conf, is_train=False):
        self.ida_aug_conf = ida_aug_conf
        self.is_train = is_train
        self.size = cfg["img_size"]
        self.camera_names = cfg['camera_names']
        self.resize_func = T.Resize(size=self.size)
        assert len(self.size)==2
        self.undistort = cfg["undistort"]
        self.unreal_coord = cfg["unreal_coord"]
        self.use_depth = False if 'use_depth' not in cfg else cfg['use_depth']
        self.use_seg = False if 'use_seg' not in cfg else cfg['use_seg']
        ## Undistort mat
        mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (1600, 900), 5)
        mapx = (torch.from_numpy(mapx) - 800) / 800
        mapy= (torch.from_numpy(mapy) - 450) / 450
        self.num_cams = cfg["num_cams"]
        self.temporal_len = cfg["queue_length"]
        self.map_grid = torch.stack([mapx, mapy], dim=-1).unsqueeze(0).repeat(self.temporal_len*self.num_cams, 1, 1, 1)
        if self.use_depth or self.use_seg:
            self.map_grid_depth = torch.stack([mapx, mapy], dim=-1).unsqueeze(0).repeat(self.num_cams, 1, 1, 1)
        ##Initialize mats for all cameras
        self.cam_intrinsic = torch.from_numpy((np.stack([newcameramtx.copy() for _ in range(self.num_cams)], axis=0) if self.undistort else np.stack([mtx.copy() for _ in range(self.num_cams)], axis=0)).astype(np.float32))
        self.lidar2cam = torch.from_numpy(np.stack([LIDAR2CAM[name] for name in self.camera_names], axis=0).astype(np.float32))
        self.lidar2img = torch.from_numpy((np.stack([UNDISTORT_LIDAR2IMG[name] for name in self.camera_names], axis=0) if self.undistort else np.stack([LIDAR2IMG[name] for name in self.camera_names], axis=0)).astype(np.float32))

    def sample_ida_augmentation(self):
        """Generate ida augmentation values based on ida_config."""
        H, W = self.ida_aug_conf['H'], self.ida_aug_conf['W']
        fH, fW = self.ida_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.ida_aug_conf['resize_lim'])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.random.uniform(*self.ida_aug_conf['bot_pct_lim'])) *
                newH) - fH
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.ida_aug_conf['rand_flip'] and np.random.choice([0, 1]):
                flip = True
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int(
                (1 - np.mean(self.ida_aug_conf['bot_pct_lim'])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
        return resize, resize_dims, crop, flip

    def __call__(self, queue):
        #loading from queue
        imgs_list = np.stack([each['img'].data for each in queue], axis=0)
        T, N, h, w, c = imgs_list.shape
        ## undistort
        imgs_list = torch.from_numpy(imgs_list).to(dtype=torch.get_default_dtype()).view(-1, h, w, c).permute(0, 3, 1, 2)
        if self.undistort:
            imgs_list = torch.nn.functional.grid_sample(imgs_list, self.map_grid, align_corners=False)
        imgs_list = imgs_list.view(T,N,c,h,w)
        ## depth label  
        if self.use_depth:
            depth = torch.from_numpy(np.stack(queue[-1]['depth'])).to(dtype=torch.get_default_dtype()).view(-1,1,h,w)
            if self.undistort:
                depth = torch.nn.functional.grid_sample(depth, self.map_grid_depth, align_corners=False)[:,0]
        if self.use_seg:
            seg = torch.from_numpy(np.stack(queue[-1]['seg'])).to(dtype=torch.get_default_dtype()).view(-1,1,h,w)
            if self.undistort:
                seg = torch.nn.functional.grid_sample(seg, self.map_grid_depth, align_corners=False)[:,0]

        img_list_aug = []
        ida_list = []
        depth_list = []
        seg_list = []
        for cam_id in range(N):
            resize, resize_dims, crop, flip = self.sample_ida_augmentation()
            if self.use_depth:
                depth_aug = depth_transform(
                        depth[cam_id], resize_dims,
                        crop, flip)
                depth_list.append(depth_aug)
            if self.use_seg:
                seg_aug = depth_transform(
                        seg[cam_id], resize_dims,
                        crop, flip)
                seg_list.append(seg_aug)
            imgs, idas = [], []
            for frame_id in range(T):
                img_aug, ida_mat = img_transform(
                            imgs_list[frame_id][cam_id],
                            resize=resize,
                            resize_dims=resize_dims,
                            crop=crop,
                            flip=flip,
                            rotate=0,
                        )
                imgs.append(img_aug)
                idas.append(ida_mat)
            imgs, idas = torch.stack(imgs), torch.stack(idas)
            img_list_aug.append(imgs)
            ida_list.append(idas)
        
        img_list_aug = torch.stack(img_list_aug).permute(1,0,2,3,4)   
        ida_list = torch.stack(ida_list).permute(1,0,2,3)
        if self.use_depth:
            queue[-1]['depth'] = torch.stack(depth_list)
        if self.use_seg:
            queue[-1]['seg'] = torch.stack(seg_list)
        ## camera mat
        for i in range(len(queue)):     # Save origin values, and they will be transformed in fpn_lss.py finally
            queue[i]["img_metas"].data['cam_intrinsic'] = self.cam_intrinsic        
            queue[i]["img_metas"].data['lidar2cam'] = self.lidar2cam
            queue[i]["img_metas"].data['lidar2img'] = self.lidar2img
            queue[i]["img"] = img_list_aug[i]
            queue[i]["img_metas"].data['img_shape'] = [imgs_list.shape[-2:]] * self.num_cams
            queue[i]["img_metas"].data['ori_shape'] = [imgs_list.shape[-2:]] * self.num_cams
            queue[i]["img_metas"].data['ida_mats'] = ida_list[i]
        return queue

        
        

def img_transform(img, resize, resize_dims, crop, flip, rotate=0):
    ida_rot = torch.eye(2)
    ida_tran = torch.zeros(2)

    (w,h) = resize_dims
    resize_func = T.Resize((h,w))
    # adjust image
    img = resize_func(img)
    x0,y0,x1,y1 = crop
    
    img = img[..., y0:y1, x0:x1]
    if flip:
        img = torch.flip(img, dims=[-1])

    # post-homography transformation
    ida_rot *= resize
    ida_tran -= torch.Tensor(crop[:2])
    if flip:
        A = torch.Tensor([[-1, 0], [0, 1]])
        b = torch.Tensor([crop[2] - crop[0], 0])
        ida_rot = A.matmul(ida_rot)
        ida_tran = A.matmul(ida_tran) + b
    A = get_rot(rotate / 180 * np.pi)
    b = torch.Tensor([crop[2] - crop[0], crop[3] - crop[1]]) / 2
    b = A.matmul(-b) + b
    ida_rot = A.matmul(ida_rot)
    ida_tran = A.matmul(ida_tran) + b
    ida_mat = ida_rot.new_zeros(4, 4)
    ida_mat[3, 3] = 1
    ida_mat[2, 2] = 1
    ida_mat[:2, :2] = ida_rot
    ida_mat[:2, 3] = ida_tran
    return img, ida_mat

def get_rot(h):
    return torch.Tensor([
        [np.cos(h), np.sin(h)],
        [-np.sin(h), np.cos(h)],
    ])

def depth_transform(depth, resize_dims, crop, flip):
    (w,h) = resize_dims
    resize_func = T.Resize((h,w))
    # adjust image
    depth = resize_func(depth.unsqueeze(0))
    x0,y0,x1,y1 = crop
    depth = depth[..., y0:y1, x0:x1]
    if flip:
        depth = torch.flip(depth, dims=[-1])

    return depth[0]