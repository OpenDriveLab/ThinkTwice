
# Copyright (c) OpenMMLab. All rights reserved.
import torch
import numpy as np
from mmcv.parallel import DataContainer as DC


from mmdet3d.core.points import BasePoints
from mmdet.datasets.builder import PIPELINES
from mmdet.datasets.pipelines import to_tensor
from torch import Tensor


@PIPELINES.register_module()
class CarlaFormatBundle(object):
    def __init__(self, ):
        return
    def __call__(self, results):
        if 'points' in results:
            if isinstance(results['points'], BasePoints):
                results['points'] = DC(results['points'].tensor)
            else:
                results['points'] = DC(torch.Tensor(results['points']))
        if 'img' in results:
            if isinstance(results['img'], list):
                results['img'] = np.stack(results['img'], axis=0)
        return results
    def __repr__(self):
        return self.__class__.__name__



@PIPELINES.register_module()
class CarlaCollect(object):
    def __init__(self,
                 keys,
                 meta_keys=('can_bus', 'ori_shape', 'intrin_mats', 'lidar2img',
                            'depth2img', 'cam2img', 'pad_shape',
                            'scale_factor', 'flip', 'pcd_horizontal_flip',
                            'pcd_vertical_flip', 'frame_idx', 'img_filename',
                            'img_norm_cfg', 'bda_mat', 'sample_idx',
                            'ida_mats', 'sensor2ego_mats', 'pts_filename',
                            'scene_token')):
        self.keys = keys
        self.meta_keys = meta_keys

    def __call__(self, results):
        """Call function to collect keys in results. The keys in ``meta_keys``
        will be converted to :obj:`mmcv.DataContainer`.

        Args:
            results (dict): Result dict contains the data to collect.

        Returns:
            dict: The result dict contains the following keys
                - keys in ``self.keys``
                - ``img_metas``
        """
        data = {}
        img_metas = {}
        for key in self.meta_keys:
            if key in results:
                img_metas[key] = results[key]
        data['img_metas'] = DC(img_metas, cpu_only=True)
        for key in self.keys:
            if key in results:
                data[key] = results[key]
        return data

    def __repr__(self):
        """str: Return a string that describes the module."""
        return self.__class__.__name__ + \
            f'(keys={self.keys}, meta_keys={self.meta_keys})'

