import os
from unittest import result
import cv2
import copy
import mmcv
import torch
import random
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms as T

from nuscenes.eval.common.utils import Quaternion
import pickle
from mmcv.parallel import DataContainer as DC
from mmdet.datasets import DATASETS, CustomDataset
from mmdet3d.datasets.pipelines import Compose

from .base_dataset import BaseDataset
import json
import math
import io

@DATASETS.register_module()
class CarlaDataset(BaseDataset):
    r"""Carla Dataset.
    """
    def __init__(self, 
                cfg,
                used_town,
                pipeline, 
                is_local,
                full_queue_pipeline,
                test_mode=False,
                **kwargs):
        super(CarlaDataset).__init__()
        self.test_mode = test_mode
        self.pred_len = cfg["pred_len"]
        ## Pipeline for single frame
        self.pipeline = Compose(pipeline)
        ## Pipeline for a sequence of history frames
        self.full_queue_pipeline = Compose(full_queue_pipeline)
        self.resize_to21x21 = T.Resize(size=(21, 21))
        self.is_local = bool(cfg['is_local']) ## Ignore this, we train out model on a cluster with ceph
        self.is_full = False ## 189K or 2M dataset
        if "is_full" in cfg:
            self.is_full = bool(cfg["is_full"]) ## 2M frames dataset setting
        dataset_size_fname = "dataset_metadata.pkl"

        with open("../dataset/"+dataset_size_fname, "rb") as f:
            route_length_dict = pickle.load(f)

        self.history_query_index_lis = cfg["history_query_index_lis"]
        route_start_index = -self.history_query_index_lis[0]
        self.data_infos = [] #tuple: (route_folder, current_timestep)
        
        ### Prepare for dataloader to shuffle data
        town_sampel_cnt_dict = {}
        for single_used_town in used_town:
            if single_used_town not in route_length_dict: ## No data collected
                print(single_used_town, "Not Found")
                continue
            now_town = single_used_town[:6] ## town_name without index
            if now_town not in town_sampel_cnt_dict:
                town_sampel_cnt_dict[now_town] = 0
            is_val_town = ("02" in now_town or "05" in now_town)
            ##When we only use the 189K dataset
            if (not self.is_full or is_val_town) and town_sampel_cnt_dict[now_town] > cfg["max_sample_per_town"][now_town]:
                continue
            for route in route_length_dict[single_used_town]:
                ##When we only use the 189K dataset
                if (not self.is_full or is_val_town) and town_sampel_cnt_dict[now_town] > cfg["max_sample_per_town"][now_town]:
                    break
                ##Not use the first few frames
                if route_length_dict[single_used_town][route] < route_start_index+1+self.pred_len:
                    continue
                if self.is_local:
                    route_path = route[:3] + "dataset/"+route[3:]
                else:
                    route_path = route
                new_data_info = [(route_path, current_time_step) for current_time_step in range(route_start_index, route_length_dict[single_used_town][route]-self.pred_len)]
                self.data_infos += new_data_info
                town_sampel_cnt_dict[now_town] += len(new_data_info)
        print(town_sampel_cnt_dict)
        if not self.is_local: ## ignore this line (we train our model on a cluster with ceph)
            ceph_conf = '~/petreloss.conf'
            from petrel_client.client import Client
            self.client = Client(ceph_conf)
        self.flag = np.ones(len(self), dtype=np.uint8)
        self.cfg = cfg
    
    def load_json(self, fname):
        if self.is_local:
            with open(fname, "r") as f:
                return json.load(f)
        else:
            return json.loads(self.client.get(fname))

    def load_npy(self, fname):
        if self.is_local:
            with open(fname, "rb") as f:
                return np.load(fname, allow_pickle=True)
        else:
            return np.load(io.BytesIO(self.client.get(fname)), allow_pickle=True)

    def offset_then_rotate(self, target_2d_world_coor, ref_2d_wolrd_coor, ref_yaw):
        final_coor = target_2d_world_coor - ref_2d_wolrd_coor
        R = np.array([
            [np.cos(ref_yaw), -np.sin(ref_yaw)],
            [np.sin(ref_yaw), np.cos(ref_yaw)]
        ])
        return np.einsum("ij,kj->ki", R.T, final_coor)

    ## Preprocess for single frame
    def get_data_info(self, data_folder, route_index, is_current):
        results = {}
        results["scene_token"] = data_folder
        results["frame_idx"] = route_index
        measurements = self.load_json(os.path.join(data_folder, "measurements", f"{str(route_index).zfill(4)}.json"))
        ego_theta = measurements["theta"] if not np.isnan(measurements["theta"]) else 0# fix for theta=nan in some measurements
        ego_theta = ego_theta - np.pi/2 #Follow https://github.com/dotchen/LAV/blob/23e2f1be3b1b43593761bc7ea7beabda1086b253/team_code/lav_agent.py
        results["input_theta"] = ego_theta
        results["input_x"] = measurements["y"] ## All coordinates are in the ego coordinate system (go front=vertically up in BEV similar to Roach)
        results["input_y"] = -measurements["x"] ## All coordinates are in the ego coordinate system (go front=vertically up in BEV similar to Roach)
        ego_xy = np.stack([results["input_x"], results["input_y"]], axis=-1)
        
        if is_current:
            future_measurements_lis = []
            for future_index in range(1, self.pred_len+1):
                future_measurements_lis.append(self.load_json(os.path.join(data_folder, "measurements", f"{str(route_index+future_index).zfill(4)}.json")))
            future_x =  np.array([_["y"] for _ in future_measurements_lis])
            future_y = -np.array([_["x"] for _ in future_measurements_lis])
            results['waypoints'] = self.offset_then_rotate(np.stack([future_x, future_y], axis=-1), ego_xy, ego_theta)
            results["future_speed"] = [_["speed"] for _ in future_measurements_lis]

        results["speed"] = measurements["speed"]
        results["can_bus"] = np.zeros(18)
        results["can_bus"][0] = results["input_x"] ## Gloabal Coordinate
        results["can_bus"][1] = results["input_y"] ## Global Coordinate
        accel = np.array(measurements["acceleration"])
        accel[:2] = self.offset_then_rotate(np.array(accel[:2])[np.newaxis, :], np.array([0, 0]), ego_theta).squeeze(0)
        results["can_bus"][7:10] = accel
        results["can_bus"][10:13] = measurements["angular_velocity"]
        results["can_bus"][13] = measurements["speed"]
        results["can_bus"][-2] = ego_theta
        results["can_bus"][-1] = ego_theta / np.pi * 180
        
        x_target = measurements["y_target"]  ## All coordinates are in the ego coordinate system (go front=vertically up in BEV similar to Roach)
        y_target = -measurements["x_target"]  ## All coordinates are in the ego coordinate system (go front=vertically up in BEV similar to Roach)
        results['target_point'] = self.offset_then_rotate(np.array([[x_target, y_target]]), ego_xy, ego_theta).squeeze(0)
        results['target_point_aim'] = results["target_point"]
        # VOID = -1
        # LEFT = 1
        # RIGHT = 2
        # STRAIGHT = 3
        # LANEFOLLOW = 4
        # CHANGELANELEFT = 5
        # CHANGELANERIGHT = 6
        command = measurements["target_command"]
        if command < 0:
            command = 4
        command -= 1
        results['target_command_raw'] = torch.tensor(command).long()
        assert command in [0, 1, 2, 3, 4, 5]
        cmd_one_hot = [0] * 6
        cmd_one_hot[command] = 1
        results['target_command'] = torch.tensor(cmd_one_hot)

        results["pts_filename"] = os.path.join(data_folder, "lidar", f"{str(route_index).zfill(4)}.npy")
        if is_current:
            ##Distill Feature
            current_supervision = self.load_npy(os.path.join(data_folder, "supervision", f"{str(route_index).zfill(4)}.npy")).item()
            ## [throttle, steer, brake]
            results["action"] = current_supervision["action"]
            results["action_mu"] = current_supervision["action_mu"]
            results["action_sigma"] = current_supervision["action_sigma"]
            ## Brake by rules
            if current_supervision['only_ap_brake']:
                results['action_mu'][0] = 0.8
                results['action_sigma'][0] = 5.5

            ## Label
            future_supervision_lis = []
            for future_index in range(1, self.pred_len+1):
                future_supervision_lis.append(self.load_npy(os.path.join(data_folder, "supervision", f"{str(route_index+future_index).zfill(4)}.npy")).item())

            future_only_ap_brake = [_["only_ap_brake"] for _ in future_supervision_lis]

            results['future_action_mu'] = [_["action_mu"] for _ in future_supervision_lis]  ## pred_len x 2
            results['future_action_sigma'] =[_["action_sigma"] for _ in future_supervision_lis] ## pred_len x 2
            results['future_action'] = np.stack([_["action"] for _ in future_supervision_lis], axis=0) ## pred_len x (control.throttle, control.steer, control.brake)
            # use the average value of roach braking action when the brake is only performed by the rule-based detector
            for future_index in range(self.pred_len):
                if future_only_ap_brake[future_index]:
                    results['future_action_mu'][future_index][0] = 0.8
                    results['future_action_sigma'][future_index][0] = 5.5
            results["value"] = current_supervision["value"]
            results["feature"] = current_supervision["features"]
            results["grid_feature"] = current_supervision["cnn_features"]
            results["future_feature"] = [_["features"] for _ in future_supervision_lis]
            results["future_value"] = [_["value"] for _ in future_supervision_lis]
            results["future_grid_feature"] = [_["cnn_features"] for _ in future_supervision_lis]
        return results

    def prepare_train_data(self, index):
        """Returns the item at index idx. """
        data_queue = []
        # temporal aug      # random choose 3 frames from last 4 frames
        #prev_indexs_list = list(range(index-self.queue_length, index))
        #random.shuffle(prev_indexs_list)
        #prev_indexs_list = sorted(prev_indexs_list[1:], reverse=True)
        prev_indexs_list = self.history_query_index_lis

        data_folder, route_index = self.data_infos[index]
        input_dict = self.get_data_info(data_folder, route_index, is_current=True)
        if input_dict is None:
            return None

        example = self.pipeline(input_dict)
        data_queue.insert(0, example)
        # Load prev frames, not load the current
        for i in prev_indexs_list[:-1][::-1]:
            input_dict = self.get_data_info(data_folder, route_index + i, is_current=False)
            input_dict["is_local"] = self.is_local
            example = self.pipeline(input_dict)
            data_queue.insert(0, copy.deepcopy(example))
        return union2one(self.full_queue_pipeline, data_queue)
    def __getitem__(self, idx):
        while True:
            data = self.prepare_train_data(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue
            return data

    def _rand_another(self, idx):
        """Randomly get another item with the same flag.

        Returns:
            int: Another index of item with the same flag.
        """
        pool = np.where(self.flag == self.flag[idx])[0]
        return np.random.choice(pool)

    def prepare_test_data(self, index):
        input_dict = self.get_data_info(index)
        example = self.pipeline(input_dict)
        return example



def get_ego_shift(delta_x, delta_y, ego_angle):
    # obtain rotation angle and shift with ego motion
    translation_length = np.sqrt(delta_x ** 2 + delta_y ** 2)
    translation_angle = np.arctan2(delta_y, delta_x) / np.pi * 180
    bev_angle = ego_angle - translation_angle
    shift_y = translation_length * np.cos(bev_angle / 180 * np.pi)
    shift_x = translation_length * np.sin(bev_angle / 180 * np.pi) 
    return shift_x, shift_y


### Multiple frames
def union2one(full_queue_pipeline, queue):
    queue = full_queue_pipeline(queue)
    imgs_list = torch.stack([each['img'] for each in queue])
    points_size = [each['points'].data.shape[0] for each in queue]
    metas_map = []
    prev_pos = None
    prev_angle = None
    for i, each in enumerate(queue):
        meta = copy.deepcopy(each['img_metas'].data)
        meta['points_size'] = points_size
        if i == 0: ## Current Frame
            meta['prev_bev'] = False
            prev_pos = copy.deepcopy(meta['can_bus'][:3])
            prev_angle = copy.deepcopy(meta['can_bus'][-1])
            meta['can_bus'][:3] = 0
            meta['can_bus'][-1] = 0
        else:
            meta['prev_bev'] = True
            tmp_pos = copy.deepcopy(meta['can_bus'][:3])
            tmp_angle = copy.deepcopy(meta['can_bus'][-1])
            meta['can_bus'][:3] -= prev_pos
            meta['can_bus'][-1] -= prev_angle
            prev_pos = copy.deepcopy(tmp_pos)
            prev_angle = copy.deepcopy(tmp_angle)
        metas_map.append(meta)

    # sweep2key transformation
    metas_map[-1]['curr2key'] = torch.eye(4)
    metas_map[-1]['currlidar2keycam'] = metas_map[-1]['lidar2cam']
    key_x, key_y = queue[-1]['img_metas'].data['can_bus'][:2]
    key_yaw = queue[-1]['img_metas'].data['can_bus'][-2]
    for i in range(len(queue)-2, -1, -1):
        curr_x = queue[i]['img_metas'].data['can_bus'][0]
        curr_y = queue[i]['img_metas'].data['can_bus'][1]
        curr2key_x, curr2key_y = get_ego_shift(
            key_x - curr_x,
            key_y - curr_y,
            key_yaw / np.pi * 180
        )
        curr_yaw = queue[i]['img_metas'].data['can_bus'][-2]
        curr2key_angle = key_yaw - curr_yaw 
        # get transmation mats
        R = torch.eye(4)
        R[:2,:2] = torch.Tensor([[np.cos(curr2key_angle), np.sin(curr2key_angle)],
                    [-np.sin(curr2key_angle), np.cos(curr2key_angle)]])
        T = torch.eye(4)
        T[0,3], T[1,3] = curr2key_x, curr2key_y
        curr2key = R @ T
        metas_map[i]['curr2key'] = curr2key
        metas_map[i]['currlidar2keycam'] = metas_map[i]['lidar2cam'] @ curr2key

    # dense-fusion
    points = queue[-1]['points'].data     
    points = torch.cat([points,torch.zeros(points.shape[0],1)], dim=1)
    points[:,4] = 0
    points_list = [points]
    for i in range(len(queue)-2, -1, -1):
        points_sweep = copy.deepcopy(queue[i]['points'].data)
        points_sweep = torch.cat([points_sweep,torch.zeros(points_sweep.shape[0],1)], dim=1)
        curr2key = metas_map[i]['curr2key']
        points_sweep[:, :4] = (curr2key@points_sweep[:, :4].T).T
        timestamp = i-(len(queue)-1)
        points_sweep[:,4] = timestamp
        points_list.append(points_sweep)
        points = torch.cat(points_list).unsqueeze(0)
        queue[-1]['points'] = DC(points, cpu_only=False, stack=True)
        
    queue[-1]['img'] = DC(imgs_list,
                            cpu_only=False, stack=True)
    queue[-1]['img_metas'] = DC(metas_map, cpu_only=True)
    queue = queue[-1]
    return queue
