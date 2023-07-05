# Prepare Dataset

## Script to Collect for One Xml File
Due to the huge size of the dataset (189K frames - 8TB, 2M frames - 85TB), we provide the instructations to generate the dataset. If the size is still too large for your device, techniques to reduce the size of files could be explored. We look forward to your contributions:). 
Update: If your disk space is limited, you could consider trying the techniques mentioned in the compression section of this awesome repo [carla_garge](https://github.com/autonomousvision/carla_garage/blob/main/docs/engineering.md).

The command to collect data from one .xml file is:

```shell
## In the ThinkTwice/ directory
port_for_carla=22023 ## Change the port for each running script to avoid cofliction
port_for_traffic_manager=22033 ## Change the port for each running script to avoid cofliction
route_file=town05_00
is_resume=False ## If there is the corresponding json file in the folder collect_data_json, you could set it as True to continue after the last finished route.
is_local=True
agent_name=roach_ap_agent_data_collection
scenario_file=all_towns_traffic_scenarios_no256
cuda_device=0
CUDA_VISIBLE_DEVICES=$cuda_device nohup bash ./leaderboard/scripts/collect_data.sh $port_for_carla $port_for_traffic_manager $route_file $is_resume $is_local $agent_name  $scenario_file 2>&1 > $route_file.log &
```

or simply:

```shell
CUDA_VISIBLE_DEVICES=0 nohup bash ./leaderboard/scripts/collect_data.sh 22023 22033 town05_00 False True roach_ap_agent_data_collection all_towns_traffic_scenarios_no256  2>&1 > town05_00.log &
```

Note that one Carla simulator will be opened by the script and there is no need to open Carla manually. **You should manually kill that Carla to avoid *error: address is used* after the collection is done or reports errors.**

## Collect Whole Dataset

For 189K training data + the rest of open loop valdiation data = 400k frames in TCP, please collect data with .xmls in [leaderboard/data/routes_for_open_loop_training](../leaderboard/data/routes_for_open_loop_training) with suffix *00, 01, 02, val*. During open-looped running, we train on town 01,03, 04, 06 and validate on town 02, 05 as in LAV, Transfuser, TCP.

For 2M frames data, please collect data with all .xmls in [leaderboard/data/routes_for_open_loop_training](../leaderboard/data/routes_for_open_loop_training). During open-looped running, we train on all routes except those with suffix *val*.


## Generate More Routes
If you want to collect more data with random routes, you could use [dataset/tools/generate_random_routes.py](../dataset/tools/generate_random_routes.py)

```shell
CUDA_VISIBLE_DEVICES=0  python generate_random_routes.py --save_file tmp.xml --town Town02 --route_num 5 --port 22023
```

## Generate Metadata for Open-Loop Training
After collecting data by the expert, we filter out those frames where expert made mistakes and then generate the meta information for all routes together:
```shell
#In ThinkTwice/dataset/tools directory
python generate_metadata.py #It could take a day for the 2M frames dataset.
```

## Additional Notes

- We strongly recommnd you **first collecting a small number of data** (for example, one route with ~20 frames) to make sure that the functions of open-loop training and closed-loop evaluation work well (by thorough visualizations and test runs).  Otherwise, it could be time-consuming and frustrating to find out that there are some bugs or missing information (for example, we do not use semantic lidar sensor in our code).

- Since we do not use any bounding boxs in our code, **if you would like to add some functions about bounding box, please first make sure that the collected information is correct and enough for your desired usage.** Test runs and visualizations as mentioned in the previous note is strongly recommended. 

- There are several coordinate system within the code: Unreal, Roach, Lidar, Camera, GNSS, Control, etc. If you want to modify or add any new component, visualizations to check whether they align with existing ones are strongly recommended

- You could modify [leaderboard/team_code/roach_ap_agent_data_collection.py](../leaderboard/team_code/roach_ap_agent_data_collection.py) to save the information you need for training. 

- For the size of images, we collect images at 900x1600 and we downsample them into 448x896 to save GPU memory. Thus, you may change the collected images' resolution (as well as depth and segmentation labels) to save disk space.

- For traffic light segmentation in images, we wrote rules regarding color to generate GT, which might not be accurate. You could consider use the build-in API of Carla to generate the GT.
  
- To reduce the data collection cost, you could first only collect data from Town01, Town03, Town04, and Town06 for training as in Appendix B Table 1 of [TCP](https://arxiv.org/pdf/2206.08129.pdf) and do not conduct open-loop validation, which could approximately half the needed data. When the model achieves *current_throttle_brake_offset < 0.1* and *longitudinal_offset < 0.2* in training set, it is considered fitting well.
