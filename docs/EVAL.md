# Clsoed-Loop Evaluation

To evaluate in the town05long:
```shell
## In the ThinkTwice/ directory
port_for_carla=22023 ## Change the port for each running script to avoid cofliction
port_for_traffic_manager=22033 ## Change the port for each running script to avoid cofliction
team_agent=thinktwice_agent
is_resume=False ## If there is the corresponding json file in the folder closed_loop_eval_log, you could set it as True to continue after the last finished route.
is_local=True
ckpt_and_config_path=open_loop_training/ckpt/thinktwice.pth+open_loop_training/configs/thinktwice.py
scenario_file=all_towns_traffic_scenarios_no256
cuda_device=0
setting_name=thinktwice_town05long
CUDA_VISIBLE_DEVICES=$cuda_device nohup bash ./leaderboard/scripts/evaluation_town05long.sh $port_for_carla $port_for_traffic_manager $team_agent $is_resume $is_local $ckpt_and_config_path $scenario_file $setting_name 2>&1 > $setting_name.log &
```

or simply:
```shell
## In the ThinkTwice/ directory
CUDA_VISIBLE_DEVICES=0 nohup bash ./leaderboard/scripts/evaluation_town05long.sh 22023 22033 thinktwice_agent  False True open_loop_training/ckpt/thinktwice.pth+open_loop_training/configs/thinktwice.py all_towns_traffic_scenarios_no256 thinktwice_town05long 2>&1 > thinktwice_town05long.log &
```

To evaluate in the longest6, you can simply use:
```shell
## In the ThinkTwice/ directory
CUDA_VISIBLE_DEVICES=0 nohup bash ./leaderboard/scripts/evaluation_longest6.sh 23023 23033 thinktwice_agent  False True open_loop_training/ckpt/thinktwice.pth+open_loop_training/configs/thinktwice.py longest6_eval_scenarios thinktwice_longest6 2>&1 > thinktwice_longest6.log &
```

Note that the evaluation result is in the directory **closed_loop_eval_log/results_$setting_name.json** and the visualizations and recordings for debug (top-down view, front view, and canbus) are in the directory **closed_loop_eval_log/eval_log/$setting_name**.

Warning: The visualizations and recordings could take lots of disk space. Please monitor those folders in the [closed_loop_eval_log/eval_log/](../closed_loop_eval_log/eval_log/) and delete those useless ones in time. You could also modify the **save** function of [leaderboard/team_code/thinktwice_agent.py](../leaderboard/team_code/thinktwice_agent.py) to change the saved information during evaluation.