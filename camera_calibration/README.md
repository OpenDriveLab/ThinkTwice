# Camera Calibration

Under the current setting, all four cameras have **fov 150**. If you want to change their fov, you could use the following code to estimate the corresponding distortion parameters. More discussions here: https://github.com/carla-simulator/carla/issues/3412.

1. Build Carla from source code with commit 0c97e9a5de5b35b759cc5b0955801244ed76791f:
   - Guidance https://carla.readthedocs.io/en/0.9.10/build_linux/
   - git clone the latest commit
   - git checkout -b branch_id commit_id
   - make launch == ./CarlaUE4.sh
   
   Note that the Carla built from source and its cooresponding environment (Python, environment variables, etc) are only used for estimating the distortion parameters. In all other experiments, we use the official pre-build version 0.9.10.1 and the environment under our setting. If there is any problem during building the Carla, please refer to https://carla.readthedocs.io/en/0.9.10/build_linux/.

2. Condcut Checkboard Calibration
   - Download the checkboard asset for Carla Town3
   https://github.com/AbanobSoliman/IBISCape and put it in the suggested folder of Carla  from source
   - In the Carla UE4 editor, click make launch -> Compile -> Build -> Play
   - **python collect_data_for_calibration.py** (sensor setting) to collect image for calibration (For more information about the data collecting process, please refer to tutorials about checkboard camera calibration: https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html and the instruction in the original code https://github.com/AbanobSoliman/IBISCape)
   - **python calculate_distortion_parameters.py** (line 26-35 to calculate intrinsic matrix, line 32 to select image index for calibration) to output distortion matrix
   - Put an image collected by your Carla into this folded and rename it as "test.png".
   - **python undistort.py** (line 4-10 to set intrinsics matrix, line 14 to input the distortion matrix from the last step) to verify the effectiveness of the undistortion. Note that the Tangential Distortion effects need extra attention when collect images.


## Images without Distortion
If your application does not involve official benchmarks such town05Long, Longest6 or Leaderboard, you could eliminate all distortions. In [leaderboard/leaderboard/autoagents/agent_wrapper.py](../leaderboard/leaderboard/autoagents/agent_wrapper.py), set all *lens_circle_multiplier* to 0.0, all *lens_circle_falloff* to 5.0, and all *chromatic_aberration_intensity* to 0.0. 

Reference: https://carla.readthedocs.io/en/latest/ref_sensors/


## Acknowledgements

The calibration code is based on [IBISCape](https://github.com/AbanobSoliman/IBISCape) and please check out their awsome code.