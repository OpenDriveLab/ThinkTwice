# Installation
Modified from mmdetection3d, BEVFormer, UniAD

**a. Env: Create a conda virtual environment and activate it.**
```shell
conda create -n thinktwice python=3.7 -y ## Must be py3.7 required by Carla 9.10
conda activate thinktwice
```

**b. Torch: Install PyTorch and torchvision following the [official instructions](https://pytorch.org/).**
```shell
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
```

**c. GCC: Make sure gcc>=5 in conda env.**
```shell
conda install -c omgarcia gcc-6
conda install libgcc -y
conda install -c conda-forge libcxxabi -y
```

**d. CUDA: Before installing MMCV family, you need to set up the environment variables in ~/.bashrc (for compiling some operators on the gpu).**
```shell
export CUDA_HOME=YOUR_CUDA_PATH/
# Eg: export CUDA_HOME=/mnt/cuda-11.3/
export PATH=$PATH:YOUR_CUDA_PATH//bin
# Eg: export CUDA_HOME=/mnt/cuda-11.3/bin
export LD_LIBRARY_PATH=YOUR_CUDA_PATH/lib64:YOUR_CONDA_PATH/envs/thinktwice/lib
# Eg: export LD_LIBRARY_PATH=/mnt/cuda-11.3/lib64:/mnt/miniconda3/envs/thinktwice/lib
export LD_PRELOAD=YOUR_CONDA_PATH/envs/thinktwice/lib/libstdc++.so.6.0.29
# Eg: export LD_PRELOAD=/mnt/miniconda3/envs/thinktwice/lib/libstdc++.so.6.0.29
```


**e. Install mmcv-full, mmdet, and mmseg**
```shell
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.12/index.html
pip install mmdet==2.28.2
pip install mmsegmentation==0.30.0
```


**f. Install mmdet3d from source code.**
```shell
git clone https://github.com/open-mmlab/mmdetection3d.git -b 1.0 ## Note that 1.1 is incompatible
cd mmdetection3d
pip install cumm-cu113
pip install spconv-cu113 ## For fast lidar model
MMCV_WITH_OPS=1 FORCE_CUDA=1 pip install -v -e . ## Must have a GPU
```


**g. Other Requirements.**
```shell
conda install -c anaconda protobuf -y
conda install matplotlib -y
conda install requests -y
conda install tabulate -y
conda install protobuf -y
conda install -c anaconda setuptools -y
conda install -c conda-forge libjpeg-turbo -y
conda install -c anaconda decorator -y
conda install shapely -y
conda install -c anaconda ephem -y
conda install -c conda-forge omegaconf -y
conda install -c conda-forge hydra -y
conda install -c anaconda scikit-image -y
conda install -c conda-forge opt_einsum -y
conda install -c conda-forge mpi4py -y
conda install h5py -y
conda install -c conda-forge importlib-metadata -y
conda install -c conda-forge zipp -y
pip install gym==0.17.2
pip install imgaug
pip install opencv-python
pip install opencv-contrib-python
pip install pygame
pip install py_trees==0.8.1
pip install dictor
pip install gym==0.17.2
pip install stable-baselines3==0.8.0
pip install numpy --upgrade

pip install shapely==1.6.4.post2  --force-reinstall  ### Important! Higher version of shapely may make the CARLA crash mysteriously!
```

**h. Install Carla. (From Roach)**
```shell
mkdir carla
cd carla
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.10.1.tar.gz
tar -xvzf CARLA_0.9.10.1.tar.gz
echo "export CARLA_ROOT=/mnt/carla" >> /mnt/.bashrc ## Set to your own Carla path and ~/.bashrc in your system
cd Import && wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/AdditionalMaps_0.9.10.1.tar.gz
cd .. && bash ImportAssets.sh
rm CARLA_0.9.11.tar.gz Import/AdditionalMaps_0.9.10.1.tar.gz
rm Import/AdditionalMaps_0.9.10.1.tar.gz Import/AdditionalMaps_0.9.11.tar.gz
cd && source .bashrc

echo "$CARLA_ROOT/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg" >> YOUR_CONDA_PATH/envs/thinktwice/lib/python3.7/site-packages/carla.pth
```

**i. Clone ThinkTwice.**
```shell
git clone git@github.com:OpenDriveLab/ThinkTwice.git
cd ThinkTwice/open_loop_training
python setup.py develop ## Compile CUDA function for LSS from BEVDepth
```
