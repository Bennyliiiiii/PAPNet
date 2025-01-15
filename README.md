
# PAPNet

#### PAPNet: Point-enhanced Attention-aware Pillar Network for 3D Object Detection in Autonomous Driving

## 1. Installation

### Requirements
All the codes are tested in the following environment:
* Linux (tested on Ubuntu 20.04)
* Python 3.6+
* PyTorch 1.1 or higher (tested on PyTorch 1.10)
* CUDA 11.1
* [`spconv v1.2`](https://github.com/traveller59/spconv) 

### Install `pcdet v0.5`
NOTE: Please re-install `pcdet v0.5` by running `python setup.py develop` even if you have already installed previous version.

a. Clone this repository.
```shell
git clone https://github.com/open-mmlab/OpenPCDet.git
```

b. Install the dependent libraries as follows:

[comment]: <> (* Install the dependent python libraries: )

[comment]: <> (```)

[comment]: <> (pip install -r requirements.txt )

[comment]: <> (```)

Install the SparseConv library, we use the implementation from [`[spconv]`](https://github.com/traveller59/spconv). 

c. Install this `pcdet` library and its dependent libraries by running the following command:
```shell
python setup.py develop
```

## 2. Getting Started

The dataset configs are located within [tools/cfgs/dataset_configs](../tools/cfgs/dataset_configs), 
and the model configs are located within [tools/cfgs](../tools/cfgs) for different datasets. 

### KITTI Data
* Please download the official [KITTI 3D object detection](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) dataset and organize the downloaded files as follows (the road planes could be downloaded from [[road plane]](https://drive.google.com/file/d/1d5mq0RXRnvHPVeKx6Q612z0YRO1t2wAp/view?usp=sharing), which are optional for data augmentation in the training):
* If you would like to train [CaDDN](../tools/cfgs/kitti_models/CaDDN.yaml), download the precomputed [depth maps](https://drive.google.com/file/d/1qFZux7KC_gJ0UHEg-qGJKqteE9Ivojin/view?usp=sharing) for the KITTI training set

```
PAPNet
├── dataset_data
│   ├── kitti
│   │   │── ImageSets
│   │   │── training
│   │   │   ├──calib & velodyne & label_2 & image_2 & (optional: planes) & (optional: depth_2)
│   │   │── testing
│   │   │   ├──calib & velodyne & image_2
├── pcdet
├── tools
```

* Generate the data infos by running the following command: 
```python 
python -m PAPNet.datasets.kitti.kitti_dataset create_kitti_infos tools/cfgs/dataset_configs/kitti_dataset.yaml
```

    Now, we have datasets as follows:
    ```
    kitti
        |- training
            |- calib (#7481 .txt)
            |- image_2 (#7481 .png)
            |- label_2 (#7481 .txt)
            |- velodyne (#7481 .bin)
            |- velodyne_reduced (#7481 .bin)
        |- testing
            |- calib (#7518 .txt)
            |- image_2 (#7518 .png)
            |- velodyne (#7518 .bin)
            |- velodyne_reduced (#7518 .bin)
        |- kitti_gt_database (# 19700 .bin)
        |- kitti_infos_train.pkl
        |- kitti_infos_val.pkl
        |- kitti_infos_trainval.pkl
        |- kitti_infos_test.pkl
        |- kitti_dbinfos_train.pkl
    
    ```

### Real World Data (Note: Real-world data is unlabeled.)
If you want to test model on real-world object detection,  you should prepare the data like this:
* Download data (The data is saved in .bag format in folder **real_word_data**.)
* Convert the .bag data into .bin files. (You should modify the specific file name and file path correctly.)
```
python bag2bin.py
```
* Convert the .bin file from the previous step into the standard KITTI format. (You should modify the specific file name and file path correctly.)
```
python bin2kittibin.py
```

## 3. Training & Testing

### Test the pretrained models
* Test with a pretrained model: 
```
python test.py --cfg_file ${CONFIG_FILE} --batch_size ${BATCH_SIZE} --ckpt ${CKPT}
```


### Train a model

* Train with a single GPU:
```
python train.py --cfg_file ${CONFIG_FILE}
```


## Acknowledements

**To be continued**......Thanks for the open souce code [OpenPCDet](https://github.com/open-mmlab/OpenPCDet)！




