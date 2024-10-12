# WideHRNet: an Efficient Model for Human Pose Estimation Using Wide Channels in Lightweight High-Resolution Network.

## Abstract 
Human pose estimation is a task that involves locating the body joints in an image. Current deep learning models accurately estimate the locations of these joints. However, they struggle with smaller joints, such as the wrist and ankle, leading to lower accuracy. To address this problem, current models add more layers and make the model deeper to achieve higher accuracy. However, this solution adds complexity to the model. Therefore, we present an efficient network that can estimate small joints by capturing more features by increasing the network’s channels. Our network structure follows multiple stages and multiple branches while maintaining high-resolution output along the network. Hence, we called this network Wide High-Resolution Network (WideHRNet). WideHRNet provides several advantages. First, it runs in parallel and provides a high-resolution output. Second, unlike heavyweight networks, WideHRNet obtains superior results using a few layers. Third, the complexity of WideHRNet can be controlled by adjusting the hyperparameter of expansion channels. Fourth, the performance of WideHRNet is further enhanced by adding the attention mechanism. Experimental results on the MPII dataset show that the WideHRNet outperforms state-of-the-art efficient models, achieving 88.47% with the attention block.

You can read the full paper [here](https://ieeexplore.ieee.org/abstract/document/10707605)

<img width="960" height="512" src="/resources/WideHRNet.jpg"/>

# Usage 
The code was developed and tested on Ubuntu 22.04. We used 1 RTX 3060 GPU card to train and test the model. We also trained the WideHRNet model using 8 NVIDIA V100 GPU cards. Other platforms or GPU cards are not fully tested.

## Requirements
- Linux 

## Training 
We have trained our model on the MPII dataset using 1 RTX 3060 GPU card. After a while, we had a higher graphics card (8 NVIDIA V100 GPU) available, which allowed us to train our model on the COCO dataset.

Use the following command to train the model
```shell
bash ./tools/dist_train.sh <Config PATH> <NUM GPUs> --seed 0
```
Examples:
### Single machine
```shell
# Training on the MPII dataset
bash ./tools/dist_train.sh ./configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/widehrnet_18_mpii_256x256.py 1 --seed 0
```

### Multiple machines
```shell
#  Training on the COCO dataset
bash ./tools/dist_train.sh ./configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/widehrnet_18_coco_256x192.py 8 --seed 0 
```

## Testing
Use the following command to test the model
```shell
bash ./tools/dist_test.sh <Config PATH> <Checkpoint PATH> <NUM GPUs>
```
Examples:
```shell
# Testing on the MPII dataset
bash ./tools/dist_test.sh ./configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/widehrnet_18_mpii_256x256.py  ../work_dirs/epoch_210.pth 1
```

```shell
#  Testing on the COCO dataset
bash ./tools/dist_test.sh ./configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/widehrnet_18_coco_256x192.py  ../work_dirs/epoch_210.pth 1
```

## Get the computational complexity
```shell
python3 ./tools/summary_network.py ./configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/widehrnet_18_mpii_256x256.py 
```


# TODO List
- [ ] Upload WideHRNet code
- [x] Add abstract and link of WideHRNet paper
- [ ] Add results table (MPII and COCO datasets)
- [ ] Upload checkpoints
- [x] Add environment setup 
- [x] Add instructions on how to train and test the model
- [x] Add acknowledgement
- [x] Add citation 


# Acknowledgement
Thanks to:
- [MMPose](https://github.com/open-mmlab/mmpose)
- [LiteHRNet](https://github.com/HRNet/Lite-HRNet)

# Citation
If you use our code or models in your research, please cite with:
```
@article{samkari2024widehrnet,
  title={WideHRNet: an Efficient Model for Human Pose Estimation Using Wide Channels in Lightweight High-Resolution Network},
  author={Samkari, Esraa and Arif, Muhammad and Alghamdi, Manal and Al Ghamdi, Mohammed A},
  journal={IEEE Access},
  year={2024},
  publisher={IEEE}
}

```

