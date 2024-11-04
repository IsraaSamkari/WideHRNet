# WideHRNet: an Efficient Model for Human Pose Estimation Using Wide Channels in Lightweight High-Resolution Network.

## Abstract 
Human pose estimation is a task that involves locating the body joints in an image. Current deep learning models accurately estimate the locations of these joints. However, they struggle with smaller joints, such as the wrist and ankle, leading to lower accuracy. To address this problem, current models add more layers and make the model deeper to achieve higher accuracy. However, this solution adds complexity to the model. Therefore, we present an efficient network that can estimate small joints by capturing more features by increasing the network’s channels. Our network structure follows multiple stages and multiple branches while maintaining high-resolution output along the network. Hence, we called this network Wide High-Resolution Network (WideHRNet). WideHRNet provides several advantages. First, it runs in parallel and provides a high-resolution output. Second, unlike heavyweight networks, WideHRNet obtains superior results using a few layers. Third, the complexity of WideHRNet can be controlled by adjusting the hyperparameter of expansion channels. Fourth, the performance of WideHRNet is further enhanced by adding the attention mechanism. Experimental results on the MPII dataset show that the WideHRNet outperforms state-of-the-art efficient models, achieving 88.47% with the attention block.

You can read the full paper [here](https://ieeexplore.ieee.org/abstract/document/10707605)

<img width="960" height="512" src="/resources/WideHRNet.jpg"/>
Building block, where (a) is the proposed block that is inspired by various blocks, including (b) the conditional channel weighting (CCW) and (c) the inverted residual block. The stride value of all these blocks is 1. Conv: convolution, BN: batch normalization, SE: squeeze-excitation block, CRW: cross-resolution weights block, and SW: spatial weights block.

## Results and models
### Results on MPII val set
Using groundtruth bounding boxes. The metric is PCKh.  The value of the channel expansion and SE reduction ratio is 4 and 4, respectively.
| Model  | Input Size | #Params | FLOPs | PCKh | config | log | weight |
| :----------------- | :-----------: | :------: | :------: |:------: | :------: |  :------: |  :------: |
| Wide-HRNet-18 | 256x256 | 2.7M | 0.96G | 87.7 | [config](WideHRNet/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/widehrnet_18_mpii_256x256.py) |  [log](https://drive.google.com/drive/folders/1dCMll1oy_dD3jTJnAUD4ll_50mCpnJcK?usp=sharing) |  [weight](https://drive.google.com/drive/folders/1dCMll1oy_dD3jTJnAUD4ll_50mCpnJcK?usp=sharing) |
| Wide-HRNet-18 + SE| 256x256 | 4.4M | 0.97G | 88.47 | [config](WideHRNet/configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/widehrnet_18_se_mpii_256x256.py)  |  [log](https://drive.google.com/drive/folders/1Gj4YTWLM4QpfUtUP9CouG13nTmPppeKb?usp=sharing) |  [weight](https://drive.google.com/drive/folders/1Gj4YTWLM4QpfUtUP9CouG13nTmPppeKb?usp=sharing) |


### Results on COCO val2017
Using detection results from a detector that obtains 56 mAP on the person. The value of the channel expansion and SE reduction ratio is 4 and 4, respectively.
| Model  | Input Size | #Params | FLOPs | AP | AR | config | log | weight |
| :----------------- | :-----------:  | :------: | :-----------: | :------: |:------: | :------: |  :------: |  :------: |
| Wide-HRNet-18 + SE| 256x192 |4.4M | 0.9G | 69.8 | 75.6 | [config](https://drive.google.com/drive/folders/1SB0x19DvXZSRZ2ptiRx90YSUrwbNPR34?usp=sharing) |  [log](https://drive.google.com/drive/folders/1SB0x19DvXZSRZ2ptiRx90YSUrwbNPR34?usp=sharing) |  [weight](https://drive.google.com/drive/folders/1SB0x19DvXZSRZ2ptiRx90YSUrwbNPR34?usp=sharing) |



# Usage 
The code was developed and tested on Ubuntu 22.04. We used 1 RTX 3060 GPU card to train and test the model. We also trained the WideHRNet model using 8 NVIDIA V100 GPU cards. Other platforms or GPU cards are not fully tested.

## Requirements
- Linux
- Python 3.8 
- mmcv 1.4.8
- PyTorch 1.9.0
- cuda >= 11.1 

```shell
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install mmcv-full==1.4.8 -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html
pip install mmdet==2.28.2
pip install mmpose==0.29.0
# then clone this repository
pip install -r requirements.txt
pip install -v -e .
```

After installing these libraries, install timm and einops, i.e.,
```shell
pip install timm==0.4.9 einops
```

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
bash ./tools/dist_train.sh ./configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/widehrnet_18_se_coco_256x192.py 8 --seed 0 
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
bash ./tools/dist_test.sh ./configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/coco/widehrnet_18_se_coco_256x192.py  ../work_dirs/epoch_210.pth 1
```

## Get the computational complexity
```shell
python3 ./tools/summary_network.py ./configs/body/2d_kpt_sview_rgb_img/topdown_heatmap/mpii/widehrnet_18_mpii_256x256.py 
```


# TODO List
- [x] Upload WideHRNet code
- [x] Add abstract and link of WideHRNet paper
- [x] Add results table (MPII and COCO datasets)
- [x] Upload checkpoints
- [ ] Update checkpoints
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

