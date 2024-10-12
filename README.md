# [WideHRNet: an Efficient Model for Human Pose Estimation Using Wide Channels in Lightweight High-Resolution Network](https://ieeexplore.ieee.org/abstract/document/10707605).

## Abstract 
Human pose estimation is a task that involves locating the body joints in an image. Current deep learning models accurately estimate the locations of these joints. However, they struggle with smaller joints, such as the wrist and ankle, leading to lower accuracy. To address this problem, current models add more layers and make the model deeper to achieve higher accuracy. However, this solution adds complexity to the model. Therefore, we present an efficient network that can estimate small joints by capturing more features by increasing the network’s channels. Our network structure follows multiple stages and multiple branches while maintaining high-resolution output along the network. Hence, we called this network Wide High-Resolution Network (WideHRNet). WideHRNet provides several advantages. First, it runs in parallel and provides a high-resolution output. Second, unlike heavyweight networks, WideHRNet obtains superior results using a few layers. Third, the complexity of WideHRNet can be controlled by adjusting the hyperparameter of expansion channels. Fourth, the performance of WideHRNet is further enhanced by adding the attention mechanism. Experimental results on the MPII dataset show that the WideHRNet outperforms state-of-the-art efficient models, achieving 88.47% with the attention block.


# TODO List
- [x] Upload WideHRNet code
- [x] Add abstract and link of WideHRNet paper
- [ ] Add comparison table (MPII and COCO datasets)
- [ ] Add environment setup 
- [ ] Add instructions on how to train and test the model
- [ ] Add Acknowledgement
- [ ] Add citation 
