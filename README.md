## Edge-aware Object Pixel-level Representation Tracking

###### This is a PyTorch implementation of our proposed EATtracker. 

## Prerequisites
-pytorch 1.1.0

-python 3.7

-cudatoolkit 10.0

Usage
--------------------------
### Download Datasets
1. Download the youtube-vos dataset, put it into /EATtracker/data/.
2. Download the DAVIS dataset, put it into /EATtracker/dataset/.

### Download models 
1. Please download the model trained by authors. [BaiDuYun]：https://pan.baidu.com/s/1g3yKi_qz16USF4rKnLLXEA?pwd=zqg3 提取码：zqg3 
2. Please download the pre-trained model. Put it into /EATtracker/pretrain/. [BaiDuYun]：https://pan.baidu.com/s/1VgN-o1yVkOMD9ptDy0qT6w?pwd=8dg0 
提取码：8dg0 


### Test
1. run the './tracking/test_oceanplus.py'.
### Train 
1. run the './tracking/train_oceanplus_wyt.py'.

```
If you reference this paper, please cite the following publication:
Peiguang Jing, Zijian Huang, Jing Liu, Yating Wang, Jiexiao Yu,"Edge-aware object pixel-level representation tracking", to appear in Journal of Visual Communication and Image Representation.
```
