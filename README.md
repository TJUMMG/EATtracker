## Edge-aware Object Pixel-level Representation Tracking

###### This is a PyTorch implementation of our proposed EATtracker. 

Copyright(c) 2022 Jing Liu
```
If you reference this paper, please cite the following publication:
Peiguang Jing, Zijian Huang, Jing Liu, Yating Wang, Jiexiao Yu,"Edge-aware object pixel-level representation tracking", to appear in Journal of Visual Communication and Image Representation.
```


## Prerequisites
-pytorch 1.1.0

-python 3.7

-cudatoolkit 10.0

Usage
--------------------------
### Download Datasets
1. Download the youtube-vos dataset, put it into /EATtracker/data/. [BaiDuYun]：https://pan.baidu.com/s/1WMB0q9GJson75QBFVfeH5A password: sf1m (SiamMask)
2. Download the DAVIS dataset, put it into /EATtracker/dataset/. [BaiDuYun]：https://pan.baidu.com/s/1JTsumpnkWotEJQE7KQmh6A password: c9qp (SiamMask)

### Download models 
1. Please download the model trained by authors. Put it into /EATtracker/tracking/snapshot/. [BaiDuYun]：https://pan.baidu.com/s/1g3yKi_qz16USF4rKnLLXEA?pwd=zqg3 password:zqg3 
2. Please download the pre-trained model. Put it into /EATtracker/pretrain/. [BaiDuYun]：https://pan.baidu.com/s/1VgN-o1yVkOMD9ptDy0qT6w?pwd=8dg0 
password:8dg0 


### Test
1. run the '/EATtracker/tracking/test_oceanplus.py'.

    The test results will be output under the /EATtracker/tracking/result/.
### Train 
1. run the '/EATtracker/tracking/train_oceanplus_wyt.py'.

    The training generated model will be saved in /EATtracker/tracking/snapshot/.

