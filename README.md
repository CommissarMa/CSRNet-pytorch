# CSRNet-Simple-Pytorch
This is an simple and clean implemention of CVPR 2018 paper ["CSRNet: Dilated Convolutional Neural Networks for Understanding the Highly Congested Scenes"](https://arxiv.org/abs/1802.10062).  

## Requirement
1. Install pytorch 1.0.0+
2. Install tensorboardX
3. Clone this repository  
    ```git
    git clone https://github.com/CommissarMa/CSRNet-pytorch.git
    ```
    We'll call the directory that you cloned CSRNet-pytorch as ROOT.

## Data Setup
1. Download ShanghaiTech Dataset from
[Dropbox](https://www.dropbox.com/s/fipgjqxl7uj8hd5/ShanghaiTech.zip?dl=0) or [Baidu Disk](https://pan.baidu.com/s/101mNo_Vz21IwDYnYTnLQpw) (code: a2v8)   
2. Put ShanghaiTech Dataset in 'ROOT/data'. 
You can find two python scripts in 
'data_preparation' folder which are used to generate ground truth density-map for 
ShanghaiTech PartA and PartB respectively. (Mind that you need move the script to corresponding 
sub dataset folder like 'ROOT/data/part_A_final' and run it)  
## Train
1. Modify the dataset root in 'config.py'   
2. Run 'train.py'
3. Open the command line and type in 'tensorboard --logdir=ROOT/runs', then browse 'localhost:6006' to see the visual result. 

## Testing
1. Run 'test.py' for calculate MAE of test images or just show an estimated density-map. 

## Other notes
1. We trained the model and got 67.74 MAE at 124-th epoch on ShanghaiTech PartA. Got 8.68 
MAE on PartB at 94-th epoch. 
2. If you are new to crowd counting, we recommand you to take a look at [Crowd_counting_from_scratch](https://github.com/CommissarMa/Crowd_counting_from_scratch) first. It is a tutorial for crowd counting beginner.