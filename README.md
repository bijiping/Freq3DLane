<br />
<p align="center">
  <h3 align="center"><strong>Freq-3DLane: 3D Lane Detection From Monocular Images via Frequency-Aware Feature Fusion</strong></h3>
</p>

This repo is the official PyTorch implementation for paper: [Freq-3DLane: 3D Lane Detection From Monocular Images via Frequency-Aware Feature Fusion](https://ieeexplore.ieee.org/abstract/document/10995224). 

![pipeline](images/pipeline.png)
In this paper, we present Freq-3DLane, a simple yet effective end-to-end 3D lane detection framework. Freq-3DLane performs 3D lane detection by fusing multi-scale information and leveraging the frequency characteristics of features to enhance perception. Additionally, we introduce an attention-guided spatial transform fusion module to further improve detection performance. Extensive experiments demonstrate that Freq-3DLane achieves impressive results. We believe that our work has the potential to make a positive contribution to society and lay the groundwork for future research advancements.

## News
- [2025/04/24] Freq-3DLane is accepted by IEEE Transactions on Intelligent Transportation Systems.
- [2025/04/24] Freq-3DLane has been published in the IEEE Transactions on Intelligent Transportation Systems with Early Access.
- [2025/09/16] Freq-3DLane has been officially published in the IEEE Transactions on Intelligent Transportation Systems.
- [2026/01/19] We have open-sourced the source code of Freq-3DLane.

## Installation
To run our code, make sure you are using a machine with at least one GPU. Setup the enviroment , follow these steps:

#### **Step 1.** Create a conda virtual environment and activate it
```
conda create -n freq3dlane python=3.9 -y
conda activate freq3dlane
conda install pytorch==1.10.0 torchvision==0.11.0 cudatoolkit=11.1 -c pytorch -y
```

#### **Step 2.** Install Freq3DLane
```
git clone https://github.com/bijiping/Freq3Dlane.git
cd Freq3DLane
pip install -r requirements.txt
```




## Data Preparation
- Please refer to [Apollo 3D Lane Synthetic](https://github.com/yuliangguo/3D_Lane_Synthetic_Dataset) for downloading Apollo 3D Lane Synthetic Dataset. For example: download OpenLane dataset to /Dataset/Apollosim
- Please refer to [OpenLane](https://github.com/OpenPerceptionX/OpenLane) for downloading OpenLane Dataset. For example: download OpenLane dataset to /Dataset/OpenLane


The data folders are organized as follows:
```
├── Dataset/
|   └── Apollosim
|       └── images/...
|       └── data_splits  
|           └── standard
|               └── train.json
|               └── test.json 
|           └── illus_chg/...
|           └── rare_subset/...
\
|   └── OpenLane
|       └── images/...
|       └── lane3d_1000
|           └── training/...
|           └── validation/...
|           └── test
|               └── up_down_case
|               └── curve_case
|               └── extreme_weather_case
|               └── night_case
|               └── intersection_case
|               └── merge_split_case
|               └── The corresponding 6 txt files...
```
