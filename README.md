# 3D Structure-guided Network for Tooth Alignment in 2D Photograph (BMVC 2023)

This repository includes our code for the paper ***'3D Structure-guided Network for Tooth Alignment in 2D Photograph'*** in BMVC 2023. 
### [Webpage](https://proceedings.bmvc2023.org/322/) | [Paper](https://arxiv.org/abs/2310.11106) | [Code](https://github.com/douyl/2DToothAlignment/tree/master)

## Method overview
<img src="./Code/config/Method%20overview.png"  width="500" />
<img src="./Code/config/Result%20overview.png"  width="500" />


## High-Level structure
The code is organized as follows:
* [Code/main.py](./Code/main.py) is the main program for generating photographs.
* [Code/Stage2/ckpt](./Code/Stage2/ckpt) we provide link to download model weights. See details in [Code/Stage2/ckpt/download_ckpt.txt](./Code/Stage2/ckpt/download_ckpt.txt).
* [Code/Stage3/ckpt](./Code/Stage3/ckpt) we provide link to download model weights. See details in [Code/Stage3/ckpt/download_ckpt.txt](./Code/Stage3/ckpt/download_ckpt.txt).
* [Data/](./Data) is the directory for saving input images, we provide one testing case here.
* [Output/](./Output) is the directory for saving output images.

## How to Use

### Get started
We run with Python 3.7, you can set up a conda environment with all dependencies.

### Download model weights
* Refer to the links in [Code/Stage2/ckpt/download_ckpt.txt](./Code/Stage2/ckpt/download_ckpt.txt), download the model weights and put as ***Code/Stage2/ckpt/ckpt_contour2contour_mixed_v2_ContourSegm_4000.pth***.
* Refer to the links in [Code/Stage3/ckpt/download_ckpt.txt](./Code/Stage3/ckpt/download_ckpt.txt), download the model weights and put as ***Code/Stage3/ckpt/ckpt_contour2tooth_v2_ContourSegm_facecolor_lightcolor_10000.pth***.

### Prepare data
Prepare some facial photographs for testing and then put them under path [Data/](./Data). Here [Data/case1.jpg](./Data/case1.jpg) is an example.

### Usage
Simply use the following command to run our code. You will see the results in [Output/prediction](./Output/prediction) and [Output/processing](./Output/processing).
```python
   cd Code
   python main.py -i ../Data/case1.jpg
```

## Citation

If our code or models help your work, please cite our [paper](https://arxiv.org/abs/2310.11106):
```BibTeX
@inproceedings{Dou_2023_BMVC,
author    = {Yulong Dou and Lanzhuju Mei and Dinggang Shen and Zhiming Cui},
title     = {3D Structure-guided Network for Tooth Alignment in 2D Photograph},
booktitle = {34th British Machine Vision Conference 2023, {BMVC} 2023, Aberdeen, UK, November 20-24, 2023},
publisher = {BMVA},
year      = {2023},
url       = {https://papers.bmvc2023.org/0322.pdf}
}
```
