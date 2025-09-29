<div align="center">
<h1>A Unified Solution to Video Fusion: From Multi-Frame Learning to Benchmarking</h1>
<h3>NeurIPS 2025 (Spotlight)</h3>

<a href="https://vfbench.github.io/">
  <img src="src/images/badge-website.svg" alt="Website">
</a>
<a href="https://arxiv.org/abs/2505.19858">
  <img src="https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv" alt="arXiv">
</a>
<a href="https://share.phys.ethz.ch/~pf/zixiangdata/">
  <img src="https://img.shields.io/badge/Dataset-Link-orange?logo=googledrive" alt="Dataset">
</a>
<a href="https://huggingface.co/prs-eth/rollingdepth-v1-0">
  <img src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Model-green" alt="Hugging Face Model">
</a>
<p>
<a href="https://zhaozixiang1228.github.io/">Zixiang Zhao</a><sup>1</sup>, 
<a href="https://haowenbai.github.io/">Haowen Bai</a><sup>2</sup>, 
<a href="http://www.kebingxin.com/">Bingxin Ke</a><sup>1</sup>, 
<a href="https://openreview.net/profile?id=~Yukun_Cui2">Yukun Cui</a><sup>2</sup>, 
<a href="https://openreview.net/profile?id=~Lilun_Deng1">Lilun Deng</a><sup>2</sup>, 
<a href="https://yulunzhang.com/">Yulun Zhang</a><sup>3</sup>, 
<a href="https://cszn.github.io/">Kai Zhang</a><sup>4</sup>, 
<a href="https://scholar.google.com/citations?user=FZuNgqIAAAAJ&hl=en">Konrad Schindler</a><sup>1</sup>
</p>

<p>
<sup>1</sup>ETH Zurich· 
<sup>2</sup>Xi'an Jiaotong University· 
<sup>3</sup>Shanghai Jiao Tong University· 
<sup>4</sup>Nanjing University.
</p>

<img src="src/images/teaser.jpg" alt="Project Teaser" width="800">

</div>


<!-- # 🌐 A Unified Solution to Video Fusion: From Multi-Frame Learning to Benchmarking

**NeurIPS 2025 (Spotlight)**

[![Website](src/images/badge-website.svg)](https://vfbench.github.io/)
[![arXiv](https://img.shields.io/badge/arXiv-Paper-b31b1b?logo=arxiv)](https://arxiv.org/abs/2505.19858)
[![Dataset](https://img.shields.io/badge/Dataset-Link-orange?logo=googledrive)](https://share.phys.ethz.ch/~pf/zixiangdata/)
[![Hugging Face Model](https://img.shields.io/badge/🤗%20Hugging%20Face-Model-green)](https://huggingface.co/prs-eth/rollingdepth-v1-0)

Code and dataset for ***A Unified Solution to Video Fusion: From Multi-Frame Learning to Benchmarking (NeurIPS 2025 Spotlight)***.

[Zixiang Zhao](https://zhaozixiang1228.github.io/)<sup>1</sup>, 
[Haowen Bai](https://haowenbai.github.io/)<sup>2</sup>, 
[Bingxin Ke](http://www.kebingxin.com/)<sup>1</sup>,
[Yukun Cui](https://openreview.net/profile?id=~Yukun_Cui2)<sup>2</sup>,
[Lilun Deng](https://openreview.net/profile?id=~Lilun_Deng1)<sup>2</sup>, 
[Yulun Zhang](https://yulunzhang.com/)<sup>3</sup>,
[Kai Zhang](https://cszn.github.io/)<sup>4</sup>,
[Konrad Schindler](https://scholar.google.com/citations?user=FZuNgqIAAAAJ&hl=en)<sup>1</sup>.

<sup>1</sup>ETH Zurich
<sup>2</sup>Xi'an Jiaotong University
<sup>3</sup>Shanghai Jiao Tong University
<sup>4</sup>Nanjing University -->

## 📢 News
<!-- 2024-09-28: Dataset VF-Bench is released.<br>
2024-09-28: Code for UniVF is released.<br> -->
2025-09-18: Paper is accepted to **NeurIPS 2025 (Spotlight)**. <br>
2024-05-26: Paper is on arXiv.<br>

## 🛠️ Setup
The inference code was tested on: Debian 12, Python 3.10.16, CUDA 12.8, GeForce RTX 4090

### 📦 Repository
```bash
git clone https://github.com/Zhaozixiang1228/VF-Bench.git
cd VF-Bench
```

### 🐍 Python environment
Create python environment:
```bash
# with venv
python -m venv venv/vfbench
source venv/vfbench/bin/activate

# or with conda
conda create --name vfbench python=3.12
conda activate vfbench
```

### 💻 Dependencies
Install dependicies: 
```bash
pip install -r requirements.txt
```
We also recommand [pyav](https://github.com/PyAV-Org/PyAV) for video I/O, which relies on [ffmpeg](https://www.ffmpeg.org/) (tested with version 5.1.7-0+deb12u1).

## 🏃 Run inference (for demo or your videos) 
All scripts are designed to run from the project root directory.

### 📷 Prepare demo videos
```bash
bash script/download_demo_data.sh
```
These example videos are to be used as demo, also shown in our project homepage.
### ⬇ Checkpoint cache
The checkpoints are stored in the `ETH Cloud Disk` and `Hugging Face cache`. Use the following script to download the checkpoint weights locally:
```bash
bash script/download_demo_data.sh
```
### 🏃 Obtain fusion videos
```bash
bash script/test_demo.sh
```

## 🚀 Run inference (for academic comparisons)
### 🎮 Prepare training & test datasets
```bash
bash script/download_demo_data.sh
```
or Baidu Disk.
### ⬇ Checkpoint cache
The checkpoints are stored in the `ETH Cloud Disk` and `Hugging Face cache`. Use the following script to download the checkpoint weights locally:
```bash
bash script/download_demo_data.sh
```

### 🏄 Testing
```bash
# for Multi-Exposure Video Fusion
python test.py --task_name MEF --dataset_name DAVIS --exp_path checkpoint/MEF

# for Multi-Focus Video Fusion
python test.py --task_name MFF --dataset_name YouTube --exp_path checkpoint/MFF

# for Infrared-Visible Video Fusion
python test.py --task_name IVF --dataset_name VTMOT --exp_path checkpoint/IVF

# for Medical Video Fusion
python test.py --task_name MVF --dataset_name Harvard --exp_path checkpoint/MVF
```
### ⚙️ Inference settings
#### Passing these inference arguments will overwrite the preset settings:
<!-- - `--res` or `--processing-resolution`: the maximum resolution (in pixels) at which image processing will be performed. If set to 0, processes at the original input image resolution.
- `--refine-step`: number of refinement iterations to improve accuracy and details. Set to 0 to disable refinement.
- `--snip-len` or `--snippet-lengths`: number of frames to analyze in each snippet.
- `-d` or `--dilations`: spacing between frames for temporal analysis, could have multiple values e.g. `-d 1 10 25`. -->

## 🏋️ Training UniVF
After successfully preparing the environment and dataset as described above, Run training script:
```bash
# for Multi-Exposure Video Fusion
python train.py --task_name MEF

# for Multi-Focus Video Fusion
python train.py --task_name MFF

# for Infrared-Visible Video Fusion
python train.py --task_name IVF

# for Medical Video Fusion
python train.py --task_name MVF
```
### 🔧 Training settings


## 🎓 Citation
```bibtex
@InProceedings{zhao2025unified,
    title={A Unified Solution to Video Fusion: From Multi-Frame Learning to Benchmarking},
    author={Zhao, Zixiang and Bai, Haowen and Ke, Bingxin and Cui, Yukun and Deng, Lilun and Zhang, Yulun and Zhang, Kai and Schindler, Konrad},
    booktitle = {Advances in Neural Information Processing Systems (NeurIPS)},
    year={2025}
}
```


## 🙏 Acknowledgments
We would like to express our sincere gratitude to the following excellent works:
* [SEA-RAFT](https://github.com/princeton-vl/SEA-RAFT)
* [Marigold](https://github.com/prs-eth/Marigold)


<!-- ## 🎫 License

This code of this work is licensed under the Apache License, Version 2.0 (as defined in the [LICENSE](LICENSE.txt)).

The model is licensed under RAIL++-M License (as defined in the [LICENSE-MODEL](LICENSE-MODEL.txt))

By downloading and using the code and model you agree to the terms in [LICENSE](LICENSE.txt) and [LICENSE-MODEL](LICENSE-MODEL.txt) respectively.  -->
