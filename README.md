# Zero-Shot Video Translation and Editing with Frame Spatial-Temporal Correspondence

[Shuai Yang](https://williamyang1991.github.io/), [Junxin Lin](https://github.com/Sunnycookies), [Yifan Zhou](https://zhouyifan.net/), [Ziwei Liu](https://liuziwei7.github.io/) and [Chen Change Loy](https://www.mmlab-ntu.com/person/ccloy/)<br>
[**Paper**](https://arxiv.org/abs/2512.03905) | [**Project**](
https://williamyang1991.github.io/projects/FRESCOv2/) |[**Input Data and Video Results**](https://drive.google.com/file/d/12LkAEmzRBgSMKqQqgHX5_LmhYI59ZRAZ/view?usp=sharing)<br>

**Abstract:** *The remarkable success in text-to-image diffusion models has motivated extensive investigation of their potential for video applications. Zero-shot techniques aim to adapt image diffusion models for videos without requiring further model training. Recent methods largely emphasize integrating interframe correspondence into attention mechanisms. However, the soft constraint applied to identify the valid features to attend is insufficient, which could lead to temporal inconsistency. In this paper, we present FRESCO, which integrates intra-frame correspondence with inter-frame correspondence to formulate a more robust spatial-temporal constraint. This enhancement ensures a consistent transformation of semantically similar content between frames. Our method goes beyond attention guidance to explicitly optimize features, achieving high spatial-temporal consistency with the input video, significantly enhancing the visual coherence of manipulated videos. We verify FRESCO adaptations on two zero-shot tasks of video-to-video translation and text-guided video editing. Comprehensive experiments demonstrate the effectiveness of our framework in generating high-quality, coherent videos, highlighting a significant advance over current zero-shot methods.*

**Features**:<br>

- **Robustness**: more robust to large and quick motion and long video translation compared with the [previous version](https://github.com/williamyang1991/FRESCO)
- **Flexibility**: compatible with different frame manipulation ([Plug-and-Play](https://github.com/MichalGeyer/plug-and-play)) and synthesis ([TokenFlow](https://github.com/omerbt/TokenFlow)) methods compared with the [previous version](https://github.com/williamyang1991/FRESCO)

![teaser](https://github.com/user-attachments/assets/2fe83add-101a-470a-bfcb-90f5e5b1e3ad)


## Updates
- [10/2025] Code is released.

## TODO
- [ ] Update readme
- [ ] Upload paper to arXiv, release related material

## Installation

1. Clone the repository. 

```shell
git clone https://github.com/sunnycookies/FRESCO.git
cd FRESCO
```

2. You can simply set up the environment with pip based on [requirements.txt](./requirements.txt)
    - Create a conda environment and install torch >= 2.0.0. Here is an example script to install torch 2.0.0 + CUDA 11.8 :
    ```shell
    conda create --name diffusers python==3.8.5
    conda activate diffusers
    pip install torch==2.0.0 torchvision==0.15.1 --index-url https://download.pytorch.org/whl/cu118
    ```
    - Run `pip install -r requirements.txt` in an environment where torch is installed.
    - We have tested on python 3.12.2, torch 2.9.0 and diffusers 0.19.3
    - If you use new versions of diffusers, you need to modify [my_forward()](https://github.com/Sunnycookies/FRESCO/blob/6bc94a618ff654b2f669a46dedbf513278ed2f42/src/diffusion_hacked.py#L620)

3. Run the installation script. The required models will be downloaded in `./model`, `./src/ControlNet/annotator` and `./src/ebsynth/deps/ebsynth/bin`.
    - Requires access to huggingface.co

```shell
python install.py
```

4. You can run the demo with `run_fresco.py`

```shell
python run_fresco.py ./config/config_bread.yaml
```

5. For issues with Ebsynth, please refer to [issues](https://github.com/williamyang1991/Rerender_A_Video#issues)

## (1)Inference

We provide a flexible script `run_fresco.py` to run our method. Set the options via a config file. For example,

```shell
python run_fresco.py ./config/config_bread.yaml
```
We provide  `preprocess.py` taken from [TokenFlow](https://github.com/omerbt/TokenFlow) for producing the intermediate results of DDIM inversion. The script can be run as 

```
python preprocess.py ./config/config_bread.yaml
```

For convenience, we combine the arguments of `run_fresco.py` and `preprocess.py` into one configuration file. We provide some examples of the config in `config` directory. For detailed arguments, please refer to the instructions in `run_fresco.ipynb`.

We provide a gadget `frame2video.py` to combine several input frame sequences from different source videos into a single video. The length of the output video is equal to the shortest length of input frame sequences.

```python
frames2video.py [-h] -r ROOT [ROOT ...] -o OUT [-f FPS] [-n NAME]

options:
  -h, --help            show this help message and exit
  -r ROOT [ROOT ...], --root ROOT [ROOT ...]
                        Directories of input frame sequences
  -o OUT, --out OUT     Path to output video
  -f FPS, --fps FPS     The FPS of output video
  -n NAME, --name NAME  Name of output video
```

The following is an example:

```python
python frames2video.py -r result/bread/keys result/deer/keys -o result/blend -f 24 -n blend_video
```

We provide a separate Ebsynth python script `video_blend.py` with the temporal blending algorithm introduced in [Stylizing Video by Example](https://dcgi.fel.cvut.cz/home/sykorad/ebsynth.html) for interpolating style between key frames. It can work on your own stylized key frames independently of our FRESCO algorithm.

```python
video_blend.py [-h] [--output OUTPUT] [--fps FPS] [--key_ind KEY_IND [KEY_IND ...]] [--key KEY] [--n_proc N_PROC] [-ps] [-ne] [-tmp] name

positional arguments:
  name                  Path to input video

optional arguments:
  -h, --help            show this help message and exit
  --output OUTPUT       Path to output video
  --fps FPS             The FPS of output video
  --key_ind KEY_IND [KEY_IND ...]
                        key frame index
  --key KEY             The subfolder name of stylized key frames
  --n_proc N_PROC       The max process count
  -ps                   Use poisson gradient blending
  -ne                   Do not run ebsynth (use previous ebsynth output)
  -tmp                  Keep temporary output
```
An example
```
python video_blend.py ./result/bread/ --key keys --key_ind 0 11 23 33 49 60 72 82 93 106 120 137 151 170 182 193 213 228 238 252 262 288 299  --output ./result/bread/blend.mp4 --fps 24 --n_proc 4 -ps
```

For the details, please refer to our previous work [Rerender-A-Video](https://github.com/williamyang1991/Rerender_A_Video/tree/main?tab=readme-ov-file#our-ebsynth-implementation). (The mainly difference is the way of specifying key frame index)

## (2)Results

### Long video manipulation

https://github.com/user-attachments/assets/c6582b56-983d-4c04-b689-5ef2e0818508

### Text-guided video translation

https://github.com/user-attachments/assets/3289a107-d152-46bb-a162-db461d2a35ab


## Citation

If you find this work useful for your research, please consider citing our paper:

```bibtex
@article{yang2025fresco,
 title = {Zero-Shot Video Translation and Editing with Frame Spatial-Temporal Correspondence},
 author = {Yang, Shuai and Lin, Junxin and Zhou, Yifan and Liu, Ziwei and Loy, Chen Change},
 journal = {arXiv preprint arXiv:2512.03905},
 year = {2025}
}
```


## Acknowledgements

The code is mainly developed based on [Rerender-A-Video](https://github.com/williamyang1991/Rerender_A_Video), [ControlNet](https://github.com/lllyasviel/ControlNet), [Stable Diffusion](https://github.com/Stability-AI/stablediffusion), [GMFlow](https://github.com/haofeixu/gmflow), [Ebsynth](https://github.com/jamriska/ebsynth), [Plug-and-Play](https://github.com/MichalGeyer/plug-and-play) and [TokenFlow](https://github.com/omerbt/TokenFlow).
