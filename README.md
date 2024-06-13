<base target="_blank"/>

# Video Harmonization with Triplet Spatio-Temporal Variation Patterns

Here we provide the PyTorch implementation and pre-trained model of our latest version.

## Prerequisites

- Linux
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN

## Video Harmonization
### Train/Test
- Download [HYouTube](https://github.com/bcmi/Video-Harmonization-Dataset-HYouTube) dataset.

- Train our VTT model:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --model basestv2 --netG vth --name experiment_name --deformable_depth x --gpt_depth x --dataset_root <dataset_dir> --batch_size x --init_port xxxx --loss_T --save_iter_model
```
- Test our VTT model:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model basestv2 --netG  vth  --name experiment_name --deformable_depth x --gpt_depth x --dataset_root <dataset_dir> --batch_size 1  --init_port xxxx
```


### Apply a pre-trained model
- Download pre-trained models from [BaiduCloud](https://pan.baidu.com/s/15e_98vYmm3ojIKrY33u29g?pwd=h63l) (access code: h63l), and put `latest_net_G.pth` in the directory `checkpoints/vth_harmonization`. Run:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model basestv2 --netG  vth  --name vth_harmonization --deformable_depth 2 --gpt_depth 2 --dataset_root <dataset_dir> --batch_size 1  --init_port xxxx
```

### Evaluation
To evaluate the spatial consistency, run:
```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/ih_evaluation.py --dataroot <dataset_dir> --result_root results/experiment_name/test_latest/images/ --evaluation_type hyt --dataset_name HYT
```
To evaluate the temporal consistency, run:
```bash
python tc_evaluation.py --dataset_root <dataset_dir> --experiment_name experiment_name --mode 'v' --brightness_region 'foreground'
```

### Real composite image harmonization
More compared results can be found at [BaduCloud](https://pan.baidu.com/s/15e_98vYmm3ojIKrY33u29g?pwd=h63l) (access code: h63l).

## Video Enhancement
### Train/Test
- Download [SDSD](https://github.com/dvlab-research/SDSD) dataset.

- Train our VTT model:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --model basestv2 --netG vth --name experiment_name   --deformable_depth x --gpt_depth x  --dataset_root <dataset_dir> --batch_size x --init_port xxxx --n_frames 5 --loss_T --save_iter_model
```
- Test our VTT model:
```bash
CUDA_VISIBLE_DEVICES=1 python test.py --model basestv2 --netG vth --name experiment_name --deformable_depth x --gpt_depth x --dataset_root <dataset_dir> --batch_size 1 --init_port xxxx --n_frames 15
```


### Apply a pre-trained model
- Download pre-trained models from [BaiduCloud](https://pan.baidu.com/s/15e_98vYmm3ojIKrY33u29g?pwd=h63l) (access code: h63l), and put `latest_net_G.pth` in the directory `checkpoints/vth_enhancement`. Run:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model basestv2 --netG vth --name vth_enhancement --deformable_depth 2 --gpt_depth 2 --dataset_root <dataset_dir> --batch_size 1 --init_port xxxx --n_frames 15
```

### Evaluation
To evaluate the spatial consistency, run:
```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/ve_evaluation.py --dataroot <dataset_dir> --result_root results/experiment_name/test_latest/images/input/
```
To evaluate the temporal consistency, run:
```bash
python tc_evaluation.py --dataset_root <dataset_dir> --experiment_name experiment_name --mode 'v' --brightness_region 'image'
```

## Video Demoireing
### Train/Test
- Download [video demoireing](https://daipengwa.github.io/VDmoire_ProjectPage/) dataset.

- Train our VTT model:
```bash
CUDA_VISIBLE_DEVICES=0 python train.py --model basestv2 --netG  vth  --name experiment_name --deformable_depth x --gpt_depth x --dataset_root <dataset_dir> --batch_size x  --init_port xxxx --n_frames 5  --loss_T   --save_iter_model
```
- Test our VTT model:
```bash
CUDA_VISIBLE_DEVICES=1 python test.py --model basestv2 --netG  vth  --name experiment_name --deformable_depth x --gpt_depth x --dataset_root <dataset_dir>  --batch_size 1 --init_port xxxx --n_frames 20
```


### Apply a pre-trained model
- Download pre-trained models from [BaiduCloud](https://pan.baidu.com/s/15e_98vYmm3ojIKrY33u29g?pwd=h63l) (access code: h63l), and put `latest_net_G.pth` in the directory `checkpoints/vth_demoireing`. Run:
```bash
CUDA_VISIBLE_DEVICES=0 python test.py --model basestv2 --netG  vth  --name vth_demoireing --deformable_depth 2 --gpt_depth 2 --dataset_root <dataset_dir> --batch_size 1 --init_port xxxx --n_frames 20
```
### Evaluation
To evaluate the spatial consistency, run:
```bash
CUDA_VISIBLE_DEVICES=0 python evaluation/vd_evaluation.py --dataroot <dataset_dir> --result_root results/experiment_name/test_latest/images/test/source/ 
```
To evaluate the temporal consistency, run:
```bash
python tc_evaluation.py --dataset_root <dataset_dir> --experiment_name experiment_name --mode 'v' --brightness_region 'image'
```

<!--# Bibtex
If you use this code for your research, please cite our papers.

```
@InProceedings{Guo_2021_ICCV,
    author    = {Guo, Zonghui and Guo, Dongsheng and Zheng, Haiyong and Gu, Zhaorui and Zheng, Bing and Dong, Junyu},
    title     = {Image Harmonization With Transformer},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {14870-14879}
}
```-->

# Acknowledgement
For some of the data modules and model functions used in this source code, we need to acknowledge the repositories of [HarmonyTransformer](https://github.com/zhenglab/HarmonyTransformer), [Swin3D](https://github.com/SwinTransformer/Video-Swin-Transformer), [
VEN-Retinex](https://github.com/dvlab-research/SDSD) and [VDRTC](https://daipengwa.github.io/VDmoire_ProjectPage/). 
