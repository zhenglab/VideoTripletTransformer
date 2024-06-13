# CUDA_VISIBLE_DEVICES=4 python evaluation/ve_evaluation.py --dataroot ../datasets/SDSD/indoor_np/ --result_root results/vth_MAE75_Deori_selfnorm_addnorm_2H_6L_LT_L2/test_latest/images/input/
from PIL import Image
import numpy as np
import os
import torch
import argparse
import cv2
import pytorch_ssim
import torchvision.transforms.functional as tf
import torchvision
import torch.nn.functional as f
from skimage import data, img_as_float
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import mean_squared_error as mse
from tqdm import tqdm
import utils as utls
from PIL import Image
import lpips
from torchvision.transforms import InterpolationMode
import glob

"""parsing and configuration"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='test', help='train or test ?')
    parser.add_argument('--dataroot', type=str, default='', help='dataset_dir')
    parser.add_argument('--result_root', type=str, default='', help='dataset_dir')
    parser.add_argument('--dataset_name', type=str, default='VE', help='dataset_name')
    parser.add_argument('--evaluation_type', type=str, default="our", help='evaluation type')
    parser.add_argument('--ssim_window_size', type=int, default=11, help='ssim window size')

    return parser.parse_args()



def main(dataset_name = None):
    cuda = True if torch.cuda.is_available() else False
    IMAGE_SIZE = np.array([1024,1024])
    opt.dataset_name = dataset_name
    if 'indoor' in opt.dataroot:  # ../datasets/SDSD/indoor_np/
        test_dir = ['pair13','pair15','pair21','pair23','pair31','pair33','pair50','pair52','pair58','pair60','pair68','pair70']
    elif 'outdoor' in opt.dataroot:
        test_dir = ['pair1','pair5','pair14','pair36','pair46','pair48','pair49','pair60','pair62','pair63','pair66','pair75','pair76']
    else:
        raise NotImplementedError('dataroot Error')

    root_LQ = os.path.join(opt.result_root, 'input')
    root_GT = os.path.join(opt.dataroot, 'GT')

    harmonized_paths = []
    real_paths = []
    
    count = 0
    psnr_scores = 0
    ssim_scores = 0
    lpips_scores = 0
    image_size = 256
    lpips_net_metric_alex = lpips.LPIPS(net='alex').cuda()
    for subfolder in test_dir:
        subfolder_LQ = os.path.join(opt.result_root, subfolder)
        subfolder_GT = os.path.join(opt.result_root, subfolder)
        img_paths_LQ = sorted(glob.glob(os.path.join(subfolder_LQ, '*harmonized*')))
        img_paths_GT = sorted(glob.glob(os.path.join(subfolder_GT, '*real*')))
        assert len(img_paths_LQ) == len(img_paths_GT), 'Different number of images in LQ and GT folders'
        harmonized_paths += img_paths_LQ
        real_paths += img_paths_GT

    for i, harmonized_path in enumerate(tqdm(harmonized_paths)):
        count += 1

        harmonized = Image.open(harmonized_path).convert('RGB')
        real = Image.open(real_paths[i]).convert('RGB')

        if real.size[0] != image_size:
            harmonized = tf.resize(harmonized,[image_size,image_size], InterpolationMode.BICUBIC)
            real = tf.resize(real,[image_size,image_size], InterpolationMode.BICUBIC)

        harmonized_np = np.array(harmonized, dtype=np.float32)
        real_np = np.array(real, dtype=np.float32)

        harmonized = tf.to_tensor(harmonized_np).unsqueeze(0).cuda()
        real = tf.to_tensor(real_np).unsqueeze(0).cuda()

        psnr_score = psnr(real_np, harmonized_np, data_range=255)
        ssim_score, fssim_score = pytorch_ssim.ssim(harmonized, real, window_size=opt.ssim_window_size, mask=None)
        lpips_score = lpips_net_metric_alex.forward(harmonized/255.0, real/255.0, normalize=True).item()

        psnr_scores += psnr_score
        ssim_scores += ssim_score
        lpips_scores += lpips_score

    psnr_scores_mu = psnr_scores/count
    ssim_scores_mu = ssim_scores/count
    lpips_scores_mu = lpips_scores/count


    print(count)
    mean_sore = "%s PSNR %0.4f | SSIM %0.4f | LPIPS %0.4f" % (opt.dataset_name, psnr_scores_mu,ssim_scores_mu,lpips_scores_mu)
    print(mean_sore)    

    # return None
    return psnr_scores_mu

def generstr(dataset_name='ALL'): 
    datasets = ['HCOCO','HAdobe5k','HFlickr','Hday2night','IHD']
    if dataset_name == 'newALL':
        datasets = ['HCOCO','HAdobe5k','HFlickr','Hday2night','HVIDIT','newIHD']
    for i, item in enumerate(datasets):
        print(item)
        mse_scores_mu,fmse_scores_mu, psnr_scores_mu,fpsnr_scores_mu = main(dataset_name=item)
        

if __name__ == '__main__':
    opt = parse_args()
    if opt is None:
        exit()
    if opt.dataset_name == "ALL":
        generstr()
    elif opt.dataset_name == "newALL":
        generstr(dataset_name='newALL')
    else:
        main(dataset_name=opt.dataset_name)
