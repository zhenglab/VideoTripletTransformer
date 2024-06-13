# CUDA_VISIBLE_DEVICES=1 python evaluation/vd_evaluation.py --dataroot ../datasets/video-demoireing_v2/tcl/ --result_root results/ori_vth_1Cross_Deori_selfnorm_addnorm_2H_6L_LT_L2/test_latest/images/test/source/  
# CUDA_VISIBLE_DEVICES=1 python evaluation/vd_evaluation.py --dataroot ../datasets/video-demoireing_v2/iphone/ --result_root results/ori_vth_1Cross_Deori_selfnorm_addnorm_2H_6L_LT_L2_iphone/test_latest/images/test/source/  
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
import lpips

"""parsing and configuration"""
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', type=str, default='test', help='train or test ?')
    parser.add_argument('--dataroot', type=str, default='', help='dataset_dir')
    parser.add_argument('--result_root', type=str, default='', help='dataset_dir')
    parser.add_argument('--dataset_name', type=str, default='VD', help='dataset_name')
    parser.add_argument('--evaluation_type', type=str, default="our", help='evaluation type')
    parser.add_argument('--ssim_window_size', type=int, default=11, help='ssim window size')

    return parser.parse_args()



def main(dataset_name = None):
    cuda = True if torch.cuda.is_available() else False
    IMAGE_SIZE = np.array([1024,1024])
    opt.dataset_name = dataset_name
    # files = opt.dataroot+opt.dataset_name+'_'+opt.phase+'.txt'
    files = opt.dataroot+opt.phase+'_list.txt'
    comp_paths = []
    harmonized_paths = []
    # mask_paths = []
    real_paths = []
    root_folder_in = opt.dataroot+'indoor_np/input'
    root_folder_out = opt.dataroot+'outdoor_np/input'

    img_names = sorted([file.split('.')[0] for file in os.listdir(opt.dataroot+'test/source') if (file.endswith('.jpg') or file.endswith('.png'))])

    for img_name in img_names:
        harmonized_path = os.path.join(opt.result_root, 'test_source_'+img_name+'_harmonized.jpg')
        if os.path.exists(harmonized_path):
            real_path = harmonized_path.replace('harmonized','real')
            comp_path = harmonized_path.replace('harmonized','comp')
        real_paths.append(real_path)
        harmonized_paths.append(harmonized_path)
        comp_paths.append(comp_path)

    count = 0
    psnr_scores = 0
    ssim_scores = 0
    lpips_scores = 0
    image_size = 256
    lpips_net_metric_alex = lpips.LPIPS(net='alex').cuda()

    for i, harmonized_path in enumerate(tqdm(harmonized_paths)):
        count += 1

        harmonized = Image.open(harmonized_path).convert('RGB')
        real = Image.open(real_paths[i]).convert('RGB')
        if real.size[0] != image_size:
            harmonized = tf.resize(harmonized,[image_size,image_size], interpolation=Image.BICUBIC)
            real = tf.resize(real,[image_size,image_size], interpolation=Image.BICUBIC)

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
