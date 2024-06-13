import os.path
import torch
import torchvision.transforms.functional as tf
import torch.nn.functional as F
from PIL import Image, ImageStat
import numpy as np
import torchvision.transforms as transforms
import random
import cv2
import matplotlib.pyplot as plt
import glob
from skimage.metrics import mean_squared_error as mse
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser(description='some parameters about temporal consistency evaluation.')

parser.add_argument('--dataset_root', type=str, default='../datasets/HYouTube/', help='path of dataset')
parser.add_argument('--experiment_name', type=str, default='swin3d_skipnorm_patch_v2_1H6L_win4_p1_LT_L2', help='folder name in the results folder')
parser.add_argument('--mode', type=str, default='v', help='v, rgb, gray or hsv')
parser.add_argument('--brightness_region', type=str, default='foreground', help='forground or image')
args = parser.parse_args()

def extract_image_brightness(args, frame_path, mask_path):
    im = cv2.imread(frame_path)
    # RGB
    if args.mode == 'rgb':
        v = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
    # HSV
    elif args.mode == 'hsv':
        v = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)  # [256, 256, 3]
    # V in HSV
    elif args.mode == 'v':
        hsv_img = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)  # [256, 256, 3]
        v = cv2.split(hsv_img)[2]  # [256, 256]
    # gray
    elif args.mode == 'gray':
        v = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    else:
        raise NotImplementedError('Mode Error')

    return v.astype(np.float32)

def extract_brightness_with_hsv(args, frame_path, mask_path):
    im = cv2.imread(frame_path)
    # RGB
    if args.mode == 'rgb':
        v = cv2.cvtColor(im,cv2.COLOR_BGR2RGB)
        mask = np.tile(mask.reshape(256,256,1),(1,1,3))
    # HSV
    elif args.mode == 'hsv':
        v = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)  # [256, 256, 3]
        mask = np.tile(mask.reshape(256,256,1),(1,1,3))
    # V in HSV
    elif args.mode == 'v':
        hsv_img = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)  # [256, 256, 3]
        v = cv2.split(hsv_img)[2]  # [256, 256]
        mask = np.array(Image.open(mask_path).convert('1').resize((256,256)))
    # gray
    elif args.mode == 'gray':
        v = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        mask = np.array(Image.open(mask_path).convert('1').resize((256,256)))
    else:
        raise NotImplementedError('Mode Error')

    fg_v = v * mask
    fg_area = np.sum(mask)
    fg_v_mean = np.sum(fg_v) / fg_area
    return fg_v_mean

testfile = args.dataset_root+'test_list.txt'
video_objs = []
with open(testfile, 'r') as f:
    for line in f.readlines():
        video_objs.append(line.rstrip().split(' ')[-1].replace('synthetic_composite_videos/', ''))  # ['ff692fdc56/object_1', 'ff773b1a1e/object_0']


count = 0
gradient_mse_scores = 0.0
gradient_mse_score_list = []
results_root = 'results/' + args.experiment_name + '/test_latest/images/'

for i,video_obj in enumerate(tqdm(video_objs)):
    video_obj_path = results_root + video_obj

    frame_paths_dict = {}
    frame_paths_dict['real'] = glob.glob(os.path.join(video_obj_path, '*real*'))
    frame_paths_dict['harmonized'] = glob.glob(os.path.join(video_obj_path, '*harm*'))
    for value in frame_paths_dict.values():
        value.sort()
    
    
    videos_cls = ['real', 'harmonized']
    bright_gradients_dict = {}
    bright_values_dict = {}
    for video_cls in videos_cls:
        bright_values_dict[video_cls] = []
    for video_cls, frame_paths in frame_paths_dict.items():
        for frame_path in frame_paths:
            frame_index = frame_path.split('/')[-1].replace((video_obj.replace('/','_')+'_'),'')[:5]
            mask_path = args.dataset_root + 'foreground_mask/' + video_obj + '/' + frame_index + '.png'

            if args.brightness_region == 'image':
                bright_value = extract_image_brightness(args, frame_path, mask_path)
            elif args.brightness_region == 'foreground':
                bright_value = extract_brightness_with_hsv(args, frame_path, mask_path)
            else:
                raise NotImplementedError('Brightness Region Error')
            
            bright_values_dict[video_cls].append(bright_value)
        if args.brightness_region == 'image':
            bright_values_dict[video_cls] = np.stack(bright_values_dict[video_cls], axis=0) #[20,256,256]
            bright_gradients_dict[video_cls] = (np.array(bright_values_dict[video_cls][1:]) - np.array(bright_values_dict[video_cls][:-1])).mean(axis=(-1,-2)) #[19,256,256]
        elif args.brightness_region == 'foreground':
            bright_gradients_dict[video_cls] = np.array(bright_values_dict[video_cls][1:]) - np.array(bright_values_dict[video_cls][:-1])
        else:
                raise NotImplementedError('Brightness Region Error')

    count += 1

    mse_score = pow((bright_gradients_dict['real']-bright_gradients_dict['harmonized']),2)
    RTC_1 = mse_score
    harm_real_diff = np.abs(bright_gradients_dict['harmonized']-bright_gradients_dict['real'])
    mu = np.mean(harm_real_diff)
    harm_real_diff = harm_real_diff - mu
    harm_real_diff[harm_real_diff < 0] = 0
    RTC_2 = pow(harm_real_diff, 2)
    gradient_mse_score = (RTC_1 + RTC_2).mean()
    gradient_mse_scores += gradient_mse_score
    gradient_mse_info = (video_obj, gradient_mse_score)
    gradient_mse_score_list.append(round(gradient_mse_score,2))

gradient_mse_score_mu = gradient_mse_scores / count

if args.brightness_region == 'foreground':
    print('fR-RTC:',round(gradient_mse_score_mu, 2))
else:
    print('R-RTC:',round(gradient_mse_score_mu, 2))