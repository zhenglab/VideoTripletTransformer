"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
# import os.path
# import torch
import torchvision.transforms.functional as tf
import torch.nn.functional as F
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
# from util import util

import glob
import os.path as osp
import torch
import torch.utils.data as data
import data.util as util
import random
import cv2
import os

class SDSDDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.add_argument('--is_train', type=bool, default=True, help='whether in the training phase')
        parser.set_defaults(max_dataset_size=float("inf"), new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        # self.opt = opt
        # self.cache_data = opt['cache_data']  # true
        self.n_frames = 5
        self.dataset_root = opt.dataset_root
        self.GT_root, self.LQ_root = self.dataset_root + 'GT', self.dataset_root + 'input'
        # Generate data info and cache data
        self.isTrain = opt.isTrain
        self.image_size = opt.crop_size

        if 'outdoor_np' in self.dataset_root:
            testing_dir = ['pair1','pair5','pair14','pair36','pair46','pair48','pair49','pair60','pair62','pair63','pair66','pair75','pair76']
        elif 'indoor_np' in self.dataset_root:
            testing_dir = ['pair13','pair15','pair21','pair23','pair31','pair33','pair50','pair52','pair58','pair60','pair68','pair70']

        # read data:
        subfolders_LQ = util.glob_file_list(self.LQ_root)
        subfolders_GT = util.glob_file_list(self.GT_root)
        self.video_info_list = []

        for subfolder_LQ, subfolder_GT in zip(subfolders_LQ, subfolders_GT):  # '../dataset/SDSD/indoor_np/GT/pair9'
            subfolder_name = osp.basename(subfolder_GT)  # pair9
            if self.isTrain:
                if (subfolder_name in testing_dir):
                    continue
            else:
                if (subfolder_name not in testing_dir):
                    continue

            img_paths_LQ = util.glob_file_list(subfolder_LQ)
            img_paths_GT = util.glob_file_list(subfolder_GT)  # ['../dataset/SDSD/indoor_np/GT/pair10/0306.npy', ...]

            self.video_info_list.append({
                'subfolder_LQ': subfolder_LQ,
                'img_paths_LQ': img_paths_LQ,
                'img_paths_GT': img_paths_GT
            })
            assert len(img_paths_LQ) == len(img_paths_GT), 'Different number of images in LQ and GT folders'
        if self.isTrain:
            self.video_info_list =  self.video_info_list * 40
            random.shuffle(self.video_info_list)
        # get the image paths of your dataset;
          # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0, 0, 0), (1, 1, 1))
        ]
        self.transforms = transforms.Compose(transform_list)
        # print(len(self.image_paths))
        # assert 1==0
    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        video_number = self.video_info_list[index]['subfolder_LQ']
        video_info = self.video_info_list[index]
        if self.isTrain:
            frames_each_video = len(video_info['img_paths_LQ'])
            idx = get_ref_index(frames_each_video, self.n_frames)
            comp_img_paths, real_img_paths = [], []
            for i in idx:
                LQ_path_src = video_info['img_paths_LQ'][i]
                comp_img_paths.append(LQ_path_src)
                GT_path_src = video_info['img_paths_GT'][i]
                real_img_paths.append(GT_path_src)
            comp_img_paths = sorted(comp_img_paths)
            real_img_paths = sorted(real_img_paths)
            image_in_gt = comp_img_paths
        else:
            comp_img_paths = sorted(video_info['img_paths_LQ'])[:30]
            real_img_paths = sorted(video_info['img_paths_GT'])[:30]
            image_in_gt = comp_img_paths

        real, comp = [], []
        flip = False
        if np.random.rand() > 0.5 and self.isTrain:
            flip = True
            
        for comp_img_path, real_img_path in zip(comp_img_paths, real_img_paths):

            comp_frame = np.load(comp_img_path)
            real_frame = np.load(real_img_path)
            # some images have 4 channels
            if comp_frame.shape[2] > 3 or real_frame.shape[2] > 3:
                comp_frame = comp_frame[:, :, :3]
                real_frame = real_frame[:, :, :3]
            # BGR to RGB
            comp_frame = comp_frame[:, :, [2, 1, 0]]
            real_frame = real_frame[:, :, [2, 1, 0]]
            # from numpy to Image
            comp_frame = Image.fromarray(comp_frame).convert('RGB')
            real_frame = Image.fromarray(real_frame).convert('RGB')
            
            if comp_frame.size[0] != self.image_size:
                comp_frame = tf.resize(comp_frame, [self.image_size, self.image_size])
                real_frame = tf.resize(real_frame, [self.image_size, self.image_size])

            comp_frame = self.transforms(comp_frame)
            real_frame = self.transforms(real_frame)

            comp.append(comp_frame)
            real.append(real_frame)
        if len(comp) == 0:
            print(video_number)
            assert 0
        comp, real = torch.stack(comp), torch.stack(real)  # [t, c, h, w]

        if flip:
            comp, real = tf.hflip(comp), tf.hflip(real)

        return comp, real,image_in_gt



    def __len__(self):
        """Return the total number of images."""
        return len(self.video_info_list)

def get_ref_index(length, sample_length):
    if random.uniform(0, 1) > 0.2:
        ref_index = random.sample(range(length), sample_length)
        ref_index.sort()
    else:
        pivot = random.randint(0, length-sample_length)
        ref_index = [pivot+i for i in range(sample_length)]
    return ref_index