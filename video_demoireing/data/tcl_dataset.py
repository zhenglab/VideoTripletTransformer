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
import os.path
import torch
import torchvision.transforms.functional as tf
import torch.nn.functional as F
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from util import util
import random


class TCLDataset(BaseDataset):
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
        self.isTrain = opt.isTrain
        self.image_size = opt.crop_size
        self.frames_each_video = 60
        self.dataset_root = opt.dataset_root
        self.n_frames = 5
        
        video_list = []
        if not self.isTrain:
            self.image_list = sorted([file for file in os.listdir(self.dataset_root+'test/source') if (file.endswith('.jpg') or file.endswith('.png'))])
            for i in range(len(self.image_list)):
                video_number = self.image_list[i].split('/')[-1][0:7]
                if video_number not in video_list:
                    video_list.append(video_number)
            self.image_list = video_list
        else:
            self.image_list = sorted([file for file in os.listdir(self.dataset_root+'train/source') if (file.endswith('.jpg') or file.endswith('.png'))])
            for i in range(len(self.image_list)):
                video_number = self.image_list[i].split('/')[-1][0:7]
                if video_number not in video_list:
                    video_list.append(video_number)
            self.image_list = video_list*10
            random.shuffle(self.image_list)
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
        image_in_gt = self.image_list[index] 
        if self.isTrain:
            comp_img_paths = []
            video_number = self.image_list[index] 
            idx = get_ref_index(self.frames_each_video, self.n_frames)
            for i in idx:
                path_src = self.dataset_root + 'train/source/' + video_number + '%05d' % i + '.jpg'
                comp_img_paths.append(path_src)
            comp_img_paths = sorted(comp_img_paths)
            image_in_gt = comp_img_paths
        else:
            video_number = self.image_list[index] 
            comp_img_paths = sorted([self.dataset_root+'test/source/'+file for file in os.listdir(self.dataset_root+'test/source') if (file.startswith(video_number))])
            image_in_gt = comp_img_paths
        real, comp = [], []
        flip = False
        if np.random.rand() > 0.5 and self.isTrain:
            flip = True
            
        for comp_img_path in comp_img_paths:
            real_img_path = comp_img_path.replace('source','target')

            comp_frame = Image.open(comp_img_path).convert('RGB')
            real_frame = Image.open(real_img_path).convert('RGB')
            
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
        return len(self.image_list)
    
def get_ref_index(length, sample_length):
    if random.uniform(0, 1) > 0.5:
        ref_index = random.sample(range(length), sample_length)
        ref_index.sort()
    else:
        pivot = random.randint(0, length-sample_length)
        ref_index = [pivot+i for i in range(sample_length)]
    return ref_index