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

class HytSampleDataset(BaseDataset):
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
        # self.image_paths = []
        self.video_object_paths = []
        self.isTrain = opt.isTrain
        self.image_size = opt.crop_size
        if opt.isTrain==True:
            #self.real_ext='.jpg'
            print('loading training file')
#             self.trainfile = opt.dataset_root+opt.dataset_name+'_train.txt' #opt.dataset_root='HYouTube/' trainfile=train_list
            self.trainfile = opt.dataset_root+'train_list.txt'
            with open(self.trainfile,'r') as f:
                    for line in f.readlines():
                        video_object_path = os.path.join(opt.dataset_root, line.rstrip().split(' ')[-1])
                        self.video_object_paths.append(video_object_path)
                        
        elif opt.isTrain==False:
            #self.real_ext='.jpg'
            print('loading test file')
#             self.trainfile = opt.dataset_root+opt.dataset_name+'_test.txt'
            self.trainfile = opt.dataset_root+'test_list.txt'
            with open(self.trainfile,'r') as f:
                    for line in f.readlines():
                        video_object_path = os.path.join(opt.dataset_root, line.rstrip().split(' ')[-1])
                        self.video_object_paths.append(video_object_path)
        # get the image paths of your dataset;
          # You can call sorted(make_dataset(self.root, opt.max_dataset_size)) to get all the image paths under the directory self.root
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        transform_list = [
            transforms.ToTensor(),
            transforms.Normalize((0, 0, 0), (1, 1, 1))
        ]
        self.transforms = transforms.Compose(transform_list)
        # self.video_object_paths = self.video_object_paths[:20]
    def __getitem__(self, index):
        video_object_path = self.video_object_paths[index] # HYouTube/synthetic_composite_videos/003234408d/object_0
        img_paths = os.listdir(video_object_path)
        img_paths.sort()
        
        if self.isTrain:
            idx = get_ref_index(len(img_paths), self.opt.n_frames)
            selected_frames = [img_paths[i] for i in idx]
            comp_img_paths = [video_object_path+'/'+img_path for img_path in selected_frames]
        else:
            selected_frames = [i for i in range(len(img_paths))]
            comp_img_paths = [video_object_path+'/'+img_path for img_path in img_paths]
        comp, real, mask, inputs = [], [], [], []
            
        for comp_img_path in comp_img_paths:
            real_img_path = comp_img_path.replace('synthetic_composite_videos','real_videos')
            real_img_path = real_img_path.replace(('/'+comp_img_path.split('/')[-2]), '')
            mask_img_path = comp_img_path.replace('synthetic_composite_videos','foreground_mask')
            mask_img_path = mask_img_path.replace(os.path.splitext(mask_img_path)[1],'.png')

            comp_frame = Image.open(comp_img_path).convert('RGB')
            real_frame = Image.open(real_img_path).convert('RGB')
            mask_frame = Image.open(mask_img_path).convert('1')

            if comp_frame.size[0] != self.image_size:
                comp_frame = tf.resize(comp_frame, [self.image_size, self.image_size])
                real_frame = tf.resize(real_frame, [self.image_size, self.image_size])
                mask_frame = tf.resize(mask_frame, [self.image_size, self.image_size])

            comp_frame = self.transforms(comp_frame)
            mask_frame = tf.to_tensor(mask_frame)
            real_frame = self.transforms(real_frame)
            inputs_frame = torch.cat([comp_frame,mask_frame],0)

            comp.append(comp_frame)
            real.append(real_frame)
            mask.append(mask_frame)
            inputs.append(inputs_frame)

        inputs, comp, real, mask = torch.stack(inputs), torch.stack(comp), torch.stack(real), torch.stack(mask)  # [t, c, h, w]

        if np.random.rand() > 0.5 and self.isTrain:
            inputs, comp, mask, real = tf.hflip(inputs), tf.hflip(comp), tf.hflip(mask), tf.hflip(real)

        # return inputs, comp, real, video_object_path, mask, selected_frames
        return comp, real, video_object_path, mask,selected_frames
    

    def __len__(self):
        """Return the total number of images."""
        return len(self.video_object_paths)
    
def get_ref_index(length, sample_length):
    if random.uniform(0, 1) > 0.5:
        ref_index = random.sample(range(length), sample_length)
        ref_index.sort()
    else:
        pivot = random.randint(0, length-sample_length)
        ref_index = [pivot+i for i in range(sample_length)]
    return ref_index