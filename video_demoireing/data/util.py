import os
import random
import numpy as np
import glob
import torch
import cv2

def glob_file_list(root):
    return sorted(glob.glob(os.path.join(root, '*')))

def index_generation(crt_i, max_n, N, padding='reflection'):
    """Generate an index list for reading N frames from a sequence of images
    Args:
        crt_i (int): current center index
        max_n (int): max number of the sequence of images (calculated from 1)
        N (int): reading N frames
        padding (str): padding mode, one of replicate | reflection | new_info | circle
            Example: crt_i = 0, N = 5
            replicate: [0, 0, 0, 1, 2]
            reflection: [2, 1, 0, 1, 2]
            new_info: [4, 3, 0, 1, 2]
            circle: [3, 4, 0, 1, 2]

    Returns:
        return_l (list [int]): a list of indexes
    """
    max_n = max_n - 1
    n_pad = N // 2
    return_l = []

    for i in range(crt_i - n_pad, crt_i + n_pad + 1):
        if i < 0:  # crt_i=1, (-1,0,1,2,3)
            if padding == 'replicate':
                add_idx = 0  # (0,0,1,2,3)
            elif padding == 'reflection':
                add_idx = -i  # (1,0,1,2,3)
            elif padding == 'new_info':
                add_idx = (crt_i + n_pad) + (-i)  # (4,0,1,2,3)
            elif padding == 'circle':
                add_idx = N + i  # (4,0,1,2,3)
            else:
                raise ValueError('Wrong padding mode')
        elif i > max_n:
            if padding == 'replicate':
                add_idx = max_n
            elif padding == 'reflection':
                add_idx = max_n * 2 - i
            elif padding == 'new_info':
                add_idx = (crt_i - n_pad) - (i - max_n)
            elif padding == 'circle':
                add_idx = i - N
            else:
                raise ValueError('Wrong padding mode')
        else:  # crt_i=3, (1,6)
            add_idx = i
        return_l.append(add_idx)
    return return_l