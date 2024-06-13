import torch
import os
import itertools
import torch.nn.functional as F
from .base_model import BaseModel
from util import util
from . import harmony_networks_v3 as networks
from . import base_networks as networks_init
from einops.layers.torch import Rearrange
from util import flops
from models.lib import swin3d_transformer_v17


class BaseSTV2Model(BaseModel):
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(norm='instance', netG='TemporalTre', dataset_mode='hyt')
        if is_train:
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_T', type=float, default=50.0, help='weight for L1 loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.opt = opt
        self.postion_embedding = None
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G','G_L1']
        if opt.loss_T:
            self.loss_names = ['G','G_L1','G_T']
        self.visual_names = ['harmonized','comp','real']
        
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G'] 
        self.opt.device = self.device
        self.netG = networks.define_G(opt.netG, opt.init_type, opt.init_gain, self.opt)
        
        if self.isTrain:
            util.saveprint(self.opt, 'netG', str(self.netG))  
            # define loss functions
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionL2 = torch.nn.MSELoss()
            self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.gpt_mask = self.get_mask(5)
        else:
            self.gpt_mask = self.get_mask(self.opt.n_frames)
        reference_points_1, spatial_shapes_1, level_start_index_1 = self.get_deformable_params(5, 64, 64, self.device)
        self.deformable_att = (reference_points_1, spatial_shapes_1, level_start_index_1)
    def set_position(self, pos, patch_pos=None):
        pass
    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        self.comp = input['comp'].to(self.device)
        self.real = input['real'].to(self.device)
        self.video_object_paths = input['video_object_path']
        self.harmonized = None

    def data_dependent_initialize(self, data):
        pass
    
    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.isTrain:
            self.harmonized = self.netG(self.comp, deformable_att=self.deformable_att, gpt_mask=self.gpt_mask)
        else:
            self.harmonized = self.netG(self.test_comp, deformable_att=self.deformable_att, gpt_mask=self.gpt_mask, is_test=True)
          
    def compute_G_loss(self):
        """Calculate L1 loss for the generator"""
        self.loss_G_L1 = self.criterionL1(self.harmonized, self.real)*self.opt.lambda_L1
        self.loss_G = self.loss_G_L1
        if self.opt.loss_T:
            self.loss_G_T = self.criterionL2(self.harmonized[:,1:,:,:,:]-self.harmonized[:,:-1,:,:,:], self.real[:,1:,:,:,:]-self.real[:,:-1,:,:,:])*self.opt.lambda_T
            self.loss_G = self.loss_G + self.loss_G_T
        return self.loss_G

    def optimize_parameters(self):
        self.forward()
        # update G
        self.optimizer_G.zero_grad()
        self.loss_G = self.compute_G_loss()
        self.loss_G.backward()
        self.optimizer_G.step()
    
    def new_test(self):
        with torch.no_grad():
            outputs = []
            self.test_inputs = None
            for i in range(int(self.comp.size(1)/self.opt.n_frames)):
                self.test_comp = self.comp[:, i*self.opt.n_frames:(i+1)*self.opt.n_frames, :, :, :]
                self.forward()
                outputs.append(self.harmonized)
            self.harmonized = (torch.cat(outputs, 1)+self.comp)*0.5
            
    def get_mask(self, t):
        H,W = 64, 64
        gpt_attn_mask = get_gpt_mask(t, H, W, 16, shift_size=0).to(self.device)
        gpt_attn_mask_shift = get_gpt_mask(t, H, W, 16, shift_size=8).to(self.device)
        gpt_mask = (gpt_attn_mask, gpt_attn_mask_shift)
        return gpt_mask
    
    def get_deformable_params(self, T, H, W, device=None):
        spatial_shape = (H, W)
        spatial_shapes = torch.as_tensor(spatial_shape, dtype=torch.long, device=device)
        spatial_shapes = spatial_shapes.unsqueeze(0).repeat(T, 1)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))

        m = self.get_valid_ratio(torch.ones([1, H, W], device=device))
        valid_ratios = m.repeat(T, 1).unsqueeze(0)
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=device)
        return reference_points, spatial_shapes, level_start_index
        
    def get_valid_ratio(self, mask):
        _, H, W = mask.shape
        valid_H = torch.sum(mask[:, :, 0], 1)
        valid_W = torch.sum(mask[:, 0, :], 1)
        valid_ratio_h = valid_H.float() / H
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1)
        return valid_ratio
    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):

            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_)
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_)
            ref = torch.stack((ref_x, ref_y), -1)
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None]
        return reference_points
    

def get_gpt_mask(D, H, W, window_size, shift_size):
    img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
    h_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))
    w_slices = (slice(0, -window_size),
                slice(-window_size, -shift_size),
                slice(-shift_size, None))

    cnt = 0
    for h in h_slices:
        for w in w_slices:
            img_mask[:, h, w, :] = cnt
            cnt += 1

    mask_windows = window_partition2d(img_mask, window_size)  # nW, window_size, window_size, 1
    mask_windows = mask_windows.view(-1, window_size * window_size)
    attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
    attn_mask = attn_mask.masked_fill(attn_mask != 0, -float("Inf")).masked_fill(attn_mask == 0, float(0.0))

    length = D
    
    attn_mask = attn_mask.repeat(1, length, length)
    
    return attn_mask

def window_partition2d(x, window_size):
    """
    Args:
        x: (B, H, W, C)
        window_size (int): window size

    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows