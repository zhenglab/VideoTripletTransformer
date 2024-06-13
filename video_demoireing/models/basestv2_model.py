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
            # self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, self.netG.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
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
        self.comp = input['comp'].to(self.device)  # [b,t,c,w,h]
        self.real = input['real'].to(self.device)
        # self.real = self.real-self.comp
        self.video_object_paths = input['video_object_path']  # 长度为batchsize的列表 ../../dataset/HYouTube/synthetic_composite_videos/003234408d/object_0/00000.jpg
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
            # self.harmonized = torch.cat(outputs, 1)
            
            # img_train_mean = self.harmonized.mean(dim=4).mean(dim=3)
            # img_train_mean = img_train_mean.unsqueeze(dim=-1).unsqueeze(dim=-1)
            # img_train_mean = img_train_mean.expand_as(self.harmonized)
            # noise = self.harmonized - img_train_mean
            # self.harmonized = noise*0.2 + self.harmonized*0.8
            
            # self.harmonized = torch.div(self.harmonized, self.comp)
            # self.harmonized = torch.clamp(self.harmonized, 0, 255)
            # self.harmonized = torch.cat(outputs, 1)*0.6+self.comp*0.4 #outdoor
            self.harmonized = (torch.cat(outputs, 1)+self.comp)*0.5

    # def new_test(self):
    #     with torch.no_grad():
    #         outputs = []
    #         self.test_inputs = None
    #         for i in range(int(self.comp.size(1)/self.opt.n_frames)):
                
    #             self.test_comp = self.comp[:, i*self.opt.n_frames:(i+1)*self.opt.n_frames, :, :, :]
    #             self.test_video()
    #             outputs.append(self.harmonized)
    #         self.harmonized = torch.cat(outputs, 1)

    # def test_video(self):
    #     '''test the video as a whole or as clips (divided temporally). '''
    #     lq = self.test_comp
    #     num_frame_testing = self.opt.n_frames # 5
    #     if num_frame_testing:
    #         # test as multiple clips if out-of-memory
    #         sf = 1
    #         self.opt.tile_overlap = [0,100,100]
    #         num_frame_overlapping = self.opt.tile_overlap[0]  # num_frame_overlapping = 2
    #         not_overlap_border = False
    #         b, d, c, h, w = lq.size()
    #         # c = c - 1 if args.nonblind_denoising else c
    #         stride = num_frame_testing - num_frame_overlapping  # stride = 5-2 = 3
    #         d_idx_list = list(range(0, d-num_frame_testing, stride)) + [max(0, d-num_frame_testing)]  # [0,3,6,9,10]
    #         E = torch.zeros(b, d, c, h*sf, w*sf).to(self.device)
    #         W = torch.zeros(b, d, 1, 1, 1).to(self.device)

    #         for d_idx in d_idx_list:
    #             lq_clip = lq[:, d_idx:d_idx+num_frame_testing, ...]
    #             out_clip = self.test_clip(lq_clip)
    #             out_clip_mask = torch.ones((b, min(num_frame_testing, d), 1, 1, 1)).to(self.device)

    #             if not_overlap_border:
    #                 if d_idx < d_idx_list[-1]:
    #                     out_clip[:, -num_frame_overlapping//2:, ...] *= 0
    #                     out_clip_mask[:, -num_frame_overlapping//2:, ...] *= 0
    #                 if d_idx > d_idx_list[0]:
    #                     out_clip[:, :num_frame_overlapping//2, ...] *= 0
    #                     out_clip_mask[:, :num_frame_overlapping//2, ...] *= 0

    #             E[:, d_idx:d_idx+num_frame_testing, ...].add_(out_clip)
    #             W[:, d_idx:d_idx+num_frame_testing, ...].add_(out_clip_mask)
    #         output = E.div_(W)
    #         self.harmonized = output

    # def test_clip(self, lq):
    
    #     sf = 1
    #     window_size = [2, 8, 8] #args.window_size
    #     size_patch_testing = 256 #args.tile[1]
    #     assert size_patch_testing % window_size[-1] == 0, 'testing patch size should be a multiple of window_size.'

    #     if size_patch_testing:
    #         # divide the clip to patches (spatially only, tested patch by patch)
    #         overlap_size = self.opt.tile_overlap[1]  # overlap_size = 8
    #         not_overlap_border = False
    #         # not_overlap_border = True

    #         # test patch by patch
    #         b, d, c, h, w = lq.size()
    #         # c = c - 1 if args.nonblind_denoising else c
    #         stride = size_patch_testing - overlap_size  # stride = 256-8
    #         h_idx_list = list(range(0, h-size_patch_testing, stride)) + [max(0, h-size_patch_testing)]
    #         w_idx_list = list(range(0, w-size_patch_testing, stride)) + [max(0, w-size_patch_testing)]
    #         E = torch.zeros(b, d, c, h*sf, w*sf).to(self.device)
    #         W = torch.zeros_like(E).to(self.device)

    #         for h_idx in h_idx_list:
    #             for w_idx in w_idx_list:
    #                 in_patch = lq[..., h_idx:h_idx+size_patch_testing, w_idx:w_idx+size_patch_testing]
    #                 # out_patch = model(in_patch).detach().cpu()
    #                 with torch.no_grad():
    #                     # out_patch,self.frames, self.locations = self.netG(in_patch, spatial_pos=self.spatial_pos, three_dim_pos=self.three_dim_pos)
    #                     out_patch = self.netG(in_patch, deformable_att=self.deformable_att, gpt_mask=self.gpt_mask, is_test=True)
    
    #                 out_patch_mask = torch.ones_like(out_patch)

    #                 if not_overlap_border:
    #                     if h_idx < h_idx_list[-1]:
    #                         out_patch[..., -overlap_size//2:, :] *= 0
    #                         out_patch_mask[..., -overlap_size//2:, :] *= 0
    #                     if w_idx < w_idx_list[-1]:
    #                         out_patch[..., :, -overlap_size//2:] *= 0
    #                         out_patch_mask[..., :, -overlap_size//2:] *= 0
    #                     if h_idx > h_idx_list[0]:
    #                         out_patch[..., :overlap_size//2, :] *= 0
    #                         out_patch_mask[..., :overlap_size//2, :] *= 0
    #                     if w_idx > w_idx_list[0]:
    #                         out_patch[..., :, :overlap_size//2] *= 0
    #                         out_patch_mask[..., :, :overlap_size//2] *= 0

    #                 E[..., h_idx*sf:(h_idx+size_patch_testing)*sf, w_idx*sf:(w_idx+size_patch_testing)*sf].add_(out_patch)
    #                 W[..., h_idx*sf:(h_idx+size_patch_testing)*sf, w_idx*sf:(w_idx+size_patch_testing)*sf].add_(out_patch_mask)
    #         output = E.div_(W)
    #     return output
            
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
    # attn_mask_gpt = torch.full((length, length), -float("Inf"))
    # attn_mask_gpt = torch.triu(attn_mask_gpt, diagonal=1)
    # attn_mask_gpt = attn_mask_gpt.repeat_interleave(window_size**2, dim=0)
    # attn_mask_gpt = attn_mask_gpt.repeat_interleave(window_size**2, dim=1)
    
    # # mask self window
    # attn_mask_mae = torch.zeros(length)-float("Inf")
    # attn_mask_mae = torch.diag_embed(attn_mask_mae)
    # attn_mask_mae = attn_mask_mae.repeat_interleave(window_size**2, dim=0)
    # attn_mask_mae = attn_mask_mae.repeat_interleave(window_size**2, dim=1)
    
    
    attn_mask = attn_mask.repeat(1, length, length)
    
    # attn_mask = attn_mask+attn_mask_mae[None, :, :]
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