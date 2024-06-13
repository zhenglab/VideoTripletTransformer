import numpy as np
import os, torch
import sys
import ntpath
import time
from . import util, html, htmlfortest
from subprocess import Popen, PIPE
from PIL import Image,ImageDraw, ImageFont
from torchvision.transforms import Resize
from collections import OrderedDict
if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError


def save_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()  # results/experiment_name/test_latest/images
    #增
    name_parts = image_path[-1].split('/')  # ../../dataset/HYouTube/synthetic_composite_videos/003234408d/object_0/00000.jpg
    name = name_parts[-3]+'_'+name_parts[-2]+'_'+name_parts[-1]  # 003234408d_object_0_00000.jpg
    name = os.path.splitext(name)[0]  # .splitext分离文件名与扩展名，返回元组(003234408d_object_0_00000, .jpg)
#    short_path = ntpath.basename(image_path[0])
#    name = os.path.splitext(short_path)[0]

    # webpage.add_header(name)
    

    for label, im_data in visuals.items():  # test中传来的visuals是visual_ones字典，label为'mask'等，im_data为单个tensor
        # print(im_data.shape)
        im = util.tensor2im(im_data)
        image_name = '%s_%s.jpg' % (name, label)  # 003234408d_object_0_00000_real
        sub_dir = os.path.join(image_dir,name_parts[-3],name_parts[-2])  # results/experiment_name/test_latest/images/003234408d/object_0
        if not os.path.exists(sub_dir):
            os.makedirs(sub_dir)
        # web_dir = os.path.join(sub_dir, '%s_%s' % (name_parts[-3], name_parts[-2]))  # define the website directory
        
        save_path = os.path.join(sub_dir, image_name)
        util.save_image(im, save_path, aspect_ratio=aspect_ratio)
        
    
        #增
        # if label=='mask':
        #     sub_dir = os.path.join(image_dir,label,name_parts[-3],name_parts[-2])
        #     if not os.path.exists(sub_dir):
        #         os.makedirs(sub_dir)
        # elif label=='harmonized':
        #     sub_dir = os.path.join(image_dir,label,name_parts[-3],name_parts[-2])
        #     if not os.path.exists(sub_dir):
        #         os.makedirs(sub_dir)            
        # elif label=='comp':
        #     sub_dir = os.path.join(image_dir,label,name_parts[-3],name_parts[-2])
        #     if not os.path.exists(sub_dir):
        #         os.makedirs(sub_dir)            
        # elif label=='real':
        #     sub_dir = os.path.join(image_dir,label,name_parts[-3],name_parts[-2])
        #     if not os.path.exists(sub_dir):
        #         os.makedirs(sub_dir)            
        # save_path = os.path.join(sub_dir, image_name)
        #增

def save_train_feature_images(visuals, image_dir, epoch):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """

    # one_image  = 64
    # padding_num = 0
    # one_image_w = one_image+2*padding_num
    # w_grp = 4
    # h_grp = 4*5
    
    # # save_path = os.path.join(sub_dir, name)
    # frames = visuals['frames'][0,...]
    # result = Image.new('RGB', (w_grp*one_image_w-padding_num*2, h_grp*one_image_w+20),"white")
    # # print(result.size)
    # # result_1 = Image.new('RGB', (w_grp*one_image_w-padding_num*2, h_grp*one_image_w+10),"white")
    # for i in range(0, 5):
    #     locations = visuals['locations'][0,i,...]
    #     frames_label = visualize_train_landmark(frames, locations)
    #     one_image_h = one_image_w+4
    #     result.paste(frames_label[0], box=(0*one_image_w, 4*i*one_image_h))
        
    #     result.paste(frames_label[1], box=(2*one_image_w, 4*i*one_image_h))

    #     result.paste(frames_label[2], box=(0*one_image_w, (4*i+1)*one_image_h))
    #     result.paste(frames_label[3], box=(2*one_image_w, (4*i+1)*one_image_h))
    #     result.paste(frames_label[4], box=(0*one_image_w, (4*i+2)*one_image_h))
    # img_path = os.path.join(image_path, str(epoch)+'_merge.jpg')
    # result.save(img_path,  quality=75)
    # return img_path

    one_image  = 256
    padding_num = 4
    one_image_w = one_image+2*padding_num
    w_grp = 9
    h_grp = 5
    result = Image.new('RGB', (w_grp*one_image_w-padding_num*2, h_grp*one_image_w),"white")
    # save_path = os.path.join(sub_dir, name)
    for i in range(0, 5):
        img_1 = tensor_to_image(visuals['mask'][0, i,:,:,:])
        comp = tensor_to_image(visuals['comp'][0, i,:,:,:])
        real = tensor_to_image(visuals['real'][0, i,:,:,:])
        harmonized = tensor_to_image(visuals['harmonized'][0, i,:,:,:])
        result.paste(Image.fromarray(img_1), box=(0, i*one_image_w))
        result.paste(Image.fromarray(comp), box=(one_image_w, i*one_image_w))
        result.paste(Image.fromarray(real), box=(2*one_image_w, i*one_image_w))
        result.paste(Image.fromarray(harmonized), box=(3*one_image_w, i*one_image_w))
        
        # frames = visuals['frames'][0,...].detach()
        if 'locations' in visuals.keys():
            locations = visuals['locations'][0,i,...].detach()
            frames_label = visualize_landmark(visuals['harmonized'][0,:5,...], locations)
            for j in range(len(frames_label)):
                result.paste(frames_label[j], box=((3+j+1)*one_image_w, i*one_image_w))
        
    img_path = os.path.join(image_dir, epoch+'_merge.jpg')
    result.save(img_path,  quality=75)

def visualize_train_landmark(features, landmarks):
    """ plot landmarks on image
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return: plots of images with landmarks on
    """
    images = []
    t = features.size(0)
    features = (torch.nn.functional.tanh(torch.mean(features, dim=1,keepdim=True))+1)/2
    landmarks = 64*landmarks[10,40,0,...].cpu()
    for i in range(0,t):
        feature = features[i]
        # feature = torch.nn.functional.avg_pool2d(feature, (4,4))
        landmark = landmarks[i]
        feature =  Image.fromarray(tensor_to_image(feature))
        draw = ImageDraw.Draw(feature)
        landmark = np.array(landmark)
        for point in landmark:
            draw.point((point[0],point[1]),fill = (255, 255, 0))
        draw.point((10,40),fill = (255, 0, 0))
        
        # feature = feature.resize([256,256], Image.BICUBIC)
        images.append(feature)
    return images 

def save_feature_images(webpage, visuals, image_path, aspect_ratio=1.0, width=256):
    """Save images to the disk.

    Parameters:
        webpage (the HTML class) -- the HTML webpage class that stores these imaegs (see html.py for more details)
        visuals (OrderedDict)    -- an ordered dictionary that stores (name, images (either tensor or numpy) ) pairs
        image_path (str)         -- the string is used to create image paths
        aspect_ratio (float)     -- the aspect ratio of saved images
        width (int)              -- the images will be resized to width x width

    This function will save images stored in 'visuals' to the HTML file specified by 'webpage'.
    """
    image_dir = webpage.get_image_dir()  # results/experiment_name/test_latest/images
    #增
    name_parts = image_path[-1].split('/')  # ../../dataset/HYouTube/synthetic_composite_videos/003234408d/object_0/00000.jpg
    name = name_parts[-3]+'_'+name_parts[-2]+'_'+name_parts[-1]  # 003234408d_object_0_00000.jpg
    name = os.path.splitext(name)[0]  # .splitext分离文件名与扩展名，返回元组(003234408d_object_0_00000, .jpg)
#    short_path = ntpath.basename(image_path[0])
#    name = os.path.splitext(short_path)[0]

    one_image  = 256
    padding_num = 4
    one_image_w = one_image+2*padding_num
    w_grp = 9
    h_grp = 5
    result = Image.new('RGB', (w_grp*one_image_w-padding_num*2, h_grp*one_image_w),"white")
    # save_path = os.path.join(sub_dir, name)
    for i in range(0, 5):
        img_1 = tensor_to_image(visuals['mask'][0, i,:,:,:])
        comp = tensor_to_image(visuals['comp'][0, i,:,:,:])
        real = tensor_to_image(visuals['real'][0, i,:,:,:])
        harmonized = tensor_to_image(visuals['harmonized'][0, i,:,:,:])
        result.paste(Image.fromarray(img_1), box=(0, i*one_image_w))
        result.paste(Image.fromarray(comp), box=(one_image_w, i*one_image_w))
        result.paste(Image.fromarray(real), box=(2*one_image_w, i*one_image_w))
        result.paste(Image.fromarray(harmonized), box=(3*one_image_w, i*one_image_w))
        
        frames = visuals['frames'][0,...]
        locations = visuals['locations'][0,i,...]
        frames_label = visualize_landmark(visuals['harmonized'][0,:5,...], locations)
        for j in range(len(frames_label)):
            result.paste(frames_label[j], box=((3+j+1)*one_image_w, i*one_image_w))
        
    img_path = os.path.join(image_dir, name+'_merge.jpg')
    result.save(img_path,  quality=75)

def tensor_to_image(image,mean=0, std=1,imtype=np.uint8):
    image_numpy = image.detach().cpu().float().numpy()
    if image_numpy.shape[0] == 1:
        image_numpy = np.tile(image_numpy, (3, 1, 1))
    image_numpy = np.transpose(image_numpy, (1, 2, 0)) * std + mean
    image_numpy = image_numpy*255.0
    image_numpy = np.clip(image_numpy, 0,255)
    return image_numpy.astype(imtype)
def visualize_landmark(features, landmarks):
    """ plot landmarks on image
    :param image_array: numpy array of a single image
    :param landmarks: dict of landmarks for facial parts as keys and tuple of coordinates as values
    :return: plots of images with landmarks on
    """
    images = []
    t = features.size(0)
    # features = (torch.nn.functional.tanh(torch.mean(features, dim=1,keepdim=True))+1)/2.0
    # landmarks = 64*landmarks[10,40,0,...].cpu()
    landmarks = 16*landmarks[4,4,0,...].cpu()
    for i in range(0,t):
        feature = features[i]
        # feature = torch.nn.functional.avg_pool2d(feature, (4,4))
        feature = torch.nn.functional.avg_pool2d(feature, (16,16))
        landmark = landmarks[i]
        feature =  Image.fromarray(tensor_to_image(feature))
        draw = ImageDraw.Draw(feature)
        
        draw.point((4,4),fill = (255, 0, 0))
        
        landmark = np.array(landmark)
        for point in landmark:
            draw.point((point[0],point[1]),fill = (255, 255, 0))
            
        feature = feature.resize([256,256], Image.BICUBIC)
        images.append(feature)
    return images 


def save_test_html(path,visual_names,width):
    home = path+"/images"
    files = sorted(os.listdir(home))
    for fi in files:
        home_fi = os.path.join(home,fi)
        if os.path.isdir(home_fi):
            fi_1 = sorted(os.listdir(home_fi))
            webpage1 = htmlfortest.HTML(home_fi, 'Experiment = %s' % (fi))
            for fi_2 in fi_1:
                home_fi_fi_2 = os.path.join(home_fi,fi_2)
                if os.path.isdir(home_fi_fi_2):
                    fi_3 = sorted(os.listdir(home_fi_fi_2))
                    for fi_4 in fi_3:
                        if 'harmonized' in fi_4:
                            name_parts = fi_4[:-14]
                            webpage1.add_header('%s' % name_parts[:-1])
                            ims, txts, links = [], [], []
                            for label in visual_names:
                                image_name = os.path.join(fi_2, name_parts+label+".jpg")
                                ims.append(image_name)
                                txts.append(label)
                                links.append(image_name)
                            webpage1.add_images(ims, txts, links, width=width)
            webpage1.save()



class Visualizer():
    """This class includes several functions that can display/save images and print/save logging information.

    It uses a Python library 'visdom' for display, and a Python library 'dominate' (wrapped in 'HTML') for creating HTML files with images.
    """

    def __init__(self, opt):
        """Initialize the Visualizer class

        Parameters:
            opt -- stores all the experiment flags; needs to be a subclass of BaseOptions
        Step 1: Cache the training/test options
        Step 2: connect to a visdom server
        Step 3: create an HTML object for saveing HTML filters
        Step 4: create a logging file to store training losses
        """
        self.opt = opt  # cache the option
        self.display_id = opt.display_id
        self.use_html = opt.isTrain and not opt.no_html
        self.win_size = opt.display_winsize
        self.name = opt.name
        self.port = opt.display_port
        opt.display_env = opt.name
        self.saved = False
        if self.display_id > 0:  # connect to a visdom server given <display_port> and <display_server>
            import visdom
            self.ncols = opt.display_ncols
            self.vis = visdom.Visdom(server=opt.display_server, port=opt.display_port, env=opt.display_env)
            if not self.vis.check_connection():
                self.create_visdom_connections()

        if self.use_html:  # create an HTML object at <checkpoints_dir>/web/; images will be saved under <checkpoints_dir>/web/images/
            self.web_dir = os.path.join(opt.checkpoints_dir, opt.name, 'web')
            self.img_dir = os.path.join(self.web_dir, 'images')
            print('create web directory %s...' % self.web_dir)
            util.mkdirs([self.web_dir, self.img_dir])
        # create a logging file to store training losses
        self.log_name = os.path.join(opt.checkpoints_dir, opt.name, 'loss_log.txt')
        with open(self.log_name, "a") as log_file:
            now = time.strftime("%c")
            log_file.write('================ Training Loss (%s) ================\n' % now)

    def reset(self):
        """Reset the self.saved status"""
        self.saved = False

    def create_visdom_connections(self):
        """If the program could not connect to Visdom server, this function will start a new server at port < self.port > """
        cmd = sys.executable + ' -m visdom.server -p %d &>/dev/null &' % self.port
        print('\n\nCould not connect to Visdom server. \n Trying to start a server....')
        print('Command: %s' % cmd)
        Popen(cmd, shell=True, stdout=PIPE, stderr=PIPE)

    def display_current_results(self, visuals, epoch, save_result,dec="image"):
        """Display current results on visdom; save current results to an HTML file.

        Parameters:
            visuals (OrderedDict) - - dictionary of images to display or save
            epoch (int) - - the current epoch
            save_result (bool) - - if save the current results to an HTML file
        """
        if self.display_id > 0:  # show images in the browser using visdom
            ncols = self.ncols
            if ncols > 0:        # show all the images in one visdom panel
                ncols = max(ncols, len(visuals))
                h, w = next(iter(visuals.values())).shape[:2]
                table_css = """<style>
                        table {border-collapse: separate; border-spacing: 4px; white-space: nowrap; text-align: center}
                        table td {width: % dpx; height: % dpx; padding: 4px; outline: 4px solid black}
                        </style>""" % (w, h)  # create a table css
                # create a table of images.
                title = dec
                label_html = ''
                label_html_row = ''
                images = []
                idx = 0
                mask = visuals['mask']
                h = w = mask.size(4)
                torch_resize = Resize([h,w]) # 定义Resize类对象
                padding = torch.nn.ZeroPad2d(96)
                for label, image in visuals.items():
                    if len(image.size()) >= 5:
                        # zero = torch.zeros_like(image)[:,:,:,:2,:]
                        # print(zero.size())
                        # print(image.size())
                        if label == 'flows':
                            image = image[0].permute(0,2,3,1).detach().cpu().float().numpy()
                            flows = []
                            for j in range(image.shape[0]):
                                tmp = util.flow_to_image(image[j])
                                flows.append(tmp)
                            image = torch.from_numpy(np.array(flows)).permute(0,3,1,2)
                            # image = torch_resize(image)
                            image = padding(image)

                            pad = torch.nn.ZeroPad2d((2, 2, 2, 2))
                            image = pad(image).transpose(1,0).flatten(1,2).permute(1,2,0)
                            image_numpy = image.numpy()
                        elif label == 'locations' or label == 'frames':
                            pass
                        else:
                            if image.size(3) != h:
                                # image = torch_resize(image.flatten(0,1))
                                image = padding(image.flatten(0,1))
                                image = image.view(mask.size(0),mask.size(1),-1,h,w)
                            pad = torch.nn.ZeroPad2d((2, 2, 2, 2))
                            image = pad(image).transpose(2,1).flatten(2,3)
                            image_numpy = util.tensor2im(image)
                    if label != 'locations' and label != 'frames':
                        label_html_row += '<td>%s</td>' % label
                        images.append(image_numpy.transpose([2, 0, 1]))
                        idx += 1
                        if idx % ncols == 0:
                            label_html += '<tr>%s</tr>' % label_html_row
                            label_html_row = ''

                white_image = np.ones_like(image_numpy.transpose([2, 0, 1])) * 255
                while idx % ncols != 0:
                    images.append(white_image)
                    label_html_row += '<td></td>'
                    idx += 1
                if label_html_row != '':
                    label_html += '<tr>%s</tr>' % label_html_row
                try:
                    self.vis.images(images, nrow=ncols, win=self.display_id + 1,
                                    padding=2, opts=dict(title=title + ' images'))
                    label_html = '<table>%s</table>' % label_html
                    self.vis.text(table_css + label_html, win=self.display_id + 2,
                                  opts=dict(title=title + ' labels'))
                except VisdomExceptionBase:
                    self.create_visdom_connections()

            else:     # show each image in a separate visdom panel;
                idx = 1
                try:
                    for label, image in visuals.items():
                        image_numpy = util.tensor2im(image)
                        self.vis.image(image_numpy.transpose([2, 0, 1]), opts=dict(title=label),
                                       win=self.display_id + idx)
                        idx += 1
                except VisdomExceptionBase:
                    self.create_visdom_connections()
            visuals_ones = OrderedDict()
            for label, im_data in visuals.items():
                visuals_ones[label] = im_data
            save_train_feature_images( visuals_ones,self.img_dir, str(epoch))
        if self.use_html and (save_result or not self.saved):  # save images to an HTML file if they haven't been saved.
            self.saved = True
            # save images to the disk
            for label, image in visuals.items(): # image.shape=[bs,t,c,h,w]
                if label == 'flows':
                    image = image[0].permute(0,2,3,1).detach().cpu().float().numpy()
                    image_numpy = util.flow_to_image(image[0])
                    # image_numpy = np.transpose(flow, (1, 2, 0))
                elif label == 'frames' or label == 'locations':
                    pass
                else:
                    bs, t, c, h, w = image.shape
                    image = image.reshape(bs*t, c, h, w)
                    image_numpy = util.tensor2im(image)
                img_path = os.path.join(self.img_dir, 'epoch%.3d_%s.jpg' % (epoch, label))
                util.save_image(image_numpy, img_path)

            # update website
            webpage = html.HTML(self.web_dir, 'Experiment name = %s' % self.name, refresh=50)
            for n in range(epoch, 0, -1):
                webpage.add_header('epoch [%d]' % n)
                ims, txts, links = [], [], []

                for label, image_numpy in visuals.items():             
                    # image_numpy = util.tensor2im(image)
                    img_path = 'epoch%.3d_%s.jpg' % (n, label)
                    ims.append(img_path)
                    txts.append(label)
                    links.append(img_path)
                webpage.add_images(ims, txts, links, width=self.win_size)
            webpage.save()

    def plot_current_losses(self, epoch, counter_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values

        Parameters:
            epoch (int)           -- current epoch
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(epoch + counter_ratio)
        self.plot_data['Y'].append([losses[k].cpu().item() for k in self.plot_data['legend']])
        X=np.stack([np.array(self.plot_data['X'])] * len(self.plot_data['legend']), 1)
        Y=np.array(self.plot_data['Y'])
        try:
            self.vis.line(
                X=X,
                Y=Y,
                opts={
                    'title': self.name + ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win=self.display_id)
        except VisdomExceptionBase:
            self.create_visdom_connections()

    # losses: same format as |losses| of plot_current_losses
    def print_current_losses(self, epoch, iters, losses, t_comp, t_data):
        """print current losses on console; also save the losses to the disk

        Parameters:
            epoch (int) -- current epoch
            iters (int) -- current training iteration during this epoch (reset to 0 at the end of every epoch)
            losses (OrderedDict) -- training losses stored in the format of (name, float) pairs
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """
        message = '(epoch: %d, iters: %d, time: %.3f, data: %.3f) ' % (epoch, iters, t_comp, t_data)
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)

        print(message)  # print the message
        with open(self.log_name, "a") as log_file:
            log_file.write('%s\n' % message)  # save the message



