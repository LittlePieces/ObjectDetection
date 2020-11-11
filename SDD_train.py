#coding=utf-8

import argparse
import torch
import torch.nn as nn
import numpy as np
import re
import torchvision as tv
from torch.autograd import Variable
import torch.nn.functional as F
import os
from PIL import Image
import cv2
from collections import OrderedDict
import time
from scipy.io import loadmat
import operator
from torch.nn.parallel import DistributedDataParallel 
import torch.distributed as dist
import torch.utils.data.distributed 

#define argument
class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--data_path', default='train',help='# path to store training data ')
        self.parser.add_argument('--save_path',type=str,default='sdd_checkpoints',help='# path to save checkpoints')
        self.parser.add_argument('--label_path',  default='label',help='# path to store label')
        self.parser.add_argument('--batch_size', type=int, default=64)
        self.parser.add_argument('--num_workers', default=2, type=int)
        self.parser.add_argument('--image_size', type=int, default=896, help='# the image size to resize')
        self.parser.add_argument('--max_epoch', type=int, default=300, help='# epoch count')
        self.parser.add_argument('--lr', type=float, default=0.0001, help='#  learning rate')
        self.parser.add_argument('--gpus', action='store_true', default=True, help='# whether to use gpu')
        self.parser.add_argument('--vis', default=True, help='# whether to use visdom visulizer')
        self.parser.add_argument('--env', type=str, default='main', help='# visdom env')
        self.parser.add_argument('--print_every', type=int, default=50, help='# batchsize interval to print error')
        self.parser.add_argument('--display_winsize', type=int, default=256,  help='display window size')
        self.parser.add_argument('--display_port', type=int, default=8097, help='# visdom port of the web display')
        self.parser.add_argument('--load_model', type=str, default=None, help='# path to load the pre-trained model')
        self.parser.add_argument('--local_rank', type=int,help='# use multi GPU to train')

    def parse(self):
                if not self.initialized:
                        self.initialize()
                self.opt = self.parser.parse_args()
                args = vars(self.opt)
                if self.opt.local_rank==0:
                    print('------------ Options -------------')
                    for k, v in sorted(args.items()):
                        print('%s: %s' % (str(k), str(v)))
                    print('-------------- End ----------------')
                return self.opt


#define dataset
image_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in image_EXTENSIONS)

def make_dataset(dir):
    images = []  
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, dirs, fnames in sorted(os.walk(dir)):
       for fname in fnames:
           path = os.path.join(root, fname)
           images.append(path)

    return images            

def transform(image):

    transforms = tv.transforms.Compose([
                                                tv.transforms.Resize((opt.image_size,opt.image_size)),
                                                tv.transforms.ToTensor(), 
                                       ])
    image=transforms(image)
    mean=image.mean()
    std=image.std()
    image=tv.transforms.Normalize([mean],[std])(image)
    return image
class MyDataset(torch.utils.data.Dataset):
    def __init__(self, validata=False):
        self.opt = opt
        
        if  validata:
            self.root = opt.validata_path
            self.dir= os.path.join(opt.validata_path)
        else:
            self.root = opt.data_path
            self.dir = os.path.join(opt.data_path) 
        self.path = make_dataset(self.dir) 
        
    
    def __getitem__(self, index) :
                image_path = self.path[index]
                label_name = re.split(r'[/]', image_path)
                label_name = os.path.join(label_name[-2],label_name[-1])
                image = Image.open(image_path).convert('L')
                image = transform(image)
                return image, label_name

    def __len__(self):
        
        return len(self.path)	


#define network

class ResidualBlock(nn.Module):
        def __init__(self, in_channels, out_channels, stride=1):
            super(ResidualBlock, self).__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)
            if in_channels != out_channels:
                    self.downsample = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=2),
                        nn.BatchNorm2d(out_channels)
                        )
            else:
                self.downsample = None

        def forward(self, x):
            identity = x
            out = self.conv1(x)
            out = self.bn1(out)
            out = self.relu(out)
            out = self.conv2(out)
            out = self.bn2(out)

            if self.downsample is not None:
                identity = self.downsample(x)
            out += identity
            out = self.relu(out)
            return out
 
class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.first = nn.Sequential(
        nn.Conv2d(1, 32, 7, 2, 3),
                nn.BatchNorm2d(32),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, 2, 1),
                nn.BatchNorm2d(32),
                nn.ReLU()
                )
        self.layer1 = self.make_layer(32, 32, 3, 1) #in_channel=64,out_channel=64,block_num=3,stride=1
        self.layer2 = self.make_layer(32, 64, 4, 2) #in_channel=64,out_channel=128,block_num=4,stride=2
        self.layer3 = self.make_layer(64, 128, 6, 2) #in_channel=128,out_channel=256,block_num=6,stride=2
        self.layer4 = self.make_layer(128, 256, 3, 2) #in_channel=256,out_channel=512,block_num=3,stride=2
        self.layer5 = ResidualBlock(256, 512, 2)    
        self.layer6 = ResidualBlock(512,1024,2)
        self.end = nn.Sequential(
               nn.Upsample(scale_factor=2, mode='nearest'),
               nn.Conv2d(1024, 512, 3, 1,1),
               nn.BatchNorm2d(512),
               nn.ReLU(),
               nn.Upsample(scale_factor=2, mode='nearest'),
               nn.Conv2d(512, 256, 3, 1,1),
               nn.BatchNorm2d(256),
               nn.ReLU(),
               nn.Upsample(scale_factor=2, mode='nearest'),
               nn.Conv2d(256, 128, 3, 1,1),
               nn.BatchNorm2d(128),
               nn.ReLU(),
               nn.Upsample(scale_factor=2, mode='nearest'),
               nn.Conv2d(128, 64, 3, 1,1),
               nn.BatchNorm2d(64),
               nn.ReLU(),
               nn.Upsample(scale_factor=2, mode='nearest'),
               nn.Conv2d(64, 32, 3, 1,1),
               nn.BatchNorm2d(32),
               nn.ReLU(),	
               nn.Upsample(scale_factor=2, mode='nearest'),
               nn.Conv2d(32, 32, 3, 1,1),
               nn.BatchNorm2d(32),
               nn.ReLU(),
               nn.Upsample(scale_factor=2,mode='nearest'),
               nn.Conv2d(32, 1, 3, 1,1),
        )     

    def make_layer(self, in_channels, out_channels, block_num, stride):
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride))
        for i in range(1, block_num):
            layers.append(ResidualBlock(out_channels, out_channels, 1))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.first(x)       
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.end(x)
        return x

 
# visulize
class Visualizer():
        def __init__(self, opt):
                self.display_id = 1
                self.win_size = 256
                self.name = 'detection loss'
                if opt.vis==True:
                        import visdom
                        self.vis = visdom.Visdom(env=opt.env, port=opt.display_port)

        def plot_current_errors(self, epoch, count_ratio, opt, errors):
                 if not hasattr(self, 'plot_data'):
                        self.plot_data = {'X':[],'Y':[], 'legend':list(errors.keys())}
                 self.plot_data['X'].append(epoch + count_ratio)
                 for k in self.plot_data['legend']:
                        errors=errors[k].cpu().numpy()
                 self.plot_data['Y'].append([errors])
                 self.vis.line(
                        X=np.stack([np.array(self.plot_data['X'])]*len(self.plot_data['legend']),1) if len(self.plot_data['X'])==1 else np.stack([np.array(self.plot_data['X'])]*len(self.plot_data['legend']),1).squeeze(),
                        Y=np.array(self.plot_data['Y']) if len(self.plot_data['Y'])==1 else np.array(self.plot_data['Y']).squeeze(),
                        opts={
                                'title': self.name + ' loss over time',
                                'legend': self.plot_data['legend'],
                                'xlabel': 'epoch',
                                'ylabel': 'loss'},
                        win=self.display_id)
def print_current_errors(epoch, i, errors,t):
                message = '(epoch: %d, iters: %d, time: %.3f) ' % (epoch, i,t)
                for k, v in errors.items():
                        message += '%s: %.8f ' % (k, v)
                print(message)

#train network
def train():
    if opt.vis:
        vis  = Visualizer(opt)
    
    SDDNet = ResNet()
    SDDNet = nn.SyncBatchNorm.convert_sync_batchnorm(SDDNet)
    dist.init_process_group(backend='nccl', init_method='env://') 
    
    if opt.load_model:
        map_location = lambda storage, loc: storage
        SDDNet.load_state_dict(torch.load(opt.load_model, map_location=map_location))
    
    criterion=nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(SDDNet.parameters(), opt.lr, betas=(0.5, 0.999)) 
    
    if opt.gpus:
        device = torch.device('cuda', opt.local_rank)
        SDDNet=SDDNet.to(device)
        criterion=criterion.to(device)
        SDDNet = torch.nn.parallel.DistributedDataParallel(SDDNet, device_ids=[opt.local_rank], output_device=opt.local_rank,find_unused_parameters=True)
    dataset = MyDataset()
    train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader = torch.utils.data.DataLoader(dataset,
                                         pin_memory=True,
                                         batch_size=opt.batch_size,
                                         shuffle=(train_sampler is None),
                                         num_workers=opt.num_workers,
                                         sampler=train_sampler,
                                         drop_last=True
                                         )

    for epoch in range(1,opt.max_epoch+1):
        epoch_iter = 0
        for ii,(image, labels)  in enumerate(dataloader):
            iter_start_time = time.time()
            epoch_iter += opt.batch_size
            inputs = Variable(image)
            optimizer.zero_grad()
            outputs = SDDNet(inputs.to(device))
            target = np.zeros((opt.batch_size,1,opt.image_size, opt.image_size))
            for l in range(0,opt.batch_size):
                mask = labels[l]
                mask = Image.open('%s/%s'%(opt.label_path,mask)).convert('L')
                mask = tv.transforms.Resize((opt.image_size,opt.image_size))(mask)
                target[l,0,:,:] = mask
            target = torch.Tensor(target)
            pre_outputs = Variable(target).to(device)
            weights=torch.empty_like(target).fill_(0.0141)
            weights[target==1]=0.9859
            criterion=nn.BCEWithLogitsLoss(weight=weights).to(device)
            loss = criterion(outputs, pre_outputs)
            loss.backward()
            optimizer.step()
            if opt.local_rank==0:
                errors = get_current_errors(loss)
                if (ii+1)% opt.print_every  == 0:
                    
                    ti = (time.time() - iter_start_time) / opt.batch_size
                    print_current_errors(epoch, epoch_iter, errors, ti)
                
                if opt.vis and (ii+1)% opt.print_every == 0:	
                    with open('training_loss.txt','a') as f:
                        vdl = 'epoch:%d training loss:%.10f'%(epoch, loss)
                        f.write(vdl + '\n')
                        f.close()
                    load_rate = float(epoch_iter)/dataset.__len__()
                    vis.plot_current_errors(epoch, load_rate, opt, errors)	
        if opt.local_rank ==0 and epoch % 2 ==0:
            torch.save(SDDNet.module.state_dict(), './%s/%s.pth'%(opt.save_path, str(epoch)))
            
    print('complete!')


def get_current_errors(loss):
    return OrderedDict([('ResnetLoss', loss.data)])

if __name__ == "__main__":
    
     
    opt = BaseOptions().parse()
    train()

