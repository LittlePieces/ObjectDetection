#coding=utf-8
import sys
sys.path.append('..')
from torch2trt import torch2trt
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
        self.parser.add_argument('--image_size', type=int, default=896, help='# the image size to resize')	
        self.parser.add_argument('--load_model', type=str, default=None, help='# load the trained model')
        self.parser.add_argument('--save_path',type=str,default='test_result/SDD',help='path to save result')
        self.parser.add_argument('--acc',default=False,help='whether to use tensorRT')
    def parse(self):
                if not self.initialized:
                        self.initialize()
                self.opt = self.parser.parse_args()
                args = vars(self.opt)
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
    if opt.acc==True:
        image=tv.transforms.Normalize(np.array([mean],dtype=float),np.array([std],dtype=float))(image)
    else:
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
        x = torch.sigmoid(x)
        return x

def test():
        print('---------------------start------------------------') 
  
        SDDNet = ResNet().eval()
        SDDNet.cuda()

        map_location = lambda storage, loc: storage	
        SDDNet.load_state_dict(torch.load(opt.load_model, map_location=map_location),False)
        dataset = MyDataset()
        dataloader = torch.utils.data.DataLoader(dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=1,
                    drop_last=False
                    )

        for ii, (image, labels) in enumerate(dataloader):
                label_name = labels[0]       
                save_path = os.path.join('test_result',re.split(r'[/]',label_name)[-2])
                if (os.path.exists(save_path)==False):
                    os.makedirs(save_path)             
                inputs = Variable(image).cuda()
                output = SDDNet(inputs) 
                output_numpy = output.data[0,0,:,:].cpu().float().numpy()
                output_numpy = output_numpy * 255.0
                output_numpy = output_numpy.astype(np.uint8)
                output_PIL = Image.fromarray(output_numpy, mode='L')
                output_PIL.save('./test_result/%s'%(label_name))
        print('--------------------complete!-----------------------')


def acc_test():
        print('---------------------start------------------------') 
        SDDNet = ResNet().eval()
        SDDNet.cuda()
        x=torch.ones(1,1,896,896).cuda()
        model_trt = torch2trt(resnet,[x])
        map_location = lambda storage, loc: storage	
        SDDNet.load_state_dict(torch.load(opt.load_model, map_location=map_location),False)
        dataset = MyDataset()
        dataloader = torch.utils.data.DataLoader(dataset,
                    batch_size=1,
                    shuffle=False,
                    num_workers=1,
                    drop_last=False
                    )
        for ii, (image, labels) in enumerate(dataloader):
                label_name = labels[0]       
                save_path = os.path.join('%s'%(opt.save_path),re.split(r'[/]',label_name)[-2])
                if (os.path.exists(save_path)==False):
                    os.makedirs(save_path)             
                inputs = Variable(image).cuda()
                output = model_trt(inputs) 
                output_numpy = output.data[0,0,:,:].cpu().float().numpy()
                output_numpy = output_numpy * 255.0
                output_numpy = output_numpy.astype(np.uint8)
                output_PIL = Image.fromarray(output_numpy, mode='L')
                output_PIL.save('./test_result/%s'%(label_name))
        print('--------------------complete!-----------------------')


if __name__ == "__main__":
    opt = BaseOptions().parse()
    if opt.acc==True:
        acc_test()
    else:
        test()
