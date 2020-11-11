import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision as tv
from torch.autograd import Variable
from PIL import Image,ImageStat
from collections import OrderedDict
import operator
import os
import re
import time
import numpy as np
from torch.nn.parallel import DistributedDataParallel 
import torch.distributed as dist
import torch.utils.data.distributed 
import sys 
sys.path.append('..')
from torch2trt import torch2trt
#define argument
class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
    def initialize(self):  
        self.parser.add_argument('--data_path',default='train',help="path to store training data")
        self.parser.add_argument('--save_path',type=str,default='test_result/icModule',help='path to save result')
        self.parser.add_argument('--image_size',type=int,default=896,help="the size to resize")
        self.parser.add_argument('--load_model',type=str,help="path to load trained model")
        self.parser.add_argument('--acc',default=False,help='whether to use tensorRT')
    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt=self.parser.parse_args()
        args=vars(self.opt)
        print('-----------Options----------')  
        for k,v in sorted(args.items()):
            print('%s:%s'%(str(k),str(v)))
        print('-------------End------------')
        return self.opt

#define dataset
def make_dataset(dir):
    images = []
    path1 = []
    path2 = []
    path3 = []
    path4 = []
    path5 = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root,dirs,fnames in os.walk(dir):
        if len(dirs)==0:
            subFolder=[]
            for fname in fnames:
                path=os.path.join(root,fname)
                subFolder.append(path)
            images.append(subFolder)
        else:
            for dirr in dirs:
                subFolder = []
                for fname in sorted(os.listdir(dir+'/'+dirr)):
                    path = os.path.join(dir+'/'+dirr+'/', fname)
                    subFolder.append(path)
                images.append(subFolder)
            break
    for i in range(0, len(images)):
        #data augmentation
        for j in range(4, len(images[i])):
            path1.extend([images[i][j-4],images[i][j]])
            path2.extend([images[i][j-3],images[i][j-1]])
            path3.extend([images[i][j-2],images[i][j-2]])
            path4.extend([images[i][j-1],images[i][j-3]])  
            path5.extend([images[i][j],images[i][j-4]])
    return path1, path2, path3, path4, path5

def transform(image):

    transforms = tv.transforms.Compose([
        tv.transforms.Resize((opt.image_size, opt.image_size)),
        tv.transforms.ToTensor(),
    ])
    image = transforms(image)

    return image
      
class MyDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.opt=opt
        self.data_path=os.path.join(opt.data_path)
        self.path1, self.path2, self.path3, self.path4, self.path5 = make_dataset(
            self.data_path)
    def __getitem__(self,index):
        
        A_path1 = self.path1[index]
        A_path2 = self.path2[index]
        A_path3 = self.path3[index]
        A_path4 = self.path4[index]
        A_path5 = self.path5[index]
        A_image1 = Image.open(A_path1).convert('L')
        A_image2 = Image.open(A_path2).convert('L')
        A_image3 = Image.open(A_path3).convert('L')
        A_image4 = Image.open(A_path4).convert('L')
        A_image5 = Image.open(A_path5).convert('L')
        image = torch.zeros(5, opt.image_size, opt.image_size)
        image[0] = transform(A_image1)
        image[1] = transform(A_image2)
        image[2] = transform(A_image3)
        image[3] = transform(A_image4)
        image[4] = transform(A_image5)

        label_name = re.split(r'[/]', A_path5)
        label_name = os.path.join(label_name[-2],label_name[-1])
        return image,label_name

    def __len__(self):
        return len(self.path1)  

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

class PostProcess(nn.Module):
    def __init__(self):
        super (PostProcess,self).__init__()
        self.layer1=ResidualBlock(5, 32, 2)
        self.layer2=ResidualBlock(32, 64, 2)
        self.layer3=ResidualBlock(64, 128, 2)
        self.pad1 = nn.Sequential(
            nn.Conv2d(128,64,3,1,1),
            nn.BatchNorm2d(64),
            nn.ReLU()
            )
        self.pad2 = nn.Sequential(
            nn.Conv2d(64,32,3,1,1),
            nn.BatchNorm2d(32),
            nn.ReLU()
            )
        self.pad3 = nn.Sequential(
            nn.Conv2d(32,1,3,1,1),
        )
    def forward(self,x):
        x=self.layer1(x)
        x=self.layer2(x)
        x=self.layer3(x)
        x=F.interpolate(x,scale_factor=2,mode='nearest')
        x=self.pad1(x)
        x=F.interpolate(x,scale_factor=2,mode='nearest')
        x=self.pad2(x)
        x=F.interpolate(x,scale_factor=2,mode='nearest')
        x=self.pad3(x)
        x=torch.sigmoid(x)

        return x
def test():
        print('---------------------start------------------------') 
        ICNet=PostProcess()
        ICNet.cuda()
        map_location = lambda storage, loc: storage	
        ICNet.load_state_dict(torch.load(opt.load_model, map_location=map_location),False)
        dataset = MyDataset()
        dataloader = torch.utils.data.DataLoader(dataset,
					batch_size=1,
					shuffle=False,
					num_workers=1,
					drop_last=False
					)
     
        for ii, (image, labels) in enumerate(dataloader):
                # network output 
                
                label_name = labels[0]
                save_path = os.path.join(opt.save_path,re.split(r'[/]',label_name)[-2])
                if (os.path.exists(save_path)==False):
                    os.makedirs(save_path)

                inputs = Variable(image).cuda()
                output = ICNet(inputs)
                output_numpy = output.data[0,0,:,:].cpu().float().numpy()
                output_numpy = output_numpy * 255.0
                output_numpy = output_numpy.astype(np.uint8)
                output_PIL = Image.fromarray(output_numpy, mode='L')
                output_PIL.save('%s/%s'%(opt.save_path,label_name))

        print('--------------------complete!-----------------------')  

def test_acc():
        print('---------------------start------------------------') 
        x=torch.ones(1,5,896,896).cuda()
        ICNet=PostProcess()
        ICNet.cuda()
        map_location = lambda storage, loc: storage	
        ICNet.load_state_dict(torch.load(opt.load_model, map_location=map_location),False)
        model_trt = torch2trt(ICNet,[x])
        dataset = MyDataset()
        dataloader = torch.utils.data.DataLoader(dataset,
					batch_size=1,
					shuffle=False,
					num_workers=1,
					drop_last=False
					)
     
        for ii, (image, labels) in enumerate(dataloader):
                # network output 
                count += 1
                label_name = labels[0]
                save_path = os.path.join('res_result',re.split(r'[/]',label_name)[-2])
                if (os.path.exists(save_path)==False):
                    os.makedirs(save_path)

                inputs = Variable(image).cuda()
                output = model_trt(inputs)
                output_numpy = output.data[0,0,:,:].cpu().float().numpy()
                output_numpy = output_numpy * 255.0
                output_numpy = output_numpy.astype(np.uint8)
                output_PIL = Image.fromarray(output_numpy, mode='L')
                output_PIL.save('%s/%s'%(opt.save_path,label_name))

        print('--------------------complete!-----------------------')                                     

if __name__ == "__main__":
    opt=BaseOptions().parse()
    if opt.acc==True:
        test_acc()
    else:
        test()
