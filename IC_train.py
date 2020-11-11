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

#define argument
class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False
    def initialize(self):  
        self.parser.add_argument('--data_path',default='train',help="path to store training data")
        self.parser.add_argument('--label_path',default='label',help="path to store label")
        self.parser.add_argument('--save_path',type=str,default='IC_checkpoints',help='path to save checkpoints')
        self.parser.add_argument('--num_workers',default=0,type=int)
        self.parser.add_argument('--batch_size',type=int,default=16)
        self.parser.add_argument('--lr',type=float,default=0.001)
        self.parser.add_argument('--max_epoch',type=int,default=200)
        self.parser.add_argument('--image_size',type=int,default=896,help="the size to resize")
        self.parser.add_argument('--vis',default=True,help="whether to use visdom")
        self.parser.add_argument('--gpus',action='store_true',default=True,help='whether to use gpu')
        self.parser.add_argument('--load_model',type=str,help="path to load pre-trained model")
        self.parser.add_argument('--local_rank',type=int)
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

        return x

# visulize
class Visualizer():
        def __init__(self, opt):
                self.display_id = 1
                self.win_size = 256
                self.name = 'detection loss'
                if self.display_id:
                        import visdom
                        self.vis = visdom.Visdom(env='main', port=8097)
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

def train():
    dist.init_process_group(backend='nccl', init_method='env://')
    device = torch.device('cuda', opt.local_rank)
    ICNet=PostProcess()
    ICNet = nn.SyncBatchNorm.convert_sync_batchnorm(ICNet)
    criterion=torch.nn.BCEWithLogitsLoss()
    optimizer=torch.optim.Adam(ICNet.parameters(),opt.lr,betas=(0.9, 0.999))
    if opt.vis:
        vis=Visualizer(opt)
    if opt.gpus:
        ICNet=ICNet.to(device)
        criterion=criterion.to(device)
    ICNet = torch.nn.parallel.DistributedDataParallel(ICNet, device_ids=[opt.local_rank], output_device=opt.local_rank,find_unused_parameters=True)
    dataset = MyDataset()
    train_sampler=torch.utils.data.distributed.DistributedSampler(dataset)
    dataloader=torch.utils.data.DataLoader(dataset,
                                          batch_size=opt.batch_size,
                                          shuffle=(train_sampler is None),
                                          num_workers=opt.num_workers,
                                          drop_last=True,
                                          pin_memory=True
                                          )
    for epoch in range(1,opt.max_epoch+1):
        epoch_iter=0
        for ii,(images,labels) in enumerate(dataloader):
            iter_start_time=time.time()
            epoch_iter+=opt.batch_size
            inputs=Variable(images)
            optimizer.zero_grad()
            outputs=ICNet(inputs.to(device))
            target=np.zeros((opt.batch_size,1,896,896))
            for l in range(0,opt.batch_size):
                label=Image.open('%s/%s'%(opt.label_path,labels[l])).convert('L')
                label=tv.transforms.Resize((opt.image_size,opt.image_size))(label)
                target[l,0,:,:]=label     
            target=torch.Tensor(target)
            pre_outputs=Variable(target).to(device)
            loss = criterion(outputs, pre_outputs)
            loss.backward()  
            optimizer.step()             
            errors = get_current_errors(loss)
            if opt.local_rank==0:
                if (ii+1)% 100  == 0:	
                    ti = (time.time() - iter_start_time) / opt.batch_size
                    print_current_errors(epoch, epoch_iter, errors, ti)		
                if opt.vis and (ii+1)% 100 == 0:	
                    with open('ICmodule_loss.txt','a') as f:
                        vdl = 'epoch:%d ICmodule_loss:%.10f'%(epoch, loss)
                        f.write(vdl + '\n')
                        f.close()
                    load_rate = float(epoch_iter)/dataset.__len__()
                    vis.plot_current_errors(epoch, load_rate, opt, errors)	
        if epoch % 1 ==0 and opt.local_rank==0:
            torch.save(ICNet.module.state_dict(), './%s/%s.pth'%(opt.save_path,str(epoch)))
    print('complete!')

def get_current_errors(loss):
	return OrderedDict([('ResnetLoss', loss.data)])                                  

if __name__ == "__main__":
    opt=BaseOptions().parse()
    train()
