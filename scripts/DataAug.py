
from torchvision import transforms
from PIL import Image,ImageStat,ImageEnhance
import cv2
import numpy
import re
import os
IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)

def make_dataset(dir):
    images = []

    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, dirs, fnames in sorted(os.walk(dir)):
        for dirr in dirs: 
            if not os.path.exists('train/flip_'+dirr):
                os.makedirs('train/flip_'+dirr)
            if not os.path.exists('label/flip_'+dirr):
                os.makedirs('label/flip_'+dirr)
            for fname in sorted(os.listdir(dir+'/'+dirr)):
                if is_image_file(fname):
                        path = os.path.join(dir+'/'+dirr+'/', fname)
                        
                        image=Image.open(path).convert('L')
                        label=Image.open('relabel/'+dirr+'/'+fname).convert('L')
                        image=transforms.RandomHorizontalFlip(p=1)(image)
                        lable=transforms.RandomHorizontalFlip(p=1)(label)
                        image.save('train/flip_'+dirr+'/'+fname)
                        label.save('label/flip_'+dirr+'/'+fname)   
if __name__=="__main__":
    make_dataset('train/')
