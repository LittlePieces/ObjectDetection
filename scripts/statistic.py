from PIL import Image
import numpy as np
import cv2
import os
import argparse

def ROC(src1,src2):
    img = cv2.imread(src1,0)
    lab = cv2.imread(src2,0)
    lab = cv2.resize(lab,(896,896))
    _,_,status,_ = cv2.connectedComponentsWithStats(img)
    num2,_,status2,_ = cv2.connectedComponentsWithStats(lab)
    ground_truth = num2-1
    detected_truth = 0
    alarm_pixel = 0

    for i in range(1,len(status)): 
        x=status[i][0]
        y=status[i][1]
        w=status[i][2]
        h=status[i][3]
        ROI=lab[y:y+h,x:x+w]

        if np.all(ROI == 0):
            alarm_pixel += status[i][4]

    for j in range(1,len(status2)):  
        x2=status2[j][0]
        y2=status2[j][1]
        w2=status2[j][2]
        h2=status2[j][3]
        ROI2=img[y2:y2+h2,x2:x2+w2]

        if np.all(ROI2 == 0) == False:
            detected_truth += 1

    return ground_truth,detected_truth,alarm_pixel

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_path",required='True',help='path to save predicetd images')
    parser.add_argument("--label_path",required='True',help='path to save label images ')
    parser.add_argument("--width",type=int,required='True',help='label size should be same as the predicted images,if not,then resize to this size. ')
    parser.add_argument("--height",type=int,required='True')
    args = parser.parse_args()

    all_GT=0
    all_PD=0
    all_FA=0
    for fname in os.listdir(args.image_path):    
        GT,pred,alarm=ROC(os.path.join(args.image_path,fname), os.path.join(args.label_path,fname))
        all_GT += GT
        all_PD += pred
        all_FA += alarm
    all_pixel = len(os.listdir(args.image_path))*args.width*args.height
    print('PD=',all_PD/all_GT,'FA=',all_FA/all_pixel)


    
    
