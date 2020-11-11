import os
import cv2
import numpy as np
path = os.walk("label/")
for root,dirs,files in path:
    for filename in files:
        I=cv2.imread(os.path.join(root,filename),0)
        I=np.where(I==0,0,1)
        cv2.imwrite(os.path.join(root,filename),I)
        print('saved to'+filename)

  
