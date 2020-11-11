

# Infrared Small and Dim Target Detection

*Our implements contain two parts ： SDDNet and IC-Module.*  

*SDDNet: Small and Dim Object Detection. Using segmentation pictures as the labels.*  

*IC-Module: Inter-frame correlation Module. Using the inter-frame correlation information aim to reduce the false alarm rate.*  

## Environments
pytorch==1.3.0 

torchvision==0.5.0

python==3.7

visdom

torch2trt (refer to ：https://github.com/NVIDIA-AI-IOT/torch2trt)  

## Inference  
To run inference, you can type the following commands:  
```python
python SDD_test.py  --data_path (image to inference)  --load_model  (trained model)   
python IC_test.py  --data_path (image to inference)  --load_model  (trained model)    
```
And the result will be saved in ```test_result/SDD``` or ```test_result/IC``` folder if not specified.  
**More parameters:**      
　　--save_path : path to save result.  
　　--acc : if True, use tensorRT to accelerate inference.  
## Training
DDP mode is adopted in both SDDNet and IC-Module, and 4 Gpus are used for training.

To run training scripts, you can type the following commands:  

```python
python -m torch.distributed.launch --nproc_per_node 4 SDD_train.py --data_path  (training dataset path)  --label_path (label path) & nohup visdom    
python -m torch.distributed.launch --nproc_per_node 4 IC_train.py --data_path  (training dataset path)  --label_path (label path) & nohup visdom  
```
And the trained model will be saved in sdd_checkpoints or ic_checkpoints folder if not specified.  

**More Parameters:**  
　　--save_path : path to save checkpoints.  
　　--vis : whether to visualize, default True.　  
　　--load_model : path to load pre-trained model.  
## Others（in scripts folder）
1. If you want to train the model with your own custom dataset, you should prepare binarized segmentation as the label. We provide the script ```binary.py``` to convert images to binarized images.  

2. We provide a script ```DataAug.py ``` to enhance prepared dataset before training, the default folders where storing the training data and labels are ‘train’ and ‘label’ respectively.  

3. We provide a function to binary result images, and you can call the ```bwfunction``` to use it.  

4. We also provide scripts to calculate PD & FA of our inference result. And the same statistical method was used in the comparative experiment.  
```python
Command : python statistic.py --image_path (path saved result) --label_path (path saved ground truth) --width 896 --height 896
```

