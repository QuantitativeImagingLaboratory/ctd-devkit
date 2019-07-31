# UHCTD-devkit

**UHCTD:** University of Houston - Camera Tampering Detection Development Kit

### Camera Tampering Detection
An unauthorized or an accidental change in the view of a surveillance camera is called a tampering. This can be the
result of a natural phenomenon or it can be induced to accomplish malicious activities. 

Most literature has classified camera tampering as **covered, defocussed, and moved**. Covered tampering occurs when the view of the camera is
blocked. Spray painting the lens, blocking it with hand, and accumulation of dust, spider webs, and droplets are exam-
ples of covered tampering. Defocussed tampering occurs when the view of the camera is blurred. Formation of fog
on lens, failures to focus, and intentionally changing the focus are examples of defocussed tampering. Moved tamper-
ing occurs when the view point of the camera has changed. This can occur as a result of strong wind, an intentionally
change in the direction of the lens with malicious intent.

UHCTD is a large scale synthetic dataset for camera tampering detection. The dataset consists of *576* tampers with over
*288* hours (*12* days, including testing and training data) of video captured from two surveillance cameras (6 days per
camera). We define four classes 

1. Normal
2. Covered
3. Defocussed
4. Moved

The dataset is available for [download](qil.uh.edu/datasets) here.

The dataset consist of images and vidoes from 2 camera (A and B). Each camera has three folder
1. Training images: Contians a unifrom distribution of normal, coverd, defocussed, and moved image intended for training
2. Training video: Contians two videos each 24 hours long intended for training 
3. Testing video: Contains four video each 24 hours long intended for testing
 
The training code, and prediciton code has been adopted from [buptchan/scene-classification](https://github.com/buptchan/scene-classification)

The code can be used to train and evaluate 4 architectures.        

1. Alexnet
2. Densenet161
3. Resnet18
4. Resnet50

These architectures are similar to the one used in [scene classififcation](https://github.com/CSAILVision/places365). However,
we replace the last layer with a fully connected layer that has four neurons, one for each class. 

---
### Training and Prediction Instructions
Requirements:
The code has been tested on the following enviroment.
- Python 3.5.2
- PyTorch 1.1.0
- Tensorflow 1.8.0

Please see requirements file

###### Training
We train the four architectures starting with pretrained weights that are trained on the [places365](https://github.com/CSAILVision/places365) dataset.
The pre-trained weights for the pytorch models can be downloaded from links below.

1. Create Training data: The training folder must contain two folders one for trianing and evaluation. 
Each folder has four folders one for each class of images. We use the following labels 0 - normal, 1 - covered, 2 - defocussed, and 3 - moved
The training folder structure is
```
.
|-- train
    |-- 0
    |-- 1
    |-- 2
    |-- 3
|-- eval
    |-- 0
    |-- 1
    |-- 2
    |-- 3
```
create_training_data.py has a script to create the dataset from the the training image folder in the dataset.
``` 
python   create_taining_data.py  
            --input    <training image folder>     
            --output    <destination folder>
```
     
2. Download pre-trained weights and copy them in to the folder models/pretrained 
    1. AlexNet - [link](http://places2.csail.mit.edu/models_places365/alexnet_places365.pth.tar)
    2. ResNet18 - [link](http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar)
    3. ResNet50 - [link](http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar)
    4. DenseNet161 - [link](http://places2.csail.mit.edu/models_places365/densenet161_places365.pth.tar)

3. Run using the train.py file
```
python  train.py 
        --train_data  <directory with train and eval folders> 
        --model_name  <modelname (alexnet, resenet18, resnet50, densenet161)>
```
The checkpoints and trained models will be saved to train folder
    

###### Prediction
Predict the labels on a video using the following command
```
python  predict.py 
        --weights   <weights files>
        --video_file    <video file>
```
The prediction are store in output.csv file