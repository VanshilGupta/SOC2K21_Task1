# SOC WNCC IITB  
## CheckPoint 1

***

### Problem :
Given a dataset of dogs and cats images, we need to implement a CNN model which could classify the dogs and cats images.

### Implemented in :  
Jupyter Notebook

### Frameworks used :  
* TensorFlow
* Numpy
* OpenCV
* Matplotlib

### Data preprocessing  
Dataset contains 1500 images of each category (dog and cat). It is splitted as 2/3 train and 1/3 validation that is train set has 1000 images and validation set has 500 images pf each category. Each image is read in grayscale mode and the pixel values are normalized between 0 and 1.

### Architecture 
Implemented three different models with some changes in the convolutional layers which is flattened out and connected to a hidden layer of size 128 and give the output in the final layer of size 2. 

NETWORK PARAMETER 
* Rectifier Linear Unit
* Adam optimzer
* Binary CrossEntropy loss
* Softmax on final output

### Results

| Model No. 	| Architecture (Number and Size of Filters) 	| Epochs 	| Acc 	| Loss 	| Val_Acc 	| Val_Loss 	| image1 	| image2 	| image3 	|
|-	|-	|-	|-	|-	|-	|-	|-	|-	|-	|
| 1  	| 8,3<br>32,3<br>64,3 	| 10 	| 0.9376 	| 0.1603 	| 0.69 	| 1.1424 	|  	|  	|  	|
| 2 (best)	| 32,3<br>64,3<br>128,3<br>256,3 	| 20 	| 0.9787 	| 0.0694 	| 0.75 	| 1.1060 	| 0 	| 1 	| 1 	|
| 3 	| 8,2<br>32,2<br>64,2<br>128,2 	| 20 	| 0.9526 	| 0.1256 	| 0.71 	| 1.5257 	|  	|  	|  	|

* Test 1 :  

![IMAGE1](resources/Prediction1.PNG)


* Test2 :  

![IMAGE2](resources/Prediction2.PNG)


* Test3 :  

![IMAGE3](resources/Prediction3.PNG)
