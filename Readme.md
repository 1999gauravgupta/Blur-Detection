<<<<<<< HEAD
# Image-Blur-Detection
Classification of Blurred and Non-Blurred Images  

**CERTH Image Blur Dataset**


> E. Mavridaki, V. Mezaris, "No-Reference blur assessment in natural images using Fourier transform and spatial pyramids", Proc. IEEE International Conference on Image Processing (ICIP 2014), Paris, France, October 2014.


The dataset consists of undistorted, naturally-blurred and artificially-blurred images for image quality
assessment purposes.
Download the dataset from here:
http://mklab.iti.gr/files/imageblur/CERTH_ImageBlurDataset.zip

Unzip and load the files into a directory named **CERTH_ImageBlurDataset**



---

## Variation of the Laplacian
**Using OpenCV2**

This method yielded an accuracy of **87.29%**

`cv2.Laplacian(img, cv2.CV_64F).var()`

The Laplacian Operator is applied on each of the images. 
The variation of the result is calculated.
If the variation is below the threshold, 400 in this case, the image is classified as blurry.
Otherwise, it is classified as non-blurry.


To run this model:

`python OpenCV_var.py`

---

## Convolutional Neural Network

To load and pickle train data and its labels:

`python load_traindata.py`

To load and pickle test data and its labels:

`python load_testdata.py`

A Convolutional Neural Network is trained which yields an accuracy of **67.70%** on the evaluation dataset.
The deep learning model has five layers.
This accuracy can further be improved by increasing the input dimensions of the first layer in the model and the number of epochs.
However, due to constraints related to computational power, I was unable to run the model.

To train the CNN:

`python CNN.py`



---


## Maximum of Laplacian
**Using OpenCV2**

An accuracy of **63.72%** is achieved using this method.

`gray = cv2.resize(cv2.imread(imagepath, cv2.IMREAD_GRAYSCALE), input_size)`

`numpy.max(cv2.convertScaleAbs(cv2.Laplacian(gray,3)))`


The result of the Laplacian Operator is converted to an absolute scale (0-255).
The max of the values is taken for each image.
The threshold is set at 215. Values lower than 215, classify the image as out-of-focus or blurry.
Greater than 215 is classified as non-blurred.

To run this model:

`python OpenCV_max.py` 



=======
 ## Problem Statement
Image quality detection has always been an arduous problem to solve in Computer Vision. Possible applications of blur detection include:
- Automatic image quality grading
- Helping professional photographers sort through 100s to 1000s of photos during a photo shoot by automatically discarding the blurry/low quality ones
- Applying OCR to real-time video streams, but only applying the expensive OCR computation to non-blurry frames

Thus it is an important task to accomplish.

## Libraries Required
- numpy ``pip install numpy``
- mattplotlib ``pip install matplotlib``
- tensorflow ``pip install tensorflow``
- keras ``pip install keras``
- PIL ``pip install Pillow``

## Theory
### CNN Classifier
It is a simple Convolutional Neural Network to implement binary classification that I used to train the model on the given dataset. The accuracy of the model came out to be about 80% because of small size of dataset.

### Kernal Convolution and SVM
I tinkered around with various kernels of edge detection because blur can be inferred from them. To explain, if there is sharp variance in color we can say that there is an edge discovered but for blurry images the variance will be low since there will be not much sharp images present. Therefore we can just use variance as a distiction parameter for basic model. I used Sobel and Laplacian kernel and came to conclusion that Laplacian kernel gave better results. Then to set a threshold I used SVM to find the best fitting line which will be most suitable to set the threshold value. Any image over that threshold will be considered sharp while below ones will be marked as blurry.
>>>>>>> 9705450f0f037723d6f0181971fd50b35adc9243
