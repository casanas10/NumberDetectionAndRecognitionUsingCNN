# NumberDetectionAndRecognitionUsingCNN
> Computer vision project used to recognize and detect street house numbers in videos and images.

## Purpose
This was the final project for my Computer Vision class. The goal was to use the http://ufldl.stanford.edu/housenumbers/ to 
1) find the numbers in the image or video
2) Draw a box around it 
3) Then recognize which digits appear in the image/video

You can find my report here. https://github.com/casanas10/NumberDetectionAndRecognitionUsingCNN/blob/master/cv_proj_report.pdf

## TechStack
* Python 
* Computer Vision
* OpenCV
* Deep Learning using Keras
* Convolutional Neural Network(CNN)
* Numpy
* Matplotlib

## Outcome
* learned how to create deep neural networks using Keras
* implemented a sliding window approach to detecting street numbers
* created several CNNs and compared their results
* compared different plots 


## Running File
- first download weights for the CNN models https://drive.google.com/open?id=1-CzrTm7OJpY_WdYyc_SaltTnv5L4Bp9r

python run_project.py

If you want to input own image, go the run_project.py file and scroll to bottom to see main function. 
Check that the process_images is not commented and put your images in the input_images/TA_Testing_Folder.
Then run the script just as before.

If you would like to run a video. Uncomment the video processing funtion. 
Put the video inside the input_videos directory and update name of the video in the run_project file.
Run the script again. 

## Directories:
graded_images folder has the 5 images for the report

TA_Testing_Folder used to input the images you want to analyze

input_videos folder used for input the TA will like to test

output folder is where video will be after processing

weights folder contains all the weights of each model

## Links:
Video link: https://youtu.be/4h1WYyUv6G8

