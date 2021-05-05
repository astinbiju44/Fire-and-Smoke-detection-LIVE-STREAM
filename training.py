from tensorflow.keras.preprocessing.image import ImageDataGenerator,img_to_array
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical,plot_model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import BatchNormalization,Conv2D,MaxPooling2D,Activation,Flatten,Dropout,Dense
from tensorflow.keras import backend as k
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
import cv2
import random
import os
import glob





#To append images and its labels
data=[]
labels=[]




#To load image files from the dataset
imagefiles=[f for f in glob.glob(r'FIRE-SMOKE-DATASET'+"/**/*",recursive=True) if not os.path.isdir(f)]
random.shuffle(imagefiles) #to balance the weight
print("Total number image readed : "+str(len(imagefiles)))



height,width,channels=96,96,3



#converting images to array and labeling into categories
for img in imagefiles:
    #print(img)
    image=cv2.imread(img)
    image=cv2.resize(image,(96,96))
    image=img_to_array(image)
    data.append(image)

    label=img.split(os.path.sep)[-2]  #every seperation is made whereever there is backslash and second last is the class name ie -2
    if label=="Neutral":
        label=0
    elif label=="Fire":
        label=1
    else:
        label=2
    #print(label)
    labels.append([label]) #[[0],[2],[1],......]



#data preprocessing
data=np.array(data,dtype="float")/255.0  #it will convert into array of values between 0 & 1
labels=np.array(labels)










