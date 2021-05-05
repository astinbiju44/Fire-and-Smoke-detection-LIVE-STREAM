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
print(len(imagefiles))








