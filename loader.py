'''
  Author       : Bao Jiarong
  Creation Date: 2020-06-13
  email        : bao.salirong@gmail.com
  Task         : VGG11, VGG13, VGG16, VGG19 Implementation
  Dataset      : MNIST Digits (0,1,...,9)
'''
import os
import sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import pandas as pd
import tensorflow as tf
import random
import cv2

def scaling_tech(x,method = "normalization"):
    if method == "normalization":
        x = (x-x.min())/(x.max()-x.min()+0.0001)
    else:
        x = (x-np.std(x))/x.mean()
    return x

def load_data(dir,width,height,shuffle=True, split_ratio = 0.8, augment_data = False):
    # Step 1: Load Data...............
    subdirs = os.listdir(dir)
    subdirs = sorted(subdirs)

    labels = []
    images = []
    img_ext = [".jpg",".png",".jpeg"]

    print("Loading data")
    n = len(subdirs)
    for i in range(n):
        filenames = os.listdir(dir+subdirs[i])
        print("class",subdirs[i],"contains",len(filenames),"images")
        for j in range(len(filenames)):
            ext = (os.path.splitext(filenames[j]))[1]
            if ext in img_ext:
                # img_filename = dir + subdirs[i]+"/" + filenames[j]
                img_filename = os.path.join(dir, subdirs[i], filenames[j])
                img   = cv2.imread(img_filename)
                image = cv2.resize(img,(height,width),interpolation = cv2.INTER_AREA)/255.0

                images.append(image)
                labels.append(i)

                # data_augmentation
                if augment_data == True:
                    img_v = image[:,::-1]
                    # img_h = image[::-1,:]
                    # img_h_v = image[::-1,::-1]
                    images.append(img_v)
                    labels.append(i)

    images = np.array(images)
    labels = np.array(labels)

    # Step 2: Normalize Data..........
    # images = scaling_tech(images,method="normalization")

    # Step 4: Shuffle Data............
    if shuffle == True:
        print("Shuffling data")
        indics = np.arange(0,len(images))
        np.random.shuffle(indics)

        labels = labels[indics]
        images = images[indics]

    # Step 5: Split Data.............
    print("Splitting data")
    m = int(len(images)*split_ratio)

    return images[:m], labels[:m], images[m:], labels[m:]
