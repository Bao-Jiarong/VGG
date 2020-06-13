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
import loader
import vgg

np.random.seed(7)
tf.random.set_seed(7)
# np.set_printoptions(threshold=np.inf)

# Input/Ouptut Parameters
width      = 224 >> 2
height     = 224 >> 2
channel    = 3
n_outputs  = 10
model_name = "models/vgg19/digists"
data_path  = "../data_img/MNIST/train/"

# Step 0: Global Parameters
epochs     = 2
lr_rate    = 0.0001
batch_size = 32

# Step 1: Create Model
# model = vgg.VGG11((height, width, channel), classes = n_outputs, filters = 8)
# model = vgg.VGG13((height, width, channel), classes = n_outputs, filters = 8)
# model = vgg.VGG16((height, width, channel), classes = n_outputs, filters = 8)
model = vgg.VGG19((height, width, channel), classes = n_outputs, filters = 8)

# Step 2: Define Metrics
model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate = lr_rate),
              loss     = tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics  = ['accuracy'])
print(model.summary())

if sys.argv[1] == "train":
    # Step 3: Load data
    X_train, Y_train, X_test, Y_test = loader.load_data(data_path,width,height,True,0.8,False)
    # Step 4: Training
    # Create a function that saves the model's weights
    cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath = model_name,
                                                     save_weights_only=True,
                                                     verbose=0, save_freq="epoch")
    # model.load_weights(model_name)
    model.fit(X_train, Y_train,
              batch_size     = batch_size,
              epochs         = epochs,
              validation_data= (X_test,Y_test),
              callbacks      = [cp_callback])

    # Step 6: Evaluation
    loss,acc = model.evaluate(X_test, Y_test, verbose = 2)
    print("Evaluation, accuracy: {:5.2f}%".format(100 * acc))

elif sys.argv[1] == "predict":
    # Step 3: Loads the weights
    model.load_weights(model_name)
    my_model = tf.keras.Sequential([model])

    # Step 4: Prepare the input
    img = cv2.imread(sys.argv[2])
    image = cv2.resize(img,(height,width),interpolation = cv2.INTER_AREA)
    images = np.array([image])
    images = loader.scaling_tech(images,method="normalization")

    # Step 5: Predict the class
    preds = my_model.predict(images)
    print(np.argmax(preds[0]))
    print(preds[0])
