import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from tensorflow.keras.layers import Input, Conv2D, MaxPool2D, Dropout, Conv2DTranspose
from skimage.io import imread, imshow
from skimage.transform import resize


TRAIN_PATH = "NuclieDataset/stage1_train"
TEST_PATH = "NuclieDataset/stage1_test"

# number of training images
n = len(os.listdir(TRAIN_PATH))

TRAIN_IMAGES_DIR = os.listdir(TRAIN_PATH)

IMG_WIDTH = 128
IMG_HEIGHT = 128
IMG_CHANNELS = 3
X_train = np.zeros((n, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                   dtype=np.uint8)
Y_train = np.zeros((n, IMG_HEIGHT, IMG_WIDTH, 1),
                   dtype=np.bool)

#img_path = TRAIN_PATH + "/" + TRAIN_IMAGES_DIR[0] + "/images/"
#img = imread(img_path + "/" + img_name)
for i in tqdm(range(n)):
   img_path = TRAIN_PATH + "/" + TRAIN_IMAGES_DIR[i] + "/images/"
   img_name = os.listdir(img_path)[0]
   img = imread(img_path + "/" + img_name)[:,:,:IMG_CHANNELS]
   img = resize(img, (IMG_HEIGHT, IMG_WIDTH),
                mode="constant", preserve_range=True)
   X_train[i] = img
   mask_path = TRAIN_PATH + "/" + TRAIN_IMAGES_DIR[i] + "/masks/"
   mask_n = os.listdir(mask_path)
   mask = np.zeros([IMG_HEIGHT, IMG_WIDTH, 1], dtype=np.bool)
   for j in range(len(mask_n)):
       mask_img = imread(mask_path + "/" + mask_n[j])
       mask_img = np.expand_dims(resize(mask_img, 
                                        (IMG_HEIGHT, IMG_WIDTH), 
                                        mode="constant",
                                        preserve_range=True), 
                                 axis=-1)
       mask = np.maximum(mask, mask_img)
   Y_train[i] = mask



imshow(X_train[1])
imshow(Y_train[1])


# number of testing images
test_n = len(os.listdir(TEST_PATH))
TEST_IMAGES_DIR = os.listdir(TEST_PATH)
X_test = np.zeros((test_n, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
                   dtype=np.uint8)

for i in tqdm(range(test_n)):
   img_path = TEST_PATH + "/" + TEST_IMAGES_DIR[i] + "/images/"
   img_name = os.listdir(img_path)[0]
   img = imread(img_path + "/" + img_name)[:,:,:IMG_CHANNELS]
   img = resize(img, (IMG_HEIGHT, IMG_WIDTH),
                mode="constant", preserve_range=True)
   X_test[i] = img
   






















