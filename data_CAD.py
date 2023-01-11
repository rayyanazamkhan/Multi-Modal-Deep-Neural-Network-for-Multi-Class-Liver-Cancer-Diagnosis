# -*- coding: utf-8 -*-
"""
Created on Tue May 14 16:57:06 2019

@author: rayyan
"""
import random
import re
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import numpy as np
import pandas as pd
import cv2 
import os
import glob
#import skimage.io as io
#import skimage.transform as trans
from PIL import Image
import natsort
import tensorflow as tf
from keras import backend as K
#import skimage as sk
#import skimage.feature.texture as skft
#from skimage import filters
from scipy import ndimage,stats
#from radiomics import featureextractor
#import SimpleITK as sitk
#import six
#import logging
#from sklearn.preprocessing import StandardScaler
#from sklearn.decomposition import KernelPCA,PCA
############################################################
##################################Main File####################################

Training = True                                   #### True/ False
Weights = True                                   #### True/ False
binary = True                                     #### True/ False
prior_weights = False
Raw_input = (512,512)
im_size = (128,128)                        #### Image size to process
path_arr_sz = (1,5) 
num_channel = 1
No_of_test_images = 821

####################
BATCH_SIZE = 5
NO_OF_TRAINING_IMAGES = 22565#821
NO_OF_VAL_IMAGES = 22565#821
No_of_epochs = 500
Num_of_train_batches = int(NO_OF_TRAINING_IMAGES/BATCH_SIZE)
Num_of_val_batches = int(NO_OF_VAL_IMAGES/BATCH_SIZE)
############################################################

####################### Pathology Features
def pathology_features(file_names,size):
    dataset = pd.read_csv('Book1.csv')
    S = dataset.iloc[:, 0].values
    X = dataset.iloc[:, 0:].values
    A=[]
    for j in range(size):
        file_name=file_names[j]
        for i in range(S.shape[0]):
            #if (file_name.find(S[i])!= -1):
            if re.search(S[i]+r"\b", file_name):
                path_data = X[i,1:]
                path_data = np.reshape(path_data,(1,)+path_data.shape)
                A.append(path_data)
    OUT=np.array(A)
    OUT *= 255.0/OUT.max()
    OUT=OUT/255
    return (OUT)


############################# Image PreProcessing
def adjustData(img,**kwargs):
    try:
        im = img[img.any(1),:]
        im = im[:,im.any(0)]                                               #Remove background except liver part
        img = Image.fromarray(im)
        img = img.resize(im_size,Image.ANTIALIAS)
        im1 = np.array(img, dtype=np.float64)
    except:
        img = Image.fromarray(img)
        img = img.resize(im_size,Image.ANTIALIAS)
        im1 = np.array(img, dtype=np.float64)
    
    im = np.reshape(im1,im1.shape+(1,))
    return im
######################## 
def get_indices_from_keras_generator(gen, batch_size):
    """
    Given a keras data generator, it returns the indices and the filepaths
    corresponding the current batch. 
    :param gen: keras generator.
    :param batch_size: size of the last batch generated.
    :return: tuple with indices and filenames
    """
    idx_left = (gen.batch_index - 1)* batch_size
    idx_right = idx_left + gen.batch_size if idx_left >= 0 else None
    indices = gen.index_array[idx_left:idx_right]
    file_names = [gen.filenames[i] for i in indices]
    return (indices, file_names)

###############################################################################
train_datagen = ImageDataGenerator(rescale = 1./255,
                                   featurewise_center = False, 
                                   samplewise_center=False,
                                   featurewise_std_normalization = False,
                                   samplewise_std_normalization=False,
                                   rotation_range = 5,
                                   width_shift_range = 0.1,
                                   height_shift_range = 0.1,
                                   shear_range = 0.1,
                                   zoom_range = 0.2,
                                   horizontal_flip = True,
                                   vertical_flip = True,
                                   fill_mode = 'nearest',
                                   preprocessing_function = None,
                                   zca_whitening=False)

val_datagen = ImageDataGenerator(rescale=1./255, 
                                featurewise_center = False,
                                featurewise_std_normalization = False,
                                preprocessing_function = None)

def trainGenerator(image_path,size,batch,classes):
    train_image_generator = train_datagen.flow_from_directory(image_path,
                                                              batch_size = batch,
                                                              class_mode= classes,
                                                              color_mode= 'grayscale',
                                                              shuffle = True,
                                                              seed = random.randint(1,100),
                                                              target_size=size)
    for (img_batch,label_batch) in train_image_generator:
        X1_train_batch = []
        for i in range(img_batch.shape[0]):
            im = img_batch[i,:,:,0]
            img = adjustData(im)
            X1_train_batch.append(img)
        if binary:
            OUT = np.array(X1_train_batch)
        else:
            indices, file_names = get_indices_from_keras_generator(train_image_generator, img_batch.shape[0])
            features_patho = pathology_features(file_names,img_batch.shape[0])
            OUT = [np.array(X1_train_batch),features_patho]
        yield (OUT,label_batch)
def train_generator(image_path,size,batch,classes):
    train_generator = val_datagen.flow_from_directory(image_path,
                                                    batch_size = batch,
                                                    class_mode= classes,
                                                    color_mode= 'grayscale',
                                                    shuffle = False,
                                                    seed = 1,
                                                    target_size=size)
    return train_generator
def valGenerator(image_path,size,batch,classes):
    val_image_generator = val_datagen.flow_from_directory(image_path,
                                                          batch_size = batch,
                                                          class_mode= classes,
                                                          color_mode= 'grayscale',
                                                          shuffle = False,
                                                          seed = 1,
                                                          target_size=size)
    for (img_batch,label_batch) in val_image_generator:
        X1_train_batch = []
        for i in range(img_batch.shape[0]):
            im = img_batch[i,:,:,0]
            img = adjustData(im)
            X1_train_batch.append(img)
        if binary:
            OUT = np.array(X1_train_batch)
        else:
            indices, file_names = get_indices_from_keras_generator(val_image_generator, img_batch.shape[0])
            features_patho = pathology_features(file_names,img_batch.shape[0]) 
            OUT = [np.array(X1_train_batch),features_patho]
        yield (OUT,label_batch)
def testGenerator(image_path,size,batch,classes):
    test_image_generator = val_datagen.flow_from_directory(image_path,
                                                           batch_size = batch,
                                                           class_mode= classes,
                                                           color_mode= 'grayscale',
                                                           shuffle = False,
                                                           seed = None,
                                                           target_size=size)
    for img_batch in test_image_generator:
        X1_train_batch = []
        for i in range(img_batch.shape[0]):
            im = img_batch[i,:,:,0]
            img = adjustData(im)
            X1_train_batch.append(img)
        if binary:
            OUT = np.array(X1_train_batch)
        else:
            indices, file_names = get_indices_from_keras_generator(test_image_generator, img_batch.shape[0])
            features_patho = pathology_features(file_names,img_batch.shape[0])
            OUT = [np.array(X1_train_batch),features_patho]
        yield (OUT)
    
