# -*- coding: utf-8 -*-
"""
Created on Tue Nov 16 17:58:28 2021

@author: rayya
"""
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import *
from keras.losses import *
from keras.layers import *
from tensorflow.keras.optimizers import *
#from keras.optimizers_v1 import Adam
#from keras.optimizers import Adam
from keras.regularizers import *
from keras.callbacks import *
from keras import backend as K
from keras.utils import *
from data_CAD import *
from keras.callbacks import *
from keras.utils import *
from sklearn.metrics import *
import itertools
##############################################################################
######### Binary Classification/ Categorical Classification ##################
if binary:
    class_type = 'categorical'#'binary'
    Activate = 'softmax'
    Weight_name = 'CAD_heatmap.h5'
    Output = 'CAD_heatmap.out'
    ################ Model Parameters##############
    # Latent_dim presents the number of distributions this is equal to number of desired output classes.  
    num_class = 2
    Train_image_path = '/home/rak038/CAD/Diagnosis/Binary_Class/Train/'
    Val_image_path = '/home/rak038/CAD/Diagnosis/Binary_Class/Val/'
    Test_image_path = '/home/rak038/CAD/Diagnosis/Test/'
    Test_save_path = '/home/rak038/CAD/Diagnosis/'
else:
    class_type = 'categorical'
    Activate = 'softmax'
    Weight_name = 'CAD_cat.h5'
    Output = 'CAD_cat.out'
    ################ Model Parameters##############
    # Latent_dim presents the number of distributions this is equal to number of desired output classes.  
    num_class = 3
    Train_image_path = '/home/rak038/CAD/Diagnosis/Categorical_Class/Train/'
    Val_image_path = '/home/rak038/CAD/Diagnosis/Categorical_Class/Val/'
    Test_image_path = '/home/rak038/CAD/Diagnosis/Test/'
    Test_save_path = '/home/rak038/CAD/Diagnosis/'

###########################################################
################# Image Input
im_input = Input(shape=(im_size[0],im_size[1],num_channel),name='hierarchical_features')

def RAYMODEL(inputs = im_input,**kwargs):
    conv1 = Conv2D(96, 5, activation = 'relu', padding = 'same', dilation_rate=(1,1), kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(96, 5, activation = 'relu', padding = 'same', dilation_rate=(2,2), kernel_initializer = 'he_normal')(conv1)
    conv1 = Conv2D(96, 5, activation = 'relu', padding = 'same', dilation_rate=(4,4), kernel_initializer = 'he_normal')(conv1)
    conv1 = (BatchNormalization())(conv1)

    ############################################################
    conv21 = Conv2D(256, 3, activation = 'relu', padding = 'same', dilation_rate=(2,2), kernel_initializer = 'he_normal')(conv1)
    conv22 = Conv2D(256, 3, activation = 'relu', padding = 'same', dilation_rate=(4,4), kernel_initializer = 'he_normal')(conv21)
    conv2 = Conv2D(256, 3, activation = 'relu', padding = 'same', dilation_rate=(8,8), kernel_initializer = 'he_normal')(conv22)
    conv2 = add([conv21,conv2])
    drop2 = (BatchNormalization())(conv2)
    #pool2 = MaxPooling2D(pool_size=(2, 2))(drop2)
    ############################################################
    #conv3 = ZeroPadding2D(padding=(1, 1))(pool2)
    conv31 = Conv2D(384, 3, activation = 'relu', padding = 'same', dilation_rate=(2,2), kernel_initializer = 'he_normal')(drop2)
    conv32 = Conv2D(384, 3, activation = 'relu', padding = 'same', dilation_rate=(4,4), kernel_initializer = 'he_normal')(conv31)
    conv3 = Conv2D(384, 3, activation = 'relu', padding = 'same', dilation_rate=(8,8), kernel_initializer = 'he_normal')(conv32)
    conv3 = add([conv31,conv3])
    drop3 = (BatchNormalization())(conv3)
    #pool3 = MaxPooling2D(pool_size=(2, 2))(drop3)
    ############################################################
    #conv4 = ZeroPadding2D(padding=(1, 1))(pool3)
    conv41 = Conv2D(384, 3, activation = 'relu', padding = 'same', dilation_rate=(1,1), kernel_initializer = 'he_normal')(drop3)
    conv42 = Conv2D(384, 3, activation = 'relu', padding = 'same', dilation_rate=(2,2), kernel_initializer = 'he_normal')(conv41)
    conv4 = Conv2D(384, 3, activation = 'relu', padding = 'same', dilation_rate=(4,4), kernel_initializer = 'he_normal')(conv42)
    conv4 = add([conv41,conv4])
    drop4 = (BatchNormalization())(conv4)
    #pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    ############################################################
    #conv5 = ZeroPadding2D(padding=(1, 1))(pool4)
    conv51 = Conv2D(256, 3, activation = 'relu', padding = 'same', dilation_rate=(1,1), kernel_initializer = 'he_normal')(drop4)
    conv52 = Conv2D(256, 3, activation = 'relu', padding = 'same', dilation_rate=(2,2), kernel_initializer = 'he_normal')(conv51)
    conv5 = Conv2D(256, 3, activation = 'relu', padding = 'same', dilation_rate=(4,4), kernel_initializer = 'he_normal')(conv52)
    conv5 = add([conv51,conv5])
    pool5 = (BatchNormalization(name='block5_drop'))(conv5)
    #pool5 = MaxPooling2D(pool_size=(2, 2),name='block5_pool')(drop5)
    ###############################################
    model = Model(inputs,pool5,name='RayModel')

    if prior_weights:
        model.load_weights('Bin0.h5')

    return model


MOD = RAYMODEL()
MOD.trainable = True
x = MOD.get_layer('block5_drop').output

features_im = Flatten()(x)

if binary:
    combine = features_im

else:
############################################################################################
##############################Pathology Features
############################################################################################
    pathology_features = Input(shape=(path_arr_sz),name='pathology_features')
################Wide and Deep Network
    wide = Flatten()(pathology_features)
    wide = BatchNormalization()(wide)
    
    flatten_layers = []
    K=8
    for c in range(path_arr_sz[1]):
        inputs_c = Lambda(lambda x:x[:,:,c])(pathology_features)
        embed_c = Embedding(path_arr_sz[1]+1,K,input_length=1,embeddings_initializer="uniform")(inputs_c)
        flatten_c = Flatten()(embed_c)
        flatten_layers.append(flatten_c)


    deep = concatenate(flatten_layers)

    hidden_units = [64, 32]
    for units in hidden_units:
        deep = Dense(units)(deep)
        deep = BatchNormalization()(deep)
        deep = Activation('relu')(deep)
        deep = Dropout(0.1)(deep)

    features_patho = concatenate([wide, deep])
############################################################

    combine = concatenate([features_im,features_patho])

############################################################    
#dense6 = Dense(4096)(combine)
#drop6 = (BatchNormalization())(dense6)
#conv6 = Activation('relu')(drop6)
#conv6 = Dropout(0.5)(conv6)
############################################################
#dense7 = Dense(4096)(conv6)
#drop7 = (BatchNormalization())(dense7)
#conv7 = Activation('relu')(drop7)
#drop7 = Dropout(0.5)(conv7)
############################################################
dense8 = Dense(num_class)(combine)
drop8 = (BatchNormalization())(dense8)
OUTPUT = Activation(Activate)(drop8)
############################################################
###Define the VAE as a Model with a custom train_step
############################################################
if binary:
    model = Model(inputs= [im_input], outputs = [OUTPUT], name='binary_classification')
else:
    model = Model(inputs= [im_input,pathology_features], outputs = [OUTPUT], name='Categorical_classification')


if(Weights):
    model.load_weights(Weight_name)

############################################################
model.summary()
if (Training):

    model.compile(optimizer = Adam(lr = 1e-6), loss = 'categorical_crossentropy', metrics = ['accuracy'])

    checkpoint = ModelCheckpoint(Weight_name, monitor='loss',
                                 verbose=1, save_best_only=True, mode='min')
    csv_logger = CSVLogger(os.path.join(Test_save_path,Output), append=True, separator=';')
    earlystopping = EarlyStopping(monitor = 'loss', verbose = 1,
                                  min_delta = 0.00001, patience = 5, mode = 'min',restore_best_weights=True)
    callbacks_list = [checkpoint, csv_logger, earlystopping]

    train_data = trainGenerator(size=Raw_input,batch=BATCH_SIZE, classes=class_type,               #class_mode = 'binary'/'categorical'
                                image_path=Train_image_path)
    #val_data = valGenerator(size=Raw_input,batch=BATCH_SIZE, classes=class_type,
    #                        image_path=Val_image_path)
    results = model.fit_generator(train_data,
                                  epochs=No_of_epochs,
                                  steps_per_epoch = Num_of_train_batches, #(NO_OF_TRAINING_IMAGES//BATCH_SIZE),
                                  validation_data = None,#val_data,
                                  validation_steps = None,#NO_OF_VAL_IMAGES,#NO_OF_VAL_IMAGES, 
                                  callbacks=callbacks_list,
                                  verbose=1)
    #train_info = train_generator(size=(28,28),batch=BATCH_SIZE, classes=class_type,               #class_mode = 'binary'/'categorical'
    #                             image_path=Val_image_path)

    #MOD.save_weights('Bin0.h5')
    #Confution Matrix and Classification Report
    #Y_pred = model.predict_generator(val_data,Num_of_val_batches)
    #y_pred = np.argmax(Y_pred, axis=1)
    #print('Confusion Matrix')
    #print(confusion_matrix(train_info.classes, y_pred))
        
    #print('Classification Report')
    #if binary:
    #    target_names = ['Normal', 'Abnormal']
    #else:
    #    target_names = ['HCC Multiple', 'HCC Single', 'mCRC']
    
    #print(classification_report(train_info.classes, y_pred, target_names=target_names))

    #label_map = train_info.class_indices
    #print('Label map:',label_map)

    #with open(os.path.join(Test_save_path,'label_map.txt'),'w') as data:
    #    data.write(str(label_map))

    #y_proba = model.predict(val_data,steps = Num_of_val_batches)#,batch_size=1)

    #y_test = np.zeros((820,), dtype=int)
    #y_test[191:504]=1
    #y_test[504:820]=2

    #classes = 3#model.classes_
    #for i in range(classes):
        # Gets the class
        # Prepares an auxiliar dataframe to help with the plots

    #    Y_TEST = [1 if y == i else 0 for y in y_test]
    #    Y_PRED = y_proba[:, i]

    #    fpr, tpr, thresholds = roc_curve(Y_TEST,Y_PRED)

    #    roc_auc_ovr = auc(fpr, tpr)


    #    np.savetxt(os.path.join(Test_save_path,"tpr%d.txt"%i),tpr)
    #    np.savetxt(os.path.join(Test_save_path,"fpr%d.txt"%i),fpr)
    #    np.savetxt(os.path.join(Test_save_path,"thresholds%d.txt"%i),thresholds)
    #    print('AUC:',roc_auc_ovr)

else:
############################ Only for testing################
    test_generator = testGenerator(size=Raw_input,batch=1, classes=None,
                                   image_path=Test_image_path)
    pred = model.predict_generator(test_generator,No_of_test_images,verbose=1)
    results = np.argmax(pred,axis=1)
    np.savetxt(os.path.join(Test_save_path,'Test.txt'), results)


    #predicted_class_indices=np.argmax(pred,axis=1)

    #labels = (train_generator.class_indices)
    #labels = dict((v,k) for k,v in labels.items())
    #predictions = [labels[k] for k in predicted_class_indices]


