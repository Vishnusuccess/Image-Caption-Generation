#!/usr/bin/env python
# coding: utf-8



import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import numpy as np 
import pandas as pd 
import tensorflow as tf 
from tensorflow import keras 
import os
from tensorflow.keras.models import Sequential 
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img 
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.layers import Conv2D, Dense, BatchNormalization, Activation, Dropout, MaxPooling2D, Flatten ,Input,add,Embedding,LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.optimizers import Adam, RMSprop, SGD 
from pickle import dump , load
from tensorflow.keras.preprocessing.image import img_to_array
#from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau,EarlyStopping
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model





model=load_model('vgg_best_model.h5')
model.make_predict_function()


model_temp=VGG16(weights='imagenet',input_shape=(224,224,3))

model_VGG = Model(inputs=model_temp.inputs, outputs=model_temp.layers[-2].output)
model_VGG.make_predict_function()

# extract features from each photo in the directory
def extract_features(img_name):
    # load the model
#     features=dict()
    # extract features from image
      #directory = os.path.join(BASE_DIR, 'Images')
    image = load_img(img_name, target_size=(224, 224))
      # convert image pixels to numpy array
    image = img_to_array(image)
      # reshape data for model
    image = image.reshape((1, image.shape[0], image.shape[1], image.shape[2]))
      # preprocess image for vgg
    image = preprocess_input(image)
      # extract features
    feature = model_VGG.predict(image, verbose=0)
    return feature
    

enc=extract_features('1000268201_693b08cb0e.jpg')
enc





import pickle
with open('tokenizer_vgg.pkl', 'rb') as f:
    tokenizer = pickle.load(f)
tokenizer

def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
              return word
    return None

# generate caption for an image
def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
          word[:-1]
          break
    in_text = in_text.replace('startseq','')
    in_text=in_text.replace('endseq','')
    return in_text

    



def caption_this_image(image):
    enc=extract_features(image)
    caption = predict_caption(model, enc, tokenizer, 35)
    return caption






