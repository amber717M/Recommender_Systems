### Importing required libraries####
from PIL import Image

import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPool2D
from tensorflow.keras.preprocessing import image
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
# from annoy import AnnoyIndex
import pickle
import os
from tqdm import tqdm
import cv2
import time
st.title('Product Recommender system')

############################ Defining Model##############################################
model=ResNet50(weights='imagenet',include_top=False, input_shape=(224,224,3))
model.trainable=False
model=tf.keras.Sequential([model,GlobalMaxPool2D()])
model.summary()

########  Code to extract features of images  ######
def extract_features(path,model):
    img=image.load_img(path, target_size=(224,224))
    img_arr=image.img_to_array(img)
    ex_img_arr=np.expand_dims(img_arr,axis=0)
    pre_pr_img=preprocess_input(ex_img_arr)
    result=model.predict(pre_pr_img).flatten()
    normal_result=result/norm(result)
    return normal_result


############## EXTRACTING FEATURES OF COMPUTER IMAGES ################
path = r"C:\Users\AM_0296\PycharmProjects\pythonProject\Computer"
Computer_images = [os.path.join(path, files) for files in os.listdir(path)]

pickle.dump(Computer_images, open('Computer_images.pkl', 'wb'))
Computer_feature_list = []
for file in tqdm(Computer_images):
    Computer_feature_list.append(extract_features(file, model))
pickle.dump(Computer_feature_list, open('Computer_fetaures.pkl', 'wb'))

############## EXTRACTING FEATURES OF PHONE IMAGES ################
path = r"C:\Users\AM_0296\PycharmProjects\pythonProject\Phone"
Phone_images = [os.path.join(path, files) for files in os.listdir(path)]

pickle.dump(Phone_images, open('Phone_images.pkl', 'wb'))
Phone_feature_list = []
for file in tqdm(Phone_images):
    Phone_feature_list.append(extract_features(file, model))
pickle.dump(Phone_feature_list, open('Phone_fetaures.pkl', 'wb'))

############## EXTRACTING FEATURES OF TV IMAGES ################
path = r"C:\Users\AM_0296\PycharmProjects\pythonProject\TV"
TV_images = [os.path.join(path, files) for files in os.listdir(path)]

pickle.dump(TV_images, open('TV_images.pkl', 'wb'))
TV_feature_list = []
for file in tqdm(TV_images):
    TV_feature_list.append(extract_features(file, model))
pickle.dump(TV_feature_list, open('TV_fetaures.pkl', 'wb'))

############## EXTRACTING FEATURES OF WACHING MACHINE IMAGES ################
path = r"C:\Users\AM_0296\PycharmProjects\pythonProject\Waching machine"
Waching_machine_images = [os.path.join(path, files) for files in os.listdir(path)]

pickle.dump(Waching_machine_images, open('Waching_machine_images.pkl', 'wb'))
Waching_machine_feature_list = []
for file in tqdm(TV_images):
    Waching_machine_feature_list.append(extract_features(file, model))
pickle.dump(Waching_machine_feature_list, open('Waching_machine_fetaures.pkl', 'wb'))


############## EXTRACTING FEATURES OF ALL IMAGES ################
path = r"C:\Users\AM_0296\PycharmProjects\pythonProject\All"
All_images = [os.path.join(path, files) for files in os.listdir(path)]

pickle.dump(All_images, open('All_images.pkl', 'wb'))
All_feature_list_list = []
for file in tqdm(All_images):
    All_feature_list.append(extract_features(file, model))
pickle.dump(All_feature_list, open('All_fetaures.pkl', 'wb'))

#####################end #########################################################

