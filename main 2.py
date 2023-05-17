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
import pickle
################################Loading Stored Features and images##################################
Computer_file_img=pickle.load(open(r'C:\Users\AM_0296\PycharmProjects\pythonProject\Computer_images.pkl','rb'))
Computer_feature_list=(pickle.load(open(r'C:\Users\AM_0296\PycharmProjects\pythonProject\Computer_fetaures.pkl','rb')))


Phone_file_img=pickle.load(open(r'C:\Users\AM_0296\PycharmProjects\pythonProject\Phone_images.pkl','rb'))
Phone_feature_list=(pickle.load(open(r'C:\Users\AM_0296\PycharmProjects\pythonProject\Phone_fetaures.pkl','rb')))


TV_file_img=pickle.load(open(r'C:\Users\AM_0296\PycharmProjects\pythonProject\TV_images.pkl','rb'))
TV_feature_list=(pickle.load(open(r'C:\Users\AM_0296\PycharmProjects\pythonProject\TV_fetaures.pkl','rb')))


Waching_machine_file_img=pickle.load(open(r'C:\Users\AM_0296\PycharmProjects\pythonProject\Waching_machine_images.pkl','rb'))
Waching_machine_feature_list=(pickle.load(open(r'C:\Users\AM_0296\PycharmProjects\pythonProject\Waching_machine_fetaures.pkl','rb')))

All_file_img= pickle.load(open(r'C:\Users\AM_0296\PycharmProjects\pythonProject\All_images.pkl','rb'))
All_feature_list =(pickle.load(open(r'C:\Users\AM_0296\PycharmProjects\pythonProject\All_fetaures.pkl','rb')))
###################### Method to Save Uploaded Image into local############################

cwd = os.getcwd()
uploads_dir = os.path.join(cwd, "uploads")
if not os.path.exists(uploads_dir):
    os.mkdir(uploads_dir)



def Save_img(upload_img):
    try:
        with open(os.path.join("uploads",upload_img.name),'wb') as f:
            f.write(upload_img.getbuffer())
        return 1
    except:
        return 0


######################## Method to Extract features of new query image#######################
model=ResNet50(weights='imagenet',include_top=False, input_shape=(224,224,3))
model.trainable=False
model=tf.keras.Sequential([model,GlobalMaxPool2D()])

def feature_extraction(path, model):
    img = image.load_img(path, target_size=(224, 224))  # Load image in size of 224,224,3
    img_arr = image.img_to_array(img)  # storing into array
    ex_img_arr = np.expand_dims(img_arr, axis=0)  ## Expanding the dimension of image
    pre_pr_img = preprocess_input(ex_img_arr)  ## preprocessing the image
    result = model.predict(pre_pr_img).flatten()  ### to make 1d vector
    normal_result = result / norm(result)  ## Normalize the result using norm func from linalg(numpy)
    return normal_result


def prod_recom(features, feature_list):
    neb = NearestNeighbors(n_neighbors=6, algorithm='brute',
                           metric='euclidean')  # using brute force algo here as data is not too big
    neb.fit(feature_list)  ## fit with feature list
    dist, ind = neb.kneighbors([
                                   features])  # return distance and index but we use index to find out nearest images from stored features vector
    return ind


upload_img = st.file_uploader("Choose an image")  # To display upload button on screen

# st.image(Image.open('##### Uploaded img path #####')
### Condition to check if image got uploaded then call save_img method to save and preprocess image followed by extract features and recommendation
if upload_img is not None:
    img_path = os.path.join("uploads", upload_img.name)
    if Save_img(upload_img):
        st.image(Image.open(upload_img))
        st.header("file uploaded successfully")
        features=feature_extraction(img_path,model)
        progress_text = "Hold on! Result will shown below."
        my_bar = st.progress(0, text=progress_text)
        for percent_complete in range(100):
            time.sleep(0.02)
            my_bar.progress(percent_complete + 1, text=progress_text) ## to add progress bar untill feature got extracted

        from tensorflow.keras.applications.resnet50 import ResNet50,preprocess_input,decode_predictions

        from tensorflow.keras.preprocessing.image import load_img,img_to_array

        # Load ResNet50 model pre-trained on ImageNet dataset

        model=ResNet50(weights='imagenet')

        # Load image and preprocess it

        image=load_img(img_path,target_size = (224,224))
        image=img_to_array(image)
        image=np.expand_dims(image,axis = 0)
        image=preprocess_input(image)

        # Predict image class

        predictions=model.predict(image)
        decoded_predictions=decode_predictions(predictions,top = 1)[0]
        CLASS = decoded_predictions[0][1]

        if CLASS in ['computer', 'laptop', 'notebook', 'laptop charger']:
            feature_list = Computer_feature_list
            file_img = Computer_file_img
        elif CLASS in ['phone', 'cell','cellular_telephone','binder','iPod','pay-phone','lighter','mobile','cellular phone']:
            feature_list = Phone_feature_list
            file_img = Phone_file_img
        elif CLASS in ['tv', 'television','web_site','LCD', 'LED']:
            feature_list = TV_feature_list
            file_img = TV_file_img

        elif CLASS in ['washer','dishwasher','washing machine']:
            feature_list= Waching_machine_feature_list
            file_img = Waching_machine_file_img
        else:
            feature_list = All_feature_list
            file_img = All_file_img

        ind = prod_recom(features, feature_list)  # calling recom. func to get 6 recommendation
        ### to create 6 section of images into the screen
        col1, col2, col3, col4, col5, col6 = st.columns(6)

        ##for each section image shown by below code
        with col1:
            st.image(Image.open(file_img[ind[0][0]]))
        with col2:
            st.image(Image.open(file_img[ind[0][1]]))
        with col3:
            st.image(Image.open(file_img[ind[0][2]]))
        with col4:
            st.image(Image.open(file_img[ind[0][3]]))
        with col5:
            st.image(Image.open(file_img[ind[0][4]]))
        with col6:
            st.image(Image.open(file_img[ind[0][5]]))

    else:
        st.header("Some error occured")





