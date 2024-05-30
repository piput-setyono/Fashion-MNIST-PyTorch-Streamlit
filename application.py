# -*- coding: utf-8 -*-
"""
Created on Wed May 29 10:27:05 2024

@author: Piput Setyono
"""

import streamlit as st
import torch
from PIL import Image, ImageOps 
import numpy as np
import matplotlib.pyplot as plt

import torchvision
from torchvision.models import mobilenet_v3_small, ResNet50_Weights
from torchvision import transforms

from captum.attr import IntegratedGradients
from captum.attr import visualization as viz
import torch.nn as nn
from load_model_mnist import MnistMobileNetV3Small, MnistMobileNetV3Large, MnistMobileNetV2, MnistResNet50

preprocess_func = transforms.Compose([
                                     transforms.Resize((224, 224)),
                                     transforms.ToTensor(),
                                     transforms.Lambda(lambda x: x.repeat(3, 1, 1))
                                     ])

categories = np.array([
    'T-shirt/top',
    'Trouser',
    'Pullover',
    'Dress',
    'Coat',
    'Sandal',
    'Shirt',
    'Sneaker',
    'Bag',
    'Ankle boot'
])

@st.cache_resource
def mnist_data_test():
    return torchvision.datasets.FashionMNIST(root="data", train=False, download=True)

def load_model():
    model = MnistMobileNetV3Small(3, 10)
    model_state_dict = torch.load("E:/fashion MNIST/models/MobileNetV3Small_imageLayer_3_best_f1")
    model.load_state_dict(model_state_dict)
    model.eval();
    return model

def make_prediction(model, processed_img):
    probs = model(processed_img.unsqueeze(0))
    probs = probs.softmax(1)
    probs = probs[0].detach().numpy()

    prob, idxs = probs[probs.argsort()[-5:][::-1]], probs.argsort()[-5:][::-1]
    return prob, idxs

def interpret_prediction(model, processed_img, target):
    interpretation_algo = IntegratedGradients(model)
    feature_imp = interpretation_algo.attribute(processed_img.unsqueeze(0), target=int(target))
    feature_imp = feature_imp[0].numpy()
    feature_imp = feature_imp.transpose(1,2,0)

    return feature_imp

## Dashboard GUI
st.title("MNIST Fashion Image Classifier")
upload = st.file_uploader(label="Upload Image:", type=["png", "jpg", "jpeg"])
    
if upload:
    img = Image.open(upload)
    
    if img.mode == "L":    
        trans = transforms.Compose([transforms.Resize((28,28))])
        img = trans(img)
        
    if img.mode == "RGB":
        trans = transforms.Compose([
                    transforms.Grayscale(),
                    transforms.Resize((28,28))
                    ])
        img = ImageOps.invert(trans(img))
        
    predict = st.button('Run Prediction')
    
    if predict:
        model = load_model()
        preprocessed_img = preprocess_func(img)
        probs, idxs = make_prediction(model, preprocessed_img)
        
        feature_imp = interpret_prediction(model, preprocessed_img, idxs[0])
        
        main_fig = plt.figure(figsize=(12,3))
        ax = main_fig.add_subplot(111)
        plt.barh(y=categories[idxs][::-1], width=probs[::-1], color=["dodgerblue"]*4 + ["tomato"])
        plt.title("Top 5 Probabilities", loc="center", fontsize=15)
        st.pyplot(main_fig, use_container_width=True)
        interp_fig, ax = viz.visualize_image_attr(feature_imp, show_colorbar=True, fig_size=(6,6))
    
        col1, col2 = st.columns(2, gap="medium")
        with col1:
            main_fig = plt.figure(figsize=(6,6))
            ax = main_fig.add_subplot(111)
            plt.imshow(img)
            plt.xticks([],[]);
            plt.yticks([],[]);
            st.pyplot(main_fig, use_container_width=True)
        
        with col2:    
            st.pyplot(interp_fig, use_container_width=True)