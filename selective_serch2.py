# -*- coding: utf-8 -*-
"""
Created on Thu Mar 16 23:08:52 2017

@author: JunTaniguchi
"""
import os
import selectivesearch
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from keras.optimizers import Adam
import numpy as np
from PIL import Image
import keras.backend.tensorflow_backend as KTF
from keras.models import model_from_json
import tensorflow as tf


os.chdir('/Users/j13-taniguchi/study_tensorflow/keras_project/read_place/project_rcnn')

from vgg_model import vgg_model

#地名のリストを作成
with open("./param/place_tokyo.txt", "r") as place_file:
    place_list = place_file.readlines()
    place_list = [place_str.strip() for place_str in place_list]
NUM_CLASSES = len(place_list)


def predict_img(img, model, place_list, input_shape):
    # 画像から特徴部分を抽出
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    countours = cv2.findContours(gray,
                                 cv2.RETR_LIST,
                                 cv2.CHAIN_APPROX_SIMPLE)[1]
    print(countours)
    for cnt in countours:
        x, y, w, h = cv2.boundingRect(cnt)
        if h > 10:
            continue
        if h < 3:
            continue
        if h * 2 > w:
            continue
        img_part = img[y:y+h, x:x+w]
        ax, place_name_list = local_classification(ax, img_part, input_shape, x, y, w, h)
    ax.imshow(img)
    return place_name_list

def local_classification(ax, img_part, input_shape, x, y, w, h):
    if img_part.shape[0] > 0 and img_part.shape[1] > 0 and img_part.shape[2] == 3:
        img_resize = cv2.resize(img_part, (input_shape[0], input_shape[1]))
        img_resize = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)
        img_list = []        
        img_list.append(img_resize) 
        np_list = np.array(img_list)
        pre_list = np_list.reshape(len(np_list), input_shape[0], input_shape[1], 1)
        result_list = model.predict(pre_list.astype(np.float32))
        print(result_list)
        place_name_list = []
        for idx, result_idx in enumerate(result_list):
            if max(result_idx) > 0.9:
                prime_result_idx = result_idx.argmax()
                part_place_name = place_list[prime_result_idx]
                rect = mpatches.Rectangle((x, y), w, h, 
                                          fill=False, 
                                          edgecolor='blue', 
                                          linewidth=1)                            
                ax.add_patch(rect)
                ax.text(x - 1,
                        y - 1,
                        part_place_name,
                        color='black',
                        fontsize=15)
                place_name_list.append(part_place_name)
        return ax, place_name_list
                
                
        
#実際に確認したいターゲットの画像を設定する。
img = cv2.imread("hrei-sign105.png")
input_shape = (50, 100, 1)

with tf.Graph().as_default():
    session = tf.Session('')
    KTF.set_session(session)
    KTF.set_learning_phase(1)
    
    with open('./param/learning_place_name.json', 'r') as model_file:
        model_json = model_file.read()
    model = model_from_json(model_json)
    model.load_weights('./param/learning_place_name.hdf5', by_name=True)
    
    result_label_list = predict_img(img, model, place_list, input_shape)
    
print(len(result_label_list))