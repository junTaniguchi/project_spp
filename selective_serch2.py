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
from keras.optimizers import Adam
import tensorflow as tf


os.chdir('/Users/JunTaniguchi/study_tensorflow/keras_project/read_place/project_spp')

from spp_model import spp_model

#地名のリストを作成
with open("./param/place_tokyo.txt", "r") as place_file:
    place_list = place_file.readlines()
    place_list = [place_str.strip() for place_str in place_list]
NUM_CLASSES = len(place_list)


def predict_img(img, model, place_list, input_shape):
    # 画像から特徴部分を抽出
    img_lbl, regions = selectivesearch.selective_search(img,
                                                        scale=500,
                                                        sigma=0.9,
                                                        min_size=10)
    # 分類器を使用して判別
    candidates = set()
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    for r in regions:
        # 特徴量抽出
        if r['rect'] in candidates:
            continue
        # ある一定pixelより小さいものについては除外する
        if r['size'] < 300:
            continue
        # ある一定pixelより大きいものについては除外する
        if r['size'] > 750:
            continue
        # 特徴量部分の座標を取得
        x, y, w, h = r['rect']
        if 2 * h > w:
            continue
        if h < 5:
            continue
        #print("hの高さ%s" % h)
        img_part = img[y:y+h, x:x+w]
        ax, place_name_list = local_classification(ax, img_part, input_shape, x, y, w, h)
    ax.imshow(img)
    return place_name_list

def local_classification(ax, img_part, input_shape, x, y, w, h):
    if img_part.shape[0] > 0 and img_part.shape[1] > 0 and img_part.shape[2] == 3:
        img_resize = cv2.cvtColor(img_part, cv2.COLOR_RGB2GRAY)
        img_list = []        
        img_list.append(img_resize) 
        np_list = np.array(img_list)
        pre_list = np_list.reshape(len(np_list), img_resize.shape[0], img_resize.shape[1], 1)
        result_list = model.predict(pre_list.astype(np.float32))
        place_name_list = []
        for idx, result_idx in enumerate(result_list):
            print(result_idx)
            if max(result_idx) > 0.2:
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

    def schedule(epoch, decay=0.9):
        return base_lr * decay**(epoch)
    base_lr = 3e-4
    optim = Adam(lr=base_lr)

    model = spp_model(input_shape, NUM_CLASSES, optim)
    model.load_weights('./param/learning_place_name.hdf5', by_name=True)
    
    result_label_list = predict_img(img, model, place_list, input_shape)
    
print(len(result_label_list))
