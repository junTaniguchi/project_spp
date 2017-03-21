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
    img_lbl, regions = selectivesearch.selective_search(img,
                                                        scale=500,
                                                        sigma=0.9,
                                                        min_size=10)
    # 分類器を使用して判別
    candidates = set()
    for r in regions:
        # 特徴量抽出
        if r['rect'] in candidates:
            continue
        # ある一定pixelより小さいものについては除外する
        if r['size'] < 300:
            continue
        # ある一定pixelより大きいものについては除外する
        if r['size'] > 1000:
            continue    
        # 特徴量部分の座標を取得
        x, y, w, h = r['rect']
        if h > w:
            continue
    #    if w / h > 1.2 or h / w > 1.2:
    #        continue
        candidates.add(r['rect'])
    
    # 抽出してきた部分を表示
    fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
    ax.imshow(img)
    print(len(candidates))
    # 
    for x, y, w, h in candidates:
        #print (x, y, w, h)
        rect = mpatches.Rectangle(
            (x, y), w, h, fill=False, edgecolor='red', linewidth=1)
        ax.add_patch(rect)
    plt.show()
    
    img_list = []
    result_list = []
    for x, y, w, h in candidates:
        img_part = img[y:y+h, x:x+w]
        #image = Image.fromarray(img_part)
        #print('%s_%s_%s_%s' % (x, y, w, h) )
        #print(img_part.shape)
        if img_part.shape[0] > 0 and img_part.shape[1] > 0 and img_part.shape[2] == 3:
            print(input_shape)
            img_resize = cv2.resize(img_part, (input_shape[0], input_shape[1]))
            img_resize = cv2.cvtColor(img_resize, cv2.COLOR_RGB2GRAY)
            #fig, ax = plt.subplots(ncols=1, nrows=1, figsize=(6, 6))
            #ax.imshow(img_resize)
            img_list.append(img_resize)
    print("img_list length : %s" % str(len(img_list)))
    np_list = np.array(img_list)
    np_list = np_list.reshape(np_list.shape[0], input_shape[0], input_shape[1], 1)
    result_list = model.predict(np_list.astype(np.float32))
    #print(result_list)
    result_label_list = []
    for idx, result_idx in enumerate(result_list):
        #print(max(result_idx))
        if max(result_idx) > 0.9:
            prime_result_idx = result_idx.argmax()
            part_place_name = place_list[prime_result_idx]
            result_label_list.append(part_place_name)
    
    return result_label_list

#実際に確認したいターゲットの画像を設定する。
img = cv2.imread("hrei-sign105.png")

with tf.Graph().as_default():
    session = tf.Session('')
    KTF.set_session(session)
    KTF.set_learning_phase(1)

    # モデルの作成
    def schedule(epoch, decay=0.9):
        return base_lr * decay**(epoch)
    base_lr = 3e-4
    optim = Adam(lr=base_lr)
    input_shape = (100, 50, 1)
    
    
    model = vgg_model(input_shape, NUM_CLASSES, optim)
    model.load_weights('./param/learning_place_name.hdf5', by_name=True)
    
    result_label_list = predict_img(img, model, place_list, input_shape)
    
print(result_label_list)