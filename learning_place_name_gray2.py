# -*- coding: utf-8 -*-
"""
Created on Tue Mar  7 16:45:08 2017
@author: j13-taniguchi
"""
import os, glob
from keras.models import Sequential
import tensorflow as tf
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from sklearn import cross_validation
import numpy as np
import keras.callbacks
import keras.backend.tensorflow_backend as KTF
from keras.utils.visualize_util import plot
from PIL import Image, ImageDraw, ImageFont
from keras.preprocessing.image import ImageDataGenerator

import shutil

path = "/Users/JunTaniguchi/study_tensorflow/keras_project/read_place/project_spp"
os.chdir(path)

log_filepath = './log'

from plot_history import plot_history
from spp_model import spp_model
#地名のリストを作成
with open("./param/japanese_lang.txt", "r") as jpn_str_file:
    jpn_str_list = jpn_str_file.read()
    jpn_str_list = [jpn_str.strip() for jpn_str in jpn_str_list]
NUM_CLASSES = len(jpn_str_list)

# 存在するnpzファイル全てを取り込む
npz = "./param/npz/jpn_str_0.npz"
# フォント画像のデータを読む
xy = np.load(npz)
X = xy["x"]
Y = xy["y"]
X /= 255
X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
Y = np_utils.to_categorical(Y, NUM_CLASSES)

input_shape = (50, 100, 1)
# 訓練データとテストデータに分割
X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y)

#VGG
#old_session = KTF.get_session()


with tf.Graph().as_default():
    session = tf.Session('')
    KTF.set_session(session)
    KTF.set_learning_phase(1)

    def schedule(epoch, decay=0.9):
        return base_lr * decay**(epoch)
    base_lr = 3e-4
    optim = keras.optimizers.Adam(lr=base_lr)
    
    model = spp_model(input_shape, NUM_CLASSES, optim) 
   
    # 中間チェックポイントのデータを一時保存するためのディレクトリを作成
    if not os.path.exists('./param/checkpoints'):
        os.mkdir('./param/checkpoints')
    # callback関数にて下記機能を追加
    #    重みパラメータの中間セーブ
    #    学習率のスケジューラ
    #    改善率が低い場合にトレーニングを終了する
    #    TensorBoardの使用 $tensorboard --logdir=/full_path_to_your_logs

    callbacks = [
                 keras.callbacks.ModelCheckpoint('./param/checkpoints/weights.{epoch:02d}-{val_loss:.2f}.hdf5', verbose=1, save_weights_only=True),
                 keras.callbacks.LearningRateScheduler(schedule),
                 keras.callbacks.EarlyStopping(monitor='val_loss', patience=0, verbose=0, mode='auto'),
                 #keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1, write_graph=True)
                ]
        
    if os.path.exists('./param/learning_jpn_str.hdf5'):
        model.load_weights('./param/learning_jpn_str.hdf5', by_name=True)
    # 学習開始
    datagen = ImageDataGenerator(featurewise_center=True,
                                 featurewise_std_normalization=True,
                                 width_shift_range=0.2,
                                 height_shift_range=0.2,
                                 rescale=1./255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 rotation_range=20,
                                 horizontal_flip=True,
                                 fill_mode='nearest')
    datagen.fit(X_train)
    nb_epoch = 50
    history = model.fit_generator(datagen.flow(X_train, y_train, batch_size=32),
                                  samples_per_epoch=len(X_train), nb_epoch=nb_epoch)
    
    score = model.evaluate(X_test, y_test, verbose=0)
    print('Learning End!!')
    print('  Test loss    :', score[0])
    print('  Test accuracy;', score[1])
    # モデルを保存
    model.save_weights('./param/learning_jpn_str.hdf5')
    # 重みパラメータをJSONフォーマットで出力
    model_json = model.to_json()
    with open('./param/learning_jpn_str.json', 'w') as json_file:
        #json.dump(model_json, json_file)
        json_file.write(model_json)

    plot_history(history)
    # モデルをpngでプロット
    plot(model,
         to_file='./param/learning_jpn_str.png', 
         show_shapes=True,
         show_layer_names=True)
    
    # 重みパラメータをJSONフォーマットで出力
    model_json = model.to_json()
    with open('./param/learning_jpn_str.json', 'w') as json_file:
        #json.dump(model_json, json_file)
        json_file.write(model_json)
    # チェックポイントとなっていたファイルを削除
    shutil.rmtree('./param/checkpoints')
    # モデルを元にテストを実施
    result_X_test = model.predict(X_test)


#KTF.set_session(old_session)
print("finish!!")
correct_count = 0
incorrect_count = 0
# 予測結果と正解のラベルを照合する
for idx1, idx_result_X in enumerate(result_X_test):
    # 予測結果のargmaxを抽出    
    result_idx = idx_result_X.argmax()
    result_label = jpn_str_list[result_idx]
    # 正解の番号を抽出
    answer_idx = y_test[idx1].argmax()
    answer_label = jpn_str_list[answer_idx]
    # 予測結果と正解の値を比較
    if result_idx == answer_idx:
        correct_message = "correct Awesome: answer: %s result: %s" %(answer_label, result_label)    
        print(correct_message)
        correct_count+=1
        continue
    # 不正解をコンソールへ表示   
    error_message = "incorrect: answer: %s result: %s" %(answer_label, result_label)    
    print(error_message)
    incorrect_count+=1

    # 不正解の画像をincorrectディレクトリへ格納する準備
    incorrect_dir = "./incorrect/%s/" %(answer_label)
    if not os.path.exists(incorrect_dir):
        os.makedirs(incorrect_dir)
    incorrect_file_name = incorrect_dir + answer_label + str(idx1) + ".png"
    # 不正解だったデータを画像化
    # idx_result_Xを非正規化
    X_test[idx1] *= 256
    X_img_array = X_test[idx1]
    reshape_X = X_img_array.reshape(X_img_array.shape[0], X_img_array.shape[1])
    img = Image.fromarray(np.uint8(reshape_X))
    img.save(incorrect_file_name) 
print("correct_count   :%s" % str(correct_count))
print("incorrect_count :%s" % str(incorrect_count))