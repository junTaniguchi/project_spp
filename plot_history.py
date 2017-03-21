# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 00:02:52 2017

@author: JunTaniguchi
"""
import os
path = "/Users/j13-taniguchi/study_tensorflow/keras_project/read_place/project_spp"
os.chdir(path)

def plot_history(history):
    # print(history.history.keys())
    import matplotlib.pyplot as plt

    # 精度の履歴をプロット
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(['acc', 'val_acc'], loc='lower right')
    filename = "./plot/history_plot_Learning_acc.png"
    plt.savefig(filename)
    plt.show()

    # 損失の履歴をプロット
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['loss', 'val_loss'], loc='lower right')
    filename = "./plot/history_plot_Learning_loss.png"
    plt.savefig(filename)
    plt.show()
    
