import os, glob
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import cv2, random

ttf_list = []
url = "/Users/JunTaniguchi/study_tensorflow/keras_project/read_place/project_spp"

os.chdir(url)

#フォンントのリストを作成
with open("./param/japanese_font_list.txt", "r") as japanese_ttf:
    jp_font_list = japanese_ttf.readlines()
    jp_font_list = [jp_font.strip() for jp_font in jp_font_list]

#文字リストのリストを作成
with open("./param/japanese_lang.txt", "r") as place_file:
    str_list = place_file.read()
    str_list = [jpn_str.strip() for jpn_str in str_list]

# サンプル画像を出力するフォルダ
for jpn_str in str_list:
    url = "./image/" + jpn_str
    if not os.path.exists(url):
        os.makedirs(url)

def generator():
    i = 0
    while True:
        yield i
        i+=1

def gaussianBlur_image(img_name, jpn_str, image_np, idx1, idx2, X, Y, font_size, x_pixel, y_pixel):
    # opencvのガウシアンフィルターを適応
    blur = cv2.GaussianBlur(image_np, (5, 5), 0)
    # グレースケール変換
    grayscale_blur = cv2.cvtColor(blur, cv2.COLOR_RGB2GRAY)
    image = Image.fromarray(grayscale_blur)
    
    # 画像ファイルを保存する
    #image.save("./image/" + jpn_str + "/" + str(idx1) + img_name + "_" + str(font_size) + str(x_pixel) + str(y_pixel) + jpn_str + '.png', 'PNG')
    data = np.asarray(image)        
    X.append(data.astype(np.float64))
    Y.append(idx2)
   
def img_make (X, Y, img_name,jp_font_list, str_list, x_pixel, y_pixel, gen):

    for idx1, jp_font in enumerate(jp_font_list):
        
        for font_size in range(y_pixel - 25, y_pixel, + 1):
        
            for idx2, jpn_str in enumerate(str_list):
                # 画像のピクセルを指定
                image = Image.new('RGB', (x_pixel, y_pixel), "black")
                draw = ImageDraw.Draw(image)
    
                # フォントの指定。引数は順に「フォントのパス」「フォントのサイズ」「エンコード」
                # メソッド名からも分かるようにTure Typeのフォントを指定する
                font = ImageFont.truetype(jp_font, font_size)

                # 表示文字のmarginを設定                
                str_width = font_size
                x_draw_pixel = (x_pixel - str_width) / 2
                y_draw_pixel = (y_pixel - font_size) / 2
                if x_draw_pixel < 0:
                    x_draw_pixel = (x_pixel - str_width)
                # 日本語の文字を入れてみる
                # 引数は順に「(文字列の左上のx座標, 文字列の左上のy座標)」「フォントの指定」「文字色」
                draw.text((x_draw_pixel, y_draw_pixel), jpn_str, font=font, fill='#ffffff')
                
                image_np = np.asarray(image)
                gaussianBlur_image(img_name, jpn_str, image_np, idx1, idx2, X, Y, font_size, x_pixel, y_pixel)
                
    return X, Y


gen = generator()

x_pixel = 25
y_pixel = 25
X = []
Y = []
X, Y = img_make(X, Y, "0", jp_font_list, str_list, x_pixel=x_pixel, y_pixel=y_pixel, gen=gen)
X, Y = img_make(X, Y, "1", jp_font_list, str_list, x_pixel=x_pixel, y_pixel=y_pixel, gen=gen)
X, Y = img_make(X, Y, "2", jp_font_list, str_list, x_pixel=x_pixel, y_pixel=y_pixel, gen=gen)
X, Y = img_make(X, Y, "3", jp_font_list, str_list, x_pixel=x_pixel, y_pixel=y_pixel, gen=gen)
X, Y = img_make(X, Y, "4", jp_font_list, str_list, x_pixel=x_pixel, y_pixel=y_pixel, gen=gen)
X, Y = img_make(X, Y, "5", jp_font_list, str_list, x_pixel=x_pixel, y_pixel=y_pixel, gen=gen)
X, Y = img_make(X, Y, "6", jp_font_list, str_list, x_pixel=x_pixel, y_pixel=y_pixel, gen=gen)
X, Y = img_make(X, Y, "7", jp_font_list, str_list, x_pixel=x_pixel, y_pixel=y_pixel, gen=gen)
X, Y = img_make(X, Y, "8", jp_font_list, str_list, x_pixel=x_pixel, y_pixel=y_pixel, gen=gen)
X, Y = img_make(X, Y, "9", jp_font_list, str_list, x_pixel=x_pixel, y_pixel=y_pixel, gen=gen)
X, Y = img_make(X, Y, "10", jp_font_list, str_list, x_pixel=x_pixel, y_pixel=y_pixel, gen=gen)
X, Y = img_make(X, Y, "11", jp_font_list, str_list, x_pixel=x_pixel, y_pixel=y_pixel, gen=gen)
X, Y = img_make(X, Y, "12", jp_font_list, str_list, x_pixel=x_pixel, y_pixel=y_pixel, gen=gen)
X, Y = img_make(X, Y, "13", jp_font_list, str_list, x_pixel=x_pixel, y_pixel=y_pixel, gen=gen)
X, Y = img_make(X, Y, "14", jp_font_list, str_list, x_pixel=x_pixel, y_pixel=y_pixel, gen=gen)
X, Y = img_make(X, Y, "15", jp_font_list, str_list, x_pixel=x_pixel, y_pixel=y_pixel, gen=gen)
X, Y = img_make(X, Y, "16", jp_font_list, str_list, x_pixel=x_pixel, y_pixel=y_pixel, gen=gen)
X, Y = img_make(X, Y, "17", jp_font_list, str_list, x_pixel=x_pixel, y_pixel=y_pixel, gen=gen)
X, Y = img_make(X, Y, "18", jp_font_list, str_list, x_pixel=x_pixel, y_pixel=y_pixel, gen=gen)
X, Y = img_make(X, Y, "19", jp_font_list, str_list, x_pixel=x_pixel, y_pixel=y_pixel, gen=gen)


X = np.array(X)
Y = np.array(Y)
np.savez("./param/npz/jpn_str_%s.npz" % (next(gen)), x=X, y=Y)
print("ok,", len(Y))

print('finish!!')
