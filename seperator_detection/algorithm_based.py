import cv2
import numpy as np
from tqdm import tqdm
import math
from PIL import Image, ImageDraw
import math
import os
import seaborn as sns

def calcul_length(img_slice):
    result = 0
    if img_slice.shape[0] < img_slice.shape[1]:
        for index in range(img_slice.shape[1]):
            result += np.max(img_slice[:, index]) / 255
        # return result
        if result >= 0.7 * img_slice.shape[1]:
            return True
        else:
            return False

    else:
        for index in range(img_slice.shape[0]):
            result += np.max(img_slice[index, :]) / 255

        if result >= 0.7 * img_slice.shape[0]:
            return True
        else:
            return False

def analysis_seprator(img,  path):
    final = {}
    img_draw = Image.open(path).convert("RGB")
    draw = ImageDraw.Draw(img_draw, "RGB")
    shuiping_result = []
    shuiping_split = []
    print('analysis horiz direction')
    # shuipingfenxi
    for index in tqdm(range(0, img.shape[0], 10)):
        img_slice = img[index:index+60, :]
        if calcul_length(img_slice) and len(shuiping_result)==0:
            shuiping_result.append(index)
        if calcul_length(img_slice) and len(shuiping_result)!=0:
            if index - shuiping_result[-1] > 200:
                shuiping_result.append(index)

    final['shuiping_1'] = shuiping_result
    for index in shuiping_result:
        draw.line((0, index, img.shape[1], index), 'red', width=4)

    # fengetuxiang_shuiping
    shuiping_result.insert(0, 0)
    if len(shuiping_result) > 1:
        for index in range(1, len(shuiping_result)):
            shuiping_split.append(img[shuiping_result[index-1]:shuiping_result[index], :])

        shuiping_split.append(img[shuiping_result[-1]:, :])
    else:
        shuiping_split.append(img)

    # chuizhi fenxi
    chuizhi_result = []
    for group_index in range(len(shuiping_split)):
        temp = []
        part = shuiping_split[group_index]
        for index_2 in tqdm(range(0, part.shape[1], 10)):
            img_slice = part[:, index_2:index_2+60]
            if calcul_length(img_slice):
                temp.append(index_2)

        chuizhi_result.append(temp)

    for index in range(len(chuizhi_result)-1):
        for index_2 in chuizhi_result[index]:
            draw.line((index_2, shuiping_result[index], index_2, shuiping_result[index+1]), 'red', width=4)
    for index_2 in chuizhi_result[-1]:
        draw.line((index_2, shuiping_result[-1], index_2, img.shape[0]), 'red', width=4)
    img_draw.save('temp/argo.png')
    print()
    pass

def mopho(path=r'sep_dataset/test/fi/images/576464_0003_23676343.jpg'):
    image = cv2.imread(path)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    cv2.imwrite("temp/gray.png", binary)
    h, w = binary.shape
    hors_k = int(math.sqrt(w)*1.2)
    vert_k = int(math.sqrt(h)*1.2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hors_k,1))
    hors = ~cv2.dilate(binary, kernel, iterations = 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1,vert_k))
    verts = ~cv2.dilate(binary, kernel, iterations = 1)
    borders = cv2.bitwise_or(hors, verts)
    analysis_seprator(borders, path)
    cv2.imwrite("temp/borders.png", borders)
    gaussian = cv2.GaussianBlur(borders, (9, 9), 0)
    edges = cv2.Canny(gaussian, 70, 150)
    cv2.imwrite("temp/edges.png", edges)

    print()

if __name__ == "__main__":
    mopho()