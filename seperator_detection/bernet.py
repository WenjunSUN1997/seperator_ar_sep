from pylena.scribo import line_detector
from pylena.scribo import VSegment, LSuperposition
from matplotlib import pyplot as plt
from skimage.io import imread
import numpy as np
from PIL import Image, ImageDraw
import cv2
import math

def line_detect(img_path=r'../article_dataset/AS_TrainingSet_NLF_NewsEye_v2/576461_0004_23676337.jpg'):
    image = cv2.imread(img_path)
    image_to_draw = Image.open(img_path).convert('RGB')
    drawer = ImageDraw.Draw(image_to_draw, 'RGB')
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    ret, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
    cv2.imwrite("temp/gray.png", binary)
    h, w = binary.shape
    hors_k = int(math.sqrt(w) * 1.2)
    vert_k = int(math.sqrt(h) * 1.2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hors_k, 1))
    hors = ~cv2.dilate(binary, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_k))
    verts = ~cv2.dilate(binary, kernel, iterations=1)
    borders = cv2.bitwise_or(hors, verts)
    cv2.imwrite("temp/borders.png", borders)
    img_in = imread(img_path)
    lines = line_detector(borders, mode='vector', blumi=200, llumi=200, min_len=200)
    image = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(image, "RGB")
    for line in lines:
        draw.line((line.x0,
                   line.y0,
                   line.x1,
                   line.y1,), 'green', width=8)
    image.save('temp/bernet.png')

    print()

if __name__ == "__main__":
    line_detect()