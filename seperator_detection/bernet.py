from pylena.scribo import line_detector
from pylena.scribo import VSegment, LSuperposition
from matplotlib import pyplot as plt
from skimage.io import imread
import numpy as np
from PIL import Image, ImageDraw

def line_detect(img_path='sep_dataset/test/fi/images/576443_0001_23676242.jpg'):
    img_in = imread(img_path)
    lines = line_detector(img_in, mode='vector', blumi=200, llumi=200, min_len=200)
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