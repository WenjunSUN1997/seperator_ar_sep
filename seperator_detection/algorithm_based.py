import cv2
import numpy as np
from tqdm import tqdm
import math
from PIL import Image, ImageDraw
import math
import os
from ast import literal_eval
from article_segmentation.sep_tool.xml_reader import XmlProcessor

def calcul_length(img_slice, direction):
    result = 0
    # if img_slice.shape[0] < img_slice.shape[1]:
    if direction == 'shuiping':
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

        if result >= 0.6 * img_slice.shape[0]:
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
        if calcul_length(img_slice, None) and len(shuiping_result)==0:
            shuiping_result.append(index)
        if calcul_length(img_slice, None) and len(shuiping_result)!=0:
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
            if calcul_length(img_slice, None):
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

def split_image(image_pice: np.array,
                direction):
    split_location = []
    if direction == 'shuiping':
        for index in tqdm(range(0, image_pice.shape[0], 30)):

                img_slice = image_pice[index:index+60, :]
                if calcul_length(img_slice, direction) and len(split_location)==0:
                    split_location.append(index)
                elif calcul_length(img_slice, direction) and len(split_location)>0:
                    if index - split_location[-1] > 200:
                        split_location.append(index)
    else:
        for index in tqdm(range(0, image_pice.shape[1], 30)):
            img_slice = image_pice[:, index:index + 60]
            if calcul_length(img_slice, direction) and len(split_location) == 0:
                split_location.append(index)
            elif calcul_length(img_slice, direction) and len(split_location) > 0:
                if index - split_location[-1] > 200:
                    split_location.append(index)

    return split_location

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

def split_link(image_path:str):
    links = []
    xml_reader = XmlProcessor(0, image_path.replace('jpg', 'xml'))
    annotation_list = xml_reader.get_annotation()
    for annotation in annotation_list:
        annotation['center'] = ([(x + y) / 2 for x, y in
                                 zip(annotation['bbox'][0], annotation['bbox'][2])])
    image = cv2.imread(image_path)
    image_to_draw = Image.open(image_path).convert('RGB')
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
    cv2.imwrite("temp/borders .png", borders)
    first_shuiping_split_points = split_image(borders, 'shuiping')
    first_shuiping_group = {'image_pieces': [],
                            'annotation': [],
                            'start_value': []}
    for index, value in enumerate(first_shuiping_split_points):
        group_temp = []
        if index == 0:
            first_shuiping_group['start_value'].append([0, value])
            first_shuiping_group['image_pieces'].append(borders[:value, :])
            for annotation in annotation_list:
                if annotation['center'][1] <= value:
                    group_temp.append(annotation)

        elif index != 0 and index <= len(first_shuiping_split_points)-1:
            first_shuiping_group['start_value'].append(
                [first_shuiping_split_points[index-1], value])
            first_shuiping_group['image_pieces'].append(
                borders[first_shuiping_split_points[index-1]:value, :])
            for annotation in annotation_list:
                if annotation['center'][1] <= value and \
                        annotation['center'][1] > first_shuiping_split_points[index-1]:
                    group_temp.append(annotation)

        first_shuiping_group['annotation'].append(group_temp)

    first_shuiping_group['image_pieces'].append(borders[first_shuiping_split_points[-1]:, :])
    first_shuiping_group['start_value'].append(
        [first_shuiping_split_points[-1], image.shape[0]])
    group_temp = []
    for annotation in annotation_list:
        if annotation['center'][1] >= first_shuiping_split_points[-1]:
            group_temp.append(annotation)

    first_shuiping_group['annotation'].append(group_temp)
    # 开始第一次垂直分割
    first_chuizhi_split = []
    for index in range(len(first_shuiping_group['annotation'])):
        result_piece = {'search': index,
                        'image_pieces': [],
                        'start_value': [],
                        'annotation': []}
        image_piece = first_shuiping_group['image_pieces'][index]
        annotation_piece = first_shuiping_group['annotation'][index]
        first_chuizhi_split_points = split_image(image_piece, 'chuizhi')
        for index, value in enumerate(first_chuizhi_split_points):
            group_temp = []
            if index == 0:
                result_piece['image_pieces'].append(image_piece[:, :value])
                result_piece['start_value'].append([0, value])
                for annotation in annotation_piece:
                    if annotation['center'][0] <= value:
                        group_temp.append(annotation)

            elif index != 0 and index <= len(first_shuiping_split_points)-1:
                result_piece['image_pieces'].append(
                    image_piece[:, first_chuizhi_split_points[index-1]:value])
                result_piece['start_value'].append([first_chuizhi_split_points[index-1], value])
                for annotation in annotation_piece:
                    if first_chuizhi_split_points[index-1] < annotation['center'][0] <= value:
                        group_temp.append(annotation)

            result_piece['image_pieces'].append(image_piece[:, first_chuizhi_split_points[-1]:])
            result_piece['start_value'].append([first_chuizhi_split_points[-1], borders.shape[1]])
            for annotation in annotation_piece:
                if annotation['center'][0] >= first_chuizhi_split_points[-1]:
                    group_temp.append(annotation)

            result_piece['annotation'].append(group_temp)

        first_chuizhi_split.append(result_piece)




    return

def split_link_by_name(image_path:str):
    links = []
    xml_reader = XmlProcessor(0, image_path.replace('jpg', 'xml'))
    annotation_list = xml_reader.get_annotation()
    for annotation in annotation_list:
        annotation['center'] = ([(x + y) / 2 for x, y in
                                 zip(annotation['bbox'][0], annotation['bbox'][2])])
    image = cv2.imread(image_path)
    image_to_draw = Image.open(image_path).convert('RGB')
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
    cv2.imwrite("temp/borders .png", borders)
    shuiping_all = [[0, 0, borders.shape[1], 0],
                    [0, borders.shape[0], borders.shape[1], borders.shape[0]]]
    chuizhi_all = [[0, 0, 0, borders.shape[0]],
                   [borders.shape[1], 0, borders.shape[1], borders.shape[0]]]
    whole_page_shuiping = split_image(borders, 'shuiping')
    image_piece_after_whole_shuiping = []
    for index, unit in enumerate(whole_page_shuiping):
        shuiping_all.append([0, unit, borders.shape[1], unit])
        if index == 0:
            image_piece = borders[:unit, :]
            bbox = [0, 0, borders.shape[1], unit]
            image_piece_after_whole_shuiping.append({'image_piece': image_piece,
                                                     'bbox': bbox})

        elif index > 0 and index <= len(whole_page_shuiping)-1:
            image_piece = borders[whole_page_shuiping[index-1]:unit, :]
            bbox = [0, whole_page_shuiping[index-1], borders.shape[1], unit]
            image_piece_after_whole_shuiping.append({'image_piece': image_piece,
                                                     'bbox': bbox})

    image_piece = borders[whole_page_shuiping[-1]:, :]
    bbox = [0, whole_page_shuiping[-1], borders.shape[1], borders.shape[0]]
    image_piece_after_whole_shuiping.append({'image_piece': image_piece,
                                             'bbox': bbox})
    # 第一次垂直分割
    inter_result_1_chuizhi = []
    for unit in image_piece_after_whole_shuiping:
        seperator = split_image(unit['image_piece'], 'chuizhi')
        for index, seperator_unit in enumerate(seperator):
            chuizhi_all.append([seperator_unit,
                                unit['bbox'][1],
                                seperator_unit,
                                unit['bbox'][3],])
            if index == 0:
                bbox = [0, unit['bbox'][1], seperator_unit, unit['bbox'][3]]
                image_piece = unit['image_piece'][:, :seperator_unit]
                inter_result_1_chuizhi.append({'bbox': bbox,
                                               'image_piece': image_piece})
            elif index > 0 and index <= len(seperator) - 1:
                bbox = [seperator[index-1], unit['bbox'][1], seperator_unit, unit['bbox'][3]]
                image_piece = unit['image_piece'][:, seperator[index-1]:seperator_unit]
                inter_result_1_chuizhi.append({'bbox': bbox,
                                               'image_piece': image_piece})

        bbox = [seperator[-1], unit['bbox'][1],  borders.shape[1], unit['bbox'][3]]
        image_piece = unit['image_piece'][:, seperator[-1]:]
        inter_result_1_chuizhi.append({'bbox': bbox,
                                       'image_piece': image_piece})
    for x in inter_result_1_chuizhi:
        drawer.rectangle(x['bbox'], outline='green', width=10)
    image_to_draw.save("temp/first_chuizhi.png")

    return



if __name__ == "__main__":
    image_path = r'../article_dataset/AS_TrainingSet_NLF_NewsEye_v2/576445_0003_23676257.jpg'
    split_link_by_name(image_path)

    # for unit in first_shuiping_group['annotation'][0]:
    #     drawer.rectangle(unit['bbox'][0] + unit['bbox'][2],
    #                      outline='red',
    #                      width=10)
    # for unit in first_shuiping_group['annotation'][1]:
    #     drawer.rectangle(unit['bbox'][0] + unit['bbox'][2],
    #                      outline='green',
    #                      width=10)
    # image_to_draw.save('temp/group.png')