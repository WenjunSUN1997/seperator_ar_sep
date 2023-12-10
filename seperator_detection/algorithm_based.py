import os

import cv2
import numpy as np
from tqdm import tqdm
from PIL import Image, ImageDraw
import math
import networkx as nx
from xml_reader import XmlProcessor
import pickle

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
        for index in range(0, image_pice.shape[0], 30):

                img_slice = image_pice[index:index+60, :]
                if calcul_length(img_slice, direction) and len(split_location)==0:
                    split_location.append(index)
                elif calcul_length(img_slice, direction) and len(split_location)>0:
                    if index - split_location[-1] > 200:
                        split_location.append(index)
    else:
        for index in range(0, image_pice.shape[1], 30):
            img_slice = image_pice[:, index:index + 60]
            if calcul_length(img_slice, direction) and len(split_location) == 0:
                split_location.append(index)
            elif calcul_length(img_slice, direction) and len(split_location) > 0:
                if index - split_location[-1] > 200:
                    split_location.append(index)

    return split_location

def remove_elements_by_indices(lst, indices):
    indices_set = set(indices)  # 将索引列表转换为集合，提高查找效率
    return [element for index, element in enumerate(lst) if index not in indices_set]

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

def judge_adjacent(center, chuizhi, shuiping):
    left = 0
    right = 0
    top = 0
    bottom = 0
#     寻找left
    mini = 1000000
    for key, value in chuizhi.items():
        if value[1] < center[1] and value[3] > center[1] and value[0] < center[0]:
            distance = center[0] - value[0]
            if distance < mini:
                mini = distance
                left = key

#     寻找right
    mini = 1000000
    for key, value in chuizhi.items():
        if value[1] < center[1] and value[3] > center[1] and value[0] > center[0]:
            distance = value[0] - center[0]
            if distance < mini:
                mini = distance
                right = key

#     寻找top
    mini = 1000000
    for key, value in shuiping.items():
        if value[0] < center[0] and value[2] > center[0] and center[1] > value[1]:
            distance = center[1] - value[1]
            if distance < mini:
                mini = distance
                top = key

#     寻找bottom
    mini = 1000000
    for key, value in shuiping.items():
        if value[0] < center[0] and value[2] > center[0] and center[1] < value[1]:
            distance = value[1] - center[1]
            if distance < mini:
                mini = distance
                bottom = key

    return (left, right, top, bottom)

def split_link_by_name(image_path:str, save_path):
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
    # cv2.imwrite("temp/gray.png", binary)
    h, w = binary.shape
    hors_k = int(math.sqrt(w) * 1.2)
    vert_k = int(math.sqrt(h) * 1.2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (hors_k, 1))
    hors = ~cv2.dilate(binary, kernel, iterations=1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vert_k))
    verts = ~cv2.dilate(binary, kernel, iterations=1)
    borders = cv2.bitwise_or(hors, verts)
    cv2.imwrite(save_path+"borders .png", borders)
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

    if len(whole_page_shuiping) > 0:
        image_piece = borders[whole_page_shuiping[-1]:, :]
        bbox = [0, whole_page_shuiping[-1], borders.shape[1], borders.shape[0]]
    else:
        image_piece = borders[:, :]
        bbox = [0, 0, borders.shape[1], borders.shape[0]]

    image_piece_after_whole_shuiping.append({'image_piece': image_piece,
                                             'bbox': bbox})
    # 第一次垂直分割
    inter_result_1_chuizhi = []
    for unit in image_piece_after_whole_shuiping:
        seperator = split_image(unit['image_piece'], 'chuizhi')
        if len(seperator) == 0 or (len(seperator)==0 and seperator[0]==0):
            continue

        for index, seperator_unit in enumerate(seperator):
            chuizhi_all.append([seperator_unit,
                                unit['bbox'][1],
                                seperator_unit,
                                unit['bbox'][3]])
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
    # for x in inter_result_1_chuizhi:
    #     drawer.rectangle(x['bbox'], outline='green', width=10)
    # image_to_draw.save("temp/first_chuizhi.png")
    # 二次水平分割
    inter_result_1_shuiping = []
    for unit in inter_result_1_chuizhi:
        shuiping_seprator = split_image(unit['image_piece'], 'shuiping')
        if len(shuiping_seprator) == 0:
            continue

        for index, seperator_unit in enumerate(shuiping_seprator):
            shuiping_all.append([unit['bbox'][0],
                                 unit['bbox'][1]+seperator_unit,
                                 unit['bbox'][2],
                                 unit['bbox'][1]+seperator_unit])
            if index == 0:
                image_piece = unit['image_piece'][:seperator_unit, :]
                bbox = [unit['bbox'][0],
                        unit['bbox'][1],
                        unit['bbox'][2],
                        unit['bbox'][1]+seperator_unit]
                inter_result_1_shuiping.append({'bbox': bbox,
                                                'image_piece': image_piece})

            elif index != 0 and index <= len(shuiping_seprator) - 1:
                image_piece = unit['image_piece'][shuiping_seprator[index-1]:shuiping_seprator[index], :]
                bbox = [unit['bbox'][0],
                        unit['bbox'][1]+shuiping_seprator[index-1],
                        unit['bbox'][2],
                        unit['bbox'][1]+shuiping_seprator[index]]
                inter_result_1_shuiping.append({'bbox': bbox,
                                                'image_piece': image_piece})

        bbox = [unit['bbox'][0],
                unit['bbox'][1]+shuiping_seprator[-1],
                unit['bbox'][2],
                unit['bbox'][3]]
        image_piece = unit['image_piece'][shuiping_seprator[-1]:, :]
        inter_result_1_shuiping.append({'bbox': bbox,
                                        'image_piece': image_piece})

    #     二次垂直分割
    for unit in inter_result_1_shuiping:
        chuizhi_seprator = split_image(unit['image_piece'], 'chuizhi')
        if len(chuizhi_seprator) == 0 or (len(chuizhi_seprator)==0 and chuizhi_seprator[0]==0):
            continue

        for chuizhi_seprator_unit in chuizhi_seprator:
            chuizhi_all.append([unit['bbox'][0]+chuizhi_seprator_unit,
                                unit['bbox'][1],
                                unit['bbox'][0]+chuizhi_seprator_unit,
                                unit['bbox'][3]])
    # 二次分割完成， 开始后续处理
    final_chuizhi_all = []
    while len(chuizhi_all) > 0:
        index_remove = [0]
        for index in range(1, len(chuizhi_all)):
            if abs(chuizhi_all[index][0] - chuizhi_all[0][0]) < 80:
                index_remove.append(index)

        length = [abs(chuizhi_all[x][1] - chuizhi_all[x][3]) for x in index_remove]
        max_index = length.index(max(length))
        final_chuizhi_all.append(chuizhi_all[index_remove[max_index]])
        chuizhi_all = remove_elements_by_indices(chuizhi_all, index_remove)

    final_shuiping_all = []
    while len(shuiping_all) > 0:
        index_remove = [0]
        for index in range(1, len(shuiping_all)):
            if abs(shuiping_all[index][1] - shuiping_all[0][1]) < 80:
                index_remove.append(index)

        length = [abs(shuiping_all[x][0] - shuiping_all[x][2]) for x in index_remove]
        max_index = length.index(max(length))
        final_shuiping_all.append(shuiping_all[index_remove[max_index]])
        shuiping_all = remove_elements_by_indices(shuiping_all, index_remove)

    index_remove = []
    for index in range(len(final_chuizhi_all)):
        if final_chuizhi_all[index][0] == final_chuizhi_all[index][2] \
                and final_chuizhi_all[index][1] == final_chuizhi_all[index][3]:
            index_remove.append(index)

    final_chuizhi_all = remove_elements_by_indices(final_chuizhi_all, index_remove)
    for index, x in enumerate(final_chuizhi_all):
        drawer.line(x, 'green', width=10)
        drawer.text((x[0], x[1]), str(index))

    for index, x in enumerate(final_shuiping_all):
        drawer.line(x, 'red', width=10)
        drawer.text((x[0], x[1]), str(index))

    image_to_draw.save(save_path+"final_lines.png")
    final_chuizhi_all = {index: value for index, value in enumerate(final_chuizhi_all)}
    final_shuiping_all = {index: value for index, value in enumerate(final_shuiping_all)}
    # 开始为各个annotation分配边
    for annotation in annotation_list:
        annotation['sign'] = judge_adjacent(annotation['center'],
                                            final_chuizhi_all,
                                            final_shuiping_all)
    # 按照边进行分组
    sign_set = set(x['sign'] for x in annotation_list)
    for index, sign in enumerate(sign_set):
        if index % 2 == 0:
            outline = 'red'
        else:
            outline = 'green'
        for annotation in annotation_list:
            if annotation['sign'] == sign:
                drawer.rectangle(annotation['bbox'][0]+annotation['bbox'][2],
                                 outline=outline,
                                 width=4)
                drawer.text(annotation['center'], str(annotation['sign']))

    # image_to_draw.save(save_path+"final_group.png")
    groups = []
    for index, sign in enumerate(sign_set):
        temp = []
        for annotation in annotation_list:
            if annotation['sign'] == sign:
                temp.append(annotation)

        groups.append(temp)

    for group_unit in groups:
        group_unit = sorted(group_unit, key=lambda x: (x['center'][0] // 300, x['center'][1]))
        for index in range(len(group_unit) - 1):
            links.append([group_unit[index]['index'],
                          group_unit[index+1]['index']])

    for chuizhi_line_index in range(len(final_chuizhi_all)):
        chuizhi_top_point = [final_chuizhi_all[chuizhi_line_index][0],
                             final_chuizhi_all[chuizhi_line_index][1]]
        chuizhi_bottom_point = [final_chuizhi_all[chuizhi_line_index][2],
                             final_chuizhi_all[chuizhi_line_index][3]]
        left_block = [x for x in annotation_list
                      if x['sign'][1] == chuizhi_line_index]
        try:
            mini_left = min(left_block, key=lambda x: (x['center'][0]-chuizhi_bottom_point[0]) ** 2
                                                      + (x['center'][1]-chuizhi_bottom_point[1]) ** 2)
        except:
            continue

        right_block = [x for x in annotation_list
                       if x['sign'][0] == chuizhi_line_index]
        try:
            mini_right = min(right_block, key=lambda x: (x['center'][0]-chuizhi_top_point[0]) ** 2
                                                      + (x['center'][1]-chuizhi_top_point[1]) ** 2)
        except:
            continue

        links.append([mini_left['index'], mini_right['index']])

    # 开始链接
    for link_unit in links:
        start_point = [x['center'] for x in annotation_list if x['index'] == link_unit[0]]
        end_point = [x['center'] for x in annotation_list if x['index'] == link_unit[1]]
        drawer.line(start_point[0] + end_point[0],
                    'red',
                    width=8)

    image_to_draw.save(save_path+'final_link.png')
    precit_nx = nx.Graph()
    precit_nx.add_nodes_from([(annotation['index'], annotation) for annotation in annotation_list])
    precit_nx.add_edges_from([(x[0], x[1]) for x in links])
    pickle.dump(precit_nx, open(save_path+'nx.nx', 'wb'))
    return

if __name__ == "__main__":
    error = 0
    save_path = 'result/'
    os.makedirs(save_path, exist_ok=True)
    image_path_list = [x for x in os.listdir('../article_dataset/AS_TrainingSet_NLF_NewsEye_v2/')
                       if '.jpg' in x]
    for image_path in tqdm(image_path_list):
        print(image_path)
        os.makedirs(save_path+image_path.replace('.jpg', '/'), exist_ok=True)
        try:
            split_link_by_name(image_path='../article_dataset/AS_TrainingSet_NLF_NewsEye_v2/'+image_path,
                               save_path=save_path+image_path.replace('.jpg', '/'))
        except:
            os.removedirs(save_path+image_path.replace('.jpg', '/'))
            error += 1
            print(error)

    # for unit in first_shuiping_group['annotation'][0]:
    #     drawer.rectangle(unit['bbox'][0] + unit['bbox'][2],
    #                      outline='red',
    #                      width=10)
    # for unit in first_shuiping_group['annotation'][1]:
    #     drawer.rectangle(unit['bbox'][0] + unit['bbox'][2],
    #                      outline='green',
    #                      width=10)
    # image_to_draw.save('temp/group.png')