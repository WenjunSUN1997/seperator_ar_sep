import os
import shutil
from pycocotools.coco import COCO
import torch
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import numpy as np
import json
import albumentations

def create_hf_dataset(coco_path, lang):
    image_id_list = []
    image_list = []
    path_list = []
    width_list = []
    height_list = []
    objects_list = []
    if lang == 'fr':
        base_path = r'../../data/AS_TrainingSet_BnF_NewsEye_v2/'
    else:
        base_path = r'../../data/AS_TrainingSet_NLF_NewsEye_v2/'

    coco = COCO(coco_path)
    imgid_list = coco.getImgIds()
    for img_id in imgid_list:
        image_id_list.append(img_id)
        anno_dict_this_img = {'id': [],
                              'area': [],
                              'bbox': [],
                              'category': []}
        image_dict = coco.loadImgs([img_id])[0]
        path_list.append(image_dict['path'].split('/')[-1])
        image_list.append(Image.open(base_path + image_dict['path'].split('/')[-1]))
        width_list.append(image_dict['width'])
        height_list.append(image_dict['height'])
        anno_ids_this_img = coco.getAnnIds(imgIds=[img_id])
        for anno_unit in coco.loadAnns(anno_ids_this_img):
            anno_dict_this_img['id'].append(anno_unit['id'])
            anno_dict_this_img['area'].append(anno_unit['area'])
            anno_dict_this_img['bbox'].append(anno_unit['bbox'])
            anno_dict_this_img['category'].append(anno_unit['category_id'])

        objects_list.append(anno_dict_this_img)

    return {'image': image_list,
            'path': path_list,
            'image_id': image_id_list,
            'width': width_list,
            'height': height_list,
            'objects': objects_list}

def formatted_anns(image_id, category, area, bbox):
    annotations = []
    for i in range(0, len(category)):
        new_ann = {
            "image_id": image_id,
            "category_id": category[i],
            "isCrowd": 0,
            "area": area[i],
            "bbox": list(bbox[i]),
        }
        annotations.append(new_ann)

    return annotations

def transform_aug_ann(image_processor):
    def __transform_aug_ann(examples):
        transform = albumentations.Compose(
                [
                    albumentations.Resize(5000, 7000)
                ],
                bbox_params=albumentations.BboxParams(format="coco", label_fields=["category"]),
            )
        image_ids = examples["image_id"]
        images, bboxes, area, categories = [], [], [], []
        for image, objects in zip(examples["image"], examples["objects"]):
            image = np.array(image.convert("RGB"))[:, :, ::-1]
            out = transform(image=image, bboxes=objects["bbox"], category=objects["category"])
            area.append(objects["area"])
            images.append(out["image"])
            bboxes.append(out["bboxes"])
            categories.append(out["category"])

        targets = [
            {"image_id": id_, "annotations": formatted_anns(id_, cat_, ar_, box_)}
            for id_, cat_, ar_, box_ in zip(image_ids, categories, area, bboxes)
        ]

        return image_processor(images=images, annotations=targets, return_tensors="pt")

    return __transform_aug_ann

def collate_fn(device):
    def __collate_fn(batch):
        result = {}
        pixel_values = torch.stack([item["pixel_values"] for item in batch]).to(device)
        try:
            pixel_mask = torch.stack([item["pixel_mask"] for item in batch]).to(device)
            result["pixel_mask"] = pixel_mask
        except:
            pass
        labels = [item["labels"] for item in batch]
        for label_unit in labels:
            for key, value in label_unit.items():
                label_unit[key] = value.to(device)
        result["pixel_values"] = pixel_values
        result["labels"] = labels
        return result

    return __collate_fn

def compute_metrics(result):
    predictions, label_ids = result
    print()

def draw(path, bbox_list):
    img = Image.open(path).convert('RGB')
    draw = ImageDraw.Draw(img, "RGB")
    for index, bbox in enumerate(bbox_list):
        print(bbox)
        draw.line(bbox, 'green', width=10)
        # draw.rectangle(bbox,
        #                fill=(0, 0, 255),
        #                outline='red',
        #                width=1)

    # img.save('../temp/sepetator_example.png')
    plt.imshow(img)
    plt.show()
    return 0

def prepare_data_for_mlsd():
    # os.mkdir('../../temp/wireframe_raw')
    # os.mkdir('../../temp/wireframe_raw/images')
    data_dict = create_hf_dataset('../../data/coco-1701423132.2024868.json', 'fr')
    loop = 1
    mlsd_format = []
    for loop_index in range(loop):
        print(loop_index)
        for data_index in range(len(data_dict['path'])):
            filename = data_dict['path'][data_index]
            # shutil.copyfile('../../data/AS_TrainingSet_NLF_NewsEye_v2/'+filename,
            #                 '../../temp/wireframe_raw/images/' + filename.split('.')[0] + '_' + str(loop_index)+'.jpg')
            lines = data_dict['objects'][data_index]['bbox']
            draw('../../data/AS_TrainingSet_BnF_NewsEye_v2/'+filename, lines)
            width = data_dict['width'][data_index]
            height = data_dict['height'][data_index]
            mlsd_format.append({'filename': filename.split('.')[0] + '_' + str(loop_index)+'.jpg',
                                'lines': lines,
                                'width': width,
                                'height': height})

    json_data = json.dumps(mlsd_format)
    # with open('../../temp/wireframe_raw/ann.json', 'w') as file:
    #     file.write(json_data)

def prepare_data_fclip(coco_path = '../../data/coco-1701461784.0087163.json',
                       lang = 'fi',
                       store_path = '../../temp/fi/'):
    os.makedirs(store_path, exist_ok=True)
    os.makedirs(store_path+'images/', exist_ok=True)
    if lang == 'fi':
        base_path = '../../data/AS_TrainingSet_NLF_NewsEye_v2/'
    else:
        base_path = '../../data/AS_TrainingSet_BnF_NewsEye_v2/'

    fclip_format = []
    coco = COCO(coco_path)
    imgid_list = coco.getImgIds()
    for img_id in imgid_list:
        lines = []
        image_dict = coco.loadImgs([img_id])[0]
        path = image_dict['path'].split('/')[-1]
        width = image_dict['width']
        height = image_dict['height']
        anno_ids_this_img = coco.getAnnIds(imgIds=[img_id])
        for anno_unit in coco.loadAnns(anno_ids_this_img):

            lines.append([anno_unit['bbox'][0],
                          anno_unit['bbox'][1],
                          anno_unit['bbox'][0]+anno_unit['bbox'][2],
                          anno_unit['bbox'][1]+anno_unit['bbox'][3]])

        # draw('../../data/AS_TrainingSet_BnF_NewsEye_v2/' + path, lines)
        shutil.copyfile(base_path+path,
                        store_path+'images/'+path)
        fclip_format.append({'filename': path,
                            'lines': lines,
                            'width': width,
                            'height': height})

    json_data = json.dumps(fclip_format)
    with open(store_path+'anno.json', 'w') as file:
        file.write(json_data)


    print('done')

if __name__ == "__main__":
    prepare_data_fclip()