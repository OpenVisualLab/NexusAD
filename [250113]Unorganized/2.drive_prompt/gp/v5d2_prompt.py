import json
import os
from pycocotools.coco import COCO
from tqdm import tqdm
import re
from collections import defaultdict
from PIL import Image
import sys
import numpy as np

sys.path.append("/data1/nemo/projects/spatial_ad/query_gen")
from prompts0811.v5d2_gp import (
    gen_query_description,
    query_task_1,
    query_task_2,
    query_task_3,
    # query_guide,
    # query_note
)


def read_jsonl(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def read_json(file_path):
    with open(file_path, "r") as file:
        return json.load(file)


def save_as_jsonl(data, output_file_path):
    with open(output_file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def read_txt_file(file_path):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            file_content = file.read()
        return file_content
    except Exception as e:
        print(f"Error reading file: {e}")
        return None


def get_subset_name(path):
    temp = path.split("/")
    return temp[0], temp[-1]


def find_key_for_value(value, data):
    for key, values_list in data.items():
        if value in values_list:
            return key
    return "other objects"


def key_map(key):
    mapmap = {
        "Vehicles": "vehicles",
        "Vulnerable Road Users": "vulnerable_road_users",
        "Traffic Signs": "traffic signs",
        "Traffic Lights": "traffic lights",
        "Traffic Cones": "traffic cones",
        "Barriers": "barriers",
        "Miscellaneous Objects": "other objects",
    }
    if key in mapmap.keys():
        return mapmap[key]
    else:
        return "other objects"



def remove_empty_entries(data):
    if isinstance(data, dict):
        return {
            k: remove_empty_entries(v)
            for k, v in data.items()
            if remove_empty_entries(v)
        }
    elif isinstance(data, list):
        return [remove_empty_entries(i) for i in data if remove_empty_entries(i)]
    else:
        return data


def clean_empty_second_level_entries(data):
    if isinstance(data, dict):
        result = {}
        for k, v in data.items():
            if isinstance(v, dict):
                # 只保留非空列表的键值对
                cleaned_sub_dict = {
                    ik: iv
                    for ik, iv in v.items()
                    if not (isinstance(iv, list) and len(iv) == 0)
                }
                result[k] = cleaned_sub_dict
            else:
                result[k] = v
        return result
    else:
        return data


def trans_objects(rec_objects, image_info):

    categorized_dict = read_json(
        "/data1/nemo/projects/spatial_ad/a_pin_jsons/object_cats.json"
    )["categories"]

    categorized_objects = {}

    for k, v in categorized_dict.items():
        categorized_objects.update({k: {}})
        for vv in v:
            categorized_objects[k].update({vv: []})
    
    categorized_objects.update({"other objects": {}})
     

    img_h, img_w = image_info["height"], image_info["width"]

    for super_cat, cat, bbox, corner_case in rec_objects:
        x, y, w, h = bbox
        x1, y1, x2, y2 = x / img_w, y / img_h, (x + w) / img_w, (y + h) / img_h
        x1, y1, x2, y2 = np.clip([x1, y1, x2, y2], 0, 1)
        
        x1, y1, x2, y2 = int(x1 * 1000), int(y1 * 1000), int(x2 * 1000), int(y2 * 1000)

        key = find_key_for_value(cat, categorized_dict)
        
        if cat not in categorized_objects[key].keys():
            categorized_objects[key].update({cat: []})
        categorized_objects[key][cat].append([x1, y1, x2, y2])

    return {key_map(key): value for key, value in categorized_objects.items()}

def depth_to_description(depth):
    return (
        "long range" if 0 <= depth < 50 else
        "mid range" if 50 <= depth < 150 else
        "short range" if 150 <= depth < 200 else
        "immediate" if 200 <= depth <= 255 else
        "unknown"
    )

def gen_objects_info(image_info, anns, categories):
    # description = f"**Detected Objects:**\n\nThe following categories of detected objects are present in the scene:\n\n"

    rec_objects = [
        [
            categories[ann["category_id"] - 1]["supercategory"],
            categories[ann["category_id"] - 1]["name"],
            ann["bbox"],
            ann.get("corner_case", False),
        ]
        for ann in anns
    ]

    formatted_targets = trans_objects(rec_objects, image_info)

    formatted_targets = clean_empty_second_level_entries(formatted_targets)

    depth_map_path = os.path.join(
        image_info["data_dir"],
        image_info["subset"],
        "depths2",
        image_info["file_name"].replace(".jpg", ".png"),
    )
    depth_map = Image.open(depth_map_path).convert("L")
    
    info_dict = {}
    for category, info in formatted_targets.items():
        info_dict[category] = {}
        for cat, cat_bbox in info.items():
            info_dict[category].update({cat: []})
            
            for bbox in cat_bbox:
                x1, y1, x2, y2 = bbox
                w,h = depth_map.size
                c1, c2 = int(w* (x2 + x1) / 2000), int(h*(y2 + y1) / 2000)
                # print(c1,c2,w,h, '-', bbox)
                depth = depth_map.getpixel((c1, c2))
                
                info_dict[category][cat].append({"bounding_box": bbox, "range": depth_to_description(depth)})

    return info_dict


def gen_traffic_description(image_info, anns, categories):

    query_description = gen_query_description(image_info)

    objects_info = gen_objects_info(image_info, anns, categories)

    query_final = query_description + str(objects_info) + '\n\n' + query_task_1

    info_json_path = os.path.join('/data1/nemo/projects/datasets/CODA/CODA-LM', image_info['sss'], image_info['subset']+'_'+image_info['file_name'].replace('.jpg', '.json'))
    super_answer = read_json(info_json_path)

    return query_final, super_answer


def calculate_median(numbers):
    sorted_numbers = sorted(numbers)
    count = len(sorted_numbers)
    if count == 0:
        return None
    mid = count // 2
    return (
        sorted_numbers[mid]
        if count % 2 == 1
        else (sorted_numbers[mid - 1] + sorted_numbers[mid]) / 2
    )


def process_data(
    ori_path, data_dir, output_file_path, test_anno_path, val_anno_path, sss="Test"
):
    data = read_jsonl(ori_path)
    coco_t = COCO(test_anno_path)
    coco_v = COCO(val_anno_path)
    categories = coco_t.loadCats(coco_t.getCatIds())
    img_ids_t = coco_t.getImgIds()
    img_ids_v = coco_v.getImgIds()

    new_data = []
    len_rec = []
    len_rec2 = []

    for item in tqdm(data):
        img_path = os.path.join(data_dir, item["image"])

        subset, file_name = get_subset_name(item["image"])
        coco = coco_t if subset == "test" else coco_v
        img_info = next(
            img
            for img in coco.loadImgs(img_ids_t if subset == "test" else img_ids_v)
            if img["file_name"] == file_name
        )

        ann_ids = coco.getAnnIds(imgIds=img_info["id"])
        anns = coco.loadAnns(ann_ids)

        img_info.update({"sss": sss})
        img_info.update({"subset": subset})
        img_info.update({"data_dir": data_dir})

        question, super_answer = gen_traffic_description(img_info, anns, categories)

        item["question"] = question
        
        item["gp"] = super_answer["general_perception"]['description and explanation']
        
        item["ds"] = super_answer["driving_suggestion"]
        
        item["question_gp"] = query_task_2
        item["question_ds"] = query_task_3
        
        gp_super_answer = super_answer["general_perception"]
        if 'description and explanation' in gp_super_answer:
            del gp_super_answer['description and explanation']

        item["struct"] = json.dumps(gp_super_answer)
                
        new_data.append(item)
        len_rec.append(len(question))
        if sss != "Test":
            len_rec2.append(len(item["answer"]))

            # processing image
        image_path_1 = item["image"]
        imgae_path_2 = item["image"].replace("images", "depths").replace(".jpg", ".png")
        # item["image"] = [image_path_1, imgae_path_2]
        item["image"] = image_path_1
        
    save_as_jsonl(new_data, output_file_path)

    print(
        max(len_rec),
        min(len_rec),
        sum(len_rec) / len(len_rec),
        calculate_median(len_rec),
    )
    if sss != "Test":
        print(
            max(len_rec2),
            min(len_rec2),
            sum(len_rec2) / len(len_rec2),
            calculate_median(len_rec2),
        )


if __name__ == "__main__":
    sss = ["Mini", "Test", "Train", "Val"][2]

    ori_path = f"/data1/nemo/projects/datasets/CODA/CODA-LM/{sss}/vqa_anno/general_perception.jsonl"
    data_dir = "/data1/nemo/projects/datasets/CODA"
    save_dir = "/data1/nemo/projects/datasets/CODA/lm0811"

    test_anno_path = "/data1/nemo/projects/datasets/CODA/test/annotations.json"
    val_anno_path = "/data1/nemo/projects/datasets/CODA/val/annotations.json"

    subsub, jsonl_name = ori_path.split("/")[-3], ori_path.split("/")[-1]
    output_file_path = os.path.join(save_dir, subsub, "v5d2_struct_gp_ds.jsonl")

    process_data(
        ori_path, data_dir, output_file_path, test_anno_path, val_anno_path, sss=sss
    )
