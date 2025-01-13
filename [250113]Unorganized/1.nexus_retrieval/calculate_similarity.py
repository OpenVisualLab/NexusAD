import os
from tqdm import tqdm
import json
import heapq
import numpy 
import matplotlib.pyplot as plt
import cv2
import math
import concurrent.futures
import torch
import torch.nn.functional as F


os.environ["CUDA_VISIBLE_DEVICES"] = "3"


def save_data(data, filename):
    with open(filename, 'w') as f:
        json.dump(data, f)

def load_data(filename):
    with open(filename, 'r') as f:
        return json.load(f)
    

def load_pooler_outputs(output_folder):
    pooler_outputs = {}
    for filename in tqdm(os.listdir(output_folder)):
        if filename.endswith('_pooler_output.pt'):
            base_filename = os.path.splitext(filename)[0].replace('_pooler_output', '')
            pooler_output_path = os.path.join(output_folder, filename)
            pooler_output = torch.load(pooler_output_path, map_location='cpu')

            pooler_outputs[base_filename] = pooler_output

    return pooler_outputs

def save_as_jsonl(data, output_file_path):
    with open(output_file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
def load_detection_results(file_path):
    detection_results = []
    with open(file_path, 'r') as file:
        for line in file:
            detection_results.append(json.loads(line))
    return detection_results

def visualize_and_save_images(original_image_path, top_k_images, save_path):
    top_k = len(top_k_images)
    cols = 3
    rows = math.ceil(top_k / cols) + 1

    fig, axes = plt.subplots(rows, cols, figsize=(15, 3 * rows))

    original_image = cv2.imread(original_image_path)
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
    original_image_name = os.path.basename(original_image_path)
    
    subset1 = original_image_path.split('/')[-3]
    
    axes[0, 1].imshow(original_image_rgb)
    axes[0, 1].set_title(f"Original: {subset1}: {original_image_name}")
    axes[0, 1].axis('off')

    if cols > 1:
        axes[0, 0].axis('off')
        axes[0, 2].axis('off')

    for i, image_info in enumerate(top_k_images):
        image_path = image_info['image_path']
        similarity_score = image_info['similarity_score']
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_name = os.path.basename(image_path)
        subset2 = image_path.split('/')[-3]
        
        row = (i // cols) + 1
        col = i % cols
        axes[row, col].imshow(image_rgb)
        axes[row, col].set_title(f"Top {i+1} Score: {similarity_score:.6f} \n{subset2}: {image_name}")
        axes[row, col].axis('off')

    # 隐藏多余的子图
    for j in range(i + 1, rows * cols):
        row = (j // cols)
        col = j % cols
        if row < axes.shape[0] and col < axes.shape[1]:
            axes[row, col].axis('off')

    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close(fig)
    

def calculate_weather_time_similarity(result1, result2):
    # 提取天气和时间信息
    weather1 = result1.get('weather', '')
    weather2 = result2.get('weather', '')
    time1 = result1.get('period', '')
    time2 = result2.get('period', '')
    
    # 简单匹配天气和时间，如果完全相同，相似度为1.0，否则为0.0
    weather_similarity = 1.0 if weather1 == weather2 else 0.0
    time_similarity = 1.0 if time1 == time2 else 0.0
    
    # 合并相似度，给出权重（如果需要）
    return 0.4 * weather_similarity + 0.6 * time_similarity

def calculate_iou_with_depth(box1, box2, depth1, depth2):
    x_left = max(box1[0], box2[0])
    y_top = max(box1[1], box2[1])
    x_right = min(box1[2], box2[2])
    y_bottom = min(box1[3], box2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box1[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - intersection_area

    iou = intersection_area / union_area

    depth_difference = abs(depth1 - depth2)
    max_depth_difference = 128
    depth_similarity = 1 - min((depth_difference / max_depth_difference) ** 2, 1)

    # 增加容忍度，减少深度差异的影响
    combined_similarity = (0.05 * iou) + (0.95 * depth_similarity)
    return combined_similarity


def compare_objects(obj_list1, obj_list2):
    similarities = []
    for obj1 in obj_list1:
        best_similarity = 0.0
        for obj2 in obj_list2:
            iou_depth_similarity = calculate_iou_with_depth(
                obj1['bounding_box'], obj2['bounding_box'],
                obj1['depth'], obj2['depth']
            )
            best_similarity = max(best_similarity, iou_depth_similarity)
        similarities.append(best_similarity)
    overall_similarity = sum(similarities) / len(similarities) if similarities else 0.0
    return overall_similarity


def compare_category(result1, result2):
    categories1 = result1.keys()
    categories2 = result2.keys()
    
    similarities = []
    for large_category in categories1:
        if large_category in categories2:
            subcategories1 = result1[large_category]
            subcategories2 = result2[large_category]
            for sub_category in subcategories1:
                if sub_category in subcategories2:
                    similarity = compare_objects(subcategories1[sub_category], subcategories2[sub_category])
                    similarities.append(similarity)
             
    category_similarity = sum(similarities) / len(similarities) if similarities else 0.0

    return category_similarity

def calculate_category_overlap(result1, result2):
    # 定义类别权重，降低car的权重
    category_weights = {
        "vehicles/car": 0.5,  # 降低car的权重
        "vehicles/bus": 1.0,  # 其他类别使用默认权重
        "traffic lights/traffic_light": 3.0
    }
    
    def extract_categories(detection, include_subcategories=True):
        categories = set()
        for major_category, subcategories in detection.items():
            if subcategories:
                if include_subcategories:
                    for subcategory in subcategories.keys():
                        categories.add(f"{major_category}/{subcategory}")
                else:
                    categories.add(major_category)
        return categories

    # 提取小类别和大类别
    categories1 = extract_categories(result1, include_subcategories=True)
    categories2 = extract_categories(result2, include_subcategories=True)
    
    major_categories1 = extract_categories(result1, include_subcategories=False)
    major_categories2 = extract_categories(result2, include_subcategories=False)

    # 计算小类别重合度
    small_overlap = categories1.intersection(categories2)
    small_total_categories = categories1.union(categories2)

    weighted_small_overlap = 0
    weighted_small_total = 0

    for category in small_total_categories:
        weight = category_weights.get(category, 1.0)
        weighted_small_total += weight
        if category in small_overlap:
            weighted_small_overlap += weight

    small_overlap_score = weighted_small_overlap / weighted_small_total if weighted_small_total else 1.0

    # 计算大类别重合度
    major_overlap = major_categories1.intersection(major_categories2)
    major_total_categories = major_categories1.union(major_categories2)

    major_overlap_score = len(major_overlap) / len(major_total_categories) if major_total_categories else 1.0

    # 最终重合度得分，可以调整权重
    final_overlap_score = 0.8 * small_overlap_score + 0.2 * major_overlap_score

    return final_overlap_score, small_overlap, major_overlap



test_vectors = load_pooler_outputs('/data1/nemo/projects/spatial_ad/aaa_com/pts/test')
val_vectors = load_pooler_outputs('/data1/nemo/projects/spatial_ad/aaa_com/pts/val')

def requir_pooler_outputs(result):
    img_n = result['image']
    base_filename = os.path.splitext(os.path.basename(img_n))[0]
    if 'test' in img_n:
        return test_vectors[base_filename]
    elif 'val' in img_n:
        return val_vectors[base_filename]
    else:
        return None
    

def calculate_image_similarity(result1, result2):
    tensor1 = requir_pooler_outputs(result1)
    tensor2 = requir_pooler_outputs(result2)
    
    similarity = F.cosine_similarity(tensor1, tensor2)
    # print(similarity.item())

    return similarity.item()

def combine_values(A, B):
    weight_A = 8 + 8 * (A ** 2)  
    weight_B = 2 + 2 * (B ** 2)  

    return (weight_A * A + weight_B * B) / (weight_A + weight_B)

def calculate_similarity(result1, result2):
    
    detection1 = result1['detection']
    detection2 = result2['detection']
    
    image_similarity = calculate_image_similarity(result1, result2)
    
    category_overlap_score, o1, o2 = calculate_category_overlap(detection1, detection2)
    # print(category_overlap_score, overlap)
    # aa
    
    category_similarity = compare_category(detection1, detection2)
    
    weather_time_similarity = calculate_weather_time_similarity(result1, result2)
    
    det_similarity = (
        0.4 * category_overlap_score +
        0.4 * category_similarity +
        0.2 * weather_time_similarity
    )
    
    if image_similarity > 0.99:
        final_similarity = (
            0.99 * image_similarity + 
            0.01 * det_similarity
        )
    elif image_similarity > 0.98:
        final_similarity = (
            0.9 * image_similarity + 
            0.1 * det_similarity
        )
    elif image_similarity > 0.97:
        final_similarity = (
            0.8 * image_similarity + 
            0.2 * det_similarity
        )
    else:
        final_similarity = (
            0.2 * image_similarity + 
            0.8 * det_similarity
        ) 
    
    return final_similarity

def find_top_k_similar(querys, bases, top_k=5, subsub= 'mini'):
    data_dir = '/data1/nemo/projects/datasets/CODA'
    
    gen_prompt_list = []
    for query in tqdm(querys):
        
        results = []
        for base in bases:
            similarity_score = calculate_similarity(query, base)
            results.append({
                # 'base_id': base['question_id'],
                # 'base_image': base['image'],
                'similarity_score': similarity_score,
                'base': base
            })
        
        top_k_results = heapq.nlargest(top_k, results, key=lambda x: x['similarity_score'])
        gen_prompt_list.append({'query': query, 'topk': top_k_results})
        
        top_k_images = []
        for rank, result in enumerate(top_k_results, 1):
            top_k_images.append({'image_path': os.path.join(data_dir, result['base']['image']), 'similarity_score': result['similarity_score']})
        
        original_image_path = os.path.join(data_dir, query['image'])
        save_path = os.path.join('/data1/nemo/projects/spatial_ad/aaa_com/find_s_vis/', f'{subsub}_final2', os.path.basename(original_image_path))
        # print(save_path) 

        visualize_and_save_images(original_image_path, top_k_images[:9], save_path)
        
    save_as_jsonl(gen_prompt_list, f'/data1/nemo/projects/spatial_ad/aaa_com/d_rank/{subsub}_v8_rank10.jsonl')
    
        
if __name__ =="__main__":
    sub = ['Mini', 'Test'][1]
    
    subsub = 'mini' if sub =='Mini' else 'test'
    
    # sub, subsub = 'Val', 'val'
    
    query_d_path = f'/data1/nemo/projects/spatial_ad/aaa_com/detections/{sub}_detection.jsonl' 
    base_d_path_1 = '/data1/nemo/projects/spatial_ad/aaa_com/detections/Train_detection.jsonl'
    base_d_path_2 = '/data1/nemo/projects/spatial_ad/aaa_com/detections/Val_detection.jsonl'
    
    querys = load_detection_results(query_d_path)

    bases = load_detection_results(base_d_path_1)+load_detection_results(base_d_path_2)
    # bases = load_detection_results(base_d_path_1)
    
    find_top_k_similar(querys, bases, top_k=10, subsub=subsub)
