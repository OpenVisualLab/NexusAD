import json
import pandas as pd
import numpy as np
from transformers import pipeline, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from glob import glob
from tqdm import tqdm
import heapq


def image_path_list(data_root, subset):
    jsons_dir = os.path.join(data_root, 'CODA-LM', subset)
    jsons_list = sorted(glob(os.path.join(jsons_dir, '*.json')))
    img_set_and_ids = [os.path.basename(f)[:-5].split('_') for f in jsons_list]
    return [os.path.join(data_root, a, 'images', b+'.jpg') for a,b in img_set_and_ids] 
    
def save_to_jsonl(data, file_path):
    with open(file_path, 'w') as jsonl_file:
        for item in data:
            jsonl_file.write(json.dumps(item) + '\n')
    print(f"Data saved to {file_path}")

def read_json(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data

def path_2_json_des(image_path, embedder, subset='Test'):
    data_root = '/data2/datasets/drivelm/data/coda'
    subsub, img_id = image_path.split('/')[-3], image_path.split('/')[-1][:-4]
    
    json_path = os.path.join(data_root, 'CODA-LM', subset, subsub+'_'+img_id+'.json')
    info = read_json(json_path)
    
    gp_info = info['general_perception']['description and explanation']
    # gp_info.pop('description and explanation')

    return np.mean(embedder(str(gp_info)[:512])[0],axis=0)


def nexus_ad(info, embedder):
    gp_info = ['general_perception']
    cat_list = ["vehicles", "vulnerable_road_users", "traffic signs", "traffic lights", "traffic cones", "barriers", "other objects"]
    
    for cat in cat_list:
        for 
        
    
    pass
    

def path_2_json(image_path, embedder, subset='Test'):
    data_root = '/data2/datasets/drivelm/data/coda'
    subsub, img_id = image_path.split('/')[-3], image_path.split('/')[-1][:-4]
    
    json_path = os.path.join(data_root, 'CODA-LM', subset, subsub+'_'+img_id+'.json')
    info = read_json(json_path)
    
    gp_info = info['driving_suggestion']
    # gp_info.pop('description and explanation')

    return np.mean(embedder(str(gp_info)[:512])[0],axis=0)



def cosine_similarity(vec1, vec2):

    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)
  
  
def get_subset_paths(embedder):
    subset = ['Train', 'Val', 'Test', 'Mini']
    data_root = '/data2/datasets/drivelm/data/coda'
    
    info_list = {}
    for sub in subset:
        paths = image_path_list(data_root, sub)
        info_list[sub] = [[path_2_json(path, embedder, sub), path] for path in tqdm(paths)]
    return info_list
    

def gen_rank(querys, bases, save_file_path, top_k=10):
    gen_prompt_list = []
    for query, query_img_path in tqdm(querys, desc="Processing queries"):
        
        results = []
        for base, base_img_path in bases:
            similarity = cosine_similarity(query, base)
            results.append({'similarity_score': similarity, 'base': base_img_path})
        
        top_k_results = heapq.nlargest(top_k, results, key=lambda x: x['similarity_score'])
        gen_prompt_list.append({'query': query_img_path, 'topk': top_k_results})
            
    save_to_jsonl(gen_prompt_list, save_file_path)


def main_final_gp(top_k=10):  #TODO
    embedder = pipeline('feature-extraction', model='/data1/nemo/projects/pre_weights/bert', device=1)
    
    all_infos = get_subset_paths(embedder)
    train_infos, val_infos, test_infos = all_infos['Train'], all_infos['Val'], all_infos['Test']
    
    save_dir = '/data2/datasets/drivelm/similarity_rank/bert_ds'
    
    save_path = os.path.join(save_dir, 'test_t_v_top_k.jsonl')
    gen_rank(test_infos, train_infos+val_infos, save_path)
    
    save_path = os.path.join(save_dir, 'train_v_top_k.jsonl')
    gen_rank(train_infos, val_infos, save_path)
    
    save_path = os.path.join(save_dir, 'val_t_top_k.jsonl')
    gen_rank(val_infos, train_infos, save_path)
    
    
if __name__ == "__main__":
    main_final_gp()
