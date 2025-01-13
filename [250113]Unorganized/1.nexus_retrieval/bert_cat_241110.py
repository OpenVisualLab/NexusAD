import json
import pandas as pd
import numpy as np
from transformers import pipeline, BertTokenizer
from sklearn.metrics.pairwise import cosine_similarity
import os
from glob import glob
from tqdm import tqdm
import heapq
import bert_score


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


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)


def nexus_ad(query_info, base_info):
    gp_q, gp_b = query_info['general_perception'], base_info['general_perception']
    cat_list = ["vehicles", "vulnerable_road_users", "traffic signs", "traffic lights", "traffic cones", "barriers", "other objects"]
    
    sim_rec = {}
    for cat in cat_list:
        cand = [str(aaa) for aaa in gp_q[cat]]
        ref = [str(bbb) for bbb in gp_b[cat]]
        
        if not cand or not ref:
            sim_rec[cat] = -1 if cand or ref else 0
            continue
        
        temp_list_1 = []
        
        for c in cand:
            temp_list = []
            for r in ref:
                P, R, F1 = bert_score.score([c], [r], lang="en", 
                                            verbose=False, 
                                            model_type='bert-large-uncased',
                                            device = 'cuda:1')
            temp_list.append(F1.numpy())
            temp_list_1.append(max(temp_list))
            
        sim_rec[cat] = np.mean(np.array(temp_list_1))
    
    filtered_values = [v for v in sim_rec.values() if v != 0]
    if filtered_values:
        overall_similarity = sum(filtered_values) / len(filtered_values)
    else:
        overall_similarity = 0    
        
    return overall_similarity
    

def path_2_json(qb):
    image_path, subset = qb
    data_root = '/data2/datasets/drivelm/data/coda'
    subsub, img_id = image_path.split('/')[-3], image_path.split('/')[-1][:-4]
    
    json_path = os.path.join(data_root, 'CODA-LM', subset, subsub+'_'+img_id+'.json')
    info = read_json(json_path)

    return info


def get_subset_paths():
    subset = ['Train', 'Val', 'Test', 'Mini']
    data_root = '/data2/datasets/drivelm/data/coda'
    
    info_list = {}
    for sub in subset:
        paths = image_path_list(data_root, sub)
        info_list[sub] = [(p, sub) for p in paths]
    return info_list
    

def gen_rank(querys, bases, save_file_path, top_k=10):
    gen_prompt_list = []
    for query in tqdm(querys, desc="Processing queries"):
        results = []
        query_info = path_2_json(query)
        for base in tqdm(bases):
            base_info = path_2_json(base)
            
            similarity = nexus_ad(query_info, base_info)
            
            results.append({'similarity_score': similarity, 'base': base[0]})
        
        top_k_results = heapq.nlargest(top_k, results, key=lambda x: x['similarity_score'])
        gen_prompt_list.append({'query': query[0], 'topk': top_k_results})
            
    save_to_jsonl(gen_prompt_list, save_file_path)


def main_final_gp(top_k=10):  #TODO
    # embedder = pipeline('feature-extraction', model='/data1/nemo/projects/pre_weights/bert', device=1)

    all_infos = get_subset_paths()
    train_infos, val_infos, test_infos = all_infos['Train'], all_infos['Val'], all_infos['Test']
    
    save_dir = '/data2/datasets/drivelm/similarity_rank/bert_category'
    
    save_path = os.path.join(save_dir, 'test_t_v_top_k.jsonl')
    gen_rank(test_infos, train_infos+val_infos, save_path)
    
    save_path = os.path.join(save_dir, 'train_v_top_k.jsonl')
    gen_rank(train_infos, val_infos, save_path)
    
    save_path = os.path.join(save_dir, 'val_t_top_k.jsonl')
    gen_rank(val_infos, train_infos, save_path)
    
    
if __name__ == "__main__":
    main_final_gp()
