import json
import os
import re
from tqdm import tqdm
import numpy as np
from glob import glob
import heapq
from transformers import pipeline, BertTokenizer


def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]

def save_jsonl(file_path, file_data):
    with open(file_path, 'w') as f:
        for item in file_data:
            f.write(json.dumps(item) + '\n')

def read_json(file_path):
    with open(file_path, 'r') as f:
     return json.load(f)

def read_txt(file_path):
    with open(file_path, 'r') as f:
        return f.read()

def save_to_jsonl(data, file_path):
    with open(file_path, 'w') as jsonl_file:
        for item in data:
            jsonl_file.write(json.dumps(item) + '\n')
    print(f"Data saved to {file_path}")
    

def symmetric_kl_divergence(h_d, h_e, epsilon = 1e-12):
    p = np.array(h_d, dtype=np.float32)
    q = np.array(h_e, dtype=np.float32)
    p = p / (p.sum()+epsilon)
    q = q / (q.sum()+epsilon)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / (m + epsilon)))  
    kl_qm = np.sum(q * np.log(q / (m + epsilon)))  
    symmetric_kl = 0.5 * (kl_pm + kl_qm)
    return symmetric_kl  
    
def main():
    vl_all = read_jsonl("/data2/datasets/drivelm/similarity_rank/inetrnvl/test_t_v_top_all.jsonl")
    print(vl_all[0]['topk'][:10])
    pass


def image_path_list(data_root, subset):
    jsons_dir = os.path.join(data_root, 'CODA-LM', subset)
    jsons_list = sorted(glob(os.path.join(jsons_dir, '*.json')))
    img_set_and_ids = [os.path.basename(f)[:-5].split('_') for f in jsons_list]
    return [os.path.join(data_root, a, 'images', b+'.jpg') for a,b in img_set_and_ids] 

def get_subset_paths():
    subset = ['Train', 'Val', 'Test', 'Mini']
    data_root = '/data2/datasets/drivelm/data/coda'
    
    info_list = {}
    for sub in subset:
        paths = image_path_list(data_root, sub)
        info_list[sub] = [(p, sub) for p in paths]
    return info_list


def path_2_vec(qb):
    image_path, subset = qb
    data_root = '/data2/datasets/drivelm/data/coda'
    subsub, img_id = image_path.split('/')[-3], image_path.split('/')[-1][:-4]
    
    json_path = os.path.join(data_root, 'CODA-LM', subset, subsub+'_'+img_id+'.json')
    info = read_json(json_path)
    
    return info

cat_list = ["vehicles", "vulnerable_road_users", "traffic signs", "traffic lights", "traffic cones", "barriers", "other objects"]
embedder = pipeline('feature-extraction', model='/data1/nemo/projects/pre_weights/bert/bert-base-uncased', device=1)

def extract_cat(info):
    iii = info['general_perception']
    # iii.pop("description and explanation")
    
    temp_list = []
    for cat in cat_list:
        if iii[cat] == []:
            temp_list.append(0)
        else:
            temp_list.append(1) # len(iii[cat])
        
    return np.array(temp_list)+1e-3
    

def calculate_cat_kl(q_info, b_info):
    q_c, b_c = extract_cat(q_info), extract_cat(b_info)
    return symmetric_kl_divergence(q_c, b_c)
    

def nexus_ad(query,  q, base, b, vl_all_q):
    q_i, q_sub = query
    b_i, b_sub = base

    kl = calculate_cat_kl(q, b)
    kl_similarity = np.exp(-kl)
    
    assert q_i == vl_all_q['query']
    
    
    init_similarity = next((item['similarity_score'] for item in vl_all_q['topk'] if item['base'] == b_i), None)

    return init_similarity*kl_similarity
    


def gen_rank(querys, bases, save_file_path, vl_all=None, top_k=10):
    
    gen_prompt_list = []

    querys_v = [path_2_vec(q) for q in tqdm(querys)]
    bases_v = [path_2_vec(b) for b in tqdm(bases)]
    
    for query, q_v, vl_all_q in tqdm(zip(querys, querys_v, vl_all), total=len(querys), desc="Processing queries"):
        results = []

        for base, b_v in zip(bases, bases_v):
            
            similarity = nexus_ad(query, q_v, base, b_v, vl_all_q)

            results.append({'similarity_score': similarity, 'base': base[0]})
        
        top_k_results = heapq.nlargest(top_k, results, key=lambda x: x['similarity_score'])
        gen_prompt_list.append({'query': query[0], 'topk': top_k_results})
            
    save_to_jsonl(gen_prompt_list, save_file_path)
    

def main_final(top_k=10):  #TODO

    all_infos = get_subset_paths()
    train_infos, val_infos, test_infos = all_infos['Train'], all_infos['Val'], all_infos['Test']
    
    save_dir = '/data2/datasets/drivelm/similarity_rank/241110_rank'
    
    # save_path = os.path.join(save_dir, 'test_t_v_top_k.jsonl')
    # vl_all = read_jsonl("/data2/datasets/drivelm/similarity_rank/inetrnvl/test_t_v_top_all.jsonl")
    # gen_rank(test_infos, train_infos+val_infos, save_path, vl_all=vl_all)
    
    
    save_path = os.path.join(save_dir, 'trian_v_top_k.jsonl')
    vl_all = read_jsonl("/data2/datasets/drivelm/similarity_rank/inetrnvl/train_v_top_k_all.jsonl")
    gen_rank(train_infos, val_infos, save_path, vl_all=vl_all)
    
    save_path = os.path.join(save_dir, 'val_t_top_k.jsonl')
    vl_all = read_jsonl("/data2/datasets/drivelm/similarity_rank/inetrnvl/val_t_top_k_all.jsonl")
    gen_rank(val_infos, train_infos, save_path, vl_all=vl_all)
    
    

if __name__ == "__main__":
    main_final()