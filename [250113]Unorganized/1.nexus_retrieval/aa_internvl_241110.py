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
import torch


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


# def cosine_similarity_cpu(vec1, vec2):
#     dot_product = np.dot(vec1, vec2)
#     norm_vec1 = np.linalg.norm(vec1)
#     norm_vec2 = np.linalg.norm(vec2)
#     return dot_product / (norm_vec1 * norm_vec2)

def cosine_similarity_cpu(vec1, vec2):
    vec1 = vec1.flatten()
    vec2 = vec2.flatten()
    
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    return dot_product / (norm_vec1 * norm_vec2)

# def kl_divergence(p, q):
#     p = np.array(p, dtype=np.float64)
#     q = np.array(q, dtype=np.float64)
#     q = np.where(q == 0, 1e-10, q)
#     return np.sum(p * np.log(p / q))

# def symmetric_kl_divergence(h_d, h_e):
#     kl_de = kl_divergence(h_d, h_e)
#     kl_ed = kl_divergence(h_e, h_d)
#     return 0.5 * (kl_de + kl_ed)

# def jensen_shannon_divergence(p, q):
#     p = np.array(p, dtype=np.float64)
#     q = np.array(q, dtype=np.float64)
#     m = 0.5 * (p + q)
#     return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def kl_divergence(p, q):
    q = torch.where(q == 0, torch.tensor(1e-10, device=p.device), q)
    return torch.sum(p * torch.log(p / q), dim=-1)

def jensen_shannon_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * kl_divergence(p, m) + 0.5 * kl_divergence(q, m)

def cosine_similarity(x, y):
    dot_product = torch.sum(x * y, dim=-1)
    norm_x = torch.norm(x, dim=-1)
    norm_y = torch.norm(y, dim=-1)
    return dot_product / (norm_x * norm_y)

def path_2_pt(qb):
    pts_dir = "/data1/nemo/projects/output/pts"
    image_path, subset = qb
    data_root = '/data2/datasets/drivelm/data/coda'
    subsub, img_id = image_path.split('/')[-3], image_path.split('/')[-1][:-4]
    # print(subsub, img_id)
    
    pt_path = os.path.join(pts_dir, subsub, img_id+'_pooler_output.pt')
    data = torch.load(pt_path, map_location=torch.device('cpu'))

    return np.array(data.to(torch.float32))


def get_subset_paths():
    subset = ['Train', 'Val', 'Test', 'Mini']
    data_root = '/data2/datasets/drivelm/data/coda'
    
    info_list = {}
    for sub in subset:
        paths = image_path_list(data_root, sub)
        info_list[sub] = [(p, sub) for p in paths]
    return info_list
    

def gen_rank(querys, bases, save_file_path, top_k=9268):
    gen_prompt_list = []
    for query in tqdm(querys, desc="Processing queries"):
        results = []
        query_pt = path_2_pt(query)
        for base in bases:
            base_pt = path_2_pt(base)
            
            similarity = cosine_similarity_cpu(query_pt, base_pt)
            results.append({'similarity_score': float(similarity), 'base': base[0]})
        
        top_k_results = heapq.nlargest(top_k, results, key=lambda x: x['similarity_score'])
        gen_prompt_list.append({'query': query[0], 'topk': top_k_results})
            
    save_to_jsonl(gen_prompt_list, save_file_path)


def main_final_gp(top_k=10):  #TODO

    all_infos = get_subset_paths()
    train_infos, val_infos, test_infos = all_infos['Train'], all_infos['Val'], all_infos['Test']
    
    save_dir = '/data2/datasets/drivelm/similarity_rank/inetrnvl'
    
    # save_path = os.path.join(save_dir, 'test_t_v_top_all.jsonl')
    # gen_rank(test_infos, train_infos+val_infos, save_path)
    
    save_path = os.path.join(save_dir, 'train_v_top_k_all.jsonl')
    gen_rank(train_infos, val_infos, save_path)
    
    save_path = os.path.join(save_dir, 'val_t_top_k_all.jsonl')
    gen_rank(val_infos, train_infos, save_path)
    
    
if __name__ == "__main__":
    main_final_gp()
