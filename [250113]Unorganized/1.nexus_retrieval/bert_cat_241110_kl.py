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


embedder = pipeline('feature-extraction', model='/data1/nemo/projects/pre_weights/bert/bert-base-uncased', device=1)
cat_list = ["vehicles", "vulnerable_road_users", "traffic signs", "traffic lights", "traffic cones", "barriers", "other objects"]

def embedder_a(t):
    return np.mean(embedder(str(t)[:512])[0],axis=0)

def embedder_b(info):
    
    gp_qb = info['general_perception']
    
    cat_vec = {}
    for cat in cat_list:
        cat_vec[cat] = []
        for ddd in gp_qb[cat]:
            # cat_vec[cat].append(embedder_a(ddd["description"]))
            cat_vec[cat].append(ddd)
    return cat_vec

# def kl_divergence(p, q):
#     p = np.array(p, dtype=np.float64)
#     q = np.array(q, dtype=np.float64)
#     q = np.where(q == 0, 1e-10, q)
#     return np.sum(p * np.log(p / q))

# def symmetric_kl_divergence(h_d, h_e):
#     kl_de = kl_divergence(h_d, h_e)
#     kl_ed = kl_divergence(h_e, h_d)
#     return 0.5 * (kl_de + kl_ed)   

def symmetric_kl_divergence(h_d, h_e):
    # 使用对称 KL 散度计算
    p = np.array(h_d, dtype=np.float64)
    q = np.array(h_e, dtype=np.float64)
    p = p / (p.sum()+1e-10)
    q = q / (q.sum()+1e-10)
    m = 0.5 * (p + q)
    kl_pm = np.sum(p * np.log(p / (m + 1e-10)))  
    kl_qm = np.sum(q * np.log(q / (m + 1e-10)))  
    symmetric_kl = 0.5 * (kl_pm + kl_qm)
    return symmetric_kl
      

def jaccard_similarity(vec1, vec2):
    intersection = np.minimum(vec1, vec2).sum()
    union = np.maximum(vec1, vec2).sum()
    return intersection / union

def nexus_ad(q_v, b_v):   
    
    kl_q, kl_b = [], []
     
    sim_rec = {}
    for cat in cat_list:
        cand = q_v[cat]
        ref = b_v[cat]
        
        kl_q.append(len(cand))
        kl_b.append(len(ref))
        
        # if not cand or not ref:
        #     sim_rec[cat] = -1 if cand or ref else 0
        #     continue
        
        # temp_list_1 = [] 
        # for cc in cand:
        #     temp_list = []
        #     for rr in ref:
        #         sim = cosine_similarity(cc, rr)
        #         temp_list.append(sim)
        #     temp_list_1.append(max(temp_list))
            
        # sim_rec[cat] = np.mean(np.array(temp_list_1))
    
    # filtered_values = [v for v in sim_rec.values() if v != 0]
    # if filtered_values:
    #     overall_similarity = sum(filtered_values) / len(filtered_values)
    # else:
    #     overall_similarity = 0  
    # print(kl_q, kl_b)
    alpha = 0.5
    symmetric_kl = jaccard_similarity(kl_q, kl_b)
    
    # kl_similarity = np.exp(-symmetric_kl)
    # combined_similarity = alpha * kl_similarity + (1 - alpha) * overall_similarity
    # # (1+0.5*cat_similarity)
    # +(1/cat_similarity)
        
    # return float(overall_similarity)
    return np.mean(symmetric_kl)

def path_2_vec(qb):
    image_path, subset = qb
    data_root = '/data2/datasets/drivelm/data/coda'
    subsub, img_id = image_path.split('/')[-3], image_path.split('/')[-1][:-4]
    
    json_path = os.path.join(data_root, 'CODA-LM', subset, subsub+'_'+img_id+'.json')
    info = read_json(json_path)
    # print(json_path)
    return embedder_b(info)
    # return info


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

    querys_v = [path_2_vec(q) for q in tqdm(querys)]
    bases_v = [path_2_vec(b) for b in tqdm(bases)]
    
    for query, q_v in tqdm(zip(querys, querys_v), total=len(querys), desc="Processing queries"):
        results = []

        for base, b_v in zip(bases,bases_v):
            
            similarity = nexus_ad(q_v, b_v)

            results.append({'similarity_score': similarity, 'base': base[0]})
        
        top_k_results = heapq.nlargest(top_k, results, key=lambda x: x['similarity_score'])
        gen_prompt_list.append({'query': query[0], 'topk': top_k_results})
            
    save_to_jsonl(gen_prompt_list, save_file_path)


def main_final_gp(top_k=10):  #TODO

    all_infos = get_subset_paths()
    train_infos, val_infos, test_infos = all_infos['Train'], all_infos['Val'], all_infos['Test']
    
    save_dir = '/data2/datasets/drivelm/similarity_rank/b_kl_cat2'
    
    save_path = os.path.join(save_dir, 'test_t_v_top_k.jsonl')
    gen_rank(test_infos, train_infos+val_infos, save_path)
    
    # save_path = os.path.join(save_dir, 'train_v_top_k.jsonl')
    # gen_rank(train_infos, val_infos, save_path)
    
    # save_path = os.path.join(save_dir, 'val_t_top_k.jsonl')
    # gen_rank(val_infos, train_infos, save_path)
    
    
if __name__ == "__main__":
    main_final_gp()
