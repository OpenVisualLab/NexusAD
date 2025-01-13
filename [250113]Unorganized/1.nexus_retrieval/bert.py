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

def path_2_json(image_path, embedder, subset='Test'):
    data_root = '/data2/datasets/drivelm/data/coda'
    
    subsub, img_id = image_path.split('/')[-3], image_path.split('/')[-1][:-4]
    
    json_path = os.path.join(data_root, 'CODA-LM', subset, subsub+'_'+img_id+'.json')
    info = read_json(json_path)
    gp_info = info['general_perception']
    gp_info.pop('description and explanation')
    
    return np.mean(embedder(str(gp_info)[:512])[0],axis=0)

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2)
  

def main(top_k=10):
    subset = ['Train', 'Val', 'Test', 'Mini']
    data_root = '/data2/datasets/drivelm/data/coda'
    
    save_dir = '/data2/datasets/drivelm/similarity_rank/bert_embedding'
    
    query_test_paths = image_path_list(data_root, subset[2])
    
    base_train_paths =  image_path_list(data_root, subset[0])  
    base_val_paths =  image_path_list(data_root, subset[1])  
    
    embedder = pipeline('feature-extraction', model='/data1/nemo/projects/pre_weights/bert', device=1)
    
    query_test_infos = [[path_2_json(d, embedder, 'Test'), d] for d in tqdm(query_test_paths)]
    base_train_infos = [[path_2_json(d, embedder, 'Train'), d] for d in tqdm(base_train_paths)]
    base_val_infos = [[path_2_json(d, embedder, 'Val'), d] for d in tqdm(base_val_paths)]

    base_tv_infos = base_train_infos+base_val_infos
    gen_prompt_list = []
    for query, query_img_path in tqdm(query_test_infos, desc="Processing queries"):
        results = []
        for base, base_img_path in base_tv_infos:
            similarity = cosine_similarity(query, base)
            results.append({'similarity_score': similarity, 'base': base_img_path})
        
        top_k_results = heapq.nlargest(top_k, results, key=lambda x: x['similarity_score'])
        gen_prompt_list.append({'query': query_img_path, 'topk': top_k_results})
            
    save_to_jsonl(gen_prompt_list, f"/data2/datasets/drivelm/similarity_rank/test_vt_bert_top_10.jsonl")
    
def main_4_ft(top_k=10):
    subset = ['Train', 'Val', 'Test', 'Mini']
    data_root = '/data2/datasets/drivelm/data/coda'
    
    save_dir = '/data2/datasets/drivelm/similarity_rank/bert_embedding'
    
    query_test_paths = image_path_list(data_root, subset[2])
    
    base_train_paths =  image_path_list(data_root, subset[0])  
    base_val_paths =  image_path_list(data_root, subset[1])  
    
    embedder = pipeline('feature-extraction', model='/data1/nemo/projects/pre_weights/bert', device=1)
    
    query_test_infos = [[path_2_json(d, embedder, 'Test'), d] for d in tqdm(query_test_paths)]
    base_train_infos = [[path_2_json(d, embedder, 'Train'), d] for d in tqdm(base_train_paths)]
    base_val_infos = [[path_2_json(d, embedder, 'Val'), d] for d in tqdm(base_val_paths)]
    
    gen_prompt_list = []
    for query, query_img_path in tqdm(base_train_infos, desc="Processing queries"):
        results = []
        for base, base_img_path in base_val_infos:
            similarity = cosine_similarity(query, base)
            results.append({'similarity_score': similarity, 'base': base_img_path})
        
        top_k_results = heapq.nlargest(top_k, results, key=lambda x: x['similarity_score'])
        gen_prompt_list.append({'query': query_img_path, 'topk': top_k_results})
            
    save_to_jsonl(gen_prompt_list, f"/data2/datasets/drivelm/similarity_rank/train_v_bert_top_10.jsonl")
    
    gen_prompt_list = []
    for query, query_img_path in tqdm(base_val_infos, desc="Processing queries"):
        results = []
        for base, base_img_path in base_train_infos:
            similarity = cosine_similarity(query, base)
            results.append({'similarity_score': similarity, 'base': base_img_path})
        
        top_k_results = heapq.nlargest(top_k, results, key=lambda x: x['similarity_score'])
        gen_prompt_list.append({'query': query_img_path, 'topk': top_k_results})
            
    save_to_jsonl(gen_prompt_list, f"/data2/datasets/drivelm/similarity_rank/val_t_bert_top_10.jsonl")
    
    
if __name__ == "__main__":
    # main()
    main_4_ft()
