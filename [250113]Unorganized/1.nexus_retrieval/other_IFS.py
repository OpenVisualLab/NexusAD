

from PIL import Image
import imagehash
import heapq

from skimage.metrics import structural_similarity as ssim
from multiprocessing import Pool, cpu_count
import cv2
import numpy as np
from glob import glob
import os
from tqdm import tqdm
import json


def _hash(img1_path, img2_path):
    hash1 = imagehash.phash(Image.open(img1_path))
    hash2 = imagehash.phash(Image.open(img2_path))
    distance = hash1 - hash2
    similarity = 1 - (distance / len(hash1.hash) ** 2)
    return similarity


def _ssim(img1_path, img2_path):
    
    image1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    image2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    score, _ = ssim(image1, image2, full=True)
    
    return score



def _mse(img1_path, img2_path, n=128):
    image1 = cv2.resize(cv2.imread(img1_path), (n, n)).astype("float")
    image2 = cv2.resize(cv2.imread(img2_path), (n, n)).astype("float")
      
    if image1.shape != image2.shape:
        image2 = cv2.resize(image2, (image1.shape[1], image1.shape[0]))

    err = np.mean((image1.astype("float") - image2.astype("float")) ** 2)
    similarity = 1 - (err / 65025)
    return max(0, similarity)


def calculate_similarity(img1_path, img2_path, sim_method):
    return sim_method(img1_path, img2_path), img2_path


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

def main(top_k=10, method=_mse):
    print(str(method))
    subset = ['Train', 'Val', 'Test', 'Mini']
    data_root = '/data2/datasets/drivelm/data/coda'
    
    # query_img_paths = image_path_list(data_root, subset[2])
    
    # base_img_paths =  image_path_list(data_root, subset[0]) \
    #     + image_path_list(data_root, subset[1])
        
    query_img_paths = image_path_list(data_root, subset[1])
    
    base_img_paths =  image_path_list(data_root, subset[0]) 
    
    gen_prompt_list = []
    print(cpu_count())
    
    with Pool(cpu_count()//2) as pool:
        for query in tqdm(query_img_paths, desc="Processing queries"):
            tasks = [(query, base, method) for base in base_img_paths]
            results = pool.starmap(calculate_similarity, tasks)
            results = [{'similarity_score': similarity, 'base': base} for similarity, base in results]
                            
            top_k_results = heapq.nlargest(top_k, results, key=lambda x: x['similarity_score'])
            gen_prompt_list.append({'query': query, 'topk': top_k_results})
            
    save_to_jsonl(gen_prompt_list, f"/data2/datasets/drivelm/similarity_rank/val_t_mse_128_top_10.jsonl")

if __name__ == "__main__":
    main()