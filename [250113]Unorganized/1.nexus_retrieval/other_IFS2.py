import numpy as np
import cv2
from glob import glob
import os
from tqdm import tqdm

def image_path_list(data_root, subset):
    jsons_dir = os.path.join(data_root, 'CODA-LM', subset)
    jsons_list = sorted(glob(os.path.join(jsons_dir, '*.json')))
    img_set_and_ids = [os.path.basename(f)[:-5].split('_') for f in jsons_list]
    return [os.path.join(data_root, a, 'images', b + '.jpg') for a, b in img_set_and_ids]

def mse_single(q_img, b_imgs):
    q_img = q_img.astype("float")
    b_imgs = b_imgs.astype("float")

    q_img_expanded = np.expand_dims(q_img, axis=0)
    err = np.mean((b_imgs - q_img_expanded) ** 2, axis=(1, 2, 3))

    similarities = 1 - (err / 65025)
    similarities = np.clip(similarities, 0, 1)

    return similarities

def main(n=128):
    subset = ['Train', 'Val', 'Test', 'Mini']
    data_root = '/data2/datasets/drivelm/data/coda'

    query_img_paths = image_path_list(data_root, subset[0])
    base_img_paths = image_path_list(data_root, subset[1])

    q_imgs = np.array([cv2.resize(cv2.imread(b), (n,n)) for b in tqdm(query_img_paths)], dtype=np.float32)
    b_imgs = np.array([cv2.resize(cv2.imread(b), (n,n)) for b in tqdm(base_img_paths)], dtype=np.float32)

    results = []
    for q_img in tqdm(q_imgs):
        similarities = mse_single(q_img, b_imgs)
        results.append(similarities)

    for similarities in results:
        print(len(similarities))

if __name__ == "__main__":
    main()
