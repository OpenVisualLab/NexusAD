import json
from dataclasses import dataclass, field
from tqdm import tqdm

@dataclass
class DatasetPaths:
    base_path: str = "/data2/datasets/drivelm/data/qas/new/"
    dataset_types: list = field(default_factory=lambda: ["train", "val", "test"])
    data_categories: list = field(default_factory=lambda: ["general_perception", "driving_suggestion", "region_perception"])
    prefix: str = "new"

    def __post_init__(self):
        self.paths = {
            f"{dataset_type}_{category}": f"{self.base_path}{self.prefix}_{dataset_type}_{category}.jsonl"
            for dataset_type in self.dataset_types
            for category in self.data_categories
        }

    def get_path(self, dataset_type: str, category: str) -> str:
        return self.paths.get(f"{dataset_type}_{category}", "路径不存在")


def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def save_jsonl(file_path, file_data):
    with open(file_path, 'w') as f:
        for item in file_data:
            f.write(json.dumps(item) + '\n')
    

def extract_vqa(infos, image_path):
    info = next((item for item in infos if item.get("images")[0] == image_path), None)

    return info['images'][0], info['query'], info['response'], info['question_id']

def query_base(query_subset, task):
    paths = DatasetPaths()
    test_gp = paths.get_path("test", task)  
    train_gp = paths.get_path("train", task)  
    val_gp = paths.get_path("val", task)

    train_infos = read_jsonl(train_gp)
    val_infos = read_jsonl(val_gp)
    test_infos = read_jsonl(test_gp)
    
    if query_subset =='test':
        query_infos = test_infos
        base_infos = train_infos+val_infos
    elif query_subset == 'train':
        query_infos = train_infos
        base_infos = val_infos
    elif query_subset == 'val':
        query_infos = val_infos
        base_infos = train_infos
    else:
        print('!'*20)
        
    return query_infos, base_infos

def build_with_no_image(subset, task, rank_jsonl, save_file_path, top_k=1):
    query_infos, base_infos = query_base(subset, task)
    
    rec_list = []
    for rank in tqdm(read_jsonl(rank_jsonl)):
        query_path = rank['query']
        base_paths = [rank['topk'][i]['base'] for i in range(top_k)]
        
        v, q, a, q_id = extract_vqa(query_infos, query_path)
        history = []
        images = []
        
        for base_path in base_paths:
            v_base, q_base, a_base, _ = extract_vqa(base_infos, base_path)
            history.append([q_base, a_base])
            images.append(base_path)
        
        images.append(query_path)
        
        rec_list.append({
            "question_id": q_id, 
            "query": '<image>'+q, 
            "response": a, 
            "history": history,
            "images": [query_path]
        })

    save_jsonl(save_file_path, rec_list)
    

def main_4_ft(rank_jsonl, task, top_k=1):
    
    rank_jsonl_train_v = "/data2/datasets/drivelm/similarity_rank/bert/train_v_bert_top_10.jsonl"
    rank_jsonl_val_t = "/data2/datasets/drivelm/similarity_rank/bert/val_t_bert_top_10.jsonl"
    rank_jsonl_test_vt = "/data2/datasets/drivelm/similarity_rank/bert/test_vt_bert_top_10.jsonl"
    
    
    save_file_path = "/data2/datasets/drivelm/similarity_rank/bert/for_ft/test_vt_gp_top_no_img.jsonl"
    build_with_no_image('test', task, rank_jsonl_test_vt, save_file_path, top_k=1)
    
    save_file_path = "/data2/datasets/drivelm/similarity_rank/bert/for_ft/train_v_gp_top_no_img.jsonl"
    build_with_no_image('train', task, rank_jsonl_train_v, save_file_path, top_k=1)
    
    save_file_path = "/data2/datasets/drivelm/similarity_rank/bert/for_ft/val_t_gp_top_no_img.jsonl"
    build_with_no_image('val', task, rank_jsonl_val_t, save_file_path, top_k=1)
    
    

if __name__ == "__main__":
    rank_jsonl = "/data2/datasets/drivelm/similarity_rank/test_tv_ssim_top_10.jsonl"
    tasks = ['general_perception', 'region_perception', 'driving_suggestion']

    main_4_ft(rank_jsonl, tasks[0], top_k=1)
    





