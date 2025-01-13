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


def find_the_info(base_path, database):
    res, pred = next(((item['response'], item['prediction']) for item in database if item['images'][0] == base_path), None)
    # return  json.dumps(res),  json.dumps(pred)
    return  str(res),  str(pred)
    


def build_with_no_image(subset, task, rank_jsonl, save_file_path, preds, top_k=1):
    pred_test, pred_train, pred_val = preds
    
    if subset == 'test':
        query_pred = pred_test
        base_pred = pred_train+pred_val
    elif subset == 'train':
        query_pred = pred_train
        base_pred = pred_val
    elif subset == 'val':
        query_pred = pred_val
        base_pred = pred_train
    else:
        print(subset*20)
    
    
    guided_prompt_0 = "Please revise this answer:\n"
    
    guided_prompt_1 = "You will be provided with several historical examples, each of which includes a poor model response (query) and a human-annotated corrected response (response). Based on these historical examples, learn how to improve the current response."
    
    guided_prompt_2 = " You will also be given an image along with the current response to improve. Your task is to analyze the provided response in combination with the given image, identify the issues, and generate a higher-quality version of the response."
    
    ori_gp_prompt = " There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please describe each object's appearance, position, direction, and explain why it affects the ego car's behavior."
    
    ori_ds_prompt = " There is an image of traffic captured from the perspective of the ego car. Focus on objects influencing the ego car's driving behavior: vehicles (cars, trucks, buses, etc.), vulnerable road users (pedestrians, cyclists, motorcyclists), traffic signs (no parking, warning, directional, etc.), traffic lights (red, green, yellow), traffic cones, barriers, miscellaneous(debris, dustbin, animals, etc.). You must not discuss any objects beyond the seven categories above. Please provide driving suggestions for the ego car based on the current scene."
    
    # TODO for infer!
    if task == "general_perception":
        if top_k == 0:
            new_query = '<image> ' + ori_gp_prompt
        else:
            new_query = guided_prompt_1 + guided_prompt_2 + '<image>' + ori_gp_prompt
    elif task == "driving_suggestion":
        if top_k == 0:
            new_query = '<image> ' + ori_ds_prompt
        else:
            new_query = guided_prompt_1 + guided_prompt_2 + '<image>' + ori_ds_prompt
    else:
        print(task)
            
    query_infos, base_infos = query_base(subset, task)
    
    
    rec_list = []
    for rank in tqdm(read_jsonl(rank_jsonl)):
        query_path = rank['query']
        
        res_q, pred_q = find_the_info(query_path, query_pred) 
        
        base_paths = [rank['topk'][i]['base'] for i in range(top_k)]
        
        v, q, a, q_id = extract_vqa(query_infos, query_path)
        history = []
        images = []
        
        for base_path in base_paths:
            res, pred = find_the_info(base_path, base_pred)  
            history.append([guided_prompt_0 + pred,  res])
            images.append(base_path)
        images.append(query_path)
        
        new_query = new_query + '\n' + guided_prompt_0 + pred_q
        
        rec_list.append({
            "question_id": q_id, 
            "query": new_query, 
            "response": a, 
            "history": history,
            "images": [query_path]
        })

    save_jsonl(save_file_path, rec_list)
    

def main_4_ft(top_k=1):
    tasks = ['general_perception', 'region_perception', 'driving_suggestion']
    
    pred_train_gp = read_jsonl("/data2/datasets/drivelm/241105_train_val/str_train_gp_run.jsonl")
    pred_val_gp = read_jsonl("/data2/datasets/drivelm/241105_train_val/str_val_gp_run.jsonl")
    pred_test_gp = read_jsonl("/data2/datasets/drivelm/output2/nexusad/241027_structure/struct_vl_8b_test_gp_ft_4.jsonl")
    
    pred_train_ds = read_jsonl("/data2/datasets/drivelm/241105_train_val/str_train_ds_run.jsonl")
    pred_val_ds = read_jsonl("/data2/datasets/drivelm/241105_train_val/str_val_ds_run.jsonl")
    pred_test_ds = read_jsonl("/data2/datasets/drivelm/output2/nexusad/241027_structure/struct_vl_8b_test_ds_ft_4.jsonl")
    
    
    # -----  gp -----
    task = tasks[0]
    rank_jsonl_test_vt = "/data2/datasets/drivelm/similarity_rank/241110_rank/test_t_v_top_k.jsonl"
    rank_jsonl_train_v = "/data2/datasets/drivelm/similarity_rank/241110_rank/trian_v_top_k.jsonl"
    rank_jsonl_val_t = "/data2/datasets/drivelm/similarity_rank/241110_rank/val_t_top_k.jsonl"
    
    
    save_file_path = "/data2/datasets/drivelm/similarity_rank/241110_rank_for_ft/test_vt_gp_v1_2.jsonl"
    build_with_no_image('test', task, rank_jsonl_test_vt, save_file_path, [pred_test_gp, pred_train_gp, pred_val_gp], top_k=2)
    
    save_file_path = "/data2/datasets/drivelm/similarity_rank/241110_rank_for_ft/train_v_gp_v1_2.jsonl"
    build_with_no_image('train', task, rank_jsonl_train_v, save_file_path, [pred_test_gp, pred_train_gp, pred_val_gp], top_k=2)
    
    save_file_path = "/data2/datasets/drivelm/similarity_rank/241110_rank_for_ft/val_t_gp_v1_2.jsonl"
    build_with_no_image('val', task, rank_jsonl_val_t, save_file_path, [pred_test_gp, pred_train_gp, pred_val_gp], top_k=2)
        
    
    # -----  ds -----
    task = tasks[2]
    
    save_file_path = "/data2/datasets/drivelm/similarity_rank/241110_rank_for_ft/test_vt_ds_v1_2.jsonl"
    build_with_no_image('test', task, rank_jsonl_test_vt, save_file_path, [pred_test_ds, pred_train_ds, pred_val_ds], top_k=2)
    
    save_file_path = "/data2/datasets/drivelm/similarity_rank/241110_rank_for_ft/train_v_ds_v1_2.jsonl"
    build_with_no_image('train', task, rank_jsonl_train_v, save_file_path, [pred_test_ds, pred_train_ds, pred_val_ds], top_k=2)
    
    save_file_path = "/data2/datasets/drivelm/similarity_rank/241110_rank_for_ft/val_t_ds_v1_2.jsonl"
    build_with_no_image('val', task, rank_jsonl_val_t, save_file_path, [pred_test_ds, pred_train_ds, pred_val_ds], top_k=2)
       
    
    

if __name__ == "__main__":
    main_4_ft()
    





