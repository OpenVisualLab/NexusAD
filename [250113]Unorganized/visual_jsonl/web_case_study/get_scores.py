import json
import os
import re
from tqdm import tqdm

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

def score_score(content):
    match = re.search(r'Rating:\s*\[\[(\d+)\]\]', content)
    if match:
        return int(match.group(1)) 
    else:
        return -1
    

def get_score(score_dir, image_path):
    subsub, img_id = image_path.split('/')[-3], image_path.split('/')[-1][:-4]
    content = read_txt(os.path.join(score_dir, subsub+'_'+img_id+'.txt'))
    return score_score(content), content
    

def get_label(label_dir, image_path):
    subsub, img_id = image_path.split('/')[-3], image_path.split('/')[-1][:-4]
    content = read_json(os.path.join(label_dir, subsub+'_'+img_id+'.json'))
    gp, ds = content["general_perception"]["description and explanation"], content["driving_suggestion"]
    return gp, ds

def extract_str(data_str, start_key=None):
    
    start_key = "'description and explanation': " if start_key==None else 'driving_suggestion'
    
    start_index = data_str.find(start_key)

    if start_index == -1:
        print(data_str)
        return data_str
    
    start_index += len(start_key)

    if data_str[start_index] == '"':
        quote_char = '"'
    elif data_str[start_index] == "'":
        quote_char = "'"
    else:
        print(data_str)
        print('-'*100)
        return None

    end_index = data_str.find(quote_char, start_index + 1)

    # description_and_explanation = data_str[start_index + 1:end_index]
    description_and_explanation = data_str[start_index + 1:]

    return description_and_explanation[:-2]


def extract_gp_ds(data_str, key=None):
    key = "description and explanation" if key==None else "driving_suggestion"

    data = json.loads(data_str.replace("'", '"'))
    
    return data[key]

    
def get_jsonl_info():
    
    h =  '/data2/datasets/drivelm/data/qas/new/new_test_general_perception.jsonl'
    
    label_dir = '/data2/datasets/drivelm/data/coda/CODA-LM/Test'
    
    aaaa_gp = '/data1/nemo/projects/NexusAD/extra/gpt/gpt_4o_gp_run.jsonl'
    aaaa_ds = '/data1/nemo/projects/NexusAD/extra/gpt/gpt_4o_ds_run.jsonl'
    
    
    b_gp = '/data2/datasets/drivelm/241113_icl_swift/post_test_icl_gp_v1_3_ft_4_his_3.jsonl'
    b_ds = '/data2/datasets/drivelm/241113_icl_swift/post_test_icl_ds_v1_3_ft_6_his_3.jsonl'
    
    c_gp = '/data2/datasets/drivelm/output2/nexusad/241028_qwen_max/eval_test_general_perception.jsonl'
    c_ds = '/data2/datasets/drivelm/output2/nexusad/241028_qwen_max/eval_test_driving_suggestion.jsonl'
    
    d_gp = '/data2/datasets/drivelm/output2/nexusad/241025_ori_temp/ori_vl_8b_test_gp_2316.jsonl'
    d_ds = '/data2/datasets/drivelm/output2/nexusad/241025_ori_temp/ori_vl_8b_test_ds_ft_4.jsonl'
    
    gp_human = read_jsonl(h)

    
    a_gp_score_dir = aaaa_gp.replace('.jsonl', '')
    a_ds_score_dir = aaaa_ds.replace('.jsonl', '')
    
    b_gp_score_dir = b_gp.replace('.jsonl', '')
    b_ds_score_dir = b_ds.replace('.jsonl', '')
    
    c_gp_score_dir = c_gp.replace('.jsonl', '')
    c_ds_score_dir = c_ds.replace('.jsonl', '')
    
    d_gp_score_dir = d_gp.replace('.jsonl', '')
    d_ds_score_dir = d_ds.replace('.jsonl', '')
    

    # print(gp_human[0].keys(), a_score_dir)
    
    data_groups = []
    for info in tqdm(zip(gp_human, read_jsonl(aaaa_gp),  read_jsonl(aaaa_ds),
        read_jsonl(b_gp),read_jsonl(b_ds),
        read_jsonl(c_gp), read_jsonl(c_ds),
        read_jsonl(d_gp), read_jsonl(d_ds))):
        gt_info, a_gp_info, a_ds_info, b_gp_info, b_ds_info, c_gp_info, c_ds_info, d_gp_info, d_ds_info = info
        
        image = gt_info['images'][-1]
        
        gt_gp, gt_ds = get_label(label_dir, image)
        
        # -------
        
        aa_gp = a_gp_info['prediction']
        a_gp_score, nbbb = get_score(a_gp_score_dir, image)
        aa_ds = a_ds_info['prediction']
        a_ds_score, nbbb = get_score(a_ds_score_dir, image)
        
        # ------
        
        b_gp = b_gp_info['prediction']
        b_ds = b_ds_info['prediction']
        
        # b_gp = extract_str(b_gp)
        
        b_gp_score, nbbb = get_score(b_gp_score_dir, image)
        b_ds_score, nbbb = get_score(b_ds_score_dir, image)
        
        
        # -------
        
        c_gp = c_gp_info['prediction']
        c_ds = c_ds_info['prediction']
        
        
        c_gp_score, nbbb = get_score(c_gp_score_dir, image)
        c_ds_score, nbbb = get_score(c_ds_score_dir, image)
        
        
        # ---------
        
        d_gp = d_gp_info['prediction']
        d_ds = d_ds_info['prediction']
        
        
        d_gp_score, nbbb = get_score(d_gp_score_dir, image)
        d_ds_score, nbbb = get_score(d_ds_score_dir, image)
        
        
        data_groups.append({
            'image_path': image,
            'models':{
                'Human': [gt_gp, gt_ds],
                'GPT-4o': [(aa_gp, a_gp_score), (aa_ds, a_ds_score)],
                'NexusAD': [(b_gp, b_gp_score), (b_ds, b_ds_score)],
                'C': [(c_gp, c_gp_score), (c_ds, c_ds_score)],
                'D': [(d_gp, d_gp_score), (d_gp, d_gp_score)]  
            }
        })
    return data_groups


    
if __name__ =="__main__":
    get_jsonl_info()