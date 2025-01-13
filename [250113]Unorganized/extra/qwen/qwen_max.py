import os
from openai import OpenAI
import dashscope
import json
from tqdm import tqdm

os.environ['https_proxy'] = ''   

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]


def qwen_max(query, image_path):
    messages=[{"role": "user",
           "content": [
        {
            "image": image_path
        },
        {"text": query},
        ]}]

    return dashscope.MultiModalConversation.call(
        api_key="sk-xxx",
        model='qwen-vl-max',
        messages=messages
        )
    

output_dir = "/data2/datasets/drivelm/output2/nexusad/241028_qwen_max"

input_jsons = [
    "/data2/datasets/drivelm/data/qas/original/original_test_general_perception.jsonl",
    "/data2/datasets/drivelm/data/qas/original/original_test_driving_suggestion.jsonl",
    "/data2/datasets/drivelm/data/qas/original/original_test_region_perception.jsonl"
               ]

json_infos = read_jsonl(input_jsons[2])

out_file_path = os.path.join(output_dir, os.path.basename(input_jsons[2]))

with open(out_file_path, 'a') as jsonl_file:
    for info in tqdm(json_infos):
        # if info['question_id'] < 658:
        #     continue
        query, image_path = info['query'], info['images'][0]
        response = qwen_max(query, image_path)
        info['res'] = response        
        jsonl_file.write(json.dumps(info) + '\n')
    
    print(f"Data saved to {out_file_path}")
