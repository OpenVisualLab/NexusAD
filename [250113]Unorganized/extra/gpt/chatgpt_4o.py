import os
import openai
import dashscope
import json
from tqdm import tqdm

import openai
import base64
from concurrent.futures import ThreadPoolExecutor

os.environ['https_proxy'] = ''   

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]
    
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

openai.api_key = "sk-xxx"
openai.base_url = "https://api.gpt.ge/v1/"
model = "gpt-4o-mini"


# client = OpenAI(api_key=api_key, base_url = base_url)

def vqa_task(image_path, query):
    base64_image = encode_image(image_path)
    
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": query,
                },
                {
                "type": "image_url",
                "image_url": {
                    "url":  f"data:image/jpeg;base64,{base64_image}"
                },
                },
            ],
            }
        ],
        )

    return response

def batch_vqa_task(image_query_pairs, max_workers=10):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [
            executor.submit(vqa_task, image_path, query)
            for image_path, query in image_query_pairs
        ]
        for future in futures:
            results.append(future.result())
    return results


def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]

def save_jsonl(file_path, file_data):
    with open(file_path, 'w') as f:
        for item in file_data:
            f.write(json.dumps(item) + '\n')
            
def main(vqa_infos, file_path):
    
    with open(file_path, 'a') as jsonl_file:
        for vqa in tqdm(vqa_infos):
            v,q =  vqa['images'][0], vqa['query']
            pred_ori = vqa_task(v,q)
            pred = pred_ori.choices[0].message.content
            vqa['pred_ori'] = {
                "model": pred_ori.model,
                "completion_tokens": pred_ori.usage.completion_tokens,
                "prompt_tokens": pred_ori.usage.prompt_tokens,
                "total_tokens": pred_ori.usage.total_tokens
                               }
            vqa['prediction'] = pred
            jsonl_file.write(json.dumps(vqa) + '\n')
    print(f"Data saved to {file_path}")

        
if __name__ == "__main__":
    # vqa_infos = read_jsonl('/data2/datasets/drivelm/data/qas/original/original_test_driving_suggestion.jsonl')     
    # file_path = '/data1/nemo/projects/NexusAD/extra/gpt/gpt_4o_ds_run.jsonl'
    # main(vqa_infos, file_path)
    
    # vqa_infos = read_jsonl('/data2/datasets/drivelm/data/qas/original/original_test_general_perception.jsonl')     
    # file_path = '/data1/nemo/projects/NexusAD/extra/gpt/gpt_4o_gp_run.jsonl'
    # main(vqa_infos, file_path)
    
    vqa_infos = read_jsonl('/data2/datasets/drivelm/data/qas/original/original_test_region_perception.jsonl')     
    file_path = '/data1/nemo/projects/NexusAD/extra/gpt/gpt_4o_rp_run.jsonl'
    main(vqa_infos, file_path)