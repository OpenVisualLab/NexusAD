import os 
import json
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
    


import openai
import base64

os.environ['https_proxy'] = 'http://10.16.0.81:8888'   

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        return [json.loads(line) for line in file]
    
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

openai.api_key = "sk-LBqxw6MZHvq0YwZf9c0f0937692842C6B2C3D3B43dFbF176"
openai.base_url = "https://api.gpt.ge/v1/"
model = "gpt-4o-mini"


def qa_task(query):
    response = openai.chat.completions.create(
        model=model,
        messages=[{
            "role": "user",
            "content": [
                {
                "type": "text",
                "text": query,
                },],}],)
    return response


def return_prompt(json_data, key=None):
    prompt = f"""Analyze the following JSON data to evaluate the contribution of each category to the overall "{key}" summary. For each category ('vehicles,' 'vulnerable road users,' 'traffic signs,' 'traffic lights,' 'traffic cones,' 'barriers,' and 'other objects'), assess how much its "description" and "explanation" contribute to shaping the final understanding. Assign values from 0 to 1: the first for the "description" contribution, the second for the "explanation."

JSON data:
{json_data}

Return only the following format:

contribution_values = {{
    "vehicles": [xx, xx],
    "vulnerable_road_users": [xx, xx],
    "traffic signs": [xx, xx],
    "traffic lights": [xx, xx],
    "traffic cones": [xx, xx],
    "barriers": [xx, xx],
    "other_objects": [xx, xx]
}}

Use [0.0, 0.0] for categories with no contribution, and do not include any additional explanation or comments.
"""

    return prompt

def main(test_json_path, save_json_path, key=None):

    with open(save_json_path, 'a') as jsonl_file:
        for info in tqdm(read_jsonl(test_json_path)):
            json_data = info['response']
            
            prompt = return_prompt(json_data, key)
            
            pred_ori = qa_task(prompt)
            pred = pred_ori.choices[0].message.content
            
            info['pred_ori'] = {
                "model": pred_ori.model,
                "completion_tokens": pred_ori.usage.completion_tokens,
                "prompt_tokens": pred_ori.usage.prompt_tokens,
                "total_tokens": pred_ori.usage.total_tokens
                               }
            info['scores'] = pred
            jsonl_file.write(json.dumps(info) + '\n')
    print(f"Data saved to {save_json_path}")


if __name__ == "__main__":
    # test_json_path = "/data2/datasets/drivelm/data/qas/new/new_test_general_perception.jsonl"
    # save_json_path = "/data1/nemo/projects/NexusAD/1.nexus_retrieval/11111/svae_dir/gp_scores.jsonl"
    
    # main(test_json_path, save_json_path, key="description and explanation")
    
    
    test_json_path = "/data2/datasets/drivelm/data/qas/new/new_test_driving_suggestion.jsonl"
    save_json_path = "/data1/nemo/projects/NexusAD/1.nexus_retrieval/11111/svae_dir/ds_scores.jsonl"
    
    main(test_json_path, save_json_path, key="driving_suggestion")
    