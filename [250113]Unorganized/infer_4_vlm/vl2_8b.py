import os
import time
import torch
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type
)
from swift.tuners import Swift
from swift.utils import seed_everything


def vl_8b_2():
    device_map = {
    'vision_model': 0,
    'language_model.model.tok_embeddings': 0,
    }

    # Define device mappings for language_model layers
    for i in range(32):
        device_map[f'language_model.model.layers.{i}'] = 1
        
    # Additional mappings
    device_map.update({
        'language_model.model.norm': 1,
        'language_model.output': 1,
        'mlp1': 1,
    })
    return device_map

def vl_26b_2():
    device_map = {
    'vision_model': 0,
    'language_model.model.tok_embeddings': 0,
    }

    for i in range(48):
        if i < 16:
            device_map[f'language_model.model.layers.{i}'] = 0
        else:
            device_map[f'language_model.model.layers.{i}'] = 1
            
        
    # Additional mappings
    device_map.update({
        'language_model.model.norm': 1,
        'language_model.output': 1,
        'mlp1': 1,
    })
    return device_map

def init_vl2_model(ckpt_dir=None, model_path='/data1/nemo/projects/pre_weights/InternVL2-8B', max_new_tokens=4096):

    model_type = model_path.split('/')[-1].lower()
    template_type = get_default_template_type(model_type)
    
    model, tokenizer = get_model_tokenizer(
        model_type, 
        torch.bfloat16, 
        model_id_or_path=model_path,
        # model_kwargs={'device_map': vl_26b_2(),}
        model_kwargs={'device_map': 'auto',}   
    )
    # print(model.hf_device_map)
    # aa
    
    if ckpt_dir:
        print(f'Loading checkpoint from {ckpt_dir}')
        model = Swift.from_pretrained(model, ckpt_dir, inference_mode=True)
    
    model.generation_config.max_new_tokens = max_new_tokens
    
    template = get_template(template_type, tokenizer)
    
    seed_everything(42)

    return model, template

def run_vl2_model(model, template, query, images):
    system_prompt = "You are an Autonomous Driving AI assistant. You receive an image that consists of six surrounding camera views. The layout is as follows: The first row contains three images: FRONT LEFT, FRONT, FRONT RIGHT. The second row contains three images: BACK RIGHT, BACK, BACK LEFT . Your task is to analyze these images and provide insights or actions based on the visual data."
    response, _ = inference(model, template, query, images=images, system=system_prompt)
    # response, _ = inference(model, template, query, images=images, system=None)
    
    return response

def main(ckpt_dir=None, query=None, image_paths=None, gpu_id='0'):

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id  # 设置 GPU
    
    # 默认查询和图片路径
    if query is None:
        query = (
            "You are a professional anomaly detection system. In this task, you will receive "
            "two image frames from a campus scene. The first image is a normal scene, and your "
            "job is to assign appropriate anomaly scores based on the degree of anomaly in the "
            "second image. The possible score range is 0.0 to 1.0, where 0.0 indicates no "
            "abnormalities and 1.0 indicates severe abnormalities. Please provide the abnormal "
            "score in the form of [[xx]] and explain why."
        )
    
    if image_paths is None:
        image_paths = ['/data1/nemo/projects/240919_TJR/encoder_vit/0000_resized.jpg']
    
    # 记录开始时间
    start_time = time.time()

    # 初始化模型和模板
    model, template = init_vl2_model(ckpt_dir)
    
    # 记录初始化时间
    print(f'Model initialization time: {time.time() - start_time:.2f} seconds')

    # 执行推理
    response = run_vl2_model(model, template, query, image_paths)
    
    # 输出结果
    print(f'Query: {query}')
    print(f'Response: {response}')
    print(f'Total elapsed time: {time.time() - start_time:.2f} seconds')

if __name__ == "__main__":
    # os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    main(
        ckpt_dir=None,  # 如果有检查点目录，提供路径
        gpu_id='0'  # 使用的 GPU ID
    )
