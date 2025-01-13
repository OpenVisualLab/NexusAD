import os
import time
import torch
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type
)
from swift.tuners import Swift
from swift.utils import seed_everything

def vl_8b_4():
    device_map = {
    'vision_model': 0,
    'language_model.model.tok_embeddings': 0,
    }

    # Define device mappings for language_model layers
    for i in range(32):
        if i < 8:
            device_map[f'language_model.model.layers.{i}'] = 1
        elif i < 16:
            device_map[f'language_model.model.layers.{i}'] = 2
        elif i < 24:
            device_map[f'language_model.model.layers.{i}'] = 3
        else:
            device_map[f'language_model.model.layers.{i}'] = 0

    # Additional mappings
    device_map.update({
        'language_model.model.norm': 2,
        'language_model.output': 2,
        'mlp1.0': 2,
        'mlp1.1': 2,
        'mlp1.2': 2,
        'mlp1.3': 2,
    })
    return device_map

def init_vl2_model(ckpt_dir=None, model_path='/data1/nemo/projects/pre_weights/InternVL2-8B'):
    """
    初始化 VL2 模型，加载权重并设置模型配置。
    """
    model_path = '/data1/nemo/projects/pre_weights/InternVL2-8B' if model_path == '' or model_path is None else model_path
    
    model_types = {
        "InternVL2-8B": "internvl2-8b",
        "LLaVA-1.5-7B-hf":  "llava1_6-mistral-7b-instruct",
        "LLaVa-NeXT-7B": "llava1_6-vicuna-7b-instruct",
        "Qwen-VL-7B": "qwen-vl-chat",
        "Qwen2-VL-7B-Instruct": "qwen2-vl-7b-instruct",
        "MiniCPM-V-2_6": "minicpm-v-v2_6-chat",
        "MiniCPM-Llama3-V-2_5": "minicpm-v-v2_5-chat",
        "DeepSeek-VL-7B": "deepseek-vl-7b-chat",
        "CogVLM2-19B": "cogvlm2-19b-chat" # require transformers<4.42
    }
    
    
    model_type = model_types[os.path.basename(model_path)]
    
    template_type = get_default_template_type(model_type)
    
    # 获取模型和 tokenizer
    model, tokenizer = get_model_tokenizer(
        model_type, 
        torch.bfloat16, 
        model_id_or_path=model_path,
        model_kwargs={'device_map': 'auto'}
        # model_kwargs={'device_map': vl_8b_4()}
    )
    
    # 如果指定了 ckpt_dir，加载预训练权重
    if ckpt_dir:
        print(f'Loading checkpoint from {ckpt_dir}')
        model = Swift.from_pretrained(model, ckpt_dir, inference_mode=True)
    
    # 设置模型生成参数
    # model.generation_config.max_new_tokens = 4096
    model.generation_config.max_new_tokens = 8192
    template = get_template(template_type, tokenizer)
    
    # 设置随机种子
    seed_everything(42)

    return model, template


def run_vl2_model(model, template, query, images, history=None):
    # query = 'Please note that the images and information presented earlier are part of the historical context and are meant for reference only. Your task is to disregard these historical images and focus on the subsequent images. Analyze and respond based on the content of these new images, paying close attention to the details and providing relevant insights or conclusions.\n'+ query
    history = None if history == [] else history
    response, history = inference(model, template, query, images=images, history=history)
    
    return response


def main(ckpt_dir=None, query=None, image_paths=None, history=None, gpu_id='0'):

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id  # 设置 GPU
    
    start_time = time.time()

    model, template = init_vl2_model(ckpt_dir=ckpt_dir)
    
    print(f'Model initialization time: {time.time() - start_time:.2f} seconds')

    response = run_vl2_model(model, template, query, image_paths[1:], history=history[1:])
    
    print(f'Query: {query}')
    print('-'*20)
    print(f'Response: {response}')
    print(f'Total elapsed time: {time.time() - start_time:.2f} seconds')
    
    response = run_vl2_model(model, template, query, image_paths[2:])
    
    print('-'*20)
    print(f'Response: {response}')
    print(f'Total elapsed time: {time.time() - start_time:.2f} seconds')

if __name__ == "__main__":
    main(
        ckpt_dir=None,  # 如果有检查点目录，提供路径
        gpu_id='0'  # 使用的 GPU ID
    )
