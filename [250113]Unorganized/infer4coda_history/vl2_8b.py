import os
import time
import torch
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type
)
from swift.tuners import Swift
from swift.utils import seed_everything

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
    )
    
    # 如果指定了 ckpt_dir，加载预训练权重
    if ckpt_dir:
        print(f'Loading checkpoint from {ckpt_dir}')
        model = Swift.from_pretrained(model, ckpt_dir, inference_mode=True)
    
    # 设置模型生成参数
    model.generation_config.max_new_tokens = 4096
    template = get_template(template_type, tokenizer)
    
    # 设置随机种子
    seed_everything(42)

    return model, template

def run_vl2_model(model, template, query, images, history=None):
    history = None if len(history) == 0 else history

    response, _ = inference(model, template, query, images=images, history=history)
    # response, _ = inference(model, template, query, images=images, history=history[:1])
    
    return response

def main(ckpt_dir=None, query=None, image_paths=None, gpu_id='0'):
    """
    主函数，用于初始化模型并进行推理。
    """
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
    main(
        ckpt_dir=None,  # 如果有检查点目录，提供路径
        gpu_id='0'  # 使用的 GPU ID
    )
