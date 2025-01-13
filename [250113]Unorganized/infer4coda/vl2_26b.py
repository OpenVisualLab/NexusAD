import os

import time
from swift.llm import (
    get_model_tokenizer, get_template, inference, ModelType, get_default_template_type
)

# from swift.llm import (
#     ModelType, get_lmdeploy_engine, get_default_template_type,
#     get_template, inference_lmdeploy, inference_stream_lmdeploy
# )

# from lmdeploy import pipeline, TurbomindEngineConfig
from swift.tuners import Swift
from swift.utils import seed_everything
import torch

def vl_26b_2():
    device_map = {
    'vision_model': 0,
    'language_model.model.tok_embeddings': 0,
    }

    # Define device mappings for language_model layers
    for i in range(48):
        if i < 0:
            device_map[f'language_model.model.layers.{i}'] = 0
        elif i < 24:
            device_map[f'language_model.model.layers.{i}'] = 1
        else:
            device_map[f'language_model.model.layers.{i}'] = 2

    # Additional mappings
    device_map.update({
        'language_model.model.norm': 0,
        'language_model.output': 0,
        'mlp1.0': 0,
        'mlp1.1': 0,
        'mlp1.2': 0,
        'mlp1.3': 0,
    })
    return device_map

def init_vl2_model(ckpt_dir=None):
    model_type = "internvl2-26b"
    template_type = get_default_template_type(model_type)
        
    model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16,
                                    model_kwargs={'device_map': vl_26b_2()},
                                    model_id_or_path='/data1/nemo/projects/pre_weights/InternVL2-26B')
    if ckpt_dir is not None:
        print('loading: ', ckpt_dir)
        model = Swift.from_pretrained(model, ckpt_dir, inference_mode=True)
        
    model.generation_config.max_new_tokens = 4096
    template = get_template(template_type, tokenizer)
    seed_everything(42)

    return model, template


def run_vl2_model(model, template, query, image_path):
    
    response, _ = inference(model, template, query, images=[image_path])
    
    return response

if __name__ == "__main__":
    import time
    temp = time.time()
    query = '请描述这张图像'
    image_path = '/home/oem/codalm/datasets/CODA/val/images/0001.jpg'
    
    model, template = init_vl2_model()
    print(time.time()-temp)
    
    response = run_vl2(model, template, query, image_path)
    
    print(f'query: {query}')
    print(f'response: {response}')
    print(time.time()-temp)
    
    
    # init_vl2_lmdeploy()