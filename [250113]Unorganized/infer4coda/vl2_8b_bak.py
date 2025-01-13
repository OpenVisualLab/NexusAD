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

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

def init_vl2_model(ckpt_dir=None):
    model_type = "internvl2-8b"
    template_type = get_default_template_type(model_type)
        
    model, tokenizer = get_model_tokenizer(model_type, torch.bfloat16,
                                    # model_kwargs={'device_map': "auto"},
                                    model_id_or_path='/data1/nemo/projects/pre_weights/InternVL2-8B')
    if ckpt_dir is not None:
        print('loading: ', ckpt_dir)
        model = Swift.from_pretrained(model, ckpt_dir, inference_mode=True)
        
    model.generation_config.max_new_tokens = 2048
    template = get_template(template_type, tokenizer)
    seed_everything(42)

    return model, template


def run_vl2_model(model, template, query, images):
    
    response, _ = inference(model, template, query, images=images)
    
    return response

if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import time
    temp = time.time()
    query = "You are a professional anomaly detection system. In this task, you will receive two image frames from a campus scene. The first image is a normal scene, and your job is to assign appropriate anomaly scores based on the degree of anomaly in the second image. The possible score range is 0.0 to 1.0, where 0.0 indicates no abnormalities and 1.0 indicates severe abnormalities. Please provide the abnormal score in the form of [[xx]] and explain why."
    
    image_path1 = '/data1/nemo/projects/240919_TJR/encoder_vit/0000_resized.jpg'
    image_path2 = '/data1/nemo/projects/240919_TJR/encoder_vit/0296_resized.jpg'
    
    
    model, template = init_vl2_model()
    print(time.time()-temp)
    
    response = run_vl2_model(model, template, query, [image_path1])
    
    print(f'query: {query}')
    print(f'response: {response}')
    print(time.time()-temp)
    
    
    # init_vl2_lmdeploy()