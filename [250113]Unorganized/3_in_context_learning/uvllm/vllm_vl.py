import os

from swift.llm import (
    ModelType, get_vllm_engine, get_default_template_type,
    get_template, inference_vllm, convert_to_base64
)


def init_model(model_id_or_path='/data2/datasets/drivelm/output3/241103_gp_icl/checkpoint-1737-merged'):
    model_type = ModelType.internvl2_8b
    # model_id_or_path = '/data1/nemo/projects/pre_weights/InternVL2-8B'

    llm_engine = get_vllm_engine(model_type, 
                                model_id_or_path=model_id_or_path, 
                                gpu_memory_utilization=0.9, 
                                #  max_model_len=60000, 
                                max_model_len=32768, 
                                limit_mm_per_prompt={"image": 2})

    template_type = get_default_template_type(model_type)
    template = get_template(template_type, llm_engine.hf_tokenizer)

    llm_engine.generation_config.max_new_tokens = 4096
    return llm_engine, template

def run_model(llm_engine, template, query, images, history=None):
    # images = ["/data2/datasets/drivelm/data/coda/test/images/2520.jpg",'/data2/datasets/drivelm/data/coda/test/images/0001.jpg']
    images = convert_to_base64(images=images)['images']
    
    request_list = [{'query': query, 'history': history, 'images': images}]

    resp_list = inference_vllm(llm_engine, template, request_list)

    return resp_list[0]['response']