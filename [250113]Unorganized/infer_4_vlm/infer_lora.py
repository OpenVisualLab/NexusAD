import argparse
import json
from tqdm import tqdm
import os

from vl2_8b import init_vl2_model, run_vl2_model


def load_json_lines(json_path):
    with open(json_path, 'r') as f:
        return json.load(f)

def load_jsonl_file(jsonl_path):
    with open(jsonl_path, 'r') as infile:
        return [json.loads(line) for line in infile]
    

def write_responses_to_file(outf, responses, data_batch):
    for response, data in zip(responses, data_batch):
        data['prediction'] = response
        outf.write(json.dumps(data) + '\n')


def main(json_path, out_json_path, model_dir=None, ckpt_dir=None):
    
    model, template = init_vl2_model(ckpt_dir=ckpt_dir, max_new_tokens=4096, model_path=model_dir)
    
    print(f'Processing: {json_path}')
    print(f'Saving in: {out_json_path}')
    
    json_info = load_jsonl_file(json_path)
    
    results = []
    for info in tqdm(json_info):
        qa_id = info['id']
        image_paths = info['image']
        question = info['query']
        gt_answer = info['response']

        # question='你收到了几张图片?并描述所有图片' 
        answer = run_vl2_model(model, template, question, image_paths)
        
        new_info = {'id': qa_id, 'image': image_paths, 'question': question, 'gt_answer': gt_answer, 'answer':answer}
        results.append(new_info)
    
    with open(out_json_path, 'w', encoding='utf-8') as outf:  
       json.dump(results, outf, indent=4, ensure_ascii=False)   
    
    
    print(f'Finished writing to {out_json_path}')


if __name__ == "__main__":
    os.environ['INPUT_SIZE'] = '448'
    os.environ['MAX_NUM'] = '12'
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"
    
    parser = argparse.ArgumentParser(description='Process a subset and task of the CODA dataset using a specified model.')
    # parser.add_argument('--data_root', type=str, default='', help='Root directory of datasets')
    
    parser.add_argument('--json_path', type=str, default='/data2/datasets/drivelm/val_data/swift_v1_1_val_nus.jsonl', help='Path to the input JSON file')
    parser.add_argument('--out_json_path', type=str, default='/data1/nemo/projects/241009_DriveLM/241121_test3/submission_4.json', help='Path to the output JSON file')
    
    parser.add_argument('--model_dir', type=str, default='/data1/nemo/projects/pre_weights/InternVL2-26B', help='Directory of model weights')
    parser.add_argument('--ckpt_dir', type=str, default='/data2/datasets/drivelm/output/internvl2-26b/v11-20241117-140830/checkpoint-3681', help='Directory of model weights')

    args = parser.parse_args()
    main(args.json_path, args.out_json_path, args.model_dir, args.ckpt_dir)