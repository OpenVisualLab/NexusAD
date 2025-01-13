import json
import os
from tqdm import tqdm

def process_json_files(data_dir, label_dir, qa_dir, subset, task, for_name):
    json_path_in = os.path.join(label_dir, f"{subset}/vqa_anno/{task}.jsonl")
    json_path_out = os.path.join(qa_dir, f"{for_name}/{for_name}_{subset.lower()}_{task}.jsonl")

    print(f"Input Path: {json_path_in}")
    print(f"Output Path: {json_path_out}")

    with open(json_path_in, 'r') as f:
        content = f.read()
    lines = content.splitlines()

    with open(json_path_out, 'w') as outf:
        for line in tqdm(lines):
            line = json.loads(line.strip())
            # print(line)
            name = line['image'].split('/')[0] + '_' + line['image'].split('/')[-1].split('.')[0]
            new_line = {
                'question_id': line['question_id'],
                'query': line['question'],
                'response': line['answer'],
                'images': [os.path.join(data_dir, line['image'])]
            }

            outf.write(json.dumps(new_line) + '\n')

def process_json_with_structure(data_dir, label_dir, qa_dir, subset, task, for_name):
    json_path_in = os.path.join(label_dir, f"{subset}/vqa_anno/{task}.jsonl")
    json_path_out = os.path.join(qa_dir, f"{for_name}/{for_name}_{subset.lower()}_{task}.jsonl")

    print(f"Input Path: {json_path_in}")
    print(f"Output Path: {json_path_out}")

    with open(json_path_in, 'r') as f:
        content = f.read()
    lines = content.splitlines()

    with open(json_path_out, 'w') as outf:
        for line in tqdm(lines):
            line = json.loads(line.strip())
            
            image_parts = line['image'].split('/')
            prefix = image_parts[0]
            base_name = image_parts[-1].split('.')[0]

            if task == 'region_perception':
                b_name, _, object_id = base_name.split('_')
                name = prefix + '_' + b_name
            else:
                name = prefix + '_' + base_name

                
            with open(os.path.join(label_dir, subset, name + '.json'), 'r') as f:
                content2 = f.read()
            content2 = json.loads(content2)
            struct_info = content2["general_perception"]
            
            if task == 'general_perception':
                pass
            elif task == 'region_perception':
                struct_info = {}
                struct_info['region perception'] = content2['region_perception'][str(object_id)]['description and explanation']
                struct_info['category_name'] = content2['region_perception'][str(object_id)]['category_name']
            elif task == 'driving_suggestion':
                _ = struct_info.pop('description and explanation')
                struct_info['driving_suggestion'] = content2['driving_suggestion']
                
            new_line = {
                'question_id': line['question_id'],
                'query': line['question'],
                'response': str(struct_info),
                'images': [os.path.join(data_dir, line['image'])]
            }

            outf.write(json.dumps(new_line) + '\n')

if __name__ == "__main__":
    data_dir = '/data2/datasets/drivelm/data/coda'
    label_dir = os.path.join(data_dir, "CODA-LM")
    qa_dir = '/data2/datasets/drivelm/data/qas'

    subsets = ['Mini', 'Test', 'Train', 'Val']
    tasks = ['general_perception', 'region_perception', 'driving_suggestion']
    # for_name = "original"

    # for subset in subsets:
    #     for task in tasks:
    #         process_json_files(data_dir, label_dir, qa_dir, subset, task, for_name)

    
    print('---'*100)
    
    for_name = "new"
    for subset in subsets:
        for task in tasks:
            # if task == 'region_perception':
            #     continue
            process_json_with_structure(data_dir, label_dir, qa_dir, subset, task, for_name)
