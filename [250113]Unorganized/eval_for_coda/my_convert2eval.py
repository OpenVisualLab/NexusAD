import os
import json
import argparse
from tqdm import tqdm


def convert(prediction_path, label_dir="", subset="", api_info=None):
    if label_dir=="":
        label_dir="/data2/datasets/drivelm/data/coda/CODA-LM"
        
    if subset =="":
        subset = 'Test'
    
    reference_path = os.path.join(label_dir, subset)
        
    # --- load ground truth ---
    reference_data = {}
    for each in tqdm(os.listdir(reference_path), desc='Loading GT'):
        if not each.endswith('.json'):
            continue
        with open(os.path.join(reference_path, each), "r", encoding='utf-8') as f:
            each_data = json.load(f)
        reference_data[each[:-5]] = each_data

    # --- load prediction --- 
    prediction_data = [json.loads(q) for q in open(os.path.expanduser(prediction_path), "r", encoding='utf-8')]
    for each in tqdm(prediction_data, desc='Processing prediction'):
        
        if api_info["eval_key"] == "prediction":
            image_name = each['images'][0].split('/')[-1]
        elif api_info["eval_key"] == 'response':
            image_name = each['image'].split('/')[-1]
        else:
            image_name = each['image'].split('\\')[-1]
        
        json_name = 'test_' + image_name.split('_')[0]
        object_id = image_name.split('_')[-1][:-4]
        assert object_id.isdigit()
        each['label_name'] = reference_data[json_name]['region_perception'][object_id]['category_name']
        # each['image'] = each['images'][0]
        

    # --- save converted prediction ---
    save_path = prediction_path[:-6] + '_w_label.jsonl'
    with open(save_path, "w", encoding='utf-8') as file:
      for each in prediction_data:
          file.write(json.dumps(each) + "\n")
    return save_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference_path", type=str, default="/data1/nemo/projects/240922_AD/datasets/labels/CODA-LM/Local")
    parser.add_argument("--prediction_path", type=str, default="/data1/nemo/projects/240922_AD/output/out_jsons/240924/ori/v0_ft10_local_rp.jsonl")
    args = parser.parse_args()
    
    convert(args.reference_path, args.prediction_path)
    