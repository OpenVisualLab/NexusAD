
import json
from tqdm import tqdm
import re
import os
import ast

def extract_description_and_explanation(data_str, start_key=None):
    
    start_key = "'description and explanation': " if start_key==None else start_key
    
    start_index = data_str.find(start_key)

    if start_index == -1:
        print(data_str)
        return data_str
    
    start_index += len(start_key)

    if data_str[start_index] == '"':
        quote_char = '"'
    elif data_str[start_index] == "'":
        quote_char = "'"
    else:
        print(data_str)
        print('-'*100)
        return None

    end_index = data_str.find(quote_char, start_index + 1)

    # description_and_explanation = data_str[start_index + 1:end_index]
    description_and_explanation = data_str[start_index + 1:]

    return description_and_explanation[:-2]



def replace_prediction_in_jsonl(input_file, output_file):
    
    if '_rp_' in input_file:
        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
            lines = infile.readlines()
            for line in tqdm(lines, desc="Processing", unit="line"):
                data = json.loads(line)
                pred = ast.literal_eval(data['prediction'])
                data['prediction'] = pred.get('region perception')
                if pred.get('region perception') == "":
                    print(data)
                
                json.dump(data, outfile, ensure_ascii=False)
                outfile.write('\n')
    else:
        if '_gp_' in input_file:
            start_key = "'description and explanation': "
        elif '_rp_' in input_file:
            # start_key = "'region perception description and explanation':"
            pass
        elif '_ds_' in input_file:
            start_key = "'driving_suggestion': "
        else:
            print('CHECK PATH!', input_file)
        
        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
            lines = infile.readlines()
            for line in tqdm(lines, desc="Processing", unit="line"):
                data = json.loads(line)
                
                data['prediction'] = extract_description_and_explanation(data['prediction'], start_key)
                
                json.dump(data, outfile, ensure_ascii=False)
                outfile.write('\n')

if __name__ == "__main__":
    input_files = [
        '/data2/datasets/drivelm/output4/res_tab_retrieval/run_ds_mse.jsonl',
                    '/data2/datasets/drivelm/output4/res_tab_retrieval/run_ds_ssim.jsonl'
        
        
        ]

    for input_file in input_files:
        output_file = os.path.join(os.path.dirname(input_file), 'post_' + os.path.basename(input_file))

        replace_prediction_in_jsonl(input_file, output_file)

        print(f"Processing complete. The modified data has been saved to {output_file}")
