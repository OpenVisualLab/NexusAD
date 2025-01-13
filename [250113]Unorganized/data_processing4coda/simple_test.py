import json


def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line) for line in f]
    

def save_jsonl(file_path, file_data):
    with open(file_path, 'w') as f:
        for item in file_data:
            f.write(json.dumps(item) + '\n')
            
            
def fusion_gp_ds():
    gp_path = '/data2/datasets/drivelm/output2/nexusad/241027_structure/post_struct_vl_8b_test_gp_ft_4.jsonl'
    ds_path = '/data2/datasets/drivelm/output2/nexusad/241027_structure/post_struct_vl_8b_test_ds_ft_4.jsonl'
    save_path = '/data2/datasets/drivelm/241105/struct_ds_1111.jsonl'
    
    gp_info = read_jsonl(gp_path)
    ds_info = read_jsonl(ds_path)
    
    gp_ds = []
    for gp, ds in zip(gp_info, ds_info):
        new = {'general perception': gp['prediction'],
               'driving suggestion': ds['prediction']
               }
        ds['prediction'] = str(new)
        gp_ds.append(ds)
    
    save_jsonl(save_path, gp_ds)
        
        
if __name__ == "__main__":
    fusion_gp_ds()