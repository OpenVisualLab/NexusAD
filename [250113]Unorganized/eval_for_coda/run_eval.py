import os


from my_convert2eval import convert
from evaluation.stage1_eval_batch import stage_1_eval
from evaluation.stage2_eval_batch import stage_2_eval
from evaluation.stage3_eval_batch import stage_3_eval


tasks = ['general_perception', 'region_perception', 'driving_suggestion']


def my_eval(prediction_path, save_path=None, task=None, pred_key=None):
    api_info = {
        "num_workers":50,
        "model_name": "gpt-4o-mini",
        # "model_name": "gpt-4o",
        # "api_key": "sk-xxxx",
        # "api_base_url": "https://api.bianxie.ai/v1",
        "api_key": "sk-xxxx",
        "api_base_url": "https://api.gpt.ge/v1/",
        "retry_attempts": 1,
        "eval_key": "prediction" #  "prediction" "response" "answer"
    }
    
    label_dir = "/data2/datasets/drivelm/data/coda/CODA-LM"
    subset = "Test" 
    reference_path = os.path.join(label_dir, subset)
    save_path = prediction_path.split('.')[0]  if save_path is None else save_path  # +'-4o'
    
    # ------- Eval -------
    # TODO temp for quick use
    if tasks[0] in prediction_path or '_gp_' in prediction_path:
        print("Evaluating on", tasks[0])
        stage_1_eval(reference_path, prediction_path, save_path, api_info)
    elif tasks[2] in prediction_path or '_ds_' in prediction_path:
        print("Evaluating on", tasks[2], "\nFile is", prediction_path)
        stage_2_eval(reference_path, prediction_path, save_path, api_info)
    elif tasks[1] in prediction_path or '_rp_' in prediction_path:
        print("Evaluating on", tasks[1], "\nFile is ", prediction_path)
        pred4eval_path = convert(prediction_path, api_info=api_info)
        stage_3_eval(reference_path, pred4eval_path, save_path, api_info)
    else:
        print("CHECK the JOSN path! ", prediction_path)
        
    
if __name__ == "__main__":
    os.environ['https_proxy'] = 'http://10.16.0.81:8888'   

    list_list = [
                    '/data2/datasets/drivelm/output4/res/run_ds_mse.jsonl',
                    '/data2/datasets/drivelm/output4/res/run_ds_ssim.jsonl',
                    '/data2/datasets/drivelm/output4/res/run_ds_bert.jsonl',
                    '/data2/datasets/drivelm/output4/res/run_ds_lvlm.jsonl',
                    
                    
                 ]      
    for prediction_path in list_list:
        my_eval(prediction_path)
    
    