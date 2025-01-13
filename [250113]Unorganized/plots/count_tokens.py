import tiktoken
import json
from tqdm import tqdm
# https://blog.csdn.net/qq_45476428/article/details/134796584 (code reference)
# https://platform.openai.com/tokenizer (Official platform)

def read_jsonl(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return [json.loads(line.strip()) for line in f]

def num_tokens_from_messages(message, model="gpt-4o-mini"):
    """Return the number of tokens used by a list of messages."""
    try:
        encoding = tiktoken.encoding_for_model(model)
    except KeyError:
        print("Warning: model not found. Using cl100k_base encoding.")
        encoding = tiktoken.get_encoding("cl100k_base")
    
    return len(encoding.encode(message))



def demo():
    example_messages = ["You are a helpful, pattern-following assistant that translates corporate jargon into plain English."]

    for model in [
        "gpt-4o-mini",
        "gpt-3.5-turbo-0613"
        ]:
        print(model)
        
        print(f"{num_tokens_from_messages(example_messages, model)} prompt tokens counted by num_tokens_from_messages().")
        print()

def main(count_jsonl_path):
    image_tokens = 256*6
    data_info = read_jsonl(count_jsonl_path)
    num_tokens_list = []

    for data in tqdm(data_info):
        num_tokens_list.append(num_tokens_from_messages(data['query']))
    print(sum(num_tokens_list), len(num_tokens_list))
    print(sum(num_tokens_list)/len(num_tokens_list))
    print(sum(num_tokens_list)/len(num_tokens_list)+image_tokens)
    
    
    
if __name__ == "__main__":
   count_jsonl_path = "/data2/datasets/drivelm/data/qas/original/original_test_general_perception.jsonl" 
   main(count_jsonl_path)
#    count_jsonl_path = "/data2/datasets/drivelm/data/qas/original/original_train_general_perception.jsonl" 
#    main(count_jsonl_path)
