from flask import Flask, render_template, send_file, abort, request
import json
import os

app = Flask(__name__)

# 加载 JSONL 数据
def load_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    return data

# 加载五个JSONL文件的数据
ssim_data = load_data('/data2/datasets/drivelm/similarity_rank/test_tv_mse_top_10.jsonl')
mse_data = load_data('/data2/datasets/drivelm/similarity_rank/test_tv_ssim_top_10.jsonl')
hash_data = load_data('/data2/datasets/drivelm/similarity_rank/bert_embedding/test_t_v_top_k.jsonl')
bert_data = load_data('/data2/datasets/drivelm/similarity_rank/inetrnvl/test_t_v_top_k.jsonl')
bert_ds_data = load_data('/data2/datasets/drivelm/similarity_rank/241110_rank/test_t_v_top_k.jsonl')

# 用于提供绝对路径图像的路由
@app.route('/image/<path:filename>')
def serve_image(filename):
    full_path = os.path.join('/data2/datasets/drivelm/data/coda', filename)
    if os.path.exists(full_path):
        return send_file(full_path)
    else:
        abort(404)

@app.route('/')
def index():
    query_index = int(request.args.get('query_index', 0))
    max_index = min(len(ssim_data), len(mse_data), len(hash_data), len(bert_data), len(bert_ds_data)) - 1

    # 限制索引在范围内
    if query_index < 0:
        query_index = 0
    elif query_index > max_index:
        query_index = max_index

    # 获取当前索引的数据
    query_ssim = ssim_data[query_index]
    query_mse = mse_data[query_index]
    query_hash = hash_data[query_index]
    query_bert = bert_data[query_index]
    query_bert_ds = bert_ds_data[query_index]

    # 获取各个JSONL文件的 query 图像和前两个 base 图像路径
    def get_image_data(query_data):
        query_image = query_data['query'].replace('/data2/datasets/drivelm/data/coda/', '')
        topk_images = [
            {
                "similarity_score": img["similarity_score"],
                "base": img["base"].replace('/data2/datasets/drivelm/data/coda/', '')
            }
            for img in query_data['topk'][:2]
        ]
        return query_image, topk_images

    query_image_ssim, topk_images_ssim = get_image_data(query_ssim)
    query_image_mse, topk_images_mse = get_image_data(query_mse)
    query_image_hash, topk_images_hash = get_image_data(query_hash)
    query_image_bert, topk_images_bert = get_image_data(query_bert)
    query_image_bert_ds, topk_images_bert_ds = get_image_data(query_bert_ds)

    return render_template(
        'index.html', 
        query_index=query_index, max_index=max_index,
        query_image_ssim=query_image_ssim, topk_images_ssim=topk_images_ssim,
        query_image_mse=query_image_mse, topk_images_mse=topk_images_mse,
        query_image_hash=query_image_hash, topk_images_hash=topk_images_hash,
        query_image_bert=query_image_bert, topk_images_bert=topk_images_bert,
        query_image_bert_ds=query_image_bert_ds, topk_images_bert_ds=topk_images_bert_ds
    )

if __name__ == '__main__':
    app.run(debug=True, port=5001)
