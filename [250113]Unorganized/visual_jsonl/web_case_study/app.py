from flask import Flask, render_template, request
import random
from flask import Flask, send_file, abort
import os

app = Flask(__name__)

from get_scores import get_jsonl_info

data_groups  = get_jsonl_info()
# data_groups = []
# for i in range(500):
#     data_groups.append({
#         'image_path': 'static/images/sample_image.png',  # 使用相对路径
#         'models': {
#             'Human': [f'Human Object1 description {i*2 + 1}', f'Human Object2 description {i*2 + 2}'],
#             'Model-A': [
#                 (f'Model-A Object1 description {i*2 + 1}', random.randint(0, 10)),
#                 (f'Model-A Object2 description {i*2 + 2}', random.randint(0, 10))
#             ],
#             'Model-B': [
#                 (f'Model-B Object1 description {i*2 + 1}', random.randint(0, 10)),
#                 (f'Model-B Object2 description {i*2 + 2}', random.randint(0, 10))
#             ],
#             'Model-C': [
#                 (f'Model-C Object1 description {i*2 + 1}', random.randint(0, 10)),
#                 (f'Model-C Object2 description {i*2 + 2}', random.randint(0, 10))
#             ],
#             'Model-D': [
#                 (f'Model-D Object1 description {i*2 + 1}', random.randint(0, 10)),
#                 (f'Model-D Object2 description {i*2 + 2}', random.randint(0, 10))
#             ],
#         }
#     })


# 根目录，包含不同的子目录，例如 "test" 和 "val"
BASE_DIRECTORY = '/data2/datasets/drivelm/data/coda'

@app.route('/data2/datasets/drivelm/data/coda/<set_type>/images/<path:filename>')
def serve_image(set_type, filename):
    # 构建完整文件路径
    file_path = os.path.join(BASE_DIRECTORY, set_type, 'images', filename)

    # 检查文件是否存在
    if os.path.isfile(file_path):
        return send_file(file_path)
    else:
        # 文件不存在时返回 404 错误
        abort(404)
        

@app.route('/')
def index():
    """显示分页数据"""
    page = request.args.get('page', 1, type=int)
    per_page = 1  # 每页显示10组数据
    total_pages = (len(data_groups) + per_page - 1) // per_page
    start = (page - 1) * per_page
    end = start + per_page
    paginated_data = data_groups[start:end]
    
    return render_template('index.html', data_groups=paginated_data, current_page=page, total_pages=total_pages)



if __name__ == '__main__':
    app.run(debug=True, port=1805)
