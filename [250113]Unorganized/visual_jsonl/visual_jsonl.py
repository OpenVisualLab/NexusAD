from flask import Flask, render_template, request, send_from_directory, redirect, url_for
import json
import os
import re
import cv2
import shutil

app = Flask(__name__)

# Load the JSONL data
jsonl_file = '/data1/nemo/projects/241009_DriveLM/drivelm_data_demo/test_swift.jsonl'  # Make sure the JSONL file is in the same directory or provide the correct path
data = []

with open(jsonl_file, 'r') as f:
    for line in f:
        data.append(json.loads(line))

# Extract unique scene, keyframe, and question indices for the selection form
scenes = set()
keyframes = set()
questions = set()
for item in data:
    scene, keyframe, question = item['id'].split('_')
    scenes.add(scene)
    keyframes.add(keyframe)
    questions.add(question)

scenes = sorted(scenes)
keyframes = sorted(keyframes)
questions = sorted(questions)

@app.route('/')
def index():
    item_id = request.args.get('item_id')
    item = next((d for d in data if d.get('id') == item_id), None) if item_id else None
    if not item and data:
        item = data[0]
    image_path = None
    if item:
        image_path = visualize_image(item)
    return render_template('index.html', data=data, item=item, image_path=image_path, scenes=scenes, keyframes=keyframes, questions=questions)

@app.route('/select', methods=['POST'])
def select_item():
    scene_id = request.form.get('scene_id')
    keyframe_id = request.form.get('keyframe_id')
    question_index = request.form.get('question_index')
    item_id = f"{scene_id}_{keyframe_id}_{question_index}"
    return redirect(url_for('index', item_id=item_id))

@app.route('/images/<path:filename>')
def serve_image(filename):
    image_directory = '/data1/nemo/projects/241009_DriveLM/drivelm_data_demo/data/nuscenes/stitched'
    return send_from_directory(image_directory, filename)

@app.route('/visualized_images/<path:filename>')
def serve_visualized_image(filename):
    temp_dir = '/data1/nemo/projects/241009_DriveLM/NexusAD/visual_jsonl/temp'
    return send_from_directory(temp_dir, filename)

@app.route('/next/<string:item_id>')
def next_item(item_id):
    current_index = next((index for (index, d) in enumerate(data) if d['id'] == item_id), 0)
    next_index = (current_index + 1) % len(data)
    return redirect(url_for('index', item_id=data[next_index]['id']))

@app.route('/previous/<string:item_id>')
def previous_item(item_id):
    current_index = next((index for (index, d) in enumerate(data) if d['id'] == item_id), 0)
    previous_index = (current_index - 1) % len(data)
    return redirect(url_for('index', item_id=data[previous_index]['id']))

def visualize_image(item):
    image_path = f"/data1/nemo/projects/241009_DriveLM/drivelm_data_demo/data/nuscenes/stitched/{item['image'][0].split('/')[-1]}"
    image = cv2.imread(image_path)
    if image is None:
        return None

    # Parse coordinates from the query if available
    query_coordinates = re.findall(r'<c\d+,[^,]+,(\d+),(\d+)>', item['query'])
    response_coordinates = re.findall(r'<c\d+,[^,]+,(\d+),(\d+)>', item['response'])
    all_coordinates = query_coordinates + response_coordinates

    for (x, y) in all_coordinates:
        x, y = int(x), int(y)
        cv2.circle(image, (x, y), radius=10, color=(0, 0, 255), thickness=-1)

    # Save the visualized image to a temporary directory
    temp_dir = '/data1/nemo/projects/241009_DriveLM/NexusAD/visual_jsonl/temp'
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    temp_image_path = os.path.join(temp_dir, f"{item['id']}.png")
    cv2.imwrite(temp_image_path, image)

    return f"/visualized_images/{item['id']}.png"

if __name__ == '__main__':
    app.run(debug=True)
