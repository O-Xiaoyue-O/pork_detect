from flask import Flask, jsonify, request, Response
import os
import random
import string
from pathlib import Path
import sys
from main import api_detect
from utils.general import LOGGER
from shutil import move
from flask_cors import CORS
from llm import llm
import json
import time
from threading import Thread

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLO root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
HTTP_ROOT = Path(r'C:\xampp\htdocs\results') # HTTP root directory

# check file is allowed
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

LOGGER.info('預備模型')
detected = api_detect()
LOGGER.info('準備llm')
llm_instance = llm(system_prompt="使用繁體中文回應")
app = Flask(__name__)
# Enable CORS
CORS(app)
# Set the upload folder
UPLOAD_FOLDER = 'uploads/'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/api', methods=['GET', 'POST'])
def api():
    if request.method == 'GET':
        data = {'status': '403', 'message': 'Forbidden', 'data': ''}
        return jsonify(data)
    elif request.method == 'POST':
        # Check if the POST request has the file part and option data
        if 'file' not in request.files:
            return jsonify({'status': '400', 'message': 'No file part in the request', 'data': ''})
        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': '400', 'message': 'No file part in the request', 'data': ''})
        if not file or not allowed_file(file.filename):
            return jsonify({'status': '400', 'message': 'File type not allowed', 'data': ''})
        # if 'option' not in request.form:
        #     return jsonify({'status': '400', 'message': 'No option part in the request', 'data': ''})
        # Get the image from the POST request
        # option = request.form['option']
        # Generate a random string for the image name
        dir_random_name = ''.join(random.choices(string.ascii_letters + string.digits, k=30))
        # Create a directory with the random name
        os.makedirs(ROOT / 'api_save_results' / dir_random_name)
        # Rename the image to 'image.png'
        new_image_name = str(ROOT / 'api_save_results' / dir_random_name / 'image.') + file.filename.split('.')[-1]
        save_dir = ROOT / 'api_save_results' / dir_random_name
        # Save the image in the directory
        file.save(new_image_name)
        results = detected.run(Path(new_image_name), save_dir)
        # move the results to the HTTP directory
        move(save_dir, HTTP_ROOT / dir_random_name)
        if len(results) == 0:
            print('No pork object detected')
            return jsonify({'status': '200', 'message': 'No pork object detected', 'data': 'None'})
        else:
            print('Results:', results)
            return jsonify({'status': '200', 'message': 'success', 'data': results, 'url': dir_random_name})

@app.route('/llm', methods=['GET', 'POST'])
def llm_api():
    if request.method == 'GET':
        get_data = request.args
        if 'text' not in get_data:
            json_data = "{'status': '400', 'message': 'No text part in the request', 'data': ''}"
            res = Response(json_data, content_type='application/json; charset=utf-8')
            res.status_code = 400
            return res
        text = get_data['text']
        
        if text == '':  
            json_data = "{'status': '400', 'message': 'No text part in the request', 'data': ''}"
            res = Response(json_data, content_type='application/json; charset=utf-8')
            res.status_code = 400
            return res
        
        if 'api_token' not in get_data:
            api_token = ''.join(random.choices(string.ascii_letters + string.digits, k=30))
        else:
            api_token = get_data['api_token']
        
        output, api_token = llm_instance.chat(text, api_token)
        print(f"input: {text}\noutput: {output}\napi_token: {api_token}")
        json_data = {'status': '200', 'message': 'success', 'data': [{"output": output, "api_token": api_token}]}
        res = jsonify(json_data)
        res.headers['Access-Control-Allow-Origin'] = '*'
        # l = str(len(json_data))
        # res.headers['Content-Length'] = l
        # print(json_data, l)
        return res
        
        
    if request.method == 'POST':
        if 'text' not in request.form:
            json_data = "{'status': '400', 'message': 'No text part in the request', 'data': ''}"
            res = Response(json_data, content_type='application/json; charset=utf-8')
            res.status_code = 400
            return res
        text = request.form['text']
        
        if text == '':
            json_data = "{'status': '400', 'message': 'No text part in the request', 'data': ''}"
            res = Response(json_data, content_type='application/json; charset=utf-8')
            res.status_code = 400
            return res
        
        if 'api_token' not in request.form:
            api_token = ''.join(random.choices(string.ascii_letters + string.digits, k=30))
        else:
            api_token = request.form['api_token']

        output, api_token = llm_instance.chat(text, api_token)
        print(f"input: {text}\noutput: {output}\napi_token: {api_token}")
        json_data = {'status': '200', 'message': 'success', 'data': [{"output": output, "api_token": api_token}]}
        res = jsonify(json_data)
        res.headers['Access-Control-Allow-Origin'] = '*'
        # l = str(len(json_data))
        # res.headers['Content-Length'] = l
        return res

@app.route('/chat_last_history', methods=['GET'])
def chat_last_history():
    get_data = request.args
    if 'api_token' not in get_data:
        json_data = "{'status': '400', 'message': 'No api_token part in the request', 'data': ''}"
        res = Response(json_data, content_type='application/json; charset=utf-8')
        res.status_code = 400
        return res 
    api_token = get_data['api_token']
    if os.path.exists(f"{llm_instance.save_dir}/{api_token}.json"):
        with open(f"{llm_instance.save_dir}/{api_token}.json", "r", encoding="utf-8") as f:
            messages = json.load(f)
    else:
        json_data = "{'status': '404', 'message': 'api_token not found', 'data': ''}"
        res = Response(json_data, content_type='application/json; charset=utf-8')
        res.status_code = 404
        return res
    
    return jsonify({'status': '200', 'message': 'success', 'data': messages[-1]["content"]})

def auto_remove():
    target = str(ROOT / 'llm_save_chat')
    days_threshold = 1
    basename = os.getcwd()
    
    while True:
        now = time.time()
        time_threshold = now - (days_threshold * 24 * 60 * 60)
        del_cont = 0
        for filename in os.listdir(target):
            file_path = os.path.join(target, filename)
            if os.path.isfile(file_path):
                file_mtime = os.path.getmtime(file_path)
                if file_mtime < time_threshold:
                    os.remove(file_path)
                    del_cont += 1
        
        print(f"Deleted {del_cont} files")
        time.sleep(60)
        

if __name__ == '__main__':
    Thread(target=auto_remove).start()
    app.run(debug=True, port=5000, host='0.0.0.0')