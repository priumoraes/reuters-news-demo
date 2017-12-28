import inference
import os
import sys
import json
from flask import Flask, jsonify, request


app = Flask(__name__, static_url_path='')
# app._static_folder = '/Users/priscillamoraes/git/news_demo/lr'

@app.route('/')
def welcome():
    return app.send_static_file('index.html')

@app.route('/tag', methods=['POST'])
def tag():

    create_json = request.get_json(force=True)
    #print(json.dumps(create_json))
    article, tags, title, keywords = inference.infer(create_json['url'])
    tst_json = {}
    tst_json["tags"] = tags
    tst_json["article"] = article
    tst_json["title"] = title
    tst_json["keywords"] = keywords
    return jsonify(tst_json)


port = os.getenv('PORT', '8000')
if __name__ == "__main__":
    app.run(host='127.0.0.1', port=int(port))
