import os

from flask import Flask, request, jsonify, render_template
from flask_cors import CORS, cross_origin

from TEXTSUMMARIZATION import BART

os.putenv('LANG', 'en_US.UTF-8')
os.putenv('LC_ALL', 'en_US.UTF-8')

app = Flask(__name__)
CORS(app)

@app.route("/", methods=['GET'])
@cross_origin()
def home():
    return render_template('INDEX.html')

@app.route("/predict", methods=['POST'])
@cross_origin()
def predictRoute():
    data = request.json['data']
    textsummarize = BART()
    result = textsummarize.input_id(data)
    print(result)
    return jsonify({"text" : result })



if __name__ == "__main__":
    #app.run(host='0.0.0.0', port=port)
    app.run(host='127.0.0.1', port=5000, debug=True)